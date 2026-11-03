#!/usr/bin/env python3
"""tools/jev_distill.py — distil a small base model into a JEV decision expert, no backprop.

Route B of the small-JEV program: a small base (e.g. Qwen3-0.6B, ~0.5B class) plus a
fitted decision head is a "JEV" in exactly the sense jev_decide.py serves: the head is
the verbalizer token rows of the lm_head plus a per-verbalizer bias and a per-kind
temperature, and NONE of that needs gradient training of the backbone —

    1. cases   — the 27B writes decision cases from each KB chunk, one per target team
                 (balanced labels), in the two live domains (ecom-support, cloud-ops).
    2. teacher — the production JEV-35B gateway (/decide, per-request "kb") labels every
                 case; its argmax (weighted by its confidence) is the distillation target.
    3. student — the small base answers the same prompts (jev_decide's exact format, KB
                 context included); the label tokens are forced into the top-logprobs by
                 a +100 logit bias, exactly the read-out trick jev_decide uses.
    4. fit     — per-verbalizer bias + choice temperature, full-batch gradient descent on
                 the softmax cross-entropy against the teacher labels (numpy, like the
                 expert-factory router head).
    5. emit    — a model dir (decision_head.json with the STUDENT's verbalizer ids and the
                 fitted bias, calibration.json with the fitted temperature) that drops
                 straight into `jev_decide.py --model-dir ... --serve`.

    python3 tools/jev_distill.py --student http://127.0.0.1:8972 \
        --teacher http://127.0.0.1:5500 --writer http://127.0.0.1:8902 \
        --teacher-head ../models/JEV-35B/adapter_vllm/decision_head.json \
        --kb-dir ../kb --domains ecom-support cloud-ops \
        --per-team 5 --out ../models/JEV-mini-06B
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 7
HOLDOUT = 0.25
ITERS = 600
LR = 0.3
L2 = 1e-3

DOMAINS = {  # domain -> (options, per-request kb name)
    "ecom-support": (["billing", "shipping", "technical support", "account security"], "ecom-support"),
    "cloud-ops": (["gpu-node", "scheduler", "network", "billing"], "cloud-ops"),
}


def http_json(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(base, messages, max_tokens=200, temperature=0.4):
    out = http_json(f"{base}/v1/chat/completions",
                    {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                     "chat_template_kwargs": {"enable_thinking": False}})
    return out["choices"][0]["message"]["content"] or ""


# ------------------------------------------------------------------ 1. cases

CASE_PROMPT = (
    "你是工单测试工程师。根据下面的资料,写一条应路由到指定团队的{kind}工单/告警消息,"
    "消息里要含资料中的具体事实(数字/设备名/流程名),但**不要直接出现团队名本身**,也不要写'路由到某团队'。\n"
    "只输出工单消息本身,一到两句。\n\n目标团队: {team}\n\n资料:\n{chunk}"
)

RETRY_PROMPT = (
    "刚才那条消息被误判给了「{wrong}」组而不是目标团队「{team}」。请重写:内容必须紧扣目标团队的职责特征"
    "(不要写成账号被盗/支付风控/安全验证类,除非那就是目标团队),一到两句,只输出消息本身。\n\n资料:\n{chunk}"
)


def gen_cases(writer, teacher, kb_dir, domains, per_team):
    """Rejection sampling: a case ships only when the 27B wrote it for the right team AND
    the JEV teacher routes it there — clean, balanced labels are worth more than volume."""
    cases = []
    for dom in domains:
        options, _ = DOMAINS[dom]
        chunks = []
        for p in sorted(Path(kb_dir, dom).rglob("*.md")):
            for para in re.split(r"\n\s*\n", p.read_text(encoding="utf-8")):
                if len(para.strip()) >= 60:
                    chunks.append(para.strip())
        kind = "客服" if dom == "ecom-support" else "运维"
        for team in options:
            got, tries = 0, 0
            ci = 0
            while got < per_team and tries < per_team * 3:
                ch = chunks[ci % len(chunks)]
                ci += 1
                tries += 1
                tmpl = CASE_PROMPT if tries <= per_team else RETRY_PROMPT
                msg = chat(writer, [{"role": "user",
                                     "content": tmpl.format(kind=kind, team=team, chunk=ch[:600],
                                                            wrong="")}], max_tokens=90, temperature=0.8)
                msg = " ".join(m.strip() for m in msg.strip().splitlines() if m.strip())[:220]
                if len(msg) < 15:
                    continue
                case = {"domain": dom, "options": options, "state": msg,
                        "question": "Which team should handle this ticket?", "gold_hint": team}
                t = teacher_decide(teacher, case)
                if t["label"] == options.index(team):
                    case["teacher"] = t
                    cases.append(case)
                    got += 1
                    print(f"  ✓ [{dom}/{team}] {msg[:50]}")
                else:
                    print(f"  ✗ [{dom}/{team}] -> {options[t['label']]}: {msg[:40]}")
    return cases


# ------------------------------------------------------------------ 2. teacher labels

def teacher_decide(gw, case):
    d = http_json(f"{gw}/decide", {
        "kind": "choice", "state": case["state"], "question": case["question"],
        "options": case["options"], "kb": [DOMAINS[case["domain"]][1]]})
    top = max(d["distribution"], key=lambda x: x["prob"])
    return {"label": case["options"].index(top["text"]), "weight": top["prob"]}


# ------------------------------------------------------------------ 3. student logprobs

def build_prompt(head, case, ctx):
    lines = [f"{chr(ord('A') + i)}) {t}" for i, t in enumerate(case["options"])]
    return (f"[kind] choice\n[state] {ctx}{case['state']}\n[question] {case['question']}\n"
            f"[options]\n" + "\n".join(lines) + "\n[decision]:")


def kb_context(kb_chunks, embedder_cache, case, top_k=4):
    """Grounding for the student, mirroring what the gateway will inject at serve time."""
    key = case["domain"]
    if key not in embedder_cache:
        from serve.knowledge import make_embedder
        emb = make_embedder("llama:http://127.0.0.1:8910")
        m = np.asarray(emb.embed([c["text"] for c in kb_chunks[key]]))
        embedder_cache[key] = (emb, m)
    emb, m = embedder_cache[key]
    q = emb.embed([f"{case['question']}\n{case['state']}"])[0]
    order = np.argsort(-(m @ q))[:top_k]
    return ("\n".join(f"[knowledge base {key} #{i + 1}]\n{kb_chunks[key][int(j)]['text']}"
                      for i, j in enumerate(order))
            + "\n[end of knowledge base context]\n")


def student_lps(base, case, prompt, ids, bias_hint):
    body = {"prompt": prompt, "max_tokens": 1, "temperature": 1.0,
            "top_p": 1.0, "top_k": 0, "min_p": 0.0, "logprobs": 20,
            "logit_bias": {str(i): 100 for i in ids}}
    out = http_json(f"{base}/v1/completions", body)
    content = (out["choices"][0].get("logprobs") or {}).get("content") or []
    if not content or not content[0].get("top_logprobs"):
        return None
    lps = {t["id"]: t["logprob"] for t in content[0]["top_logprobs"]}
    return [lps.get(i, -30.0) - 100.0 for i in ids]          # undo the +100 bias


# ------------------------------------------------------------------ 4. fit

def fit(all_lp, labels, weights, k_max):
    """Per-verbalizer bias (first k_max choice slots) + choice temperature. Logits are
    centered per case first: the bias must learn the label prior, not swallow a per-case
    offset the temperature would otherwise have to fight."""
    xs = []
    for lp in all_lp:
        x = np.asarray(lp, dtype=np.float64)
        xs.append(x - x.mean())
    b = np.zeros(k_max)
    log_t = np.zeros(())
    n = len(xs)
    for it in range(ITERS):
        t = math.exp(float(log_t))
        loss, gb, gt = 0.0, np.zeros(k_max), 0.0
        for x, (lab,), w in zip(xs, labels, weights):
            z = (x + b[:len(x)]) / t
            z -= z.max()
            e = np.exp(z)
            p = e / e.sum()
            loss += -w * math.log(max(p[lab], 1e-12))
            g = p.copy()
            g[lab] -= 1.0
            gb[:len(x)] += w * g / t
            gt += float(w * (g * (x + b[:len(x)])).sum() / t)
        gb = gb / n + L2 * b
        gt = gt / n
        b -= LR * gb
        log_t -= LR * gt * 0.1
        log_t = float(np.clip(log_t, math.log(0.2), math.log(5.0)))
    return b, math.exp(float(log_t)), loss / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True, help="small base llama-server (e.g. :8972)")
    ap.add_argument("--teacher", required=True, help="JEV-35B gateway (/decide, e.g. :5500)")
    ap.add_argument("--writer", required=True, help="chat model that writes cases (e.g. :8902)")
    ap.add_argument("--teacher-head", required=True, type=Path)
    ap.add_argument("--kb-dir", required=True, type=Path)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--per-team", type=int, default=5)
    ap.add_argument("--out", required=True, type=Path, help="student model dir (jev_decide --model-dir)")
    args = ap.parse_args()

    head = json.loads(args.teacher_head.read_text())
    verbalizers = head["slots"]["verbalizers"]               # e.g. [..., "A", "B", ...]
    lo, hi = head["slots"]["ranges"]["choice"]               # e.g. [8, 264)
    k_max = min(8, hi - lo)
    vs = verbalizers[lo:lo + k_max]                          # "A".."H"

    # student token ids for the verbalizers
    singles = {}
    for v in vs:
        t = http_json(f"{args.student}/tokenize", {"content": v})["tokens"]
        singles[v] = t[0]
    print(f"[head] choice slots {lo}..{lo + k_max} verbalizers {vs} -> student ids {[singles[v] for v in vs]}")

    print("[1/3] writing cases (rejection-sampled against the teacher)")
    cases = gen_cases(args.writer, args.teacher, args.kb_dir, args.domains, args.per_team)
    print(f"      {len(cases)} clean cases")

    print("[2/3] student logprobs (same prompt format the gateway will send)")
    kb_chunks = {
        d: [{"text": para.strip()} for p in sorted(Path(args.kb_dir, d).rglob("*.md"))
            for para in re.split(r"\n\s*\n", p.read_text(encoding="utf-8")) if len(para.strip()) >= 40]
        for d in args.domains}
    cache = {}
    ids_hint = [singles[v] for v in vs]
    keep = []
    for c in cases:
        n = len(c["options"])
        ctx = kb_context(kb_chunks, cache, c)
        prompt = build_prompt(None, c, ctx)
        lp = student_lps(args.student, c, prompt, ids_hint[:n], None)
        if lp is not None:
            c["lp"] = lp
            keep.append(c)
    print(f"  {len(keep)}/{len(cases)} usable")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(keep))
    n_test = max(3, int(len(keep) * HOLDOUT))
    test_idx = set(order[:n_test].tolist())
    train = [c for i, c in enumerate(keep) if i not in test_idx]
    test = [keep[i] for i in sorted(test_idx)]

    print("[3/3] fitting head (bias + temperature)")
    b, t, loss = fit([c["lp"] for c in train], [(c["teacher"]["label"],) for c in train],
                     [c["teacher"]["weight"] for c in train], k_max)
    acc = lambda cs: (sum(1 for c in cs
                          if int(np.argmax((np.asarray(c["lp"]) + b[:len(c["lp"])]) / t)) == c["teacher"]["label"])
                      / len(cs) if cs else 0.0)
    print(f"  T={t:.3f} train_CE={loss:.3f} agreement train={acc(train):.2f} holdout={acc(test):.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    # splice the fitted slice into a copy of the teacher head: the student's token ids
    # and the fitted biases for the choice slots; everything else rides along verbatim
    ids = list(head["verbalizer_ids"])
    bias = list(head["bias"])
    for i, v in enumerate(vs):
        ids[lo + i] = singles[v]
        bias[lo + i] = float(b[i])
    new_head = {"verbalizer_ids": ids, "bias": bias, "slots": head["slots"], "merged": True}
    (args.out / "adapter_vllm").mkdir(exist_ok=True)          # jev_decide's hardcoded subpath
    (args.out / "adapter_vllm" / "decision_head.json").write_text(json.dumps(new_head, ensure_ascii=False))
    cal_src = json.loads((args.teacher_head.parent.parent / "calibration.json").read_text())
    (args.out / "calibration.json").write_text(json.dumps(
        {"per_kind": {**cal_src["per_kind"], "choice": round(t, 4)}}))
    meta = {"base": args.student, "cases": len(cases), "usable": len(keep),
            "train": len(train), "holdout": len(test),
            "agreement": {"train": round(acc(train), 3), "holdout": round(acc(test), 3)},
            "temperature": round(t, 4), "built_at": time.time()}
    (args.out / "distill_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[done] {args.out}/adapter_vllm/decision_head.json + calibration.json")


if __name__ == "__main__":
    main()
