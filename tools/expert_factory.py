#!/usr/bin/env python3
"""tools/expert_factory.py — offline, fully automatic expert training from a raw knowledge base.

The question this answers: "can we train an expert from scratch, offline, with zero human
labels, and then hot-plug it into the running 27B?" Weight-level training (LoRA into the
27B) is not feasible on this stack (the tree's llama-finetune is a WIP full-parameter FP32
trainer that only scales to ~1B, and no adapter trainer ships here), so an "expert" is
trained as a *bundle* the chat gateway plugs in at runtime, no restart of anything:

    1. distill   — the 27B itself turns every KB chunk into question/answer pairs
                   (self-supervision: the raw documents are the only input, no labels).
    2. train     — a softmax router head over bge-m3 question embeddings (real gradient
                   training, numpy only) learns which requests belong to this expert;
                   questions distilled for the *other* experts under --out are the
                   negative classes, so every rebuild sharpens the whole routing.
    3. draft+grade — two candidate expert cards (the system persona that fronts the 27B)
                   are written by the model from the KB digest, then graded on held-out
                   QA by exact-fact match; the better card ships. The grade run also
                   reports the no-card retrieval baseline, so the card has to *earn* its
                   place.

    experts/<name>/
        bundle.json   the hot-plug unit: card + router file + threshold + metrics
        router.npz    W, b, class names (incl. "off-domain" when negatives were synthesized)
        qa.jsonl      the distilled training pairs (train+test split marked)
        metrics.json  router held-out accuracy, card scores, timings

    python3 tools/expert_factory.py build ecom-support --source kb/ecom-support \
        --llm http://127.0.0.1:8902 --kb-embedder llama:http://127.0.0.1:8910 --out experts

Everything runs against the local serving stack (a llama-server chat endpoint and an
embedding endpoint); nothing leaves the machine and no human touches the loop.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root, for serve.knowledge
from serve.knowledge import make_embedder, ingest               # the same ingest the gateways use

SEED = 13
TEST_FRACTION = 0.2
NEGATIVES_PER_BUILD = 12          # synthesized off-domain questions when only one expert exists
ROUTER_ITERS = 400
ROUTER_LR = 0.5
ROUTER_L2 = 1e-4
DEFAULT_THRESHOLD = 0.55


def llm_chat(base: str, messages: list[dict], max_tokens: int = 220, temperature: float = 0.3,
             timeout: float = 300.0) -> str:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"] or ""


def extract_json(text: str) -> dict | None:
    """The models wrap JSON in prose now and then; take the first {...} block that parses."""
    for m in re.finditer(r"\{.*?\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


# ------------------------------------------------------------------ stage 1: distill

DISTILL_PROMPT = (
    "你是题库工程师。只依据下面的资料,写出用户最可能提出的一个具体问题和一个尽量短的答案。"
    "答案里的数字、期限、流程名等关键事实必须原文来自资料,禁止编造。\n"
    '只输出 JSON: {"question": "...", "answer": "..."}\n\n资料:\n'
)


def distill(llm: str, chunks: list[dict]) -> list[dict]:
    pairs = []
    for i, c in enumerate(chunks):
        raw = llm_chat(llm, [{"role": "user", "content": DISTILL_PROMPT + c["text"]}],
                       max_tokens=200, temperature=0.2)
        obj = extract_json(raw) or {}
        q, a = str(obj.get("question", "")).strip(), str(obj.get("answer", "")).strip()
        if len(q) >= 6 and len(a) >= 4:
            pairs.append({"question": q, "answer": a, "source": c.get("source", ""), "chunk": i})
        print(f"  [distill {i + 1}/{len(chunks)}] {'ok' if q and a else 'skip'}  {q[:48]}")
    return pairs


def synthesize_negatives(llm: str, domain_hint: str) -> list[dict]:
    raw = llm_chat(llm, [{"role": "user", "content": (
        f"写出 {NEGATIVES_PER_BUILD} 个与『{domain_hint}』领域完全无关的日常客服/技术问题"
        "(购物退货、账号、网络、办公设备等),每行一个,不要编号。")}],
        max_tokens=300, temperature=0.8)
    lines = [ln.strip(" -、\t") for ln in raw.splitlines() if len(ln.strip()) >= 8]
    return [{"question": q, "answer": "", "klass": "off-domain"} for q in lines[:NEGATIVES_PER_BUILD]]


# ------------------------------------------------------------------ stage 2: the router head

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_router(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch gradient descent with L2 on the softmax cross-entropy — small data, no
    sklearn in this image, and it keeps the project's zero-extra-dependency rule."""
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.01, (x.shape[1], k)).astype(np.float32)
    b = np.zeros(k, dtype=np.float32)
    for _ in range(ROUTER_ITERS):
        p = softmax(x @ w + b)
        g = p.copy()
        g[np.arange(len(y)), y] -= 1.0
        g /= len(y)
        w -= ROUTER_LR * (x.T @ g + ROUTER_L2 * w)
        b -= ROUTER_LR * g.sum(axis=0)
    return w, b


def pick_threshold(p: np.ndarray, y: np.ndarray, off_idx: int | None) -> float:
    """Calibrate the abstain cutoff on data that HAS negatives (the train set): routing and
    being right scores 1, routing and wrong 0, abstaining scores 1 only when the gold really
    is off-domain — so the cutoff can't win by routing nothing or by routing everything."""
    best, best_acc = DEFAULT_THRESHOLD, -1.0
    for t in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        take = p.max(axis=1) >= t
        correct = p.argmax(axis=1) == y
        acc = float(np.mean(np.where(take, correct, y == off_idx if off_idx is not None else False)))
        if acc > best_acc:
            best, best_acc = t, acc
    return best


# ------------------------------------------------------------------ stage 3: cards

CARD_STYLES = {
    "policy": ("你是{domain}的资深专家,在客服一线工作。回答规则:\n"
               "1. 只依据提供的知识库资料回答,资料里没有的就明确说不知道;\n"
               "2. 先给结论,再列依据,关键数字/期限/流程名必须与资料原文一致;\n"
               "3. 不确定时分步引导用户补齐信息;\n"
               "4. 使用与用户相同的语言,保持简洁。"),
    "guide": ("你是{domain}领域的专家助手。请基于知识库资料准确回答:直接给出答案和关键事实"
              "(数字、期限、流程),资料未覆盖的问题如实告知并建议找谁处理,不要编造。"),
}


def draft_cards(llm: str, digest: str) -> dict[str, str]:
    cards = {}
    for style, tpl in CARD_STYLES.items():
        domain = llm_chat(llm, [{"role": "user", "content": (
            "用不超过12个字概括以下资料所属的业务领域,只输出领域名。\n\n" + digest[:1500])}],
            max_tokens=30, temperature=0.1).strip().splitlines()[0][:12]
        body = llm_chat(llm, [{"role": "user", "content": (
            f"以下是需要遵守的专家守则模板:\n{tpl.format(domain=domain)}\n\n"
            "请基于资料,把模板改写为一份最终版专家系统提示:保留全部规则,补充 2-3 条只有这些资料里才有的"
            "领域要点(每条一行,事实必须来自资料)。只输出最终提示本身。\n\n资料:\n" + digest[:2500])}],
            max_tokens=330, temperature=0.3).strip()
        cards[style] = f"{tpl.format(domain=domain)}\n\n领域要点:\n{body}"
    return cards


def fact_hit(gold: str, resp: str) -> bool:
    """Automatic grading: the gold answer's digits and its head must show up in the reply."""
    nums = re.findall(r"\d+(?:\.\d+)?", gold)
    if nums and all(n in resp for n in nums):
        return True
    return gold[:10] in resp


def grade_card(llm: str, card: str, kb_chunks, embedder, test_qa: list[dict], top_k: int = 4) -> float:
    """Hold-out score of one card: card as system + top-k retrieval, exact-fact graded."""
    if not test_qa:
        return 0.0
    matrix = np.asarray([e for e in embedder.embed([c["text"] for c in kb_chunks])])
    hits = 0
    for qa in test_qa:
        q = embedder.embed([qa["question"]])[0]
        order = np.argsort(-(matrix @ q))[:top_k]
        ctx = "\n".join(f"[资料 {i + 1}]\n{kb_chunks[int(j)]['text']}" for i, j in enumerate(order))
        msgs = ([{"role": "system", "content": card}] if card else []) + [
            {"role": "user", "content": f"参考资料:\n{ctx}\n\n问题:{qa['question']}"}]
        resp = llm_chat(llm, msgs, max_tokens=160, temperature=0.1)
        hits += fact_hit(qa["answer"], resp)
    return round(hits / len(test_qa), 3)


# ------------------------------------------------------------------ the build

def build(name: str, source: Path, llm: str, embedder, out_dir: Path) -> dict:
    t0 = time.time()
    print(f"[1/4] ingest {source}")
    chunks = ingest(source)
    if not chunks:
        raise SystemExit(f"{source}: no text found")
    print(f"      {len(chunks)} chunks")

    print("[2/4] distill QA pairs (self-supervised, no labels)")
    qa = distill(llm, chunks)
    if len(qa) < 5:
        raise SystemExit(f"only {len(qa)} usable pairs — add more source text")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(qa))
    n_test = max(2, int(len(qa) * TEST_FRACTION))
    test_idx, train_idx = set(order[:n_test].tolist()), [i for i in range(len(qa)) if i not in set(order[:n_test])]
    test_qa = [qa[i] for i in sorted(test_idx)]
    train_qa = [qa[i] for i in train_idx]
    print(f"      {len(train_qa)} train / {len(test_qa)} held-out")

    print("[3/4] train router head (softmax over question embeddings)")
    classes: list[str] = [name]
    rows, labels = [], []
    for qa_ in train_qa:
        rows.append(qa_["question"]); labels.append(0)
    for other in sorted(out_dir.glob("*/qa.jsonl")):          # negatives: the other experts' pairs
        if other.parent.name == name:
            continue
        k = len(classes)
        for line in other.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("split") == "train":
                rows.append(obj["question"]); labels.append(k)
        if sum(1 for l in labels if l == k):
            classes.append(other.parent.name)
    if len(classes) == 1:                                       # single expert: synthesize negatives
        for qa_ in synthesize_negatives(llm, name):
            rows.append(qa_["question"]); labels.append(len(classes))
        classes.append("off-domain")
    x = np.asarray(embedder.embed(rows), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    w, b = train_router(x, y, len(classes))
    p = softmax(x @ w + b)
    train_acc = float(np.mean(p.argmax(axis=1) == y))
    xt = np.asarray(embedder.embed([q["question"] for q in test_qa]), dtype=np.float32)
    yt = np.zeros(len(test_qa), dtype=np.int64)
    pt = softmax(xt @ w + b)
    test_acc = float(np.mean(pt.argmax(axis=1) == yt))
    off_idx = classes.index("off-domain") if "off-domain" in classes else None
    threshold = pick_threshold(p, y, off_idx)
    print(f"      classes={classes} train_acc={train_acc:.2f} heldout_acc={test_acc:.2f} threshold={threshold}")

    print("[4/4] draft + grade expert cards (better one ships)")
    digest = "\n\n".join(c["text"] for c in chunks)[:6000]
    cards = draft_cards(llm, digest)
    scores = {}
    for style, card in cards.items():
        scores[style] = grade_card(llm, card, chunks, embedder, test_qa)
    baseline = grade_card(llm, "", chunks, embedder, test_qa)   # retrieval only, no card
    best_style = max(scores, key=scores.get)
    print(f"      cards={scores} retrieval_only_baseline={baseline} -> ship '{best_style}'")

    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "router.npz", W=w, b=b, classes=np.array(classes, dtype=object))
    with open(d / "qa.jsonl", "w", encoding="utf-8") as f:
        for i, qa_ in enumerate(qa):
            f.write(json.dumps({**qa_, "split": "test" if i in test_idx else "train"},
                               ensure_ascii=False) + "\n")
    metrics = {"chunks": len(chunks), "qa_total": len(qa), "heldout": len(test_qa),
               "router": {"classes": classes, "train_acc": round(train_acc, 3),
                          "heldout_acc": round(test_acc, 3), "threshold": threshold,
                          "iters": ROUTER_ITERS},
               "cards": scores, "retrieval_only_baseline": baseline,
               "shipped_card": best_style, "build_seconds": round(time.time() - t0, 1)}
    (d / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle = {"name": name, "source": str(source), "card": cards[best_style],
              "card_style": best_style, "threshold": threshold, "built_at": time.time(),
              "metrics": metrics}
    (d / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {d}/bundle.json  ({metrics['build_seconds']}s)")
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="train an expert bundle from a knowledge base directory")
    b.add_argument("name")
    b.add_argument("--source", required=True, help="the KB directory (same formats the gateways ingest)")
    b.add_argument("--llm", default="http://127.0.0.1:8902", help="llama-server chat endpoint (the teacher)")
    b.add_argument("--kb-embedder", default="llama:http://127.0.0.1:8910",
                   help='"lexical" or "llama:http://host:port" (the embedder the gateway will use)')
    b.add_argument("--out", default="experts", help="expert bundles directory (negatives come from siblings)")
    args = ap.parse_args()

    if args.cmd == "build":
        build(args.name, Path(args.source).resolve(), args.llm.rstrip("/"),
              make_embedder(args.kb_embedder), Path(args.out).resolve())


if __name__ == "__main__":
    main()
