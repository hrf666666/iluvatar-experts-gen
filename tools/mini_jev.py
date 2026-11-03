#!/usr/bin/env python3
"""tools/mini_jev.py — mass-produced ≤1B JEVs: a frozen encoder plus a trained matching head.

The edge constraint (from the 2026-11-03 size-threshold experiments): a decision model at
or under 1B cannot be had by calibrating a small LLM's verbalizer logits — the signal is
not there (0.6B collapsed to label priors) and the industry recipes (mini-Jev, joyfox
0.8B, kotoba) all train weights. This pod has no trainer, so the one road that both fits
the constraint and runs on this stack is the encoder route the research converged on:

    score(option | state, question) = (U·e_query) · (V·e_option)      — low-rank bilinear

e_query is the bge-m3 embedding of the question + state + retrieved KB chunks (the same
grounding the big JEV sees), e_option the embedding of an ANCHOR text distilled once per
team from the KB ("billing: duplicate charges, refunds, payment channel faults...").
U, V (d×r) and a temperature are fitted with numpy against rejection-sampled,
teacher-labeled cases — no backprop, no GPU training, minutes per domain.

The result serves the SAME /decide API shape as jev_decide (kind=choice), runs one
embedding forward per decision (~50-100 ms, 0.5B-class compute, 467 MB model), and every
side-loadable expert gets one: `mini-jevs/<domain>/` is the wholesale unit.

    python3 tools/mini_jev.py build ecom-support --kb-dir ../kb \
        --teacher http://127.0.0.1:5500 --writer http://127.0.0.1:8902 \
        --embedder llama:http://127.0.0.1:8910 --out mini-jevs
    python3 tools/mini_jev.py --domain ecom-support --kb-dir ../kb \
        --embedder llama:http://127.0.0.1:8910 --serve 5504
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.knowledge import KBRegistry, make_embedder            # the shared retrieval stack
from tools.jev_distill import DOMAINS, gen_cases                 # rejection-sampled teacher cases

RANK = 64
ITERS = 800
LR = 1.0
L2 = 1e-4
HOLDOUT = 0.25
SEED = 11


def http_json(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(base, messages, max_tokens=300, temperature=0.3):
    out = http_json(f"{base}/v1/chat/completions",
                    {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                     "chat_template_kwargs": {"enable_thinking": False}})
    return out["choices"][0]["message"]["content"] or ""


# ------------------------------------------------------------------ anchors: the options' semantics

ANCHOR_PROMPT = (
    "根据知识库资料,为下面每个团队写一句职责锚点(30字内,写清该团队处理什么类型的问题、"
    "关键流程/数字),输出 JSON 数组,每项形如 {{\"team\": \"...\", \"anchor\": \"...\"}},不要输出别的。\n\n"
    "团队: {teams}\n\n资料:\n{digest}"
)


def write_anchors(writer, domain, kb_dir):
    options = DOMAINS[domain][0]
    digest = "\n\n".join(p.read_text(encoding="utf-8")[:900]
                         for p in sorted(Path(kb_dir, domain).rglob("*.md")))[:6000]
    raw = chat(writer, [{"role": "user",
                         "content": ANCHOR_PROMPT.format(teams=", ".join(options), digest=digest)}])
    anchors = {o: o for o in options}
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            for it in json.loads(m.group(0)):
                if it.get("team") in anchors and it.get("anchor"):
                    anchors[it["team"]] = f"{it['team']}: {it['anchor']}"
        except ValueError:
            pass
    return anchors


# ------------------------------------------------------------------ the head

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit(eq, eo, ys, ws):
    """eq: (n,d) query embeddings, eo: (n,k,d) per-case option embeddings, ys: labels, ws: weights.
    Full-batch GD on CE over logits (eq@U)·(V·eo) — U,V are d×r, temperature scalar."""
    n, d = eq.shape
    rng = np.random.default_rng(SEED)
    u = rng.normal(0, 0.05, (d, RANK)).astype(np.float64)
    v = rng.normal(0, 0.05, (d, RANK)).astype(np.float64)
    log_t = np.zeros(())
    for _ in range(ITERS):
        t = math.exp(float(log_t))
        pu = eq @ u                                  # (n,r)
        loss, gu, gv, gt = 0.0, np.zeros_like(u), np.zeros_like(v), 0.0
        for i in range(n):
            pv = eo[i] @ v                           # (k,r)
            z = (pu[i:i + 1] @ pv.T)[0] / t          # (k,)
            p = softmax(z)
            y, w = ys[i], ws[i]
            loss += -w * math.log(max(p[y], 1e-12))
            g = p.copy()
            g[y] -= 1.0
            g = g * w / t                            # (k,)
            gu += np.outer(eq[i], g @ pv)            # d×r
            gv += eo[i].T @ np.outer(g, pu[i])       # d×r
            gt += float(w * (g * z).sum())           # d log CE / d log T (z already /t)
        gu, gv, gt = gu / n + L2 * u, gv / n + L2 * v, gt / n
        u -= LR * gu
        v -= LR * gv
        log_t = float(np.clip(log_t - 0.1 * LR * gt, math.log(0.3), math.log(4.0)))
    return u, v, math.exp(float(log_t)), loss / n


def predict(u, v, t, q, opts):
    return softmax(((q @ u) @ (opts @ v).T) / t)


# ------------------------------------------------------------------ build / serve

def build(domain, kb_dir, teacher, writer, embedder_spec, out_dir, per_team):
    if domain not in DOMAINS:
        raise SystemExit(f"domain {domain}: not in {sorted(DOMAINS)}")
    t0 = time.time()
    print(f"[1/4] anchors (one-time distillation of what each team is)")
    anchors = write_anchors(writer, domain, kb_dir)
    for k, a in anchors.items():
        print(f"      {a[:64]}")

    print("[2/4] rejection-sampled teacher cases")
    cases = gen_cases(writer, teacher, kb_dir, [domain], per_team)
    if len(cases) < 12:
        raise SystemExit(f"only {len(cases)} clean cases — raise --per-team")

    print("[3/4] embeddings + head fit (numpy, no backprop)")
    emb = make_embedder(embedder_spec)
    options = DOMAINS[domain][0]
    opt_mat = np.asarray(emb.embed([anchors[o] for o in options]), dtype=np.float64)
    eq = np.asarray(emb.embed([f"{c['question']}\n{c['state']}" for c in cases]), dtype=np.float64)
    eo = np.stack([opt_mat[[options.index(t) for t in c["options"]]] for c in cases])
    ys = np.asarray([c["teacher"]["label"] for c in cases])
    ws = np.asarray([c["teacher"]["weight"] for c in cases])
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(cases))
    n_test = max(3, int(len(cases) * HOLDOUT))
    test = set(order[:n_test].tolist())
    tr = [i for i in range(len(cases)) if i not in test]
    te = sorted(test)
    u, v, t, ce = fit(eq[tr], eo[tr], ys[tr], ws[tr])
    acc = lambda idx: float(np.mean([int(np.argmax(predict(u, v, t, eq[i], eo[i]))) == ys[i] for i in idx]))
    print(f"      T={t:.3f} CE={ce:.3f} agreement train={acc(tr):.2f} holdout={acc(te):.2f}")

    d = out_dir / domain
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "head.npz", U=u, V=v, T=np.asarray(t), opt_emb=opt_mat,
             options=np.array(options, dtype=object))
    (d / "bundle.json").write_text(json.dumps({
        "domain": domain, "anchors": anchors, "options": options,
        "kb": DOMAINS[domain][1], "rank": RANK, "temperature": round(t, 4),
        "cases": len(cases), "agreement": {"train": round(acc(tr), 3), "holdout": round(acc(te), 3)},
        "built_at": time.time(), "build_seconds": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {d}/  ({time.time() - t0:.0f}s)  — serve: mini_jev.py --domain {domain} --serve <port>")


class MiniJev:
    def __init__(self, domain, kb_dir, embedder_spec, bundle_dir):
        z = np.load(bundle_dir / domain / "head.npz", allow_pickle=True)
        self.u, self.v, self.t = z["U"], z["V"], float(z["T"])
        self.opt_emb, self.options = z["opt_emb"], [str(o) for o in z["options"]]
        self.bundle = json.loads((bundle_dir / domain / "bundle.json").read_text(encoding="utf-8"))
        self.reg = KBRegistry(embedder=make_embedder(embedder_spec), root=kb_dir)
        self.reg.load(domain, self.bundle.get("kb", domain))
        self.emb = self.reg.embedder

    def decide(self, state, question, options=None, top_k=4):
        """Options may be reordered/subset at request time: anchors are matched by text,
        so the head stays permutation-equivariant over whatever the caller declares."""
        options = options or self.options
        try:
            idx = [self.options.index(o) for o in options]
        except ValueError as e:
            raise ValueError(f"unknown option: {e.args[0]}; this mini-JEV covers {self.options}")
        q_text = f"{question}\n{state}"
        hits = self.reg.search(q_text, top_k=top_k)
        if hits:                                            # the same grounding the big JEV gets
            q_text = "\n".join(h["text"] for h in hits) + "\n" + q_text
        q = np.asarray(self.emb.embed([q_text])[0], dtype=np.float64)
        p = predict(self.u, self.v, self.t, q, self.opt_emb[idx])
        order = np.argsort(-p)
        return [{"label": chr(ord("A") + i), "prob": float(p[i]), "text": options[i]}
                for i in order]


def serve(domain, kb_dir, embedder_spec, bundle_dir, port):
    mj = MiniJev(domain, kb_dir, embedder_spec, bundle_dir)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0].rstrip("/") == "/kb/list":
                self._reply(200, mj.reg.list())
            else:
                self.send_error(404)

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                if path == "/decide":
                    n = int(self.headers.get("Content-Length", 0))
                    req = json.loads(self.rfile.read(n) or b"{}")
                    if req.get("kind", "choice") != "choice":
                        self._reply(400, {"error": "mini-JEV serves kind=choice (score/noul: roadmap)"})
                        return
                    t0 = time.time()
                    dist = mj.decide(req.get("state", ""), req.get("question", ""),
                                     req.get("options") or None, req.get("kb_top_k", 4))
                    top = dist[0]
                    self._reply(200, {"answer": top["label"], "prob": top["prob"],
                                      "distribution": dist,
                                      "latency_ms": round((time.time() - t0) * 1000, 1),
                                      "kb": [domain], "model": f"mini-jev:{domain}"})
                else:
                    self.send_error(404)
            except ValueError as e:
                self._reply(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    print(f"mini-JEV [{domain}] options={mj.options} on http://0.0.0.0:{port}/decide "
          f"(frozen encoder + r{RANK} matching head, T={mj.t:.2f})", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", metavar="DOMAIN", help="train this domain's mini-JEV bundle")
    ap.add_argument("--kb-dir", default="../kb")
    ap.add_argument("--teacher", default="http://127.0.0.1:5500")
    ap.add_argument("--writer", default="http://127.0.0.1:8902")
    ap.add_argument("--embedder", default="llama:http://127.0.0.1:8910")
    ap.add_argument("--out", default="mini-jevs", type=Path)
    ap.add_argument("--per-team", type=int, default=6)
    ap.add_argument("--domain", help="serve this domain")
    ap.add_argument("--serve", type=int, metavar="PORT")
    args = ap.parse_args()
    if args.build:
        build(args.build, Path(args.kb_dir), args.teacher.rstrip("/"), args.writer.rstrip("/"),
              args.embedder, args.out, args.per_team)
    elif args.serve and args.domain:
        serve(args.domain, Path(args.kb_dir), args.embedder, args.out, args.serve)
    else:
        ap.error("--build DOMAIN or --domain X --serve PORT")


if __name__ == "__main__":
    main()
