#!/usr/bin/env python3
"""jev_decide.py — JEV-35B (AutoTrust System 1 decision model) on llama.cpp.

Replicates the System 1 read-out of the model's serve_decide.py against a plain
llama.cpp llama-server:

    prompt   = "[kind] {kind}\\n[state] {state}\\n[question] {question}"
               "\\n[options]\\nA) ...\\n[decision]:"
    read-out = softmax((logprob[label] + bias) / temperature)  over the kind's slots

The decision head is merged into the lm_head rows of the verbalizer tokens
(decision_head.json: "merged": true), so a plain text completion exposes it.

llama.cpp has no per-token-id logprob endpoint, so we bias all label token ids
by +100 (logit_bias) and read the top-N logprobs: the constant cancels in the
softmax, the non-label mass is negligible, and the returned label logprobs are
exact up to that vanishing term.

Usage:
    python3 jev_decide.py --model-dir /path/to/JEV-35B --port 8902 \
        --kind choice --question "Which city is the capital of France?" \
        --options "Paris" "London" "Berlin" "Madrid"

    # or as a server: POST /decide {"kind","state","question","options"}
    python3 jev_decide.py --model-dir /path/to/JEV-35B --port 8902 --serve 8940

Hot-pluggable knowledge bases: a decision can be grounded in one (or several)
without restarting anything. Pass --kb-dir once, then load/unload/swap KBs at
runtime on the gateway (/kb/load, /kb/unload, /kb/swap, /kb/list, /kb/search);
each /decide request picks its own with "kb", or leaves it out to use the
active set. Retrieved chunks go into the [state] block - the context the
decision is made on - marked as [knowledge base context] so the model can tell
them from the caller's own state.
"""

import argparse
import json
import math
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root, for serve.knowledge
from serve.knowledge import KBRegistry, make_embedder          # the same registry the chat server uses

TOP_K = 64  # top-N full-vocab logprobs requested; labels below this rank have negligible probability


def http_json(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class JevHead:
    """Slot layout, verbalizer token ids, bias and per-kind temperature."""

    def __init__(self, model_dir):
        with open(f"{model_dir}/adapter_vllm/decision_head.json") as f:
            head = json.load(f)
        with open(f"{model_dir}/calibration.json") as f:
            cal = json.load(f)
        self.ids = head["verbalizer_ids"]            # 264 token ids
        self.bias = head["bias"]                     # 264 biases
        self.slots = head["slots"]["ranges"]         # {"noul": [0,2], "score": [2,8], "choice": [8,264]}
        self.verbalizers = head["slots"]["verbalizers"]
        self.T = cal["per_kind"]

    def labels(self, kind, n_options):
        lo, hi = self.slots[kind]
        if kind == "choice":
            n = min(n_options, hi - lo)
            return list(range(lo, lo + n))
        return list(range(lo, hi))

    def prompt(self, kind, state, question, option_texts):
        lo = self.slots[kind][0]
        if kind == "choice":
            lines = [f"{self.verbalizers[lo + i]}) {t}" for i, t in enumerate(option_texts)]
        else:
            # serve_decide.py: noul and score have fixed option lines ("false","true" / "0".."5")
            lines = ["false", "true"] if kind == "noul" else [str(i) for i in range(6)]
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return (f"[kind] {kind}\n[state] {state}\n[question] {question}\n"
                f"[options]\n" + "\n".join(lines) + "\n[decision]:")

    def decide(self, base_url, kind, state, question, option_texts):
        slots = self.labels(kind, len(option_texts or []))
        if kind != "choice":
            option_texts = None  # the option lines are fixed; slots cover the whole range
        ids = [self.ids[s] for s in slots]
        bias = [self.bias[s] for s in slots]
        # llama.cpp reports pre-bias full-vocab logprobs, so a plain top-N read is exact for
        # the labels it covers; "logprobs" is the integer form (n_probs), not the OpenAI bool
        body = {
            "prompt": self.prompt(kind, state, question, option_texts),
            "max_tokens": 1,
            "temperature": 1.0,       # full distribution, like serve_decide.py's READ
            "top_p": 1.0, "top_k": 0, "min_p": 0.0,
            "logprobs": TOP_K,        # a label outside the top TOP_K has negligible probability
            "cache_prompt": True,     # /v1/completions does not cache by default in this build;
        }                            # without it every decision re-prefills the whole state
        lps = None
        for attempt in range(3):
            # llama-server very rarely answers logprobs:null on a slot's first task after startup;
            # a resend is served from the prompt cache (~100 ms) and comes back complete.
            out = http_json(f"{base_url}/v1/completions", body)
            content = (out["choices"][0].get("logprobs") or {}).get("content") or []
            if content and content[0].get("top_logprobs"):
                lps = {t["id"]: t["logprob"] for t in content[0]["top_logprobs"]}
                break
            time.sleep(0.1 * (attempt + 1))
        if lps is None:
            raise RuntimeError(f"upstream {base_url} returned no logprobs for the sampled token (3 tries)")
        t = self.T.get(kind, 1.0)
        zs = [(lps.get(i, -1e9) + b) / t for i, b in zip(ids, bias)]
        m = max(zs)
        es = [math.exp(z - m) for z in zs]
        z = sum(es)
        probs = [e / z for e in es]
        return [{"label": self.verbalizers[s], "prob": p, "text": (option_texts or [""])[i] if i < len(option_texts or []) else ""}
                for i, (s, p) in enumerate(zip(slots, probs))]


def kb_context(reg, names, question, state, top_k):
    """-> the retrieved block that goes in front of [state], or "". The query is the question plus the state's
    head: the question says what the decision needs, the state says the case."""
    if reg is None or not names or not (question + " " + state).strip():
        return ""
    try:
        hits = reg.search(f"{question}\n{state[:2000]}", names, top_k)
    except KeyError as e:
        raise ValueError(f"unknown knowledge base: {e.args[0]}") from None
    if not hits:
        return ""
    block = "\n".join(f"[knowledge base {h['kb']} #{i + 1} | source: {h['source']}]\n{h['text']}"
                      for i, h in enumerate(hits))
    return f"[knowledge base context]\n{block}\n[end of knowledge base context]\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="the JEV-35B checkout (decision_head.json, calibration.json)")
    ap.add_argument("--port", type=int, default=8080, help="llama-server port")
    ap.add_argument("--kind", choices=["choice", "score", "noul"], default="choice")
    ap.add_argument("--state", default="", help="context shown under [state]")
    ap.add_argument("--question", help="the decision's question (each request has its own in --serve mode)")
    ap.add_argument("--options", nargs="*", default=[], help="choice: option texts; score: level descriptions")
    ap.add_argument("--serve", type=int, metavar="PORT", help="run an HTTP /decide gateway on this port instead")
    ap.add_argument("--kb", action="append", default=[], metavar="NAME",
                    help="ground decisions in these knowledge bases (source: --kb-dir/NAME); repeatable")
    ap.add_argument("--kb-dir", help="the root knowledge bases load from (and must stay inside)")
    ap.add_argument("--kb-top-k", type=int, default=4, help="chunks retrieved per decision (default 4)")
    ap.add_argument("--kb-embedder", default="lexical", metavar="SPEC",
                    help='"lexical" (the default) or "llama:http://host:port" (a llama-server with --embedding)')
    args = ap.parse_args()
    if not args.serve and not args.question:
        ap.error("--question is required without --serve")

    base = f"http://127.0.0.1:{args.port}"
    head = JevHead(args.model_dir)
    reg = None
    if args.kb or args.kb_dir:
        try:
            reg = KBRegistry(embedder=make_embedder(args.kb_embedder), root=args.kb_dir or ".")
            for name in args.kb:
                reg.load(name, name)
            reg.swap(args.kb)
        except ValueError as e:
            raise SystemExit(f"knowledge base: {e}")

    if not args.serve:
        t0 = time.time()
        names = tuple(args.kb) if args.kb else (reg.snapshot_names() if reg else ())
        state = kb_context(reg, names, args.question, args.state, args.kb_top_k) + args.state
        result = head.decide(base, args.kind, state, args.question, args.options)
        dt = (time.time() - t0) * 1000
        top = max(result, key=lambda r: r["prob"])
        print(json.dumps({"answer": top["label"], "text": top["text"], "prob": round(top["prob"], 4),
                          "latency_ms": round(dt, 1), "distribution": result},
                         ensure_ascii=False, indent=2))
        return

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n)) if n else {}
            if not isinstance(req, dict):
                raise ValueError("send a JSON object")
            return req

        def do_GET(self):
            if self.path.split("?")[0].rstrip("/") == "/kb/list":
                self._reply(200, reg.list() if reg else
                            {"error": "started without --kb / --kb-dir; restart with --kb-dir to use knowledge bases"})
            else:
                self.send_error(404)

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                if path == "/decide":
                    self._decide(self._body())
                elif path == "/decide_batch":
                    self._decide_batch(self._body())
                elif reg is None:
                    self._reply(404, {"error": "started without --kb / --kb-dir; restart with --kb-dir to use "
                                               "knowledge bases"})
                elif path == "/kb/load":
                    req = self._body()
                    self._reply(200, reg.load(req["name"], req["source"],
                                              req.get("chunk_size", 900), req.get("overlap", 140)))
                elif path == "/kb/reload":
                    self._reply(200, reg.reload(self._body()["name"]))
                elif path == "/kb/unload":
                    self._reply(200, reg.unload(self._body()["name"]))
                elif path == "/kb/swap":
                    self._reply(200, reg.swap(self._body().get("active", "")))
                elif path == "/kb/search":
                    req = self._body()
                    names = tuple(req["kb"]) if isinstance(req.get("kb"), list) else None
                    self._reply(200, {"hits": reg.search(req["query"], names, req.get("top_k", 4))})
                else:
                    self.send_error(404)
            except ValueError as e:
                self._reply(400, {"error": str(e)})
            except RuntimeError as e:   # upstream (llama-server) failed after retries
                self._reply(502, {"error": str(e)})
            except KeyError as e:
                self._reply(404, {"error": f"unknown knowledge base: {e.args[0]}"})

        def _decide(self, req):
            # "kb" picks the request's own KBs ("auto"/left out: the active set; false: none)
            names = reg.resolve(req.get("kb")) if reg else ()
            if names is None:
                names = reg.snapshot_names() if reg else ()
            state = kb_context(reg, names, req.get("question", ""), req.get("state", ""),
                               req.get("kb_top_k", 4)) + (req.get("state", "") or "")
            t0 = time.time()
            result = head.decide(base, req.get("kind", "choice"), state,
                                 req["question"], req.get("options") or [])
            top = max(result, key=lambda r: r["prob"])
            self._reply(200, {"answer": top["label"], "prob": top["prob"], "distribution": result,
                              "latency_ms": round((time.time() - t0) * 1000, 1),
                              "kb": [kb.name for kb in (reg.snapshot(names) if reg and names else ())]})

        def _decide_batch(self, req):
            """FLIDE-style state forking on our engine: ONE state (+ one retrieval pass for
            the KB context), then N questions sent back-to-back so llama-server's prompt
            cache dedupes the shared prefix — each question pays only its own tail + one
            token of decode. Same-kind questions share the whole prefix; mixed kinds still
            work (the cache matches the longest common token prefix)."""
            t0 = time.time()
            if reg is None:
                names = ()
            else:
                names = reg.resolve(req.get("kb")) or reg.snapshot_names()
            state = req.get("state") or ""
            ctx = kb_context(reg, names, "", state, req.get("kb_top_k", 4))
            full_state = ctx + state
            out = []
            for q in req.get("questions", []):
                t1 = time.time()
                result = head.decide(base, q.get("kind", "choice"), full_state,
                                     q.get("question", ""), q.get("options") or [])
                top = max(result, key=lambda r: r["prob"])
                out.append({"question": q.get("question", ""), "answer": top["label"],
                            "prob": top["prob"], "distribution": result,
                            "latency_ms": round((time.time() - t1) * 1000, 1)})
            self._reply(200, {"decisions": out,
                              "total_ms": round((time.time() - t0) * 1000, 1),
                              "questions": len(out)})

        def log_message(self, *a):
            pass

    what = f"jev_decide gateway on http://0.0.0.0:{args.serve}/decide -> llama-server :{args.port}"
    print(what + (f" | knowledge bases: {', '.join(args.kb) or 'none active'} (hot: /kb/*)"
                  if reg else ""), file=sys.stderr)
    ThreadingHTTPServer(("0.0.0.0", args.serve), Handler).serve_forever()


if __name__ == "__main__":
    main()
