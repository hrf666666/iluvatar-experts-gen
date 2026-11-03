#!/usr/bin/env python3
"""chat_kb_gateway.py — a hot-pluggable-knowledge-base front for llama-server (the Qwen3.8-27B path on
Iluvatar, where the model is served by llama.cpp's own llama-server, not by the Strata engine).

Clients talk to this gateway exactly as they would to llama-server's OpenAI API; the gateway retrieves
from the plugged-in knowledge bases, writes the hits into the last user message (the same inject() the
Strata server uses), and forwards the request. Replies - streamed or not - pass through untouched, so
everything llama-server does (tools, JSON schemas, caching headers) keeps working.

The knowledge bases are hot: load, unload, reload and swap at runtime on the same port, no restart of
the gateway and none of llama-server. In-flight requests hold the KB snapshot they started with.

    python3 tools/chat_kb_gateway.py --upstream http://127.0.0.1:8902 --port 8960 --kb-dir ./kb --kb handbook

    # then, while it runs:
    curl -s localhost:8960/v1/kbs -X POST -H 'Content-Type: application/json' \
         -d '{"name": "freight", "source": "freight"}'
    curl -s localhost:8960/v1/kbs/swap -X POST -H 'Content-Type: application/json' -d '{"active": ["freight"]}'
    curl -s localhost:8960/v1/chat/completions -H 'Content-Type: application/json' -d \
         '{"messages": [{"role": "user", "content": "顺丰多少钱"}], "max_tokens": 64}'
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root, for serve.knowledge
from serve.knowledge import KBRegistry, make_embedder          # the same registry the Strata server uses

MAX_BODY = 10 * 1024 * 1024        # a request with images can be big; past this it is not a chat request


# --------------------------------------------------------------------------- trained experts

class ExpertHub:
    """Hot-pluggable expert bundles produced by tools/expert_factory.py: an expert = a trained
    router row + a graded expert card + the KB it fronts. Two "self-*" behaviours make the
    plug invisible to users (2026-11-04): the hub AUTODISCOVERS bundles that appear under
    --expert-dir (a factory run lands and the expert is serving — no POST, no swap), and the
    trained router MULTI-SELECTS: every active expert whose own classifier fires above its
    threshold joins the answer (cards compose into one persona, retrieval merges across
    their KBs by score), because real questions cross domains — a drunk-driving damages
    consult is criminal law AND civil law, and the caller must not have to pick."""

    def __init__(self, reg: KBRegistry, embedder, auto: bool = True):
        self.reg, self.embedder, self.auto = reg, embedder, auto
        self.experts: dict[str, dict] = {}          # name -> {bundle, W, b, classes}
        self.active: tuple[str, ...] = ()
        self.route_log: deque = deque(maxlen=50)
        self._dir_mtime = None
        self.activation = None                      # the expert-picker mini-JEV (tools/mini_jev
        self._anchor_emb: dict[str, np.ndarray] = {}   # --build-activation), options = domains

    def load(self, expert_dir: Path) -> dict:
        bundle = json.loads((expert_dir / "bundle.json").read_text(encoding="utf-8"))
        name = bundle["name"]
        head = expert_dir / "router.npz"             # a bundle with just a card is COLD-START
        if head.exists():                            # routable: activation/zero-shot only
            z = np.load(head, allow_pickle=True)
            self.experts[name] = {"bundle": bundle, "W": z["W"], "b": z["b"],
                                  "classes": [str(c) for c in z["classes"]]}
        else:
            self.experts[name] = {"bundle": bundle, "W": None, "b": None, "classes": []}
        src = Path(bundle["source"])
        kb_src = src.name if src.is_dir() else str(src)   # under the registry root by leaf name …
        try:                                                # … or the absolute path when it moved
            self.reg.load(name, src if src.exists() else kb_src)
        except ValueError:
            self.reg.load(name, kb_src)
        return bundle

    def scan(self):
        """Autodiscover: rescan --expert-dir when its mtime changes; new bundles load AND
        activate themselves. A cheap stat, not a watch loop — called at the top of routing."""
        root = getattr(self, "expert_root", None)
        if root is None or not self.auto or not root.exists():
            return
        m = root.stat().st_mtime_ns
        if m == self._dir_mtime:
            return
        self._dir_mtime = m
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if d.name.startswith("_"):              # _activation & friends are routing substrate
                h, b = d / "head.npz", d / "bundle.json"
                if h.exists() and b.exists():
                    z = np.load(h, allow_pickle=True)
                    self.activation = {"U": z["U"], "V": z["V"], "T": float(z["T"]),
                                       "options": [str(o) for o in z["options"]],
                                       "anchors": json.loads(b.read_text(encoding="utf-8"))["anchors"],
                                       "threshold": json.loads(b.read_text(encoding="utf-8"))["threshold"]}
                    print(f"[expert-auto] activation head loaded (options: "
                          f"{self.activation['options']})", file=sys.stderr, flush=True)
                continue
            if d.name in self.experts or not (d / "bundle.json").exists():
                continue
            try:
                self.load(d)
                self.active = self.active + (d.name,)
                print(f"[expert-auto] discovered + activated: {d.name}", file=sys.stderr, flush=True)
            except (ValueError, KeyError, OSError) as e:
                print(f"[expert-auto] skipped {d.name}: {e}", file=sys.stderr, flush=True)

    def swap(self, names) -> dict:
        names = (names,) if isinstance(names, str) else tuple(names or ())
        missing = [n for n in names if n not in self.experts]
        if missing:
            raise KeyError(", ".join(missing))
        self.active = names
        return self.list()

    def unload(self, name: str) -> dict:
        self.experts.pop(name, None)
        self.active = tuple(n for n in self.active if n != name)
        return self.list()

    def anchor_of(self, name: str) -> str:
        """The expert's responsibility text: bundle.anchor when the factory wrote one, else
        the card's head — enough for zero-shot routing of experts that predate the head."""
        b = self.experts[name]["bundle"]
        return b.get("anchor") or b.get("card", "")[:300]

    def route(self, question: str, cap: int = 3) -> list[tuple[str, float]]:
        """Multi-select, via the activation mini-JEV when present (ONE softmax over all
        expert anchors — a normalized, calibrated choice of which modules to side-load),
        else via the per-expert binary heads. Empty list = decline, plain KB path."""
        self.scan()
        if not self.active or not question.strip():
            return []
        q = self.embedder.embed([question])[0].astype(np.float32)
        # Primary: the per-expert binary heads — each renormalizes against its own broad
        # negative class, so an out-of-domain question declines EVERYWHERE (measured:
        # a softmax-over-experts activation head must allocate mass to some domain and
        # false-fired on movie/code requests; the independent heads abstain cleanly).
        picked = []
        # Cold-start companions: bundles without a trained head score zero-shot against
        # their anchors, through the activation head's U/V when available (a bilinear
        # matching is a better zero-shot than raw cosine; plain cosine otherwise).
        cold = [n for n in self.active if self.experts[n]["W"] is None]
        if cold:
            u = self.activation["U"] if self.activation else np.eye(len(q), dtype=np.float64)
            v = self.activation["V"] if self.activation else u
            q2 = q.astype(np.float64) @ u
            for n in cold:
                a = self._anchor_emb.setdefault(
                    n, self.embedder.embed([self.anchor_of(n)])[0].astype(np.float32))
                k = a.astype(np.float64) @ v
                cos = float(q2 @ k / (np.linalg.norm(q2) * np.linalg.norm(k) + 1e-9))
                if cos >= 0.62:                          # cold-start bar: strict, never spurious
                    picked.append((n, cos))
        for n in self.active:
            ex = self.experts[n]
            if ex["W"] is None:                      # untrained bundle: zero-shot anchors only
                continue
            z = ex["W"].T @ q + ex["b"]
            z = z - z.max()
            e = np.exp(z)
            p = e / e.sum()
            i = int(p.argmax())
            if ex["classes"][i] != n or n == "off-domain":
                continue                    # this expert's own classifier declines
            thr = ex["bundle"].get("threshold", 0.55)
            if float(p[i]) >= thr:
                picked.append((n, float(p[i])))
        picked.sort(key=lambda t: -t[1])
        picked = picked[:cap]
        self.route_log.append({"q": question[:80],
                               "experts": [n for n, _ in picked],
                               "probs": {n: round(p, 3) for n, p in picked}})
        return picked

    def list(self) -> dict:
        return {"active": list(self.active),
                "experts": [{"name": n, "card_style": e["bundle"].get("card_style"),
                             "threshold": e["bundle"].get("threshold"),
                             "metrics": e["bundle"].get("metrics", {}).get("router", {})}
                            for n, e in self.experts.items()]}


def make_handler(upstream: str, reg: KBRegistry | None, hub: ExpertHub | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                raise ValueError("body too large")
            req = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(req, dict):
                raise ValueError("send a JSON object")
            return req

        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/v1/kbs":
                self._reply(200, reg.list() if reg else
                            {"error": "started without --kb / --kb-dir; restart with --kb-dir to use "
                                      "knowledge bases"})
            elif path == "/v1/experts":
                if hub:
                    hub.scan()                                 # the listing reflects the dir, live
                self._reply(200, hub.list() if hub else
                            {"error": "started without --expert-dir; restart with it to use experts"})
            elif path == "/v1/experts/route-log":
                self._reply(200, {"recent": list(hub.route_log)} if hub else {"recent": []})
            else:                                            # /health, /props, ...: ask the model server
                self._forward("GET", None)

        def do_DELETE(self):
            path = self.path.split("?")[0].rstrip("/")
            if hub and re.fullmatch(r"/v1/experts/[\w.-]+", path):
                self._reply(200, hub.unload(path.split("/")[3]))
                return
            if reg is None:
                self._reply(404, {"error": "started without --kb / --kb-dir"})
                return
            if not re.fullmatch(r"/v1/kbs/[\w.-]+", path):
                self._reply(404, {"error": "not found"})
                return
            try:
                self._reply(200, reg.unload(path.split("/")[3]))
            except KeyError:
                self._reply(404, {"error": f"unknown knowledge base: {path.split('/')[3]}"})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                if path == "/v1/chat/completions" and reg is not None:
                    self._chat(self._body())
                elif path == "/v1/kbs" and reg is not None:
                    req = self._body()
                    self._reply(200, reg.load(req["name"], req["source"],
                                              req.get("chunk_size", 900), req.get("overlap", 140)))
                elif path == "/v1/kbs/swap" and reg is not None:
                    self._reply(200, reg.swap(self._body().get("active", "")))
                elif re.fullmatch(r"/v1/kbs/[\w.-]+/reload", path) and reg is not None:
                    self._reply(200, reg.reload(path.split("/")[3]))
                elif re.fullmatch(r"/v1/kbs/[\w.-]+/search", path) and reg is not None:
                    req = self._body()
                    names = tuple(req["kb"]) if isinstance(req.get("kb"), list) else None
                    self._reply(200, {"hits": reg.search(req["query"], names, req.get("top_k", 4))})
                elif path == "/v1/experts" and hub is not None:
                    req = self._body()
                    d = Path(req["bundle_dir"] if "bundle_dir" in req else
                             str(hub.expert_root / req["name"]))
                    self._reply(200, hub.load(d))
                elif path == "/v1/experts/swap" and hub is not None:
                    self._reply(200, hub.swap(self._body().get("active", "")))
                else:                                        # everything else: the model server's own API
                    self._forward("POST", self._body())
            except ValueError as e:
                self._reply(400, {"error": str(e)})
            except KeyError as e:
                self._reply(404, {"error": f"unknown: {e.args[0]}"})
            except OSError as e:
                self._reply(400, {"error": f"expert bundle not readable: {e}"})

        def _chat(self, req):
            """Retrieve with the last user message, inject, forward. "kb" picks the KBs ("auto"/left
            out: the active set; false: none); it and "kb_top_k" are ours and are stripped before
            forwarding. With experts active and no explicit "kb"/"expert", the TRAINED router picks
            the expert: its card fronts the model as the system persona and its KB is the retrieval
            set; below the confidence threshold the request falls through to the plain KB path."""
            messages = req.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages: a list of chat messages")
            body = {k: v for k, v in req.items() if k not in ("kb", "kb_top_k", "expert")}
            last_user_text = next((m.get("content") if isinstance(m.get("content"), str) else
                                   "".join(p.get("text", "") for p in m.get("content", [])
                                           if isinstance(p, dict) and p.get("type") == "text")
                                   for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"), "")
            expert = None                       # [] = multi-select from the trained routers
            if hub is not None:
                want = req.get("expert", "auto")
                if want is False:
                    pass                                       # explicitly off
                elif isinstance(want, str) and want not in ("", "auto"):
                    if want not in hub.experts:
                        raise ValueError(f"unknown expert: {want}")
                    expert = [want]
                elif hub.active and "kb" not in req:            # nothing asked: the heads decide
                    expert = [n for n, _ in hub.route(last_user_text)]
            names = reg.resolve(req.get("kb")) if not expert else tuple(expert)
            if names is None:
                names = reg.snapshot_names()
            if expert:
                if len(expert) == 1:
                    card = hub.experts[expert[0]]["bundle"]["card"]
                else:                                          # compose: one persona, several domains
                    card = ("你同时作为以下领域的专家回答;涉及哪个领域就用哪个领域的规则与事实,并在答案里分别说明:\n\n"
                            + "\n\n".join(f"〔领域 {n}〕\n{hub.experts[n]['bundle']['card']}" for n in expert))
                first = messages[0] if messages else None
                if isinstance(first, dict) and first.get("role") == "system":
                    first["content"] = f"{card}\n\n{first.get('content') or ''}"
                else:
                    messages.insert(0, {"role": "system", "content": card})
            if names:
                top_k = req.get("kb_top_k", 4)
                if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 32:
                    raise ValueError("kb_top_k: a whole number from 1 to 32")
                try:
                    hits = reg.search(
                        next((m.get("content") if isinstance(m.get("content"), str) else
                              "".join(p.get("text", "") for p in m.get("content", [])
                                      if isinstance(p, dict) and p.get("type") == "text")
                              for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"), ""),
                        names, top_k)
                except KeyError as e:                        # a chat request naming a missing KB: a 400, as the
                    raise ValueError(f"unknown knowledge base: {e.args[0]}") from None   # Strata server says it
                if hits:
                    block = "\n".join(f"[knowledge base {h['kb']} #{i + 1} | source: {h['source']}]\n{h['text']}"
                                      for i, h in enumerate(hits))
                    last = next(m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user")
                    if isinstance(last.get("content"), list):
                        last["content"] = [{"type": "text",
                                            "text": f"[knowledge base context]\n{block}\n"
                                                    f"[end of knowledge base context]"}] + list(last["content"])
                        for p in last["content"][1:]:        # llama-server takes OpenAI parts, not the template's
                            p.pop("source", None)
                    else:
                        last["content"] = (f"[knowledge base context]\n{block}\n[end of knowledge base context]"
                                           f"\n\n{last.get('content') or ''}")
            self._forward("POST", body)

        def _forward(self, method, body):
            """One hop to the model server; an SSE reply is piped through as it arrives."""
            data = json.dumps(body).encode() if body is not None else None
            headers = {"Content-Type": "application/json"} if data is not None else {}
            req = urllib.request.Request(upstream + self.path, data=data, headers=headers, method=method)
            try:
                r = urllib.request.urlopen(req, timeout=600)
            except urllib.error.HTTPError as e:
                self._reply(e.code, {"error": {"message": e.read().decode("utf-8", "replace")[:4000]}})
                return
            except OSError as e:
                self._reply(502, {"error": {"message": f"the model server at {upstream} is not answering: {e}"}})
                return
            with r:
                self.send_response(r.status)
                stream = (r.headers.get("Content-Type", "").startswith("text/event-stream"))
                if not stream:
                    self.send_header("Content-Length", r.headers.get("Content-Length", ""))
                for h in ("Content-Type", "Cache-Control"):
                    if r.headers.get(h):
                        self.send_header(h, r.headers[h])
                self.end_headers()
                try:
                    while True:                             # small chunks: the first token stays early
                        chunk = r.read(1024 if stream else 65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        if stream:
                            self.wfile.flush()
                except OSError:
                    pass                                     # the client went away; the model server keeps going

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="the llama-server base URL, e.g. http://127.0.0.1:8902")
    ap.add_argument("--port", type=int, default=8960)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--kb", action="append", default=[], metavar="NAME",
                    help="start with these knowledge bases active (source: --kb-dir/NAME); repeatable")
    ap.add_argument("--kb-dir", help="the root knowledge bases load from (and must stay inside)")
    ap.add_argument("--kb-embedder", default="lexical", metavar="SPEC",
                    help='"lexical" (the default) or "llama:http://host:port" (a llama-server with --embedding)')
    ap.add_argument("--expert-dir", metavar="DIR",
                    help="expert bundles from tools/expert_factory.py; hot at /v1/experts (needs --kb-dir)")
    ap.add_argument("--expert", action="append", default=[], metavar="NAME",
                    help="start with these trained experts active (bundle: --expert-dir/NAME)")
    args = ap.parse_args()

    reg = None
    hub = None
    if args.kb or args.kb_dir:
        try:
            reg = KBRegistry(embedder=make_embedder(args.kb_embedder), root=args.kb_dir or ".")
            for name in args.kb:
                reg.load(name, name)
            reg.swap(args.kb)
        except ValueError as e:
            raise SystemExit(f"knowledge base: {e}")
    if args.expert_dir and reg is None:
        raise SystemExit("--expert-dir needs --kb-dir (an expert fronts a knowledge base)")
    if args.expert_dir:
        root = Path(args.expert_dir)
        hub = ExpertHub(reg, reg.embedder)
        hub.expert_root = root
        try:
            for name in args.expert:
                hub.load(root / name)
            hub.swap(args.expert)
        except (ValueError, KeyError, OSError) as e:
            raise SystemExit(f"expert: {e}")

    print(f"chat kb gateway on http://{args.host}:{args.port} -> {args.upstream}"
          + (f" | knowledge bases: {', '.join(args.kb) or 'none active'} (hot: /v1/kbs)" if reg else "")
          + (f" | experts: {', '.join(args.expert) or 'none active'} (hot: /v1/experts)" if hub else ""),
          file=sys.stderr, flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(args.upstream.rstrip("/"), reg, hub)).serve_forever()


if __name__ == "__main__":
    main()
