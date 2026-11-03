# iluvatar-experts-gen

**[中文](README.zh-CN.md) · English**

Mass-produce **MoE experts** and **mini JEVs** on domestic GPUs (Iluvatar 天数智芯 BI-V150 / CoreX) — knowledge-base **side-loading** and **edge-grade domain decisions**, built on the **Qwen** architecture.

One base stack (Qwen-family models served by an Iluvatar-built llama.cpp) powers two fully automatic production lines:

| Line | Factory | Output | Hot-plug via | Build time |
|---|---|---|---|---|
| 💬 Chat experts | `tools/expert_factory.py` | expert bundle: trained router head + graded expert card + KB | `POST /v1/experts` on the chat gateway | ~45 s / domain |
| ⚡ Mini JEVs (≤1B decisions) | `tools/mini_jev.py` | frozen-encoder matching head + anchors + KB | `--domain X --serve PORT` | ~5 min / domain |

Both factories share the same facilities: raw documents → hot-pluggable knowledge bases, a teacher (Qwen3.8-27B or JEV-35B) for self-supervised distillation, and bge-m3 embeddings. **No backprop, no GPU training, no human labels anywhere in the loop** — every trainable piece is fitted with numpy.

## Results (measured on 2× BI-V150)

| Decision approach | Active compute | Canonical 5-test suite | Latency / decision |
|---|---|---|---|
| 0.6B LLM + head calibration | 0.6B | 1/5 (input-independent — dead) | 40–140 ms |
| 4B LLM + head calibration | 4B | 4/5 | 192–376 ms |
| **mini-JEV (frozen bge-m3 + rank-64 matching head)** | **0.5B-class, 467 MB model** | **5/5** | **25–30 ms** |
| JEV-35B teacher (reference) | 3B active | 5/5 | ~172 ms warm |

Chat side: Qwen3.8-27B at 12.6 tok/s on GPU; expert hot-swap is second-level with **zero model-process restarts** (verified by PID across load/swap/use/delete cycles).

## Quickstart

```bash
pip install numpy                          # the only dependency
bash scripts/setup-iluvatar.sh             # ivcc + CoreX symlinks + probe (per pod reset)
LLAMA_BIN=<llama.cpp-iluvatar-build>/bin bash scripts/start-expert-services.sh

# — mass-produce a chat expert from raw documents (self-distilled, rejection-sampled) —
python3 tools/expert_factory.py build my-domain --source kb/my-domain \
    --llm http://127.0.0.1:8902 --kb-embedder llama:http://127.0.0.1:8910 --out experts

# — hot-plug it into the running chat gateway (no restart of anything) —
curl -X POST localhost:5000/v1/experts -d '{"name": "my-domain"}'
curl -X POST localhost:5000/v1/experts/swap -d '{"active": ["my-domain"]}'
curl localhost:5000/v1/chat/completions -d '{"messages":[{"role":"user","content":"…"}]}'

# — mass-produce a mini JEV (frozen encoder + numpy-fitted matching head) —
python3 tools/mini_jev.py --build my-domain --kb-dir kb --out mini-jevs
python3 tools/mini_jev.py --domain my-domain --kb-dir kb --serve 5504
curl localhost:5504/decide -d '{"state":"…","question":"Which team?","options":["a","b","c"]}'
```

The repo ships with three example KBs (`kb/ecom-support`, `kb/cloud-ops`, `kb/facility-cooling`) and pre-built bundles for all of them, so the serve commands above work before you build anything.

## What an "expert" is here

- **Knowledge base** — plain files (`.md/.txt/.jsonl/...`) chunked and embedded; retrieval is a dot product against a row-normalized matrix. Hot load/unload/swap/multi-active at runtime; in-flight requests keep the snapshot they started with.
- **Chat expert bundle** — a trained softmax **router head** over question embeddings (real gradient descent, numpy) + an **expert card** (system persona, two styles drafted by the model and graded on held-out QA by exact-fact match, better one ships) + the KB it fronts.
- **Mini JEV** — a ≤1B decision expert: `logit_i = (U·e_query)·(V·e_option)` with a frozen bge-m3 encoder; options are represented by **anchor texts** distilled once from the KB, so head parameters never grow with the candidate count and requests may declare any subset/order of options. Serves the same `/decide` shape as the JEV-35B gateway (`kind=choice`).

The tools stay honest about what each layer can and cannot do: the 0.6B experiment (kept in `tools/jev_distill.py`) documents that calibration-only distillation cannot rescue a backbone with no signal, which is exactly why the mini-JEV line exists.

## Repo layout

```
serve/knowledge.py        the hot-pluggable KB engine (embedders, chunker, registry snapshots)
tools/chat_kb_gateway.py  OpenAI-compatible chat gateway: KB injection + expert hot-plug + trained routing
tools/jev_decide.py       JEV-35B decision gateway (verbalizer read-out, retry, hot KBs)
tools/expert_factory.py   chat-expert factory (self-distill QA → router head → graded card)
tools/mini_jev.py         mini-JEV factory + server (frozen encoder + matching head)
tools/jev_distill.py      teacher→LLM-student head distillation (size-threshold experiments)
tools/moe_surgery.py      offline MoE GGUF surgery (prefix-cut; expert pruning = roadmap)
iluvatar/                 ivcc (CoreX clang CUDA wrapper), corex_compat.h, probe source
kb/                       example knowledge bases
experts*/, mini-jevs/     pre-built example bundles (regenerate with the factories)
scripts/                  pod setup, stack launcher, decision demo suite
docs/iluvatar-ops.md      the hardware & build runbook (pitfalls, all learned the hard way)
```

## Roadmap

- [ ] Expert pruning in `moe_surgery.py` (keep depth, cut 256→K experts) for slim JEV-35B variants
- [ ] `score` / `noul` kinds for mini JEVs; per-request option anchoring for unseen teams
- [ ] LoRA side-load path (`POST /lora-adapters` on llama-server) once an external trainer produces adapters
- [ ] More encoders for the mini-JEV line (bge-small) to push edge footprint below 150 MB

## Acknowledgments

Built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) (Iluvatar/CoreX port), Qwen-family models, and the typed-decision ("Jev") model idea; serving ideas informed by the open small-JEV community (mini-Jev, kotoba typed-decisions, joyfox 0.8B). See `docs/iluvatar-ops.md` for the full field log.

## License

Apache-2.0
