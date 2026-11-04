# iluvatar-experts-gen

**[中文](README.zh-CN.md) · English**

In one sentence: **on domestic GPUs (Iluvatar 天数智芯), turn a pile of documents into a "domain expert" — one every few minutes, no model training, hot-swappable in seconds.**

Under the hood it runs Qwen-family models on llama.cpp (the Iluvatar-specific build). All you need is one Iluvatar card or a many-core CPU box.

## What can it do? Three examples

**Example 1: knowledge Q&A (chat experts)**
Drop your policy docs or product manuals into `kb/`, run one command. From then on users ask questions in plain language; the system retrieves the relevant passages and hands them to the LLM — answers come with sources, no hallucinated facts. Swap in new documents? No restart, effective in seconds.

**Example 2: automatic ticket routing (mini JEVs)**
Same documents, one more command, and you get a "mini JEV": a tiny 0.5B-class judgment model that decides in **25-30 ms** which team a ticket belongs to (billing? shipping? tech support?). It runs on a single edge device and is 6× faster than the big model.

**Example 3: automatic multi-domain composition**
A user asks "my friend drove drunk, hit my parked car, and refuses to pay for the repairs" — that's criminal law AND civil law at once. The system **automatically** activates both experts and merges their knowledge in the answer. The user never picks any "mode"; they just ask.

## JEV-35B decision acceleration (measured)

We ran a focused acceleration pass on JEV-35B, the big judgment model used in production. Everything was done inside llama.cpp — no engine swap, no torch:

| Scenario | Before | After | How |
|---|---|---|---|
| same decision asked again | 1.35 s | **0.1 s (13.7× faster)** | turned on prompt caching — which this engine ships OFF by default |
| first decision on a new ticket | 1.35 s | **0.9 s (32% faster)** | retrieve 2 passages instead of 4; all 5 standard tests still pass |
| several questions about one ticket | recomputed from scratch each time | shared computation | new `/decide_batch` endpoint |

We also mapped the engine's floor (documented so nobody re-sweeps it): this llama.cpp build has **no partial prefix reuse** — change one word and the whole prompt is re-processed; prefill sits at ~300 tokens/s, the MoE kernel ceiling, and thread/batch flags change nothing. The only levers that work: **shorter prompts** and **exact-resend caching**.

## Mini-JEV results, and one important measured lesson

| Approach | Size | Standard 5-test suite | Latency |
|---|---|---|---|
| 0.6B small LLM + calibration | 0.6B | 1/5 (output independent of input — dead) | — |
| 4B small LLM + calibration | 4B | 4/5 | 0.2-0.4 s |
| **the mini JEV in this repo** | **0.5B class, 467 MB model** | **5/5** | **25-30 ms** |
| JEV-35B (reference) | 3B active | 5/5 | 0.1-0.9 s |

The lesson: **you cannot get judgment ability out of a 0.6B LLM by tuning its calibration** — a backbone that small simply has no signal. The community agrees (either LoRA-train it, or start at 4B). This repo takes a third road: freeze an off-the-shelf text encoder (bge-m3) and train only a tiny "matching head" with numpy — no LLM training, no training GPU, and it ends up both the most accurate and the fastest.

## Five-minute quickstart

```bash
pip install numpy                              # the only dependency
bash scripts/setup-iluvatar.sh                 # Iluvatar environment setup (probe, symlinks)

# step 1: start the stack (chat model + embedder + gateway)
LLAMA_BIN=<llama.cpp-iluvatar-build>/bin bash scripts/start-expert-services.sh

# step 2: turn documents into a chat expert (fully automatic, ~1 min)
python3 tools/expert_factory.py build my-domain --source kb/my-domain \
    --llm http://127.0.0.1:8902 --kb-embedder llama:http://127.0.0.1:8910 --out experts

# step 3: use it immediately (nothing restarts)
curl -X POST localhost:5000/v1/experts -d '{"name": "my-domain"}'
curl -X POST localhost:5000/v1/experts/swap -d '{"active": ["my-domain"]}'
curl localhost:5000/v1/chat/completions -d '{"messages":[{"role":"user","content":"your question"}]}'

# want ticket-routing too? one command (~5 min), then 25 ms per decision
python3 tools/mini_jev.py --build my-domain --kb-dir kb --out mini-jevs
python3 tools/mini_jev.py --domain my-domain --kb-dir kb --serve 5504
curl localhost:5504/decide -d '{"state":"ticket text","question":"Which team?","options":["billing","shipping"]}'
```

The repo ships three ready-made examples (e-commerce support, GPU cloud ops, civil/criminal law) with prebuilt expert bundles and mini JEVs — **train nothing**, the serve commands above work out of the box.

## How it works (one paragraph)

Documents are chunked and embedded; retrieval is a dot product against the matrix, which is why load/swap takes milliseconds. An "expert" = a knowledge base + an auto-generated "responsibility card" (telling the LLM who it is and which rules to follow) + an auto-trained router head (deciding whether an incoming question belongs to this expert — if yes, it activates). The "mini JEV" hands the "who handles this" judgment to a frozen encoder plus a small numpy head: 25 ms per answer. **No LLM training anywhere** — every "training" step is just fitting a very small part, in minutes.

## What "expert" actually means (read this before you complain)

The "experts" in this repo are **NOT** the MoE expert weights inside the base model — those FFN blocks are untouched and untouchable here (no trainer on this stack). Our experts live in the **activation layer**: a knowledge base + a responsibility card + a trained router head. What you get beyond a bare RAG setup is the *judgment* — which domain a question belongs to, whether to compose several, when to abstain — done by genuinely trained (numpy-fitted) small heads at 25 ms.

The E in MoE still matters, implicitly: the reason one base model can answer credibly across all these domains is its MoE capacity. Operating on the E explicitly (training real weight deltas per domain) has exactly one road here: produce LoRA adapters on an external trainer and hot-mount them via llama-server's `/lora-adapters` — the interface is ready, see the roadmap.

## Known limitations (honestly)

- A newly landed expert is routable instantly (zero-shot cold start), but taking over an **adjacent domain** needs one factory rerun (~3 min) — a trained router head always out-scores a zero-shot anchor on its own vocabulary;
- routing uses "independent per-domain judgment + abstention", not one softmax over all experts — measured: the latter must allocate probability mass somewhere and mis-fires on chit-chat;
- the Iluvatar llama.cpp build has real pitfalls (card wedging, CoreX 5.0 missing libs) — `docs/iluvatar-ops.md` is the field log, every entry earned the hard way.

## Repo layout

```
serve/knowledge.py        the hot-pluggable knowledge-base engine
tools/chat_kb_gateway.py  chat gateway: retrieval + expert auto-discovery/routing/composition
tools/jev_decide.py       JEV-35B decision gateway (with the /decide_batch endpoint)
tools/expert_factory.py   chat-expert factory (~1 min per domain)
tools/mini_jev.py         mini-JEV factory + server (~5 min per domain, 25 ms per decision)
tools/jev_distill.py      small-LLM distillation experiments (where the 0.6B/4B threshold came from)
tools/moe_surgery.py      offline MoE GGUF surgery (layer cuts, expert-pruning research)
kb/                       example knowledge bases
experts*/, mini-jevs/     prebuilt bundles (regenerate with the factories anytime)
scripts/                  setup + stack launcher
docs/iluvatar-ops.md      the Iluvatar hardware & build runbook
```

## Real weight-increment experts: done once, end to end

A Qwen3-4B LoRA law expert (civil + criminal) was trained on an external RTX 3090
(46 rejection-sampled cases, 33M trainable params = 0.81%, 3 minutes), converted with
`convert_lora_to_gguf.py`, and hot-mounted on the Iluvatar pod through llama-server's
`/lora-adapters`. Verified behavior on a deliberately hard boundary set (blood-alcohol
72/65/81 mg edges, sober accidents, hit-and-run + refusal): adapter OFF 4/6 with p=0.25
(the base guessing) → runtime `POST /lora-adapters [{"id":0,"scale":1.0}]` → **6/6 with
p≥0.99**, same server PID throughout (zero restarts). **Update — vLLM-style hot-add, ported to our Iluvarian llama.cpp**: `patches/vllm-style-hot-add-lora.patch`
makes `POST /lora-adapters` accept `{"path": ..., "scale": ...}` entries that load a NEW
adapter file into the RUNNING server (task-loop-side load, single-threaded, race-free).
Measured: bare server 4/6 → one POST → 6/6, same PID. Per-request routing via the
request-level `"lora"` field also verified (global off + request-level on → correct).
Previous limits about pre-loaded-only adapters no longer apply to this build.

Interface limits measured: only
pre-loaded adapters can be managed at runtime (unknown ids silently no-op), and
`--lora-init-without-apply` does not hold on this build. Artifacts in `lora/`.

## Roadmap

- [ ] Requantize JEV-35B to Q3_K_M for another ~20-25% (bandwidth-bound regime; re-run the 5-test suite)
- [ ] score / noul decision kinds for mini JEVs
- [x] LoRA side-load path (first expert shipped, see above) (adapters trained elsewhere, hot-mounted via llama-server's /lora-adapters)

## Acknowledgments

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (the Iluvatar/CoreX port), Qwen-family models, and the typed-decision ("Jev") model idea; the mini-JEV design follows the community consensus (mini-Jev, kotoba typed-decisions, joyfox). Full field log in `docs/iluvatar-ops.md`.

## License

Apache-2.0
