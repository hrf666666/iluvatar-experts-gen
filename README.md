# iluvatar-experts-gen

**[中文](README.zh-CN.md) · English**

Turn a pile of documents into domain experts on domestic GPUs, Iluvatar 天数智芯. One expert every few minutes. The base model is never trained. Experts go live and get replaced without restarts. The stack underneath is Qwen-family models on an Iluvatar-specific build of llama.cpp.

## Three kinds of experts from one pipeline

One knowledge base yields three forms.

**Knowledge-package experts.** One command, about one minute. Produces a responsibility card, a router head and a retrieval index. When a user asks a question the system retrieves the relevant passages and hands them to the model. Answers carry sources and do not invent facts. Swapping documents takes effect in seconds with no restart. The command is `tools/expert_factory.py`.

**Mini JEVs.** One command, about five minutes. Produces a 0.5B-class micro judgment model that answers in 25 to 30 milliseconds, for example which team a ticket belongs to. It runs on a single edge device. The command is `tools/mini_jev.py`.

**LoRA weight experts.** The knowledge is written into the model weights. Data generation, training, conversion and mounting now all happen locally on the Iluvatar card in about ten minutes, with no external NVIDIA machine. The training script is `patches/train_local_iluvatar.py` and it uses the vendor-supplied torch 2.10.0+corex.5.0.0.

The three forms complement each other. Knowledge packages update in seconds and stay explainable. LoRA experts answer correctly without retrieval and resist prompt interference. Mini JEVs serve the edge and the activation layer.

## The full life cycle of a weight expert

Training uses a Qwen3-4B base with LoRA, 33M trainable parameters, 0.81 percent of the model. It converges in about five minutes on the Iluvatar card. The adapter converts to a 66MB GGUF file and mounts into a running server with one command.

The mounting interface is our port of the vLLM ecosystem feature onto the Iluvatar llama.cpp. The patch is `patches/vllm-style-hot-add-lora.patch`. It lets `POST /lora-adapters` accept entries with a path field and load a new adapter file into a running server with zero process restarts. Loading, enabling, disabling, rescaling and per-request routing are all runtime operations.

Behavioral verification used a deliberately hard boundary set with blood-alcohol levels of 72, 65 and 81 milligrams per hundred milliliters, sober accidents and hit-and-run with refusal to pay. The bare base scores 4 of 6. After hot-loading the weight expert it scores 6 of 6 and confidence rises from 0.25 to above 0.99.

## JEV-35B decision acceleration

Everything happens inside llama.cpp with no engine swap.

| Scenario | Before | After |
|---|---|---|
| same decision asked again | 1.35 s | 0.1 s |
| first decision on a new ticket | 1.35 s | 0.9 s |
| several questions about one ticket | recomputed per question | shared computation |

Two levers work. Turning on prompt caching, which this engine ships disabled. Reducing retrieved passages from four to two, which keeps all five standard tests passing. The engine limits are also mapped. This build has no partial prefix reuse, so changing one word reprocesses the whole prompt. Prefill sits at about 300 tokens per second, the MoE kernel ceiling, and thread and batch flags change nothing. Recommended JEV server flags are `-t 16 --parallel 1`.

## Mini-JEV results and one measured lesson

| Approach | Size | Five-test suite | Latency |
|---|---|---|---|
| 0.6B small LLM with calibration | 0.6B | 1 of 5 | no signal |
| 4B small LLM with calibration | 4B | 4 of 5 | 0.2 to 0.4 s |
| the mini JEV in this repo | 0.5B | 5 of 5 | 25 to 30 ms |
| JEV-35B reference | 3B active | 5 of 5 | 0.1 to 0.9 s |

The measured lesson is that tuning the calibration of a 0.6B LLM yields no judgment ability. A backbone that small carries no signal. The community either LoRA-trains it or starts at 4B. This repo takes a third road, freezing an off-the-shelf encoder, bge-m3, and training only a small matching head with numpy. No LLM training, no training GPU, best accuracy, fastest speed.

## What an expert is

The experts in this repo are not the MoE expert weights inside the base model. Those FFN blocks are untouched. Our experts live in the activation layer as a knowledge base plus a responsibility card plus trained router heads. What exceeds bare retrieval is judgment, deciding which domain a question belongs to, whether to compose several, and abstaining when none applies. Trained small heads do this in about 25 milliseconds.

The MoE capacity still matters inside the base. One base answering credibly across all domains rests on it. Turning that capacity into explicit per-domain experts is exactly the LoRA training and hot-mounting this repo now runs end to end.

## Five-minute quickstart

```bash
pip install numpy
bash scripts/setup-iluvatar.sh

LLAMA_BIN=<iluvatar-llama.cpp-build>/bin bash scripts/start-expert-services.sh

# documents to chat expert, about one minute
python3 tools/expert_factory.py build my-domain --source kb/my-domain \
    --llm http://127.0.0.1:8902 --kb-embedder llama:http://127.0.0.1:8910 --out experts

# usable immediately, nothing restarts
curl -X POST localhost:5000/v1/experts -d '{"name": "my-domain"}'
curl -X POST localhost:5000/v1/experts/swap -d '{"active": ["my-domain"]}'
curl localhost:5000/v1/chat/completions -d '{"messages":[{"role":"user","content":"your question"}]}'

# ticket-routing mini JEV, about five minutes
python3 tools/mini_jev.py --build my-domain --kb-dir kb --out mini-jevs
python3 tools/mini_jev.py --domain my-domain --kb-dir kb --serve 5504
```

The repo ships five example bases, e-commerce support, GPU cloud ops, civil law, criminal law and facility cooling, with all prebuilt artifacts. Train nothing and the serve commands work out of the box.

## How it works

Documents are chunked and embedded. Retrieval is a dot product against the matrix, so loading and switching take milliseconds. An expert is a knowledge base, an auto-generated responsibility card and an auto-trained router head. The router head decides whether a question belongs to this expert and activates it when it does. When several domains hit at once the personas compose automatically and retrieval merges across bases by score, invisibly to the user. A newly landed expert is routable immediately. Taking over an adjacent domain needs one factory rerun, because a trained router head always out-scores a zero-shot anchor on its own vocabulary. Routing uses independent per-domain judgment with abstention. One softmax over all experts must allocate probability mass somewhere and mis-fires on chit-chat, which is a measured result.

No LLM training happens anywhere. Knowledge packages and mini JEVs train only small parts with numpy. The 33M-parameter LoRA expert trains on the Iluvatar card with the vendor torch.

## Known limitations

Local training on the Iluvatar card scores 5 of 6 on the boundary set versus 6 of 6 on the 3090. The gap comes from the hand-written pure bf16 loop versus mixed precision. The fix direction is fp32 master weights.

The Iluvatar llama.cpp build has real pitfalls, probe-based card health checks, missing CoreX 5.0 libraries, pyarrow and numpy version pairing. `docs/iluvatar-ops.md` is the complete field log.

## Repo layout

```
serve/knowledge.py        the hot-pluggable knowledge-base engine
tools/chat_kb_gateway.py  chat gateway with retrieval, expert auto-discovery and routing
tools/jev_decide.py       JEV-35B decision gateway with the batch endpoint
tools/expert_factory.py   knowledge-package expert factory
tools/mini_jev.py         mini-JEV factory and server
tools/jev_distill.py      small-LLM distillation experiments
tools/moe_surgery.py      offline MoE GGUF surgery
patches/                  the vLLM-style hot-add patch and the local training script
lora/                     law LoRA expert data and eval sets
kb/                       example knowledge bases
experts*/, mini-jevs/     prebuilt bundles
scripts/                  setup and stack launcher
docs/iluvatar-ops.md      the Iluvatar hardware and build runbook
```

## Roadmap

- [ ] Requantize JEV-35B to Q3_K_M for another 20 to 25 percent in the bandwidth-bound regime
- [ ] score and noul decision kinds for mini JEVs
- [ ] raise local Iluvatar training from 5 of 6 to full marks with fp32 master weights

## Acknowledgments

Built on the Iluvatar CoreX port of llama.cpp, Qwen-family models and the typed-decision model idea. The mini-JEV design follows the public practice of mini-Jev, kotoba typed-decisions and joyfox. The full field log is `docs/iluvatar-ops.md`.

## License

Apache-2.0
