# iluvatar-experts-gen

**[中文](README.zh-CN.md) · English**

Turn documents into domain experts on domestic GPUs, Iluvatar 天数智芯. One expert every few minutes. Hot-swappable. No restarts. The stack is Qwen3.6-35B-A3B on an Iluvatar-specific build of llama.cpp.

## Three kinds of experts

**Knowledge-package experts.** One command, about one minute. Produces a responsibility card, a router head and a retrieval index. Users ask in plain language. The system retrieves relevant passages and hands them to the model. Answers carry sources and do not invent facts. The command is `tools/expert_factory.py`.

**Mini JEVs.** One command, about five minutes. Produces a 0.5B-class judgment model that answers in 25 to 30 milliseconds. Runs on a single edge device. The command is `tools/mini_jev.py`.

**LoRA weight experts.** Knowledge written directly into model weights. Data generation, training, conversion and mounting all happen locally on the Iluvatar card in about ten minutes. Training scripts are in `patches/`. Mounting goes through llama.cpp's `/lora-adapters` with zero process restarts.

The three forms complement each other. Knowledge packages update in seconds and stay explainable. LoRA experts answer correctly without retrieval. Mini JEVs serve the edge and the activation layer.

## Multimodal

`tools/multimodal_jev.py` extends the mini JEV to image plus text input. Chinese-CLIP encodes the image, bge-m3 encodes the text, and both feed the same matching head. A user sends a product damage photo with one sentence and the system routes the ticket. Evaluation score is 6 of 6.

## Runtime hot-add for weight experts

`patches/vllm-style-hot-add-lora.patch` is our port of the vLLM ecosystem feature onto the Iluvatar llama.cpp. It lets `POST /lora-adapters` accept entries with a path field and load a new adapter file into a running server. Loading, enabling, disabling, rescaling and per-request routing are all runtime operations. Measured: bare server 4 of 6, one hot-load command later 6 of 6, same process throughout.

## MoE expert profiling

`tools/moe_profile.py` hooks all 40 router gates and profiles real routing. The verdict: zero of the 256 experts are dedicated to any domain at even 50 percent purity. All domains split the experts almost evenly. Semantic structure exists but is faint. Pruning experts by domain is not possible. Expert-targeted LoRA has weak signal.

## Performance

| Scenario | Result |
|---|---|
| JEV-35B repeated decision | 0.1 s, 13.7x faster |
| JEV-35B cold decision | 0.9 s |
| Mini JEV decision | 25 to 30 ms |
| Chat generation | 58.7 tokens per second |
| LoRA hot-load | instant |

## Five-minute quickstart

```bash
pip install numpy
bash scripts/setup-iluvatar.sh
LLAMA_BIN=<iluvatar-llama.cpp-build>/bin bash scripts/start-expert-services.sh
```

Chat on `localhost:5000`, decisions on `localhost:5500`, embeddings on `localhost:8910`, model server on `localhost:8970`.

The repo ships five example bases with all prebuilt artifacts. Train nothing and the serve commands work out of the box.

## Production topology

One Iluvatar card runs Qwen3.6-35B-A3B plus bge-m3. The other card is free for multimodal CLIP or training. One model serves both the chat gateway and the decision gateway. LoRA adapters hot-mount on demand without stopping the server.

## Repo layout

```
serve/knowledge.py          hot-pluggable knowledge-base engine
tools/chat_kb_gateway.py    chat gateway
tools/jev_decide.py         JEV decision gateway
tools/expert_factory.py     knowledge-package expert factory
tools/mini_jev.py           mini-JEV factory
tools/multimodal_jev.py     multimodal mini JEV
tools/moe_profile.py        MoE routing profiler
tools/moe_surgery.py        offline MoE GGUF surgery
tools/jev_distill.py        small-LLM distillation experiments
patches/                     hot-add patch and training scripts
lora/                        law LoRA data and eval sets
kb/                          example knowledge bases
experts*/, mini-jevs/       prebuilt bundles
scripts/                     setup and stack launcher
iluvatar/                    ivcc toolchain and card probe
docs/iluvatar-ops.md         hardware runbook
```

## License

Apache-2.0
