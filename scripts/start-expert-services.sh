#!/usr/bin/env bash
# start-expert-services.sh — bring up the full serving stack for iluvatar-experts-gen
# on an Iluvatar (天数智芯) CoreX node, with automatic CPU fallback for wedged cards.
#
#   Layout (healthy cards):  card0 = bge-m3 embedder + JEV-35B   |   card1 = Qwen chat model
#   Ports: 8910 embedder · 8902 chat model · 8970 JEV model · 5000 chat gateway · 5500 decision gateway
#
# Everything is overridable via env:
#   LLAMA_BIN   path to llama.cpp build .../bin           (default: auto-detect under ../hot-remove-experts or ../llama-deploy)
#   MODELS_DIR  directory holding the GGUF models          (default: ../hot-remove-experts/models)
#   CHAT_GGUF   name under $MODELS_DIR                     (default: Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_M.gguf)
#   JEV_GGUF    name under $MODELS_DIR                     (default: JEV-35B-Q4_K_M.gguf)
#   FORCE_CPU=1 skip GPU probing, run everything on CPU
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB=$BASE/kb
EMB=http://127.0.0.1:8910

# ---- locate the llama.cpp binaries (CoreX-5.0-built tree preferred, see docs/iluvatar-ops.md) ----
if [ -z "${LLAMA_BIN:-}" ]; then
  for c in "$BASE/../hot-remove-experts/llama-deploy/llama.cpp" "$BASE/../llama-deploy/llama.cpp" "$BASE/llama.cpp"; do
    [ -x "$c/build-iluvatar-5.0/bin/llama-server" ] && LLAMA_BIN="$c/build-iluvatar-5.0/bin" && break
    [ -x "$c/build-iluvatar/bin/llama-server" ]     && LLAMA_BIN="$c/build-iluvatar/bin" && break
  done
fi
[ -x "${LLAMA_BIN:-/nonexistent}/llama-server" ] || { echo "set LLAMA_BIN to a llama.cpp Iluvatar build's bin/ (see docs/iluvatar-ops.md)"; exit 1; }
MODELS_DIR="${MODELS_DIR:-$BASE/../hot-remove-experts/models}"
CHAT_GGUF="${CHAT_GGUF:-Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_M.gguf}"
JEV_GGUF="${JEV_GGUF:-JEV-35B-Q4_K_M.gguf}"
[ -f "$MODELS_DIR/$CHAT_GGUF" ] || { echo "chat model not found: $MODELS_DIR/$CHAT_GGUF (set MODELS_DIR/CHAT_GGUF)"; exit 1; }

# ---- CoreX 5.0 images dropped libixthunk.so; 4.4-built binaries never call into it -> stub ----
STUB=$BASE/lib-stub
[ -f $STUB/libixthunk.so ] || { mkdir -p $STUB; echo 'int ixthunk_stub_placeholder;' | \
  gcc -shared -fPIC -o $STUB/libixthunk.so -x c - ; }
export LD_LIBRARY_PATH=$STUB:/usr/local/corex/lib64:/usr/local/iluvatar/lib64:$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PROBE=$BASE/bin/ivcard-probe

gpu_ok() { [ "${FORCE_CPU:-0}" = 1 ] && return 1
  [ -x "$PROBE" ] || return 1
  CUDA_VISIBLE_DEVICES=$1 timeout 40 "$PROBE" 2>/dev/null | tail -1 | grep -q 'PASS$'; }
GPU_MODE=yes
for c in 0 1; do gpu_ok $c || GPU_MODE=no; done
if [ "$GPU_MODE" = yes ]; then
  EMB_DEV="CUDA_VISIBLE_DEVICES=0"; QWEN_DEV="CUDA_VISIBLE_DEVICES=1"; JEV_DEV="CUDA_VISIBLE_DEVICES=0"
  echo "[gpu] cards 0/1 healthy — embedder+JEV on 0, chat model on 1"
else
  EMB_DEV="CUDA_VISIBLE_DEVICES=99"; QWEN_DEV="CUDA_VISIBLE_DEVICES=99"; JEV_DEV="CUDA_VISIBLE_DEVICES=99"
  echo "[cpu] GPU unavailable — everything on CPU (in CoreX an EMPTY CUDA_VISIBLE_DEVICES means ALL devices, so we pin 99)"
fi

ready() { for i in $(seq 1 240); do curl -sf -m 2 "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 5; done; return 1; }
start() { local envs=$1 log=$2; shift 2
  eval "env $envs setsid nohup $LLAMA_BIN/llama-server $* > $log 2>&1 < /dev/null &"; }

# 1) embedder :8910 ————————————————————————————————————————————
if ! curl -sf -m 2 $EMB/health >/dev/null; then
  start "$EMB_DEV" /tmp/bge-server.log -m $MODELS_DIR/bge-m3/bge-m3-Q5_K_M.gguf \
    --embedding --pooling mean -c 8192 -ub 8192 -t 16 --port 8910 --host 127.0.0.1
  ready 8910 || { echo "embedder failed:"; tail -3 /tmp/bge-server.log; exit 1; }
fi
echo "[ok] embedder :8910"

# 2) chat model :8902 ————————————————————————————————————————
if ! curl -sf -m 2 http://127.0.0.1:8902/health >/dev/null; then
  NGL=$([ "$GPU_MODE" = yes ] && echo 99 || echo 0)
  start "$QWEN_DEV" /tmp/qwen-server.log -m $MODELS_DIR/$CHAT_GGUF -ngl $NGL -t 64 -c 8192 --port 8902 --host 127.0.0.1
fi
ready 8902 && echo "[ok] chat model :8902 ($( [ "$GPU_MODE" = yes ] && echo gpu || echo cpu))" || echo "[fail] chat: $(tail -2 /tmp/qwen-server.log)"

# 3) JEV model :8970 ——————————————————————————————————————————
if [ -f $MODELS_DIR/$JEV_GGUF ] && ! curl -sf -m 2 http://127.0.0.1:8970/health >/dev/null; then
  NGL=$([ "$GPU_MODE" = yes ] && echo 99 || echo 0)
  start "$JEV_DEV" /tmp/jev-server.log -m $MODELS_DIR/$JEV_GGUF -ngl $NGL -t 64 -c 4096 --port 8970 --host 127.0.0.1
fi
[ -f $MODELS_DIR/$JEV_GGUF ] && { ready 8970 && echo "[ok] jev model :8970" || true; }

# 4) gateways —————————————————————————————————————————————————
if ! curl -sf -m 2 http://127.0.0.1:5000/v1/kbs >/dev/null 2>&1; then
  (cd $BASE && setsid nohup python3 tools/chat_kb_gateway.py \
    --upstream http://127.0.0.1:8902 --port 5000 --host 0.0.0.0 \
    --kb-dir $KB --kb ecom-support --kb-embedder llama:$EMB \
    > /tmp/chat-gateway.log 2>&1 < /dev/null &)
  sleep 2; echo "[ok] chat gateway :5000 (default kb: ecom-support)"
fi
if [ -f $MODELS_DIR/$JEV_GGUF ] && ! curl -sf -m 2 http://127.0.0.1:5500/kb/list >/dev/null 2>&1; then
  (cd $BASE && setsid nohup python3 tools/jev_decide.py \
    --model-dir $MODELS_DIR/JEV-35B --port 8970 --serve 5500 \
    --kb-dir $KB --kb ecom-support --kb-embedder llama:$EMB > /tmp/jev-gateway.log 2>&1 < /dev/null &)
  sleep 2; echo "[ok] decision gateway :5500 (default kb: ecom-support)"
fi
echo "[kbs] $(curl -s http://127.0.0.1:5000/v1/kbs)"
echo "[hint] mini-JEVs need no GPU model: python3 tools/mini_jev.py --domain ecom-support --kb-dir kb --serve 5504"
