#!/usr/bin/env bash
# start-expert-services.sh — 热插拔知识库专家服务(天数 BI-V150;卡挂死时自动回落 CPU)
#   一切路径基于 /share/hot-remove-experts(pod 重启后 /root 清空,只有 NFS 持久)。
#   布局(卡健康时): 卡0 = bge-m3 嵌入器 + JEV-35B;卡1 = Qwen3.8-27B
#   对外端口: 5000(→30338) 聊天网关;5500(→30955) JEV 决策网关
#   卡不健康(上下文创建挂死,XID 打挂): 全部改跑 CPU(无效卡号 99:CoreX 里空的 CVD 等于
#   "全部设备",仍会去碰挂卡),日志里说明,端口与 API 不变。
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 按脚本实际位置定位(2026-11-02 已从 /share 挪到 /root)
# 2026-11-02 深夜:镜像升 CoreX 5.0 后,4.4 编的二进制在 5.0 驱动上首前向挂死(雷点 1b),
# build-iluvatar-5.0/ 是 5.0 工具链重编的产物,存在即优先;不存在(旧镜像)回落 4.4 目录。
SRC=$BASE/llama-deploy/llama.cpp
if [ -x $SRC/build-iluvatar-5.0/bin/llama-server ]; then
  BIN=$SRC/build-iluvatar-5.0/bin
else
  BIN=$SRC/build-iluvatar/bin
fi
# 雷点:build 树二进制的 RPATH 是构建时的绝对路径(/root/...),挪到 /share 后失效,
# 必须把 bin 目录显式加进 LD_LIBRARY_PATH,否则报 libllama-server-impl.so 找不到。
# 雷点(2026-11-02 深夜,镜像升 CoreX 5.0.0):5.0 不再带 libixthunk.so,而 4.4 编的
# 全部二进制(探针/llama-server)DT_NEEDED 里有它——ldd -r 验证零符号实际来自该库
# (ivcc 链接时显式 -lixthunk 才被记录)。lib-stub/ 放空桩 .so 即可运行。
STUB=$BASE/lib-stub
[ -f $STUB/libixthunk.so ] || { mkdir -p $STUB; echo 'int ixthunk_stub_placeholder;' | \
  gcc -shared -fPIC -o $STUB/libixthunk.so -x c - ; }
export LD_LIBRARY_PATH=$STUB:/usr/local/corex/lib64:/usr/local/iluvatar/lib64:$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
REPO=$BASE/iluvatar-strata
KB=$BASE/kb
MD=$BASE/models
EMB=http://127.0.0.1:8910
PROBE=$BASE/bin/ivcard-probe

gpu_ok() {  # gpu_ok <卡号>: 真碰卡(建上下文+平凡内核),40s 挂死=坏卡(ixsmi 看不出)
  [ "${FORCE_CPU:-0}" = 1 ] && return 1
  [ -x "$PROBE" ] || return 1
  CUDA_VISIBLE_DEVICES=$1 timeout 40 "$PROBE" 2>/dev/null | tail -1 | grep -q 'PASS$'
}

GPU_MODE=yes
for c in 0 1; do gpu_ok $c || GPU_MODE=no; done
if [ "$GPU_MODE" = yes ]; then
  EMB_DEV="CUDA_VISIBLE_DEVICES=0"; QWEN_DEV="CUDA_VISIBLE_DEVICES=1"; JEV_DEV="CUDA_VISIBLE_DEVICES=0"
  echo "[gpu] cards 0/1 healthy — embedder+JEV on 0, Qwen3.8-27B on 1"
else
  EMB_DEV="CUDA_VISIBLE_DEVICES=99"; QWEN_DEV="CUDA_VISIBLE_DEVICES=99"; JEV_DEV="CUDA_VISIBLE_DEVICES=99"
  echo "[cpu] GPU context creation hangs on the exposed cards (wedged; needs a node reboot) —"
  echo "[cpu] running everything on CPU. Same ports, same API; slower (27B ≈ 9 tok/s on 112 cores)."
fi

ready() { for i in $(seq 1 240); do curl -sf -m 2 "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 5; done; return 1; }
start() { # start "<ENV assignments>" <logfile> <args...>
  local envs=$1 log=$2; shift 2
  eval "env $envs setsid nohup $BIN/llama-server $* > $log 2>&1 < /dev/null &"
}

# 1) 嵌入器 :8910 ————————————————————————————————————————————
if ! curl -sf -m 2 $EMB/health >/dev/null; then
  start "$EMB_DEV" /tmp/bge-server.log -m $MD/bge-m3/bge-m3-Q5_K_M.gguf \
    --embedding --pooling mean -c 8192 -ub 8192 -t 16 --port 8910 --host 127.0.0.1
  ready 8910 || { echo "embedder failed:"; tail -3 /tmp/bge-server.log; exit 1; }
fi
echo "[ok] embedder :8910 ($( [ "$GPU_MODE" = yes ] && echo gpu || echo cpu))"

# 2) Qwen3.8-27B :8902 ————————————————————————————————————
if ! curl -sf -m 2 http://127.0.0.1:8902/health >/dev/null; then
  NGL=$([ "$GPU_MODE" = yes ] && echo 99 || echo 0)
  start "$QWEN_DEV" /tmp/qwen-server.log -m $MD/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_M.gguf \
    -ngl $NGL -t 64 -c 8192 --port 8902 --host 127.0.0.1
fi
if ready 8902; then echo "[ok] qwen3.8-27b :8902 ($( [ "$GPU_MODE" = yes ] && echo gpu || echo cpu))"; else echo "[fail] qwen: $(tail -2 /tmp/qwen-server.log)"; fi

# 3) JEV-35B :8970 ——————————————————————————————————————————
if [ -f $MD/JEV-35B-Q4_K_M.gguf ]; then
  if ! curl -sf -m 2 http://127.0.0.1:8970/health >/dev/null; then
    NGL=$([ "$GPU_MODE" = yes ] && echo 99 || echo 0)
    start "$JEV_DEV" /tmp/jev-server.log -m $MD/JEV-35B-Q4_K_M.gguf \
      -ngl $NGL -t 64 -c 4096 --port 8970 --host 127.0.0.1
  fi
  if ready 8970; then echo "[ok] jev-35b :8970"; else echo "[fail] jev: $(tail -2 /tmp/jev-server.log)"; fi
else
  echo "[skip] $MD/JEV-35B-Q4_K_M.gguf missing"
fi

# 4) 网关:对外 5000(聊天)/5500(决策)——————————————————————
if ! curl -sf -m 2 http://127.0.0.1:5000/v1/kbs >/dev/null 2>&1; then
  (cd $REPO && setsid nohup python3 tools/chat_kb_gateway.py \
    --upstream http://127.0.0.1:8970 --port 5000 --host 0.0.0.0 \
    --kb-dir $KB --kb ecom-support --kb-embedder llama:$EMB \
    > /tmp/chat-gateway.log 2>&1 < /dev/null &)
  sleep 2; echo "[ok] chat gateway :5000 (default kb: ecom-support)"
fi
if [ -f $MD/JEV-35B-Q4_K_M.gguf ] && ! curl -sf -m 2 http://127.0.0.1:5500/kb/list >/dev/null 2>&1; then
  (cd $REPO && setsid nohup python3 tools/jev_decide.py \
    --model-dir $MD/JEV-35B --port 8970 --serve 5500 \
    --kb-dir $KB --kb ecom-support --kb-embedder llama:$EMB > /tmp/jev-gateway.log 2>&1 < /dev/null &)
  sleep 2; echo "[ok] jev gateway :5500 (default kb: ecom-support)"
fi
echo "[kbs] $(curl -s http://127.0.0.1:5000/v1/kbs)"
echo "[hint] 外部访问:聊天 http://<节点>:30338/v1/chat/completions;决策 http://<节点>:30955/decide"
