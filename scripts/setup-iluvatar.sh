#!/usr/bin/env bash
# setup-iluvatar.sh — one-time hardware/toolchain setup inside an Iluvatar CoreX pod.
# Idempotent: run it after every pod/image reset, then start-expert-services.sh.
set -e
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== 1/3 ivcc toolchain (CoreX clang CUDA wrapper) =="
[ -x /usr/local/bin/ivcc ] || { cp $BASE/iluvatar/ivcc /usr/local/bin/ && chmod +x /usr/local/bin/ivcc && echo "   ivcc installed"; }
[ -f /usr/local/bin/corex_compat.h ] || cp $BASE/iluvatar/corex_compat.h /usr/local/bin/
[ -e /usr/local/corex-4.4.0 ] || ln -s /usr/local/corex /usr/local/corex-4.4.0
# CoreX 5.0 ships libcuda.so.1 without the dev symlink; CMake and ivcc both need it
[ -e /usr/local/corex/lib64/libcuda.so ] || ln -sf /usr/local/corex/lib64/libcuda.so.1 /usr/local/corex/lib64/libcuda.so
echo "   ok"

echo "== 2/3 card health probe =="
echo "   probe: CUDA_VISIBLE_DEVICES=<n> timeout 40 $BASE/bin/ivcard-probe   (PASS = card usable;"
echo "   a hang means the card is wedged (XID); only a node reboot fixes it — ixsmi cannot see this)"

echo "== 3/3 python deps =="
python3 -c 'import numpy' 2>/dev/null && echo "   numpy ok" || { pip3 install -q numpy && echo "   numpy installed"; }
echo "done. Next: LLAMA_BIN=<llama.cpp iluvatar build>/bin scripts/start-expert-services.sh"
