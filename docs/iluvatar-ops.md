# Iluvatar / CoreX 运维与构建手册（Iluvatar Ops Runbook）

> Everything below was learned the hard way on BI-V150 pods (CoreX 4.4 → 5.0 images, 2026-11).
> 下面所有条目都是 BI-V150 pod 上的实测踩坑记录。English summary at the end.

## 1. 显卡健康（最重要的第一条）

- 容器透出的卡可能被 XID 打挂：症状是任何真碰卡的进程在"上下文创建"处无限挂起。**ixsmi 看不出任何异常**。
- 唯一可信判据是本仓库的探针：`CUDA_VISIBLE_DEVICES=<卡号> timeout 40 bin/ivcard-probe`——40 秒不返回即坏卡，节点重启前无解。
- `torch.cuda.is_available()` 和 ixsmi 一样走枚举层，坏卡上也返回 True，**不能当健康判据**。
- CoreX 里**空的 `CUDA_VISIBLE_DEVICES` 等于"全部设备"**。回落 CPU 必须用 `CUDA_VISIBLE_DEVICES=99`（无效卡号）。

## 2. CoreX 5.0 镜像迁移坑

- **libixthunk.so 被删**：4.4 编的二进制 DT_NEEDED 里有它但零符号实际使用——空桩即可（`start-expert-services.sh` 自动生成 `lib-stub/`）。
- **无 libcuda.so 开发软链**（只有 .so.1）：CMake 报 `CUDA::cuda_driver not found`；`ln -sf libcuda.so.1 /usr/local/corex/lib64/libcuda.so`。
- **5.0 头文件原生提供了 4.4 的三样垫片**（标量 `__ldcs`、driver entry point 类型、批量拷贝结构）——`corex_compat.h` 已用 `-DCOREX_GE_50=1` 门控（ivcc 自动检测特征宏传入）。
- 4.4 编的 llama.cpp 在 5.0 驱动上**首个前向挂死**（futex 等待 + GPU 利用率恒定）：必须用 5.0 工具链重编。

## 3. llama.cpp 构建（Iluvatar）

```bash
# 树：commit 4413a19 + iluvatar 补丁（见上游 hrf666666/iluvatar-strata）
SRC=llama.cpp; BD=$SRC/build-iluvatar-5.0
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDACXX=/usr/local/bin/ivcc \
       CUDA_PATH=/usr/local/corex-4.4.0 CUDAToolkit_ROOT=/usr/local/corex-4.4.0
cmake -B $BD -S $SRC -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/bin/ivcc \
  -DCMAKE_CUDA_ARCHITECTURES=ivcore11 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build $BD --target llama-server llama-cli llama-quantize -j 56
```

- 已打补丁的树上存在 `.ivcore_patched` 标记，构建脚本的 sed/stub 段会自动跳过；**不要**在已打补丁的树上重跑补丁。
- 产物 RPATH 是构建时绝对路径：运行时必须把 `.../bin` 加进 `LD_LIBRARY_PATH`。

## 4. 运行时怪癖

| 现象 | 处置 |
|---|---|
| llama-cli 在 GPU 上启动即挂（MoE 模型） | warmup 所致，加 `--no-warmup`；llama-server 不受影响 |
| 极偶发 `logprobs: null`（slot 首任务） | 重发即好（走前缀缓存 ~140ms）；`jev_decide.py` 已内建 3 次重试 |
| 极偶发生成中途乱码退化 | 同族毛刺，重发即好；对延迟敏感的调用建议 temperature ≤ 0.2 + 一次重试 |
| Qwen 系聊天"正文为空" | 默认开 thinking，max_tokens 全花在思考段：加到 700+ 或 `chat_template_kwargs.enable_thinking=false` |

## 5. 运维卫生

- 镜像**没有 fuser**：`fuser -k` 静默无效，旧进程占端口会让新进程悄悄死掉。清理一律先 `ps` 再 `kill <PID>`。
- `pkill -f` 的模式若出现在同一命令块里（比如同时有 `xxx.py` 和端口号字样），会匹配自己的 shell——模式加 `[]` 防自匹配，且与启动命令分开成条。
- `LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/iluvatar/lib64` 是 ixsmi/torch import 的前置条件。

## 6. 性能基线（2× BI-V150, 2026-11 实测）

| 工作负载 | 表现 |
|---|---|
| Qwen3.8-27B Q4_K_M（单卡） | 12.6 tok/s 生成，23 tok/s 提示 |
| JEV-35B 决策（热前缀缓存） | ~172 ms |
| 小 JEV（bge-m3 + 匹配头） | 25–30 ms |
| bge-m3 嵌入 | ~45 ms/条 |

---

## English summary

1. **Card health**: `ixsmi` and `torch.cuda.is_available()` cannot detect wedged cards; only `bin/ivcard-probe` (40 s timeout) can. Empty `CUDA_VISIBLE_DEVICES` means *all* devices in CoreX — pin `99` for CPU fallback.
2. **CoreX 5.0 migration**: ship a stub `libixthunk.so` (nothing actually calls it), symlink `libcuda.so`, and rebuild llama.cpp with the 5.0 toolchain — 4.4 binaries hang at the first forward pass.
3. **Build**: cmake with `ivcc` as the CUDA compiler, arch `ivcore11`; see §3 verbatim.
4. **Runtime quirks**: `llama-cli --no-warmup` on MoE; rare `logprobs:null` / mid-generation garbage both heal on resend; disable Qwen thinking or raise `max_tokens`.
5. **Ops hygiene**: no `fuser` in the image — kill by PID; beware self-matching `pkill -f` patterns.
6. **Baselines**: 27B → 12.6 tok/s; JEV-35B → 172 ms; mini-JEV → 25–30 ms per decision.
