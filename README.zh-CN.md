# iluvatar-experts-gen

**中文 · [English](README.md)**

在国产卡（天数智芯 Iluvatar BI-V150 / CoreX）上**批发 MoE 专家和小 JEV**——知识库侧载 + 对应专业领域的边缘决策，底层基于 **Qwen** 架构。

一套底座（llama.cpp Iluvatar 构建承载的 Qwen 系模型）驱动两条全自动产线：

| 产线 | 工厂 | 产物 | 热插拔方式 | 产出耗时 |
|---|---|---|---|---|
| 💬 聊天专家 | `tools/expert_factory.py` | 专家包：训练好的路由头 + 评分优选的专家卡 + 知识库 | 聊天网关 `POST /v1/experts` | ~45 秒/域 |
| ⚡ 小 JEV（≤1B 决策） | `tools/mini_jev.py` | 冻结编码器 + 匹配头 + 选项锚点 + 知识库 | `--domain X --serve 端口` | ~5 分钟/域 |

**自感知、自加载**：网关监视专家目录——工厂产物落地即上线，零运维动作。路由头对每个域独立打分、多选组合：问“酒驾撞车拒赔修车费”会自动合成刑法+民法两个专家（人设合并、检索跨库归并），用户只管用自然语言提问；域外问题自动弃权走通用路径。

两条产线共享同一套设施：原始文档 → 可热插拔知识库，教师模型（Qwen3.8-27B / JEV-35B）自监督蒸馏，bge-m3 嵌入。**全流程无反向传播、无 GPU 训练、无人工标注**——所有可训练部件均由 numpy 拟合。

## 实测结果（2× BI-V150）

| 决策方案 | 激活计算量 | 金标准 5 项 | 单次决策延迟 |
|---|---|---|---|
| 0.6B 小 LLM + 头部校准 | 0.6B | 1/5（输出与输入无关，死路） | 40–140 ms |
| 4B 小 LLM + 头部校准 | 4B | 4/5 | 192–376 ms |
| **小 JEV（冻结 bge-m3 + r64 匹配头）** | **0.5B 级，模型仅 467MB** | **5/5** | **25–30 ms** |
| JEV-35B 教师（参照） | 3B 激活 | 5/5 | 热 ~172 ms |

聊天侧：Qwen3.8-27B GPU 12.6 tok/s；专家热切换秒级生效，**模型进程零重启**（挂载/激活/使用/删除全周期以 PID 验证）。

## 快速上手

```bash
pip install numpy                          # 唯一依赖
bash scripts/setup-iluvatar.sh             # ivcc + CoreX 软链 + 探针（每次 pod 重置后跑一次）
LLAMA_BIN=<llama.cpp-iluvatar构建>/bin bash scripts/start-expert-services.sh

# — 从原始文档批发一个聊天专家（自蒸馏 + 拒绝采样，全自动）—
python3 tools/expert_factory.py build my-domain --source kb/my-domain \
    --llm http://127.0.0.1:8902 --kb-embedder llama:http://127.0.0.1:8910 --out experts

# — 运行中热插拔进聊天网关（什么都不重启）—
curl -X POST localhost:5000/v1/experts -d '{"name": "my-domain"}'
curl -X POST localhost:5000/v1/experts/swap -d '{"active": ["my-domain"]}'
curl localhost:5000/v1/chat/completions -d '{"messages":[{"role":"user","content":"…"}]}'

# — 批发一个小 JEV（冻结编码器 + numpy 拟合匹配头）—
python3 tools/mini_jev.py --build my-domain --kb-dir kb --out mini-jevs
python3 tools/mini_jev.py --domain my-domain --kb-dir kb --serve 5504
curl localhost:5504/decide -d '{"state":"…","question":"该哪个组处理?","options":["甲","乙","丙"]}'
```

仓库自带三个示例知识库（`kb/ecom-support`、`kb/cloud-ops`、`kb/facility-cooling`）和全部预构建产物——不训练任何东西，上面的 serve 命令开箱即用。

## 这里的"专家"指什么

- **知识库**——纯文件（`.md/.txt/.jsonl/...`）切块 + 嵌入，检索就是与行归一化矩阵的点积；运行时热加载/卸载/切换/多库并激活；在途请求持快照，永不被切换撕裂。
- **聊天专家包**——问题嵌入上的 softmax **路由头**（numpy 真梯度训练）+ **专家卡**（系统人设，模型起草两种风格、held-out 事实命中评分择优）+ 所挂知识库。
- **小 JEV**——≤1B 决策专家：`logit_i = (U·e_query)·(V·e_option)`，编码器冻结；选项用从知识库一次性蒸馏的**锚点文本**表示——头参数不随候选数增长，请求可任意声明选项子集与顺序；对外提供与 JEV-35B 网关同形的 `/decide` API（`kind=choice`）。

工具链对每层能力边界保持诚实：`tools/jev_distill.py` 保留了 0.6B/4B 规模阈值实验——头部校准救不了没有信号的底座，这正是小 JEV 产线存在的原因。

## 目录结构

```
serve/knowledge.py        热插拔知识库引擎（嵌入器/切块/注册表快照）
tools/chat_kb_gateway.py  OpenAI 兼容聊天网关：KB 注入 + 专家热插拔 + 训练路由分发
tools/jev_decide.py       JEV-35B 决策网关（verbalizer 读出、重试、热 KB）
tools/expert_factory.py   聊天专家工厂（自蒸馏 QA → 路由头 → 评分专家卡）
tools/mini_jev.py         小 JEV 工厂 + 服务（冻结编码器 + 匹配头）
tools/jev_distill.py      教师→LLM 学生头部蒸馏（规模阈值实验存档）
tools/moe_surgery.py      MoE GGUF 离线手术（前缀切层；剪专家 = 路线图）
iluvatar/                 ivcc（CoreX clang CUDA 包装）、corex_compat.h、探针源码
kb/                       示例知识库
experts*/、mini-jevs/     预构建示例产物（用工厂随时重建）
scripts/                  硬件初始化、服务启动、决策演示套件
docs/iluvatar-ops.md      硬件与构建手册（雷点全记录，均为实测踩坑）
```

## 路线图

- [ ] `moe_surgery.py` 剪专家（保深度、256→K），产出瘦身 JEV-35B 变体
- [ ] 小 JEV 支持 `score` / `noul` 决策类型；未见团队的按请求锚点
- [ ] LoRA 侧载通路（llama-server `POST /lora-adapters`），外部训练器产出 adapter 即接入
- [ ] 小 JEV 产线引入更小编码器（bge-small），把边缘占用压到 150MB 以下

## 致谢

基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)（Iluvatar/CoreX 移植）、Qwen 系模型与类型化决策（"Jev"）模型思想构建；小 JEV 的设计参考了开源社区（mini-Jev、kotoba typed-decisions、joyfox 0.8B）的共识。完整实测记录见 `docs/iluvatar-ops.md`。

## 许可

Apache-2.0
