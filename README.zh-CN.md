# iluvatar-experts-gen

**中文 · [English](README.md)**

在国产 GPU 天数智芯上，把一堆文档变成领域专家。几分钟产出一个，随时上线，随时更换。底层是 Qwen3.6-35B-A3B 和 llama.cpp 的天数专用构建。

## 三种专家

一份知识库可以产出三种形态。

**知识包专家。** 一条命令，约一分钟。产出职责卡片、路由头和检索索引。用户提问时系统自动检索相关段落喂给大模型，回答有出处，不编造。换文档不重启服务，秒级生效。命令是 `tools/expert_factory.py`。

**小 JEV。** 一条命令，约五分钟。产出 0.5B 规模的微型判断模型，25 到 30 毫秒给出一次决策，例如一张工单该分给哪个组。一台边缘设备就能跑。命令是 `tools/mini_jev.py`。

**LoRA 权重专家。** 知识直接写进模型权重。数据生成、训练、转换、挂载全部在天数卡本地完成，全程约十分钟。训练脚本在 `patches/` 目录。挂载通过 llama.cpp 的 `/lora-adapters` 接口热加载，服务进程零重启。

三种形态互补。知识包改文档秒级生效，可解释。LoRA 专家不依赖检索就能答对。小 JEV 负责边缘和路由激活。

## 多模态

`tools/multimodal_jev.py` 把小 JEV 扩展到图文输入。用中文 CLIP 编码图像，用 bge-m3 编码文本，拼接后进同一个匹配头。用户发一张商品损坏照片加一句话描述，系统自动分派工单。评测 6 对 6。

## 权重专家的运行时热加载

`patches/vllm-style-hot-add-lora.patch` 是我们从 vLLM 生态移植到天数 llama.cpp 的补丁。它让 `POST /lora-adapters` 接受带 path 字段的条目，把新 adapter 文件加载进正在运行的服务。加载、启用、禁用、调强度、按请求路由，全部是运行时热操作。实测裸服务 4 对 6 题，一条命令热加载后 6 对 6，同一进程零重启。

## MoE 专家画像

`tools/moe_profile.py` 钩住 40 层路由器，画像真实路由。结论是 256 个专家中没有任何一个以五成以上纯度专属于某领域。五个领域近乎均分这些专家。路由层有语义倾向但幅度极小，民法和刑法的路由分布最接近，技术和法律最疏远。按领域剪专家不可行。专家级定向 LoRA 信号很弱。

## 性能

| 场景 | 数值 |
|---|---|
| JEV-35B 重复决策 | 0.1 秒，快 13.7 倍 |
| JEV-35B 冷决策 | 0.9 秒 |
| 小 JEV 决策 | 25 到 30 毫秒 |
| 聊天生成 | 58.7 token 每秒 |
| LoRA 热加载 | 秒级生效 |

## 五分钟上手

```bash
pip install numpy
bash scripts/setup-iluvatar.sh
LLAMA_BIN=<天数llama.cpp构建>/bin bash scripts/start-expert-services.sh
```

服务启动后，聊天走 `localhost:5000`，决策走 `localhost:5500`，嵌入走 `localhost:8910`，模型服务在 `localhost:8970`。

仓库自带电商客服、GPU 云运维、民法、刑法、机房制冷五个示例库和全部预构建产物。不训练任何东西，服务命令开箱即用。

## 生产拓扑

一张天数卡跑 Qwen3.6-35B-A3B 加 bge-m3 嵌入器，另一张卡空闲可跑多模态 CLIP 或训练任务。一个模型同时供聊天网关和决策网关。LoRA 适配器按需热挂载，不停服。

## 目录

```
serve/knowledge.py          知识库引擎
tools/chat_kb_gateway.py    聊天网关
tools/jev_decide.py         JEV 决策网关
tools/expert_factory.py     知识包专家工厂
tools/mini_jev.py           小 JEV 工厂
tools/multimodal_jev.py     多模态小 JEV
tools/moe_profile.py        MoE 专家路由画像
tools/moe_surgery.py        MoE 模型离线手术
tools/jev_distill.py        小语言模型蒸馏实验
patches/                     热加载补丁和训练脚本
lora/                        法律 LoRA 数据和评测集
kb/                          示例知识库
experts*/ mini-jevs/         预构建产物
scripts/                     环境初始化和服务启动
iluvatar/                    ivcc 工具链和卡探针
docs/iluvatar-ops.md         硬件手册
```

## 许可

Apache-2.0
