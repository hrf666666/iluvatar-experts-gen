# iluvatar-experts-gen

**中文 · [English](README.md)**

在国产 GPU 天数智芯上，把一堆文档变成领域专家。几分钟产出一个，不训练底座大模型，随时上线，随时更换。底层是 Qwen 系模型和 llama.cpp 的天数专用构建。

## 三种专家，一条产线

一份知识库可以产出三种形态。

**知识包专家。** 跑一条命令，约一分钟。产出职责卡片、路由头和检索索引。用户提问时系统自动检索相关段落喂给大模型，回答有出处，不编造。换文档不重启服务，秒级生效。命令是 `tools/expert_factory.py`。

**小 JEV。** 跑一条命令，约五分钟。产出 0.5B 规模的微型判断模型，25 到 30 毫秒给出一次决策，例如一张工单该分给哪个组。一台边缘设备就能跑。命令是 `tools/mini_jev.py`。

**LoRA 权重专家。** 知识直接写进模型权重。数据生成、训练、转换、挂载现在全部在天数卡本地完成，全程约十分钟，不需要外部 NVIDIA 机器。训练脚本在 `patches/train_local_iluvatar.py`，用的是天数自带的 torch 2.10.0+corex.5.0.0。

三种形态互补。知识包改文档秒级生效，可解释。LoRA 专家不依赖检索就能答对，抗提示干扰。小 JEV 负责边缘和路由激活。

## 权重专家的完整生命周期

训练用 Qwen3-4B 底座加 LoRA，33M 可训练参数，占全模型 0.81%。在天数卡上约五分钟收敛。训练完转成 66MB 的 GGUF 文件，然后一条命令挂进正在运行的服务。

挂载接口是我们从 vLLM 生态移植到天数 llama.cpp 上的。补丁在 `patches/vllm-style-hot-add-lora.patch`。它让 `POST /lora-adapters` 接受带 path 字段的条目，把新 adapter 文件加载进运行中的服务，进程零重启。加载、启用、禁用、调强度、按请求路由，全部是运行时热操作。

行为学验证用的边界难题集包含血液酒精 72、65、81 毫克每百毫升的临界值、无酒精事故和逃逸拒赔组合。裸基座 4 对 6 题。热加载权重专家后 6 对 6，置信度从 0.25 升到 0.99 以上。

## JEV-35B 决策加速

全部在 llama.cpp 内完成，不换引擎。

| 场景 | 加速前 | 加速后 |
|---|---|---|
| 同一问题重复决策 | 1.35 秒 | 0.1 秒 |
| 全新工单首次决策 | 1.35 秒 | 0.9 秒 |
| 一张工单连问多个问题 | 每问从头算 | 共享一次计算 |

两个生效杠杆是打开 prompt 缓存和把检索段落从四段减到两段。前者因为这个引擎默认关闭缓存，后者五项标准测试依然全对。引擎边界也摸清了。这个构建没有部分前缀复用，提示改一个字就整段重算。prefill 速度约每秒 300 token，是 MoE 内核的天花板，线程和批次参数一律无效。JEV 服务推荐参数 `-t 16 --parallel 1`。

## 小 JEV 成绩与一条实测结论

| 方案 | 规模 | 五项测试 | 延迟 |
|---|---|---|---|
| 0.6B 小语言模型加校准 | 0.6B | 1 对 5 | 无信号 |
| 4B 小语言模型加校准 | 4B | 4 对 5 | 0.2 到 0.4 秒 |
| 本仓库小 JEV | 0.5B | 5 对 5 | 25 到 30 毫秒 |
| JEV-35B 参照 | 3B 激活 | 5 对 5 | 0.1 到 0.9 秒 |

实测结论是给 0.6B 小模型调参数表得不到判断能力。底座太小就没有信号，社区同行要么上 LoRA 训练要么 4B 起步。本仓库走第三条路，冻结现成编码器 bge-m3，只用 numpy 训练一个很小的匹配头。不碰大模型，不需要训练卡，效果最好，速度最快。

## 专家是什么

本仓库的专家不是底座模型里的 MoE 专家权重。那些 FFN 块一个都没动。我们的专家在激活层，是知识库加职责卡加训练过的路由头。比裸检索多出来的是判断，判断这题归哪个领域、要不要组合、都不归就弃权。这些由训练过的小头完成，25 毫秒级。

MoE 的容量在底座里隐式起作用。一个底座能像样回答所有领域，靠的就是它。要在权重上显式形成领域专家，路就是本仓库已经跑通的 LoRA 训练和热挂载。

## 五分钟上手

```bash
pip install numpy
bash scripts/setup-iluvatar.sh

LLAMA_BIN=<天数llama.cpp构建>/bin bash scripts/start-expert-services.sh

# 文档变聊天专家，约一分钟
python3 tools/expert_factory.py build my-domain --source kb/my-domain \
    --llm http://127.0.0.1:8902 --kb-embedder llama:http://127.0.0.1:8910 --out experts

# 立刻可用，什么都不重启
curl -X POST localhost:5000/v1/experts -d '{"name": "my-domain"}'
curl -X POST localhost:5000/v1/experts/swap -d '{"active": ["my-domain"]}'
curl localhost:5000/v1/chat/completions -d '{"messages":[{"role":"user","content":"你的问题"}]}'

# 工单分派小 JEV，约五分钟
python3 tools/mini_jev.py --build my-domain --kb-dir kb --out mini-jevs
python3 tools/mini_jev.py --domain my-domain --kb-dir kb --serve 5504
```

仓库自带电商客服、GPU 云运维、民法、刑法、机房制冷五个示例库和全部预构建产物。不训练任何东西，服务命令开箱即用。

## 工作原理

文档切成小段，算出语义向量，检索是一次向量点积，加载和切换都是毫秒级。每个专家等于一个知识库、一张自动生成的职责卡、一个自动训练的路由头。路由头判断用户的问题归不归这个专家管，归就激活它。多个领域同时命中时人设自动合成，检索跨库按分数归并，用户全程无感。新专家落地即可被路由。相邻领域的接管需要重跑一次工厂，因为训练过的路由头在自己的词汇领地上必然压过零样本锚点。路由层采用每域独立判断加弃权。全体专家抢一个 softmax 的方案在开放世界输入上必然误分配，问电影也会分给某个专家，这是实测结论。

全程没有大模型训练。知识包和小 JEV 的所有训练都是对很小部件的 numpy 拟合。LoRA 专家的 33M 参数训练用天数自带 torch 在天数卡上完成。

## 实测回答:MoE 里有没有领域专属专家

我们在 35B-A3B 底座上钩住全部 40 层路由器,用五个领域的语料画像了真实路由。归一化后的结论很明确。256 个专家中没有任何一个以五成以上纯度专属于某领域。五个领域近乎均分这些专家,每域分得 40 到 62 个。各层段都没有专门化。域间路由分布的 JS 散度在 0.020 到 0.064 比特之间,民法和刑法最接近,技术与法律最疏远,语义结构存在但很微弱。实践结论随之而来。按领域剪专家不可行,因为不存在可剪的领域专家。专家级定向 LoRA 信号很弱,一个领域的适配器实际会触及几乎所有专家。画像脚本是 `tools/moe_profile.py`。

## 已知限制

权重专家在天数卡上训练为 5 对 6，3090 上为 6 对 6。差距来自手写纯 bf16 训练循环和混合精度的差别，改进方向是 fp32 master weights。

天数卡 llama.cpp 构建坑不少。探针判卡、CoreX 5.0 缺库、pyarrow 与 numpy 版本配对等，`docs/iluvatar-ops.md` 有完整踩坑记录。

## 目录结构

```
serve/knowledge.py        知识库引擎，切块向量化热插拔
tools/chat_kb_gateway.py  聊天网关，检索增强加专家自动发现路由组合
tools/jev_decide.py       JEV-35B 决策网关，含批量接口
tools/expert_factory.py   知识包专家工厂
tools/mini_jev.py         小 JEV 工厂和服务
tools/jev_distill.py      小语言模型蒸馏实验存档
tools/moe_surgery.py      MoE 模型离线手术工具
patches/                  vLLM 式热加载补丁和天数卡本地训练脚本
lora/                     法律 LoRA 专家数据和评测集
kb/                       示例知识库
experts*/、mini-jevs/     预构建产物
scripts/                  环境初始化和服务启动
docs/iluvatar-ops.md      天数卡硬件与构建手册
```

## 路线图

- [ ] JEV-35B 重量化到 Q3_K_M，带宽受限场景预计再快 20 到 25 个百分点
- [ ] 小 JEV 支持 score 和 noul 决策类型
- [ ] 天数卡训练从 5 对 6 提到满分，引入 fp32 master weights

## 致谢

基于 llama.cpp 的天数 CoreX 移植、Qwen 系模型和类型化决策模型思想构建。小 JEV 设计参考了 mini-Jev、kotoba typed-decisions、joyfox 的公开做法。完整实测记录见 `docs/iluvatar-ops.md`。

## 许可

Apache-2.0
