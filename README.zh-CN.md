# iluvatar-experts-gen

**中文 · [English](README.md)**

一句话：**在国产 GPU（天数智芯）上，把一堆文档变成一个"领域专家"——几分钟一个，不训练模型，随时上线、随时更换。**

底层跑的是 Qwen 系模型 + llama.cpp（天数卡专用构建），需要的只是一块天数卡或一台多核 CPU 机器。

## 它能干什么？三个例子

**例 1：知识问答（chat 专家）**
把你的规章制度、产品手册放进 `kb/` 目录，跑一条命令。之后用户直接用大白话提问，系统自动检索相关段落喂给大模型——回答有出处、不瞎编。换一批文档？服务不重启，秒级生效。

**例 2：工单自动分派（小 JEV）**
同一份文档，再跑一条命令，得到一个"小 JEV"：一个 0.5B 规模的微型判断模型，**25-30 毫秒**就能判断一张工单该给哪个组（计费？物流？技术支持？）。一台边缘设备就能跑，速度是大模型的 6 倍以上。

**例 3：多领域自动组合**
用户问"朋友酒驾撞了我的车还拒赔修车费"——这横跨刑法和民法。系统**自动**同时激活这两个领域的专家，把两边的知识合并起来回答。用户全程无感，不需要选任何"模式"。

## JEV-35B 决策加速（实测数字）

生产用的大判断模型 JEV-35B，我们做了一轮针对性加速。全部在 llama.cpp 里完成，不换引擎、不装 torch：

| 场景 | 加速前 | 加速后 | 怎么做到的 |
|---|---|---|---|
| 同一问题重复决策 | 1.35 秒 | **0.1 秒（快 13.7 倍）** | 打开 prompt 缓存——这个引擎默认竟然是关的 |
| 全新工单首次决策 | 1.35 秒 | **0.9 秒（快 32%）** | 检索文档从 4 段减到 2 段，五项标准测试依然全对 |
| 一张工单连问多个问题 | 每问从头算一遍 | 共享一次计算 | 新增 `/decide_batch` 接口 |

顺带把引擎的底摸清了（写进文档省得后人再踩）：这个 llama.cpp 构建**没有部分前缀复用**——问题改一个字就整段重算；prefill 速度约 300 token/秒，是 MoE 内核的天花板，调线程数、批次参数统统没用。真正管用的杠杆只有两个：**把提示变短**和**精确重发缓存**。

## 小 JEV 的成绩，和一条重要的实测结论

| 判断方案 | 规模 | 标准五项测试 | 单次延迟 |
|---|---|---|---|
| 0.6B 小语言模型 + 校准 | 0.6B | 1/5（输出和输入无关，废） | — |
| 4B 小语言模型 + 校准 | 4B | 4/5 | 0.2-0.4 秒 |
| **本仓库的小 JEV** | **0.5B，模型仅 467MB** | **5/5 全对** | **25-30 毫秒** |
| JEV-35B（参照） | 3B 激活 | 5/5 | 0.1-0.9 秒 |

那条重要结论：**想靠"给 0.6B 小模型调参数表"获得判断能力，走不通**——底座太小就没有判断信号，社区同行的做法也一样（要么上 LoRA 训练，要么 4B 起步）。本仓库的小 JEV 走了第三条路：冻结一个现成的文本编码器（bge-m3），只用 numpy 训练一个很小的"匹配头"，不碰大模型、不需要训练卡，效果反而最好、速度最快。

## 五分钟上手

```bash
pip install numpy                              # 唯一依赖
bash scripts/setup-iluvatar.sh                 # 天数卡环境初始化（装探针、补软链）

# 第一步：起服务（聊天模型 + 嵌入器 + 网关）
LLAMA_BIN=<llama.cpp天数构建>/bin bash scripts/start-expert-services.sh

# 第二步：把文档变成聊天专家（全自动，约 1 分钟）
python3 tools/expert_factory.py build my-domain --source kb/my-domain \
    --llm http://127.0.0.1:8902 --kb-embedder llama:http://127.0.0.1:8910 --out experts

# 第三步：立刻可用（什么都不用重启）
curl -X POST localhost:5000/v1/experts -d '{"name": "my-domain"}'
curl -X POST localhost:5000/v1/experts/swap -d '{"active": ["my-domain"]}'
curl localhost:5000/v1/chat/completions -d '{"messages":[{"role":"user","content":"你的问题"}]}'

# 想要工单分派的小 JEV？一条命令（约 5 分钟），起服务后 25ms 出结果
python3 tools/mini_jev.py --build my-domain --kb-dir kb --out mini-jevs
python3 tools/mini_jev.py --domain my-domain --kb-dir kb --serve 5504
curl localhost:5504/decide -d '{"state":"工单内容","question":"该哪个组处理?","options":["计费","物流"]}'
```

仓库里带了三个现成的示例（电商客服、GPU 云运维、民法/刑法），预构建的专家包和小 JEV 都在，**不训练任何东西**，上面的服务命令开箱即用。

## 它是怎么工作的（一段话说清）

文档被切成小段、算出语义向量，检索就是一次向量点积——所以加载和切换都是毫秒级。每个"专家"= 一个知识库 + 一张自动生成的"职责卡片"（告诉大模型以什么身份、按什么规则回答）+ 一个自动训练的路由头（判断用户的问题归不归这个专家管，归就激活它）。"小 JEV"= 把"判断该归谁"这件事交给一个冻结的编码器加一个 numpy 小头，25 毫秒出答案。全程**没有大模型训练**：所有"训练"都是对很小部件的拟合，几分钟完成。

## 已知限制（实话实说）

- 新专家落地的瞬间即可被路由（零样本冷启动），但**相邻领域**的接管需要重跑一次工厂（约 3 分钟）——训练过的路由头在自己的词汇领地上必然压过零样本锚点；
- 路由层用的是"每域独立判断 + 弃权"，不是"全体专家抢一个 softmax"——实测后者在开放世界输入上必然误分配（问电影也会分给某个专家）；
- 天数卡的 llama.cpp 构建坑不少（探针判卡、CoreX 5.0 缺库等），`docs/iluvatar-ops.md` 有完整踩坑记录，都是实测换来的。

## 目录结构

```
serve/knowledge.py        知识库引擎（切块/向量化/热插拔）
tools/chat_kb_gateway.py  聊天网关：检索增强 + 专家自动发现/自动路由/多专家组合
tools/jev_decide.py       JEV-35B 决策网关（含 /decide_batch 批量接口）
tools/expert_factory.py   聊天专家工厂（约 1 分钟/域）
tools/mini_jev.py         小 JEV 工厂 + 服务（约 5 分钟/域，25ms 出结果）
tools/jev_distill.py      小语言模型蒸馏实验存档（0.6B/4B 规模阈值结论出处）
tools/moe_surgery.py      MoE 模型离线手术工具（GGUF 切层/剪枝研究用）
kb/                       示例知识库（电商/GPU 运维/民法/刑法/机房制冷）
experts*/、mini-jevs/     预构建产物（用工厂随时重建）
scripts/                  环境初始化、服务启动
docs/iluvatar-ops.md      天数卡硬件与构建手册（踩坑全记录）
```

## 路线图

- [ ] JEV-35B 重量化到 Q3_K_M（带宽受限场景预计再快 20-25%，需重跑五项测试验质量）
- [ ] 小 JEV 支持 score/noul 两种决策类型
- [ ] LoRA 侧载通路（外部训练器产出 adapter，经 llama-server 的 /lora-adapters 接口热挂载）

## 致谢

基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)（天数/CoreX 移植）、Qwen 系模型与类型化决策模型（"Jev"）思想构建；小 JEV 设计参考了开源社区（mini-Jev、kotoba typed-decisions、joyfox）的共识。完整实测记录见 `docs/iluvatar-ops.md`。

## 许可

Apache-2.0
