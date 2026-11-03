# 调度与配额(专家库 v2026.11)

## 显式配额与超卖

单节点显存配额按物理显存的 85% 下发(预留 ECC 与碎片)。推理池允许 1.1 倍温和超卖,但同一卡上不允许两个大模型(>20GB)实例叠加。抢占用优先级:训练 < 在线推理 < 紧急任务。

## 多实例并行

带宽受限的单流 decode(约 15 tok/s 上限)靠每卡一个实例叠加:4 卡 4 实例聚合吞吐约为单流的 18 倍。下发时同一模型的实例均匀打散到不同卡,避免单卡热点。

## 热插拔变更窗口

在线服务的模型/知识库变更有两个窗口:低峰(02:00-06:00)全量重启;高峰期只允许热操作(逐实例替换、注册表原子切换),单次影响面不超过该服务 1/N 实例。

# Scheduling and Quotas (English summary)

GPU memory quotas are issued at 85% of physical capacity; inference pools may oversubscribe 1.1x but two large (>20GB) model instances never share a card. Single-stream decode is bandwidth-bound (~15 tok/s): scale throughput with one instance per card, 4 cards giving ~18x aggregate.
