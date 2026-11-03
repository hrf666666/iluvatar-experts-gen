# GPU 节点故障处置(专家库 v2026.11)

## XID 22(非法指令)

Iluvatar BI 系列卡上报 XID 22 时,通常是设备函数调用未内联被后端降级为 RPCMOV/PCSWAP 序列引起。处置:确认编译走 ivcc 包装(强制 -O1),复现进程立即停止;已打挂的卡在该卡上进程退出前不可恢复,用 CUDA_VISIBLE_DEVICES 隔离健康卡继续服务,择机重启节点。禁止对挂卡节点反复重试训练任务。

## 显存打满与 OOM

单卡 32GB 显存打满导致进程 OOM 被杀的:推理实例检查 KV cache 上限(-c 与 -np 的乘积),按 0.9 系数下调;训练实例检查梯度累积配置。两卡以上的张量并行实例同时 OOM 的,先看 NCCL 日志排除通信假死。

## 掉卡与枚举变化

重启后 lspci 枚举变化或 ixsmi 少卡:先查 XID 历史(dmesg | grep XID),确认是单卡硬件故障还是链路问题;单卡故障直接报修,不要反复重启整机。

## 温度与功耗

BI-V150 满载 350W,持续 85°C 以上应检查风冷/水冷;温度墙触发降频(1600MHz 以下)时任务错峰。

# GPU Node Troubleshooting (English summary)

XID 22 on Iluvatar BI cards means device-function calls were not inlined (RPCMOV/PCWSAP sequences). Stop the process, isolate healthy cards with CUDA_VISIBLE_DEVICES, and schedule a node reboot; a wedged card cannot recover until the node restarts. OOM: reduce KV cache (-c * -np) by 0.9x for inference instances.
