#!/usr/bin/env python3
"""Decision LoRA for Qwen3.6-35B-A3B — trained ON THE ILUVATAR POD's CPU.
RAM 1TB holds the 70G bf16 weights. LoRA targets only the last 8 of 40 layers
(decision readout lives late), cutting backward cost to ~1/5. ~3 epochs."""
import json, random, re, time, torch
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

torch.set_num_threads(100)
BASE = "/root/q36hf"
rows = [json.loads(l) for l in open("/root/dec_sft.jsonl")]
tok = AutoTokenizer.from_pretrained(BASE)
t0 = time.time()
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
    attn_implementation="eager")
print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

# LoRA 只挂 32-39 层的注意力与共享专家投影——显式枚举模块名(此版 peft 不收正则对象)
TARGET_RE = re.compile(r"layers\.(3[2-9])\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
names = [n for n, m in model.named_modules()
         if TARGET_RE.search(n) and isinstance(m, torch.nn.Linear)]
print(f"LoRA targets: {len(names)} modules, e.g. {names[:2]}", flush=True)
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=names))
model.print_trainable_parameters()

ex = []
for r in rows:
    p = tok(r["prompt"], add_special_tokens=False)["input_ids"][:1200]
    a = tok(r["answer"], add_special_tokens=False)["input_ids"]
    assert len(a) == 1
    ex.append((p, a))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
EP = 3
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=EP*len(ex), pct_start=0.1)
step = 0
for epoch in range(EP):
    idx = list(range(len(ex))); random.Random(epoch).shuffle(idx)
    for i in idx:
        p, a = ex[i]
        ids = torch.tensor([p + a]); labels = torch.tensor([[-100]*len(p) + a])
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        if step % 10 == 0:
            print(f"step {step}/{EP*len(ex)} loss {loss.item():.4f} elapsed {time.time()-t0:.0f}s", flush=True)
model.save_pretrained("/root/adapter36")
print("ADAPTER36_SAVED", flush=True)
