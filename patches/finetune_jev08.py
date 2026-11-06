#!/usr/bin/env python3
"""LoRA fine-tune the open-source Jev-Style-0.8B on our law decision data.
Vendor torch 2.10.0+corex on the Iluvatar card. Full fine-tune is feasible at 0.8B."""
import json, random, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

torch.set_num_threads(16)
BASE = "/root/jev08-hf"
rows = [json.loads(l) for l in open("/root/hot-remove-experts/law_decisions_aug.jsonl")]
held = [json.loads(l) for l in open("/root/hot-remove-experts/law_eval.jsonl")]
LETTERS = "ABCD"

tok = AutoTokenizer.from_pretrained(BASE)
print("loading 0.8B bf16 on GPU...", flush=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map={"": 0}, attn_implementation="eager")
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.print_trainable_parameters()
model.train()

def prompt_of(c):
    lines = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(c["options"]))
    return (f"[kind] choice\n[state] {c['state']}\n[question] {c['question']}\n"
            f"[options]\n{lines}\n[decision]:")

ex = []
for c in rows:
    p = tok(prompt_of(c), add_special_tokens=False)["input_ids"]
    a = tok(c["label"], add_special_tokens=False)["input_ids"]
    assert len(a) == 1
    ex.append((p[:1200], a))

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
EP = 10
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=EP*len(ex), pct_start=0.1)
t0 = time.time()
step = 0
for epoch in range(EP):
    idx = list(range(len(ex))); random.Random(epoch).shuffle(idx)
    for i in idx:
        p, a = ex[i]
        ids = torch.tensor([p + a]).to(model.device)
        labels = torch.tensor([[-100]*len(p) + a]).to(model.device)
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        if step % 20 == 0:
            print(f"step {step}/{EP*len(ex)} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)

model.save_pretrained("/root/adapter08")
print("ADAPTER08_SAVED", flush=True)

# eval
model.eval()
letter_ids = [tok(l, add_special_tokens=False)["input_ids"][0] for l in LETTERS]
ok = 0
with torch.no_grad():
    for c in held:
        ids = torch.tensor([tok(prompt_of(c), add_special_tokens=False)["input_ids"]]).to(model.device)
        logits = model(input_ids=ids).logits[0, -1]
        pred = LETTERS[int(torch.tensor([logits[i] for i in letter_ids]).argmax())]
        ok += pred == c["label"]
        print(f"  [{'OK' if pred == c['label'] else 'X '}] want {c['label']} got {pred}: {c['state'][:36]}")
print(f"HELDOUT {ok}/{len(held)}")
