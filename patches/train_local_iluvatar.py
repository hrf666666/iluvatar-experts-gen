#!/usr/bin/env python3
"""LoRA SFT for the law expert — ON THE ILUVATAR CARD (vendor torch 2.10.0+corex.5.0.0).

Same recipe as the 3090 run, minus the `datasets` dependency (its pyarrow wants
numpy>=2; the vendor stack is built against numpy 1.26 — plain torch Dataset instead).
bf16 LoRA, no bitsandbytes (NVIDIA-only). Runs beside the production 27B on card 1:
llama-server reserves its memory at startup, so a would-be OOM hits this process
first, not production.
"""
import json, torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

BASE = "/root/law-lora-local/Qwen3-4B"
OUT = "/root/law-lora-local/adapter"
LETTERS = "ABCD"

def prompt_of(c):
    lines = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(c["options"]))
    return (f"[kind] choice\n[state] {c['state']}\n[question] {c['question']}\n"
            f"[options]\n{lines}\n[decision]:")

rows = [json.loads(l) for l in open("/root/hot-remove-experts/law_decisions.jsonl")]
held = [json.loads(l) for l in open("/root/hot-remove-experts/law_eval.jsonl")]
tok = AutoTokenizer.from_pretrained(BASE)

class LawDS(Dataset):
    def __init__(self, cases):
        self.ex = []
        for c in cases:
            p = tok(prompt_of(c), add_special_tokens=False)["input_ids"]
            a = tok(c["label"], add_special_tokens=False)["input_ids"]
            assert len(a) == 1
            self.ex.append((p, a))
    def __len__(self):
        return len(self.ex)
    def __getitem__(self, i):
        return {"p": self.ex[i][0], "a": self.ex[i][1]}

def collate(batch):
    maxlen = max(len(b["p"]) + 1 for b in batch)
    pad = tok.pad_token_id or tok.eos_token_id
    ids, labels, mask = [], [], []
    for b in batch:
        seq = b["p"] + b["a"]
        ids.append(seq + [pad] * (maxlen - len(seq)))
        labels.append([-100] * len(b["p"]) + b["a"] + [-100] * (maxlen - len(seq)))
        mask.append([1] * len(seq) + [0] * (maxlen - len(seq)))
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager", device_map={"": 0})
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
model.train()

# plain loop — transformers 5.x Trainer insists on importing datasets (pyarrow/numpy
# wars on the vendor stack); a 60-step LoRA fit needs nothing fancier than AdamW.
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1.5e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-4, total_steps=60, pct_start=0.1)
pad = tok.pad_token_id or tok.eos_token_id
ds = LawDS(rows)
import random
step = 0
accum = 0
for epoch in range(10):
    idx = list(range(len(ds)))
    random.Random(epoch).shuffle(idx)
    for i in range(0, len(idx), 4):
        for k in range(i, min(i + 4, len(idx))):
            batch = [ds[k]]
        maxlen = max(len(b["p"]) + 1 for b in batch)
        ids = torch.tensor([b["p"] + b["a"] + [pad] * (maxlen - len(b["p"]) - 1) for b in batch]).to(model.device)
        labels = torch.tensor([[-100] * len(b["p"]) + b["a"] + [-100] * (maxlen - len(b["p"]) - 1) for b in batch]).to(model.device)
        mask = torch.tensor([[1] * (len(b["p"]) + 1) + [0] * (maxlen - len(b["p"]) - 1) for b in batch]).to(model.device)
        loss = model(input_ids=ids, labels=labels, attention_mask=mask).loss / 4
        loss.backward()
        accum += 1
        if accum == 4:
            opt.step(); sched.step(); opt.zero_grad(); accum = 0
        step += 1
        if step % 5 == 0:
            print(f"step {step}/60 loss {loss.item():.4f}", flush=True)
model.save_pretrained(OUT)
print("ADAPTER_SAVED")

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
