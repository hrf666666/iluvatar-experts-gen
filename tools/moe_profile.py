#!/usr/bin/env python3
"""Expert-routing profile of JEV-35B (Qwen3.5-35B-A3B) on CPU, vendor torch.

For each domain corpus (5 KBs), forward the text and record which routed experts
each layer's gate activates (top-8 of 256). Output: counts[domain, layer, expert]
plus a per-domain affinity analysis — does the E cluster by domain or not.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

torch.set_num_threads(64)
MD = "/root/hot-remove-experts/models/JEV-35B"
KB = Path("/root/hot-remove-experts/kb")
DOMAINS = ["ecom-support", "cloud-ops", "civil-law", "criminal-law", "facility-cooling"]
MAX_TOKENS_PER_DOMAIN = 2600
TOPK = 8

tok = AutoTokenizer.from_pretrained(MD)
from transformers import Qwen3_5MoeForConditionalGeneration
print("loading 35B bf16 on CPU (70G RAM)...", flush=True)
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    MD, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, attn_implementation="eager")
model.eval()
print("loaded", flush=True)

# locate router gates: named modules ending .gate inside moe/mlp blocks
gates = {}
for name, mod in model.named_modules():
    if name.endswith(".gate") and "Router" in type(mod).__name__:
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            gates[int(m.group(1))] = (name, mod)
n_layers = max(gates) + 1
n_experts = None
counts = {d: np.zeros((n_layers, 4096), dtype=np.int64) for d in DOMAINS}  # 4096 cap, trimmed later
print(f"gates hooked: {len(gates)} layers", flush=True)

captures = []
def make_hook(layer):
    def hook(mod, inp, out):
        # router returns (logits, weights, selected_expert_indices)
        idx = out[2] if isinstance(out, tuple) else out
        captures.append((layer, idx.reshape(-1).to(torch.int64).numpy()))
    return hook
handles = []
for layer, (name, mod) in sorted(gates.items()):
    handles.append(mod.register_forward_hook(make_hook(layer)))

for d in DOMAINS:
    text = ""
    for p in sorted((KB / d).rglob("*.md")):
        text += p.read_text(encoding="utf-8") + "\n\n"
    ids = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    if len(ids) > MAX_TOKENS_PER_DOMAIN:
        ids = ids[:MAX_TOKENS_PER_DOMAIN]
    print(f"[{d}] {len(ids)} tokens", flush=True)
    captures.clear()
    with torch.no_grad():
        for i in range(0, len(ids), 512):
            chunk = ids[i:i + 512].unsqueeze(0)
            model(input_ids=chunk)
    for layer, flat in captures:
        counts[d][layer] += np.bincount(flat, minlength=4096)[:4096]

E = max(int(counts[DOMAINS[0]][l].nonzero()[0].max()) for l in range(n_layers)) + 1
C = np.stack([counts[d][:, :E].astype(np.int32) for d in DOMAINS])
np.savez_compressed("/root/moe_profile.npz", counts=C, domains=np.array(DOMAINS))
print(f"saved /root/moe_profile.npz  shape={C.shape}", flush=True)

# ------- analysis: domain affinity -------
print("\n=== 每域激活最多的专家(跨层合计 Top 10) ===")
for di, d in enumerate(DOMAINS):
    tot = C[di].sum(axis=0)
    top = np.argsort(-tot)[:10]
    print(f"{d:>16s}: " + " ".join(f"{i}({tot[i]})" for i in top))

print("\n=== 领域偏置专家:某域激活占比 > 60% 且总量 >= 50 ===")
for di, d in enumerate(DOMAINS):
    tot_all = C.sum(axis=0)                    # [expert]
    share = C[di] / np.maximum(tot_all, 1)     # 该专家激活中此域占比
    mask = (share > 0.6) & (C[di].sum(axis=0) >= 50)
    experts = np.nonzero(mask.sum(axis=0))[0]  # any layer
    n_bias = len(experts)
    tot_experts_active = len(np.nonzero(C[di].sum(axis=0))[0])
    print(f"{d:>16s}: {n_bias}/{tot_experts_active} 个专家显著偏向此域")

ov = []
for i in range(len(DOMAINS)):
    for j in range(i + 1, len(DOMAINS)):
        a = set(np.argsort(-C[i].sum(axis=0))[:100].tolist())
        b = set(np.argsort(-C[j].sum(axis=0))[:100].tolist())
        ov.append((DOMAINS[i], DOMAINS[j], len(a & b) / 100))
print("\n=== 域间 Top100 专家重合度 ===")
for a, b, r in ov:
    print(f"{a:>14s} × {b:<14s}: {r:.2f}")
