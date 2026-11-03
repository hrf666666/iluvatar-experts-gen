#!/usr/bin/env python3
"""jev_live_test.py — JEV-35B 决策网关实测(5500):带库/不带库、热切换。"""
import json, sys, time, urllib.request

GW = "http://127.0.0.1:5500"

def decide(state, question, options, **extra):
    body = {"kind": "choice", "state": state, "question": question, "options": options, **extra}
    req = urllib.request.Request(f"{GW}/decide", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    top = max(d["distribution"], key=lambda x: x["prob"])
    print(f"-> {top['label']:>10s} ({top['text'][:40]}) p={top['prob']:.3f} | gateway {d['latency_ms']}ms, "
          f"wall {time.time()-t0:.1f}s | kb={d.get('kb')}")
    for r_ in sorted(d["distribution"], key=lambda x: -x["prob"])[:3]:
        print(f"     {r_['label']:>12s} {r_['prob']:.3f}  {r_['text'][:50]}")
    return d

def post(path, body):
    req = urllib.request.Request(GW + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

OPTS = ["billing", "shipping", "technical support", "account security"]

print("=== 1) 重复扣款(默认库 ecom-support) ===")
decide("Customer message: my card was charged twice for the same order.",
       "Which team should handle this ticket?", OPTS)

print("=== 2) 地址填错(默认库) ===")
decide("Customer message: I moved apartments, the package went to my old address.",
       "Which team should handle this ticket?", OPTS)

print("=== 3) 同一工单,明确关掉知识库 ===")
decide("Customer message: my card was charged twice for the same order.",
       "Which team should handle this ticket?", OPTS, kb=False)

print("=== 4) 网关热插拔:运行中加载 cloud-ops,再切换默认库(不重启任何进程) ===")
print("   load ->", post("/kb/load", {"name": "cloud-ops", "source": "cloud-ops"})["name"])
print("   swap ->", post("/kb/swap", {"active": ["cloud-ops"]})["active"])
decide("Ops alert: training job on GPU node 7 died with XID 22, card wedged.",
       "Which on-call runbook should handle this?", ["gpu-node", "scheduler", "network", "billing"])

print("=== 5) 切回电商库 ===")
print("   swap ->", post("/kb/swap", {"active": ["ecom-support"]})["active"])
decide("Customer message: app crashes on startup every time, reinstalled twice.",
       "Which team should handle this ticket?", OPTS)
