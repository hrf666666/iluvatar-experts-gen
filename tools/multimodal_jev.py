#!/usr/bin/env python3
"""Multimodal mini JEV: Chinese-CLIP image encoder + bge-m3 text encoder → fused query
embedding → same low-rank bilinear matching head as the text-only mini JEV.

Architecture:
  e_query = concat( clip_image(path)[:512], bge_text(state)[:1024] )  → 1536-dim
  e_option = bge_text(anchor)                                          → 1024-dim
  logit_i = (U·e_query) · (V·e_option)  where U: 1536×r, V: 1024×r

The matching head handles the asymmetric input dims naturally — U projects the
fused query, V projects the text-only option anchors, and the dot product in the
rank-r latent space gives the routing score. No cross-modal alignment needed
from the encoders themselves; the head learns it.
"""
import json, math, re, sys, time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

RANK = 32
ITERS = 500
LR = 0.3
SEED = 11

# ---- encoders ----
def get_encoders():
    import torch
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    CLIP_DIR = "/root/hot-remove-experts/models/chinese-clip"
    clip_model = ChineseCLIPModel.from_pretrained(CLIP_DIR, torch_dtype=torch.float32)
    clip_proc = ChineseCLIPProcessor.from_pretrained(CLIP_DIR)
    clip_model.eval()
    
    from serve.knowledge import make_embedder
    bge = make_embedder("llama:http://127.0.0.1:8910")
    return clip_model, clip_proc, bge

def embed_image(clip_model, clip_proc, img_path):
    import torch
    img = Image.open(img_path).convert("RGB")
    inputs = clip_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = clip_model.visual_projection(out.pooler_output)
    return pooled[0].numpy().astype(np.float64)  # 512-dim

# ---- test scenario: e-commerce tickets with product images ----
def make_scenario_images():
    """Generate 4 types of 'realistic' product/ticket images."""
    imgs = {}
    
    # 1. 账单/支付问题: 模拟账单截图(数字+表格线)
    img = Image.new("RGB", (224, 224), (240, 240, 240))
    d = ImageDraw.Draw(img)
    for y in range(40, 200, 30):
        d.line([(20, y), (200, y)], fill=(180, 180, 180), width=1)
    d.text((25, 45), "¥4299.00 x2", fill=(200, 30, 30))
    d.text((25, 75), "重复扣款", fill=(200, 30, 30))
    d.text((25, 105), "¥4299.00", fill=(200, 30, 30))
    d.rectangle([15, 35, 205, 135], outline=(100, 100, 100))
    img.save("/tmp/mm_billing.png")
    imgs["billing"] = "/tmp/mm_billing.png"
    
    # 2. 物流/地址: 模拟快递单(地址+条码)
    img = Image.new("RGB", (224, 224), (255, 255, 250))
    d = ImageDraw.Draw(img)
    for i in range(20, 200, 4):
        h = 10 + (i * 7) % 30
        d.rectangle([i, 20, i + 2, 20 + h], fill=(0, 0, 0))
    d.text((25, 80), "收件: 旧地址", fill=(0, 0, 200))
    d.text((25, 100), "xx小区3栋", fill=(0, 0, 200))
    d.text((25, 130), "包裹退回", fill=(200, 0, 0))
    img.save("/tmp/mm_shipping.png")
    imgs["shipping"] = "/tmp/mm_shipping.png"
    
    # 3. 技术故障: 模拟错误弹窗(蓝屏风格)
    img = Image.new("RGB", (224, 224), (0, 80, 180))
    d = ImageDraw.Draw(img)
    d.text((30, 60), ":( 出错了", fill=(255, 255, 255))
    d.text((30, 90), "APP crashed", fill=(255, 255, 255))
    d.text((30, 120), "0x00000022", fill=(200, 200, 255))
    d.rectangle([60, 150, 160, 170], outline=(255, 255, 255))
    d.text((75, 155), "重新启动", fill=(255, 255, 255))
    img.save("/tmp/mm_crash.png")
    imgs["technical support"] = "/tmp/mm_crash.png"
    
    # 4. 账号安全: 模拟异地登录警告
    img = Image.new("RGB", (224, 224), (255, 245, 240))
    d = ImageDraw.Draw(img)
    d.text((30, 50), "⚠ 异地登录", fill=(200, 50, 50))
    d.text((30, 80), "IP: 10.x.x.x", fill=(100, 100, 100))
    d.text((30, 110), "账号已冻结", fill=(200, 50, 50))
    d.text((30, 140), "24小时", fill=(100, 100, 100))
    img.save("/tmp/mm_security.png")
    imgs["account security"] = "/tmp/mm_security.png"
    
    return imgs

# ---- matching head (asymmetric dims: query=1536, option=1024) ----
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def fit_head(eqs, eos, ys, ws, q_dim, o_dim):
    rng = np.random.default_rng(SEED)
    u = rng.normal(0, 0.05, (q_dim, RANK))
    v = rng.normal(0, 0.05, (o_dim, RANK))
    log_t = np.zeros(())
    n = len(eqs)
    for _ in range(ITERS):
        t = math.exp(float(log_t))
        loss, gu, gv, gt = 0.0, np.zeros_like(u), np.zeros_like(v), 0.0
        for i in range(n):
            pu = eqs[i] @ u  # (r,)
            pv = eos[i] @ v  # (k, r)
            z = (pu @ pv.T) / t
            p = softmax(z)
            y, w = ys[i], ws[i]
            loss += -w * math.log(max(p[y], 1e-12))
            g = p.copy(); g[y] -= 1.0; g *= w / t
            gu += np.outer(eqs[i], g @ pv)
            gv += eos[i].T @ np.outer(g, pu)
            gt += float(w * (g * z).sum())
        gu, gv, gt = gu / n + 1e-4 * u, gv / n + 1e-4 * v, gt / n
        u -= LR * gu; v -= LR * gv
        log_t = float(np.clip(log_t - 0.1 * LR * gt, math.log(0.3), math.log(4.0)))
    return u, v, math.exp(float(log_t)), loss / n

def predict(u, v, t, q, opts):
    return softmax(((q @ u) @ (opts @ v).T) / t)

def main():
    clip_model, clip_proc, bge = get_encoders()
    OPTIONS = ["billing", "shipping", "technical support", "account security"]
    ANCHORS = {
        "billing": "billing: 处理重复扣款、多扣运费、支付渠道异常、退款到账",
        "shipping": "shipping: 处理地址错误、包裹退回、配送延误、偏远地区转寄",
        "technical support": "technical support: 处理APP闪退、无法开机、屏幕坏点、功能异常",
        "account security": "account security: 处理账号被盗、异地登录、密码重置、二次验证失败",
    }
    imgs = make_scenario_images()
    
    # Encode: query = concat(image_emb[512], text_emb[1024]) → 1536
    # Option = text_emb(anchor)[1024]
    opt_mat = np.asarray(bge.embed([ANCHORS[o] for o in OPTIONS]), dtype=np.float64)
    
    # Training data: for each class, multiple (image, text) combos
    train_cases = []
    texts = {
        "billing": ["我的卡被扣了两次同样的钱", "订单多扣了运费怎么退", "支付失败了但钱扣了",
                    "重复扣款4299元未退", "银行说商家多扣了"],
        "shipping": ["我搬家了包裹寄到旧地址", "快递说地址不对退回了", "偏远地区没人配送",
                     "包裹延误一周了", "改了地址但没更新"],
        "technical support": ["APP一打开就闪退", "手机屏幕有两个坏点", "开机没反应",
                              "功能全部失灵", "更新后不能用了"],
        "account security": ["账号被别人登了", "异地登录被冻结了", "密码忘了找不回",
                             "二次验证收不到", "账号异常消费"],
    }
    for opt in OPTIONS:
        img_emb = embed_image(clip_model, clip_proc, imgs[opt])
        for txt in texts[opt]:
            txt_emb = np.asarray(bge.embed([txt])[0], dtype=np.float64)
            fused = np.concatenate([img_emb, txt_emb])  # 1536-dim
            label = OPTIONS.index(opt)
            train_cases.append({"eq": fused, "label": label})
    
    print(f"训练案例: {len(train_cases)} 条, 维度: query={train_cases[0]['eq'].shape[0]}, option={opt_mat.shape[1]}")
    
    # L2 归一化防溢出
    def l2n(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(n, 1e-9)
    for c in train_cases:
        c["eq"] = l2n(c["eq"])
    opt_mat_n = l2n(opt_mat)
    
    eqs = np.stack([c["eq"] for c in train_cases])
    eos = np.stack([opt_mat_n] * len(train_cases))
    ys = np.asarray([c["label"] for c in train_cases])
    ws = np.ones(len(train_cases))
    
    u, v, t, ce = fit_head(eqs, eos, ys, ws, eqs.shape[1], opt_mat.shape[1])
    train_acc = float(np.mean([int(np.argmax(predict(u, v, t, eqs[i], opt_mat))) == ys[i] for i in range(len(ys))]))
    print(f"T={t:.3f} CE={ce:.3f} train_acc={train_acc:.2f}")
    
    # Eval: 新的图文组合(留出的测试用例)
    eval_cases = [
        ("billing", "/tmp/mm_billing.png", "同一笔订单扣了两次4299元"),
        ("shipping", "/tmp/mm_shipping.png", "地址没改包裹送错了"),
        ("technical support", "/tmp/mm_crash.png", "重新安装了两次还是闪退"),
        ("account security", "/tmp/mm_security.png", "发现别人在用我的账号"),
        ("billing", "/tmp/mm_billing.png", "退款说好了但钱没到账"),       # 同图不同文
        ("technical support", "/tmp/mm_crash.png", "屏幕出现坏点要换货"),  # 同图不同文
    ]
    ok = 0
    for want, img_path, text in eval_cases:
        img_emb = embed_image(clip_model, clip_proc, img_path)
        txt_emb = np.asarray(bge.embed([text])[0], dtype=np.float64)
        q = np.concatenate([img_emb, txt_emb])
        q = q / np.linalg.norm(q)  # L2 归一化
        p = predict(u, v, t, q, opt_mat_n)
        pred = OPTIONS[int(np.argmax(p))]
        hit = "OK " if pred == want else "MISS"
        ok += pred == want
        print(f"{hit} want {want:>18s} got {pred:>18s} p={p.max():.3f} | {text}")
    print(f"\n多模态小 JEV: {ok}/{len(eval_cases)}")

if __name__ == "__main__":
    main()
