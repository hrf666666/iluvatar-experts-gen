#!/usr/bin/env python3
"""Generate law-decision training data for the LoRA expert (civil + criminal law case).

Task: given a free-text legal situation (JEV [state] format), pick ONE of four actions:
A) 民事索赔  B) 刑事追责  C) 民事+刑事同时  D) 仅协商解决
Prompts carry NO retrieved context — this expert's knowledge must live in the WEIGHTS.

Writer: Qwen3.8-27B drafts a scenario per target label grounded in a KB chunk.
Labeler: the same 27B WITH both law KBs (via the chat gateway, per-request kb) answers
with a single letter; a case ships only when the labeler agrees with the intent
(rejection sampling). Held-out split kept for the final eval.
"""
import json, re, sys, time, urllib.request
from pathlib import Path

WRITER = "http://127.0.0.1:8902"          # raw chat (no kb) — writes scenarios
LABELER = "http://127.0.0.1:5000"         # chat gateway with kb — labels them
KB_DIR = Path("/root/hot-remove-experts/kb")
OUT = Path("/root/hot-remove-experts/law_decisions.jsonl")

OPTIONS = ["民事索赔", "刑事追责", "民事+刑事同时", "仅协商解决"]
LETTERS = "ABCD"

W_PROMPT = {
    "A": "写一个真实感的中国民事纠纷场景(2-3句):只涉及财产/合同/侵权赔偿,应通过民事索赔解决;"
         "不要出现犯罪情节(不得有酒驾、伤人、危险驾驶等),金额或损失要具体。只输出场景本身。",
    "B": "写一个真实感的中国刑事场景(2-3句):涉及危险驾驶罪/醉驾(血液酒精含量给出具体数值≥80mg/100ml)"
         "或交通肇事,重点在刑事追责;不要写入民事索赔诉求。只输出场景本身。",
    "C": "写一个真实感的中国混合场景(2-3句):既有醉驾/危险驾驶等刑事情节(血检数值具体),"
         "又有明确的财产损失和拒赔/索赔诉求,民事刑事两条线都要处理。只输出场景本身。",
    "D": "写一个真实感的中国轻微纠纷场景(2-3句):小额损失、对方态度配合愿意赔偿、"
         "不构成任何犯罪,协商即可解决;给出具体小金额。只输出场景本身。",
}
L_PROMPT = ("根据知识库中的法律知识判断:对于下面这个场景,当事人最应该采取哪种行动?"
            "A) 民事索赔 B) 刑事追责 C) 民事+刑事同时 D) 仅协商解决\n"
            "只输出一个字母(A/B/C/D),不要输出别的。\n\n场景:{s}")


def chat(base, msgs, max_tokens=400, temperature=0.5, kb=None):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    if kb:
        body["kb"] = kb
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def label(state):
    out = chat(LABELER, [{"role": "user", "content": L_PROMPT.format(s=state)}],
               max_tokens=8, temperature=0.1, kb=["civil-law", "criminal-law"])
    m = re.search(r"[ABCD]", out.upper())
    return m.group(0) if m else None


def main(per_label=14):
    cases = []
    for lab in "ABCD":
        got, tries = 0, 0
        while got < per_label and tries < per_label * 4:
            tries += 1
            s = chat(WRITER, [{"role": "user", "content": W_PROMPT[lab]}],
                     max_tokens=160, temperature=0.9).strip()
            s = " ".join(x for x in s.splitlines() if x.strip())[:240]
            if len(s) < 20:
                continue
            got_lab = label(s)
            ok = got_lab == lab
            print(f"  [{'OK' if ok else 'X '}] want {lab} got {got_lab}: {s[:44]}")
            if ok:
                cases.append({"label": lab, "state": s,
                              "question": "当事人最应该采取哪种行动?",
                              "options": OPTIONS})
                got += 1
    import random
    random.Random(7).shuffle(cases)
    with OUT.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[done] {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
