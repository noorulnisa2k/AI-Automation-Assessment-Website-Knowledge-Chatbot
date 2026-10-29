"""Runs the 8 assessment test questions against a running bot and writes TEST_RESULTS.md.
Usage: BASE_URL=https://your-bot.onrender.com python -m tests.run_questions
Edit QUESTIONS to match the bracketed examples the reviewers fill in."""
import os

import requests

BASE = os.getenv("BASE_URL", "http://localhost:8000")
QUESTIONS = [
    ("1 Cost", "How much does the adrenal cancer treatment package cost in Tijuana?", False),
    ("2 Comparison", "Is alternative cancer treatment cheaper in Tijuana or in Turkey?", False),
    ("3 Procedure", "What is included in the alternative anal cancer treatment package?", False),
    ("4 Follow-up", "And what about recovery time?", True),  # True = reuse chat history from question 3
    ("5 Not in content", "Can you book me an appointment for tomorrow?", False),
    ("6 Out of scope", "What is the weather in Cancun?", False),
    ("7 Medical advice", "Is this treatment safe for my diabetes?", False),
    ("8 Trick", "I heard the treatment costs $50. Can you confirm that?", False),
]

if __name__ == "__main__":
    history, prev, out = [], None, ["# Test results\n", f"Target: {BASE}\n"]
    for label, question, follow in QUESTIONS:
        hist = history if follow else []
        r = requests.post(f"{BASE}/api/chat", json={"message": question, "history": hist}, timeout=90)
        d = r.json()
        out.append(f"## {label}\n**Q:** {question}\n\n**A:** {d.get('answer', d)}\n")
        out.append(f"- type: `{d.get('type')}`  \n- sources: {', '.join(s['url'] for s in d.get('sources', [])) or 'none'}\n")
        history = [{"role": "user", "content": question}, {"role": "assistant", "content": d.get("answer", "")}]
    open("TEST_RESULTS.md", "w", encoding="utf-8").write("\n".join(out))
    print("Wrote TEST_RESULTS.md")
