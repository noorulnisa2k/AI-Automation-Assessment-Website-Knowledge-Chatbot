"""Deterministic safety checks that run AFTER the LLM. Pure functions, unit-tested."""
import re

MONEY = re.compile(
    r"(?:US\$|MX\$|USD|EUR|GBP|MXN|[$€£])\s?(\d[\d,]*(?:\.\d+)?)"
    r"|(\d[\d,]*(?:\.\d+)?)\s?(?:USD|EUR|GBP|MXN|dollars?|euros?|pesos?)",
    re.I,
)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
MEDICAL = re.compile(
    r"\b(safe|safety|diabet\w*|my (?:condition|doctor|cancer|health)|should i|can i take|interact\w*|"
    r"side effects?|dose|dosage|medication|pregnan\w*|allerg\w*|cure|survival|prognosis|diagnos\w*)\b",
    re.I,
)
LEAD = re.compile(r"\b(quote|book\w*|appointment|contact|consult\w*|call me|reserve|sign me up|get started)\b", re.I)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def money_values(text: str) -> set[float]:
    out = set()
    for m in MONEY.finditer(text):
        v = _to_float(m.group(1) or m.group(2))
        if v is not None:
            out.add(v)
    return out


def numbers_in(text: str) -> set[float]:
    return {v for v in (_to_float(n) for n in NUMBER.findall(text)) if v is not None}


def prices_grounded(answer: str, context: str) -> bool:
    """Every monetary figure in the answer must literally exist in the retrieved context."""
    return money_values(answer) <= numbers_in(context)


def valid_sources(cited: list[str], retrieved: set[str]) -> list[str]:
    seen, out = set(), []
    for u in cited or []:
        if u in retrieved and u not in seen:
            seen.add(u)
            out.append(u)
    return out
