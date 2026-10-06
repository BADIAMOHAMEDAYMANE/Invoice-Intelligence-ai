"""Regex-based baseline extractor.

Two uses: (1) a sanity-check baseline to compare the fine-tuned LLM against, and
(2) a no-GPU fallback so the Streamlit demo works before a model is trained.
It deliberately does NOT extract line items (that is where the LLM should win).
"""
from __future__ import annotations

import re
from datetime import date

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_MONTHS.update({m: i for i, m in enumerate(
    ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
     "septembre", "octobre", "novembre", "décembre"], start=1)})

_AMOUNT = r"([^\d\n:]{0,4}[ ]?\d[\d \u202f\xa0.,]*\d|[^\d\n:]{0,4}[ ]?\d)"
_DATE_PAT = (r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{4}|"
             r"[A-Za-zéûô]+\s+\d{1,2},\s*\d{4}|\d{1,2}\s+[A-Za-zéûô]+\s+\d{4})")


def parse_amount(s: str) -> float | None:
    s = re.sub(r"[^\d.,\s]", "", s).strip().replace(" ", " ").replace("\xa0", " ")
    s = s.replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s:
        s = s.replace(",", ".") if re.search(r",\d{1,2}$", s) else s.replace(",", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def parse_date(s: str) -> str | None:
    s = s.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return date.fromisoformat(s).isoformat()
        m = re.fullmatch(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", s)
        if m:
            a, b, y = map(int, m.groups())
            # Default to DD/MM; fall back to MM/DD only when that is the sole valid reading.
            day, month = (b, a) if b > 12 else (a, b)
            return date(y, month, day).isoformat()
        m = re.fullmatch(r"([A-Za-zéûô]+)\s+(\d{1,2}),\s*(\d{4})", s)
        if m and m.group(1).lower() in _MONTHS:
            return date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2))).isoformat()
        m = re.fullmatch(r"(\d{1,2})\s+([A-Za-zéûô]+)\s+(\d{4})", s)
        if m and m.group(2).lower() in _MONTHS:
            return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1))).isoformat()
    except ValueError:
        return None
    return None


def _detect_currency(text: str) -> str | None:
    if "$" in text:
        return "USD"
    if "£" in text:
        return "GBP"
    if "€" in text or re.search(r"\bEUR\b", text):
        return "EUR"
    if re.search(r"\bMAD\b|\bDH\b", text):
        return "MAD"
    return None


def _labelled(text: str, labels: str, pattern: str) -> str | None:
    m = re.search(rf"^\s*(?:{labels})[^:\n]*:\s*{pattern}", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else None


def extract(text: str) -> dict:
    """Return a dict following the target schema (items always empty)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    number = _labelled(text, r"invoice\s*(?:no|#|number)|inv\s*no|facture\s*(?:n°|no)|n°\s*de\s*facture|numéro\s*de\s*facture",
                       r"(\S+)")
    inv_date = _labelled(text, r"date(?!\s*d'échéance)|invoice\s*date|issued|émise\s*le|date\s*de\s*facture", _DATE_PAT)
    due = _labelled(text, r"due|payment\s*due|échéance|date\s*d'échéance|à\s*payer\s*avant", _DATE_PAT)

    def money(labels: str) -> float | None:
        raw = _labelled(text, labels, _AMOUNT)
        return parse_amount(raw) if raw else None

    subtotal = money(r"sub[\s-]?total|total\s*ht|montant\s*ht|sous-total")
    tax = money(r"tax|vat|tva|sales\s*tax")
    total = money(r"total\s*due|amount\s*due|grand\s*total|total\s*ttc|total\s*à\s*payer|net\s*à\s*payer|total(?!\s*ht)")

    return {
        "invoice_number": number,
        "vendor": lines[0] if lines else None,
        "invoice_date": parse_date(inv_date) if inv_date else None,
        "due_date": parse_date(due) if due else None,
        "currency": _detect_currency(text),
        "items": [],
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }