"""Field-level evaluation of predictions against gold JSON.

Use it to compare: base model vs fine-tuned model vs regex baseline.
"""
from __future__ import annotations

from .utils import extract_json

SCALAR_FIELDS = ["invoice_number", "vendor", "invoice_date", "due_date", "currency",
                 "subtotal", "tax", "total"]


def _norm(v):
    if isinstance(v, str):
        return " ".join(v.split()).casefold()
    return v


def _equal(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) < 0.005
    return _norm(a) == _norm(b)


def _items_equal(pred, gold) -> bool:
    if not isinstance(pred, list) or len(pred) != len(gold):
        return False
    for p, g in zip(pred, gold):
        if not isinstance(p, dict):
            return False
        for key in ("description", "quantity", "unit_price", "total"):
            if not _equal(p.get(key), g.get(key)):
                return False
    return True


def evaluate(predictions: list, golds: list[dict]) -> dict:
    """``predictions`` are raw model strings or already-parsed dicts (None allowed)."""
    assert len(predictions) == len(golds)
    n = len(golds)
    correct = {f: 0 for f in SCALAR_FIELDS + ["items"]}
    valid_json = 0
    exact = 0

    for pred, gold in zip(predictions, golds):
        data = extract_json(pred) if isinstance(pred, str) else pred
        if not isinstance(data, dict):
            continue
        valid_json += 1
        all_ok = True
        for f in SCALAR_FIELDS:
            ok = _equal(data.get(f), gold.get(f))
            correct[f] += ok
            all_ok &= ok
        ok_items = _items_equal(data.get("items"), gold["items"])
        correct["items"] += ok_items
        all_ok &= ok_items
        exact += all_ok

    result = {f"{f}_accuracy": correct[f] / n for f in correct}
    n_fields = len(correct)
    result["field_accuracy"] = sum(correct.values()) / (n * n_fields)
    result["json_validity"] = valid_json / n
    result["exact_match"] = exact / n
    return result