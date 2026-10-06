"""Dataset splitting and prompt/chat formatting shared by training, eval and inference.

Keeping the prompt in ONE place guarantees the model sees the exact same format
at train time and at inference time.
"""
from __future__ import annotations

import json
import random

SYSTEM_PROMPT = (
    "You are an invoice parsing assistant. Read the invoice text and reply with a single "
    "JSON object only, using the keys: invoice_number, vendor, invoice_date (YYYY-MM-DD), "
    "due_date (YYYY-MM-DD or null), currency (ISO code), items (list of "
    "{description, quantity, unit_price, total}), subtotal, tax, total."
)


def build_user_prompt(invoice_text: str, instruction: str = "Extract invoice information as JSON.") -> str:
    return f"{instruction}\n\nInvoice:\n{invoice_text.strip()}"


def target_json(output: dict) -> str:
    """Canonical serialization of the target (compact: fewer tokens to learn/generate)."""
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def to_messages(record: dict, include_answer: bool = True) -> list[dict]:
    """Convert an {instruction, input, output} record to chat messages."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record["input"], record.get("instruction", ""))},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": target_json(record["output"])})
    return messages


def split_dataset(records: list[dict], seed: int = 42, train: float = 0.8, val: float = 0.1):
    """Shuffle and split into train / validation / test (default 80 / 10 / 10)."""
    records = list(records)
    random.Random(seed).shuffle(records)
    n = len(records)
    n_train, n_val = int(n * train), int(n * val)
    return (
        records[:n_train],
        records[n_train:n_train + n_val],
        records[n_train + n_val:],
    )
