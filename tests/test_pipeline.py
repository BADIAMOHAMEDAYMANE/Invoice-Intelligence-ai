import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import extract, parse_amount, parse_date  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.generator import generate_dataset  # noqa: E402
from src.preprocessing import split_dataset, to_messages  # noqa: E402
from src.schema import FIELD_ORDER  # noqa: E402
from src.utils import extract_json  # noqa: E402
from src.validation import validate_invoice  # noqa: E402

GOOD = {
    "invoice_number": "INV-1", "vendor": "ACME", "invoice_date": "2026-09-12",
    "due_date": "2026-10-12", "currency": "USD",
    "items": [{"description": "A", "quantity": 2, "unit_price": 10.0, "total": 20.0},
              {"description": "B", "quantity": 1, "unit_price": 5.0, "total": 5.0}],
    "subtotal": 25.0, "tax": 5.0, "total": 30.0,
}


def test_generator_is_deterministic_and_gold_is_consistent():
    a, b = generate_dataset(200, seed=1), generate_dataset(200, seed=1)
    assert a == b
    for rec in a:
        assert list(rec["output"]) == FIELD_ORDER
        assert validate_invoice(rec["output"]).ok, rec["output"]


def test_split_sizes_and_disjoint():
    recs = generate_dataset(100, seed=3)
    tr, va, te = split_dataset(recs)
    assert (len(tr), len(va), len(te)) == (80, 10, 10)
    assert len({r["input"] for r in tr} & {r["input"] for r in te}) <= 1  # tolerate rare dup text


def test_chat_messages():
    msgs = to_messages(generate_dataset(1)[0])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert extract_json(msgs[-1]["content"]) is not None


def test_validation_ok_and_failures():
    assert validate_invoice(GOOD).ok

    bad_total = {**GOOD, "total": 99.0}
    codes = {i.code for i in validate_invoice(bad_total).issues}
    assert "total_mismatch" in codes

    bad_item = {**GOOD, "items": [{**GOOD["items"][0], "total": 99.0}, GOOD["items"][1]]}
    codes = {i.code for i in validate_invoice(bad_item).issues}
    assert "item_total_mismatch" in codes and "subtotal_mismatch" in codes

    bad_dates = {**GOOD, "due_date": "2026-01-01"}
    assert "due_before_invoice" in {i.code for i in validate_invoice(bad_dates).issues}

    assert validate_invoice("not json at all").invoice is None
    assert validate_invoice({**GOOD, "invoice_date": "garbage"}).schema_errors


def test_extract_json_from_noisy_output():
    assert extract_json('Sure! ```json\n{"a": {"b": 1}}\n``` done') == {"a": {"b": 1}}
    assert extract_json("no json") is None


def test_baseline_parsers():
    assert parse_amount("$1,234.50") == 1234.5
    assert parse_amount("1.234,50 €") == 1234.5
    assert parse_amount("1 234,50 DH") == 1234.5
    assert parse_date("12 septembre 2026") == "2026-09-12"
    assert parse_date("September 12, 2026") == "2026-09-12"
    assert parse_date("09/24/2026") == "2026-09-24"
    assert parse_date("12/09/2026") == "2026-09-12"


def test_baseline_and_evaluation():
    txt = "ACME SOFTWARE\nInvoice #: INV-9\nDate: 2026-09-12\nSubtotal: $200.00\nTax: $40.00\nTotal: $240.00"
    d = extract(txt)
    assert (d["invoice_number"], d["subtotal"], d["tax"], d["total"], d["currency"]) == ("INV-9", 200.0, 40.0, 240.0, "USD")

    gold = [GOOD]
    perfect = evaluate([GOOD], gold)
    assert perfect["exact_match"] == 1.0 and perfect["json_validity"] == 1.0
    broken = evaluate(["garbage"], gold)
    assert broken["json_validity"] == 0.0 and broken["field_accuracy"] == 0.0
    partial = evaluate([{**GOOD, "total": 1.0}], gold)
    assert partial["total_accuracy"] == 0.0 and partial["vendor_accuracy"] == 1.0


def test_split_is_reproducible():
    recs = generate_dataset(50, seed=5)
    assert split_dataset(recs, seed=1) == split_dataset(recs, seed=1)