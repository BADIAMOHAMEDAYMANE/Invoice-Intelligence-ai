"""Evaluate the fine-tuned model on held-out test invoices and compare it to baselines.

Usage (from the project root):
    python scripts/evaluate_model.py --n 20                 # fine-tuned vs regex baseline
    python scripts/evaluate_model.py --n 20 --with-base     # + the base model without LoRA

Generation is slow on a laptop, so start small (--n 20) and raise it later. Progress is
printed per invoice. Press Ctrl+C at any time: the invoices already done are evaluated.
Results are written to results/eval_results.json and results/eval_table.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import extract as regex_extract  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.generator import generate_dataset  # noqa: E402
from src.preprocessing import split_dataset  # noqa: E402
from src.utils import extract_json  # noqa: E402
from src.validation import validate_invoice  # noqa: E402

METRICS = ["field_accuracy", "exact_match", "json_validity", "invoice_number_accuracy", "vendor_accuracy",
           "invoice_date_accuracy", "due_date_accuracy", "currency_accuracy", "subtotal_accuracy",
           "tax_accuracy", "total_accuracy", "items_accuracy", "validation_pass"]


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_model(label: str, samples: list[dict], base_model: str, adapter: str | None,
              device: str, max_new_tokens: int) -> list[str]:
    """Generate raw outputs for the samples; returns what was completed (even if interrupted)."""
    import torch

    from src.inference import InvoiceExtractor

    print(f"\nLoading {label} ...", flush=True)
    extractor = InvoiceExtractor(base_model=base_model, adapter_path=adapter, device=device,
                                 max_new_tokens=max_new_tokens)
    outputs: list[str] = []
    try:
        for i, rec in enumerate(samples, start=1):
            t0 = time.time()
            raw, data = extractor.extract(rec["input"])
            outputs.append(raw)
            print(f"  [{label}] {i}/{len(samples)} | {time.time() - t0:.0f}s | "
                  f"json={'ok' if data else 'INVALID'}", flush=True)
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(outputs)} invoices: evaluating those.", flush=True)
    del extractor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    return outputs


def score(preds: list, golds: list[dict]) -> dict:
    result = evaluate(preds, golds)
    ok = sum(1 for p in preds if validate_invoice(extract_json(p) if isinstance(p, str) else p).ok)
    result["validation_pass"] = ok / len(preds)
    return result


def table(results: dict[str, dict], n: int) -> str:
    names = list(results)
    lines = [f"Evaluation on {n} held-out test invoices", "",
             "| Metric | " + " | ".join(names) + " |",
             "|---|" + "---:|" * len(names)]
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(f"{results[k][m]:.2f}" for k in names) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="models/qwen-invoice-lora")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=20, help="number of test invoices")
    ap.add_argument("--with-base", action="store_true", help="also evaluate the base model without LoRA")
    ap.add_argument("--n-records", type=int, default=5000, help="must match training/generate_dataset.py")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    _, _, test = split_dataset(generate_dataset(args.n_records, seed=args.seed), seed=args.seed)
    samples = test[:args.n]
    device = pick_device()
    print(f"device: {device} | {len(samples)} test invoices", flush=True)

    raw: dict[str, list] = {}
    if args.with_base:
        raw["Base model"] = run_model("base", samples, args.base_model, None, device, args.max_new_tokens)
    raw["Fine-tuned"] = run_model("fine-tuned", samples, args.base_model, args.adapter, device, args.max_new_tokens)

    n_done = min(len(v) for v in raw.values())
    if n_done == 0:
        print("No invoice was completed, nothing to evaluate.")
        return
    samples = samples[:n_done]
    golds = [r["output"] for r in samples]

    preds = {"Regex baseline": [regex_extract(r["input"]) for r in samples]}
    preds.update({k: v[:n_done] for k, v in raw.items()})
    ordered = ["Regex baseline"] + (["Base model"] if "Base model" in preds else []) + ["Fine-tuned"]
    results = {k: score(preds[k], golds) for k in ordered}

    text = table(results, n_done)
    print("\n" + text, flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_table.md").write_text(text + "\n", encoding="utf-8")
    (out / "eval_results.json").write_text(json.dumps({
        "n": n_done,
        "metrics": results,
        "samples": [{"input": r["input"], "gold": r["output"],
                     **{k: (preds[k][i] if isinstance(preds[k][i], str) else json.dumps(preds[k][i]))
                        for k in ordered}} for i, r in enumerate(samples)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out / 'eval_table.md'} and {out / 'eval_results.json'}", flush=True)


if __name__ == "__main__":
    main()