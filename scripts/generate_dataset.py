"""Generate the synthetic dataset and write train/validation/test JSON files.

Usage:
    python scripts/generate_dataset.py --n 5000 --seed 42 --out data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import generate_dataset  # noqa: E402
from src.preprocessing import split_dataset  # noqa: E402
from src.utils import save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5000, help="number of invoices to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    records = generate_dataset(args.n, args.seed)
    train, val, test = split_dataset(records, seed=args.seed)

    out = Path(args.out)
    save_json(train, out / "train.json")
    save_json(val, out / "validation.json")
    save_json(test, out / "test.json")

    # A few human-readable examples to eyeball (committed to git, unlike the big files).
    with open(out / "samples.txt", "w", encoding="utf-8") as f:
        for i, rec in enumerate(train[:5], start=1):
            f.write(f"===== SAMPLE {i} (input) =====\n{rec['input']}\n\n")
            f.write(f"===== SAMPLE {i} (target) =====\n")
            f.write(json.dumps(rec["output"], ensure_ascii=False, indent=2) + "\n\n")

    print(f"train={len(train)} validation={len(val)} test={len(test)} -> {out.resolve()}")


if __name__ == "__main__":
    main()