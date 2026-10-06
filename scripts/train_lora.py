"""Standalone LoRA fine-tuning of Qwen2.5-0.5B-Instruct for invoice extraction.

A plain PyTorch training loop (no Hugging Face Trainer) that prints the time of every
step, so you can see exactly what is happening. Works on NVIDIA GPU (cuda), Apple
Silicon (mps) and CPU.

Usage (from the project root):
    python scripts/train_lora.py --train-size 600 --epochs 2
    python scripts/train_lora.py --train-size 40 --max-steps 3     # quick smoke test

Continue training an existing adapter on NEW data (more receipts), without touching the old one:
    python scripts/train_lora.py --resume models/qwen-invoice-lora-v2 --out models/qwen-invoice-lora-v3 \\
        --data-seed 43 --receipt-ratio 0.6 --train-size 320 --epochs 1 --grad-acc 2 --lr 1e-4

--data-seed picks the random stream of the training data. Keep evaluation on the default
seed (42): training on another seed means the evaluation test set is never seen in training.

The adapter is saved to models/qwen-invoice-lora (also every --save-every steps, and
when you press Ctrl+C, so an interrupted run is never lost).

Apple Silicon note: the MPS backend builds and caches a compute graph for every new tensor
shape, so training on texts of ~300 different lengths makes each step slower and slower (and
can end in a stall). Two guards keep it flat: every batch is padded to a multiple of
--pad-multiple tokens (padding is masked, the loss is unchanged), and the GPU cache is
released after each optimizer step. The step line shows the GPU memory in use.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.generator import generate_dataset  # noqa: E402
from src.preprocessing import split_dataset, target_json, to_messages  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode(tok, record: dict) -> dict:
    """Tokenize one record; the loss is computed only on the answer (the JSON)."""
    prompt_text = tok.apply_chat_template(
        to_messages(record, include_answer=False), tokenize=False, add_generation_prompt=True
    )
    answer_text = target_json(record["output"]) + tok.eos_token  # eos teaches the model to stop
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tok(answer_text, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def build_examples(tok, records: list[dict], max_len: int, name: str) -> list[dict]:
    encoded = [encode(tok, r) for r in records]
    lengths = np.array([len(e["input_ids"]) for e in encoded])
    kept = [e for e in encoded if len(e["input_ids"]) <= max_len]
    print(f"{name}: {len(kept)}/{len(encoded)} kept | tokens mean={lengths.mean():.0f} "
          f"p95={np.percentile(lengths, 95):.0f} max={lengths.max()}", flush=True)
    return kept


PAD_MULTIPLE = 64  # set from --pad-multiple; 1 disables bucketing


def collate(batch: list[dict], pad_id: int, device: str) -> dict:
    longest = max(len(e["input_ids"]) for e in batch)
    longest = -(-longest // PAD_MULTIPLE) * PAD_MULTIPLE  # round up: few distinct shapes for the GPU
    ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), longest), dtype=torch.long)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        attn[i, :n] = 1
        labels[i, :n] = torch.tensor(e["labels"])
    return {"input_ids": ids.to(device), "attention_mask": attn.to(device), "labels": labels.to(device)}


def free_memory(device: str) -> None:
    """Release cached GPU buffers (stops the slow memory growth seen on Apple Silicon)."""
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def gpu_mem_gb(device: str) -> str:
    try:
        if device == "mps":
            return f"{torch.mps.driver_allocated_memory() / 1e9:.1f} GB"
        if device == "cuda":
            return f"{torch.cuda.memory_allocated() / 1e9:.1f} GB"
    except Exception:  # noqa: BLE001 - purely informative
        pass
    return "n/a"


@torch.no_grad()
def validation_loss(model, examples: list[dict], pad_id: int, device: str, batch: int) -> float:
    model.eval()
    losses = []
    for i in range(0, len(examples), batch):
        losses.append(model(**collate(examples[i:i + batch], pad_id, device)).loss.item())
    model.train()
    return float(np.mean(losses))


def train(model, train_ex, *, pad_id, device, epochs, batch, grad_acc, lr, seed,
          max_steps=None, save_fn=None, save_every=25) -> None:
    from transformers import get_cosine_schedule_with_warmup

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    micro_per_epoch = math.ceil(len(train_ex) / batch)
    total = math.ceil(micro_per_epoch / grad_acc) * epochs
    if max_steps:
        total = min(total, max_steps)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total)), total)

    rng = random.Random(seed)
    model.train()
    step, t0 = 0, time.time()
    print(f"Training: {len(train_ex)} examples x {epochs} epochs -> {total} optimizer steps "
          f"(batch {batch} x grad_acc {grad_acc})", flush=True)

    for epoch in range(epochs):
        order = list(range(len(train_ex)))
        rng.shuffle(order)
        micro = [order[i:i + batch] for i in range(0, len(order), batch)]
        for g in range(0, len(micro), grad_acc):
            group = micro[g:g + grad_acc]
            loss_sum = 0.0
            for m, idxs in enumerate(group):
                t_micro = time.time()
                b = collate([train_ex[i] for i in idxs], pad_id, device)
                loss = model(**b).loss / len(group)
                loss.backward()
                loss_sum += loss.item()  # .item() also synchronizes the GPU, so timings are real
                if step == 0 and m < 3:
                    print(f"  first micro-batch {m + 1}: {time.time() - t_micro:.1f}s "
                          f"(seq len {b['input_ids'].shape[1]})", flush=True)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            del loss, b
            free_memory(device)
            step += 1
            elapsed = time.time() - t0
            eta = elapsed / step * (total - step)
            print(f"step {step}/{total} | epoch {epoch + 1}/{epochs} | loss {loss_sum:.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | {elapsed / 60:.1f} min elapsed | ETA {eta / 60:.1f} min"
                  f" | gpu {gpu_mem_gb(device)}", flush=True)
            if save_fn and step % save_every == 0:
                save_fn()
            if step >= total:
                return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="models/qwen-invoice-lora")
    ap.add_argument("--n-records", type=int, default=5000, help="must match generate_dataset.py to keep the same splits")
    ap.add_argument("--data-seed", type=int, default=None, help="seed of the generated training data (default: --seed)")
    ap.add_argument("--receipt-ratio", type=float, default=None, help="share of receipts in the data (default: generator default)")
    ap.add_argument("--resume", default=None, help="adapter folder to continue training from")
    ap.add_argument("--train-size", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-acc", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--pad-multiple", type=int, default=64,
                    help="pad every batch to a multiple of this many tokens (fewer GPU shapes; 1 = off)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke test)")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global PAD_MULTIPLE
    PAD_MULTIPLE = max(1, args.pad_multiple)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device()
    print(f"torch {torch.__version__} | device: {device}", flush=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_seed = args.seed if args.data_seed is None else args.data_seed
    gen_kwargs = {} if args.receipt_ratio is None else {"receipt_ratio": args.receipt_ratio}
    records = generate_dataset(args.n_records, seed=data_seed, **gen_kwargs)
    train_recs, val_recs, _test = split_dataset(records, seed=data_seed)
    n_rec = sum(r.get("kind") == "receipt" for r in train_recs[:args.train_size])
    print(f"data seed {data_seed} | {n_rec}/{min(args.train_size, len(train_recs))} training examples are receipts", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train_ex = build_examples(tok, train_recs[:args.train_size], args.max_len, "train")
    val_ex = build_examples(tok, val_recs[:50], args.max_len, "val")

    print("Loading model...", flush=True)
    try:  # newer transformers renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model = model.to(device)
    model.config.use_cache = False
    if args.resume:
        from peft import PeftModel

        print(f"Resuming from adapter: {args.resume}", flush=True)
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
        ))
    model.print_trainable_parameters()

    out = Path(args.out)

    def save() -> None:
        model.save_pretrained(out)
        tok.save_pretrained(out)

    try:
        print(f"Validation loss before training: {validation_loss(model, val_ex, pad_id, device, args.batch):.4f}", flush=True)
        train(model, train_ex, pad_id=pad_id, device=device, epochs=args.epochs, batch=args.batch,
              grad_acc=args.grad_acc, lr=args.lr, seed=args.seed, max_steps=args.max_steps,
              save_fn=save, save_every=args.save_every)
        print(f"Validation loss after training:  {validation_loss(model, val_ex, pad_id, device, args.batch):.4f}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted: saving the adapter trained so far...", flush=True)
    save()
    print(f"Adapter saved to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()