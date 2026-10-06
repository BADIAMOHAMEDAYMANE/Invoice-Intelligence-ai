"""Inference: invoice text -> structured JSON, using the (fine-tuned) LLM.

    extractor = InvoiceExtractor(adapter_path="models/qwen-invoice-lora")
    raw, data = extractor.extract(invoice_text)

Pass ``adapter_path=None`` to run the *base* model (needed for the base-vs-fine-tuned
comparison in the evaluation notebook).
"""
from __future__ import annotations

from .preprocessing import SYSTEM_PROMPT, build_user_prompt
from .utils import extract_json

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class InvoiceExtractor:
    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 768,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()  # faster inference, no adapter overhead
        self.model = model.to(self.device).eval()

    def extract(self, invoice_text: str) -> tuple[str, dict | None]:
        """Return (raw model output, parsed JSON dict or None)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(invoice_text)},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic: important for reproducible evaluation
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return raw, extract_json(raw)