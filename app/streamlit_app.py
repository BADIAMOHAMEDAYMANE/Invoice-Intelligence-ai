"""Streamlit demo: upload invoice -> OCR -> extraction -> arithmetic repair -> validation.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import baseline  # noqa: E402
from src.postprocess import repair_invoice  # noqa: E402
from src.validation import validate_invoice  # noqa: E402

st.set_page_config(page_title="Invoice Intelligence AI", page_icon="🧾", layout="wide")
st.title("🧾 Invoice Intelligence AI")
st.caption("OCR → fine-tuned LLM → arithmetic repair → structured JSON → validation")

SAMPLE = """ACME SOFTWARE
Invoice #: INV-2026-1042
Date: 12/09/2026
Due Date: 12/10/2026

Python Course 1 x $120.00 = $120.00
Cloud Hosting 1 x $80.00 = $80.00

Subtotal: $200.00
Tax: $40.00
Total: $240.00"""

# Keys that only an invoice has ("total" is excluded: a single line item has one too).
INVOICE_KEYS = {"invoice_number", "vendor", "invoice_date", "items", "subtotal"}


@st.cache_resource(show_spinner="Loading model...")
def load_llm(base_model: str, adapter_path: str):
    from src.inference import InvoiceExtractor

    return InvoiceExtractor(base_model=base_model, adapter_path=adapter_path or None)


# ------------------------------- sidebar ---------------------------------- #
with st.sidebar:
    st.header("Settings")
    ocr_backend = st.selectbox("OCR backend", ["tesseract", "paddleocr"])
    engine = st.radio("Extractor", ["Regex baseline (no GPU)", "Fine-tuned LLM"])
    base_model = adapter = ""
    if engine == "Fine-tuned LLM":
        base_model = st.text_input("Base model", "Qwen/Qwen2.5-0.5B-Instruct")
        adapter = st.text_input("LoRA adapter path", "models/qwen-invoice-lora-v5")
        st.caption("Leave the adapter empty to run the base model.")
    use_repair = st.checkbox(
        "Repair with arithmetic checks", value=True,
        help="Fixes values that break quantity x price = total, sum of lines = subtotal, "
             "subtotal + tax = total. Every change is listed.",
    )

# ------------------------------- input ------------------------------------ #
tab_upload, tab_text = st.tabs(["Upload file", "Paste text"])
text = ""
with tab_upload:
    file = st.file_uploader("Invoice (PDF / PNG / JPG)", type=["pdf", "png", "jpg", "jpeg"])
    if file is not None:
        from src.ocr import extract_text

        try:
            with st.spinner("Running OCR..."):
                text = extract_text(file.getvalue(), file.name, ocr_backend)
        except Exception as e:  # missing binary, bad file, ...
            st.error(f"OCR failed: {e}")
with tab_text:
    st.button("Use sample invoice", on_click=lambda: st.session_state.update(pasted=SAMPLE))
    pasted = st.text_area("Invoice text", height=220, key="pasted")
    if pasted.strip():
        text = pasted

if not text:
    st.info("Upload an invoice or paste its text to start.")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Extracted text")
    st.code(text, language=None)

# ------------------------------- extraction ------------------------------- #
raw = None
with st.spinner("Extracting..."):
    if engine.startswith("Regex"):
        data = baseline.extract(text)
    else:
        try:
            raw, data = load_llm(base_model, adapter).extract(text)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

# A JSON fragment that is not an invoice (e.g. a single item the parser found inside a broken answer).
if isinstance(data, dict) and not (INVOICE_KEYS & set(data)):
    st.warning("The model answer is not a complete invoice (its JSON is probably malformed or cut). "
               "See the raw output below.")
    data = None

final, notes = data, []
if use_repair and isinstance(data, dict):
    final, notes = repair_invoice(data, text)

with right:
    st.subheader("Structured JSON")
    if final is None:
        st.json({"raw_output": raw})
    elif notes:
        t_fixed, t_model = st.tabs(["Repaired", "Model output"])
        t_fixed.json(final)
        t_model.json(data)
    else:
        st.json(final)

if notes:
    with st.expander(f"🔧 {len(notes)} repair(s) applied", expanded=True):
        for n in notes:
            st.markdown(f"- {n}")
if raw is not None:
    with st.expander("Raw model output"):
        st.code(raw, language="json")

report = validate_invoice(final if final is not None else raw)

st.divider()
if report.invoice is None:
    st.error("Could not parse a valid invoice: " + "; ".join(report.schema_errors))
else:
    inv = report.invoice
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendor", inv.vendor or "—")
    c2.metric("Invoice #", inv.invoice_number or "—")
    c3.metric("Date", str(inv.invoice_date or "—"))
    c4.metric("Total", f"{inv.total:,.2f} {inv.currency or ''}" if inv.total is not None else "—")
    if inv.items:
        st.subheader("Items")
        rows = [i.model_dump() for i in inv.items]
        try:  # newer Streamlit: width="stretch"; older versions only know use_container_width
            st.dataframe(rows, width="stretch")
        except TypeError:
            st.dataframe(rows, use_container_width=True)
    if report.ok:
        st.success("✓ Invoice validated")
    else:
        for issue in report.issues:
            (st.error if issue.severity == "error" else st.warning)(f"⚠️ {issue.message}")