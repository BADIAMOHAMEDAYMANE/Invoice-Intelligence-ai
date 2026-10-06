"""OCR layer: image / PDF bytes -> plain text.

Backends:
  * "tesseract" - needs the tesseract binary (brew install tesseract tesseract-lang)
  * "paddleocr" - pip install paddleocr paddlepaddle (usually better on noisy scans)

PDFs: if the PDF has an embedded text layer we use it directly (fast and exact);
otherwise pages are rendered to images and sent through the OCR backend.

Tesseract preprocessing ("auto" mode, the default): no single cleaning recipe works for every
image (a phone photo of a crumpled receipt needs strong cleaning, a tiny clean screenshot is
damaged by it). So two versions of the image are read and the more plausible text is kept
(Tesseract confidence, minus penalties for corrupted numbers and impossible dates):
  * "light"    - EXIF rotation, grayscale, contrast stretch, white border (original resolution)
  * "enhanced" - the above + upscaling to ~300 dpi, sharpening, shadow removal and binarisation
Pass ``preprocess=False`` to OCR the image as-is.

All heavy imports are lazy so the rest of the project works without OCR installed.
"""
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
MIN_EMBEDDED_TEXT_CHARS = 30

TARGET_WIDTH = 1800   # images narrower than this are upscaled to it (Tesseract likes ~300 dpi)
MAX_SIDE = 4000       # never exceed this on the longest side (memory / speed)


# ------------------------------ preprocessing ------------------------------ #
def _otsu_threshold(gray) -> int:
    """Otsu's method on a uint8 grayscale numpy array (no OpenCV needed)."""
    import numpy as np

    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = hist.sum()
    sum_all = (np.arange(256) * hist).sum()
    w0 = sum0 = 0.0
    best, thr = -1.0, 127
    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * hist[t]
        between = w0 * w1 * (sum0 / w0 - (sum_all - sum0) / w1) ** 2
        if between > best:
            best, thr = between, t
    return thr


def preprocess_image(img, binarize: bool = True, enhance: bool = True):
    """Clean an image for OCR; returns a PIL image.

    enhance=False keeps the original resolution and only fixes rotation and contrast.
    """
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps

    img = ImageOps.exif_transpose(img)                      # phone photos carry a rotation flag
    if img.mode in ("RGBA", "LA", "P"):                     # flatten transparency on white
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    gray = ImageOps.grayscale(img)

    if enhance:
        w, h = gray.size
        scale = min(TARGET_WIDTH / w, MAX_SIDE / max(w, h))
        if scale > 1.05:                                    # upscale only, never shrink
            gray = gray.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    gray = ImageOps.autocontrast(gray, cutoff=1)            # stretch contrast, ignore 1% outliers
    if enhance:
        gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

    if enhance and binarize:
        arr = np.array(gray)
        # Local background removal: divide by a blurred copy to flatten shadows / uneven light.
        blur = np.array(gray.filter(ImageFilter.GaussianBlur(radius=max(15, gray.size[0] // 40))))
        flat = np.clip(arr.astype(float) / np.maximum(blur, 1) * 255, 0, 255).astype("uint8")
        thr = _otsu_threshold(flat)
        gray = Image.fromarray(((flat > thr) * 255).astype("uint8"))

    return ImageOps.expand(gray, border=30, fill=255)       # white margin: Tesseract reads edges better


# --------------------------------- backends -------------------------------- #
def _tesseract_read(img) -> tuple[str, float]:
    """Return (text, mean word confidence 0-100) for one image."""
    import pytesseract

    out = pytesseract.Output.DICT
    try:
        d = pytesseract.image_to_data(img, lang="eng+fra", config="--psm 6", output_type=out)
    except pytesseract.TesseractError:
        # French language data not installed (brew install tesseract-lang): fall back to English.
        d = pytesseract.image_to_data(img, lang="eng", config="--psm 6", output_type=out)
    lines: dict[tuple, list[str]] = {}
    confs: list[float] = []
    for text, conf, *key in zip(d["text"], d["conf"], d["block_num"], d["par_num"], d["line_num"]):
        if text.strip() and float(conf) >= 0:
            lines.setdefault(tuple(key), []).append(text)
            confs.append(float(conf))
    return "\n".join(" ".join(words) for words in lines.values()), (sum(confs) / len(confs) if confs else 0.0)


_DIGIT_LOOKALIKES = "SOoIlBZG"  # letters OCR confuses with digits: "25S.08" is a misread "255.08"
_NUMBERISH = re.compile(rf"[\d.,:/€$£%()+-]*[{_DIGIT_LOOKALIKES}][\d.,:/€$£%()+SOoIlBZG-]*")
_DATE_DMY = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})\b")


def _plausibility(text: str, conf: float) -> float:
    """Score an OCR result: Tesseract confidence + sanity checks that confidence alone misses.

    Binarising a tiny, already clean receipt can turn "04/05/2024" into "04/85/2024" and
    "507.50" into "597.59" while Tesseract stays just as "confident", so we also look for
    impossible dates and numbers polluted by digit-lookalike letters.
    """
    score = conf
    for tok in text.split():
        if re.search(r"\d", tok) and _NUMBERISH.fullmatch(tok) and re.search(rf"[{_DIGIT_LOOKALIKES}]", tok):
            score -= 3.0
    for d, m, _y in _DATE_DMY.findall(text):
        score += 3.0 if (1 <= int(d) <= 31 and 1 <= int(m) <= 12) else -15.0
    for _y, m, d in _DATE_ISO.findall(text):
        score += 3.0 if (1 <= int(d) <= 31 and 1 <= int(m) <= 12) else -15.0
    return score


def _ocr_tesseract(img, preprocess: bool = True) -> str:
    if not preprocess:
        return _tesseract_read(img)[0]
    results = []
    for name, enhance in (("light", False), ("enhanced", True)):
        text, conf = _tesseract_read(preprocess_image(img, enhance=enhance))
        results.append((_plausibility(text, conf), name, conf, text))
    if os.environ.get("INVOICE_OCR_DEBUG"):  # prints both candidates' scores in the terminal
        print("[ocr] " + " | ".join(f"{n}: score {s:.1f} (conf {c:.1f})" for s, n, c, _ in results), file=sys.stderr)
    return max(results, key=lambda r: r[0])[3]  # ties keep the first (light)


_paddle_engine = None


def _ocr_paddle(img) -> str:
    global _paddle_engine
    import numpy as np

    if _paddle_engine is None:
        from paddleocr import PaddleOCR

        _paddle_engine = PaddleOCR(use_angle_cls=True, lang="en")
    arr = np.array(img.convert("RGB"))
    result = _paddle_engine.ocr(arr)
    lines: list[tuple[float, float, str]] = []
    # PaddleOCR 2.x: [[ [box, (text, conf)], ... ]]; 3.x: [ {"rec_texts": [...], "rec_polys": [...]} ]
    page = result[0] if result else None
    if isinstance(page, dict):
        for text, poly in zip(page.get("rec_texts", []), page.get("rec_polys", [])):
            lines.append((float(poly[:, 1].min()), float(poly[:, 0].min()), text))
    elif page:
        for box, (text, _conf) in page:
            lines.append((min(p[1] for p in box), min(p[0] for p in box), text))
    # Group boxes into visual rows (similar y), then order left-to-right.
    lines.sort()
    rows: list[list[tuple[float, float, str]]] = []
    for item in lines:
        if rows and abs(item[0] - rows[-1][0][0]) < 12:
            rows[-1].append(item)
        else:
            rows.append([item])
    return "\n".join("  ".join(t for _, _, t in sorted(row, key=lambda r: r[1])) for row in rows)


def _ocr_image(img, backend: str, preprocess: bool = True) -> str:
    if backend == "tesseract":
        return _ocr_tesseract(img, preprocess)
    if backend == "paddleocr":
        from PIL import ImageOps

        # PaddleOCR has its own detector and was not retuned: only fix the phone-photo rotation.
        return _ocr_paddle(ImageOps.exif_transpose(img) if preprocess else img)
    raise ValueError(f"Unknown OCR backend: {backend!r}")


# ----------------------------------- files --------------------------------- #
def _pdf_to_text(data: bytes, backend: str, preprocess: bool = True) -> str:
    import fitz  # PyMuPDF
    from PIL import Image

    parts: list[str] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text").strip()
            if len(text) >= MIN_EMBEDDED_TEXT_CHARS:
                parts.append(text)
                continue
            pix = page.get_pixmap(dpi=300)  # scanned page -> OCR
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            parts.append(_ocr_image(img, backend, preprocess))
    return "\n\n".join(parts)


def extract_text(data: bytes, filename: str, backend: str = "tesseract", preprocess: bool = True) -> str:
    """Extract text from uploaded file bytes (PDF or image)."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _pdf_to_text(data, backend, preprocess)
    if ext in IMAGE_EXTS:
        from PIL import Image

        return _ocr_image(Image.open(io.BytesIO(data)), backend, preprocess)
    raise ValueError(f"Unsupported file type: {ext}")


def extract_text_from_path(path: str | Path, backend: str = "tesseract", preprocess: bool = True) -> str:
    path = Path(path)
    return extract_text(path.read_bytes(), path.name, backend, preprocess)