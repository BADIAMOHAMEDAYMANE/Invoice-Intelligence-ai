"""Repair layer: fix a model's JSON using the arithmetic every invoice must satisfy.

A small LLM reads the layout well but is weak at arithmetic and at noisy OCR digits.
Invoices carry redundancy we can exploit, deterministically and without another model:

    quantity x unit_price = line total      sum(line totals) = subtotal
    subtotal + tax = total (tax-exclusive)  or  subtotal = total, VAT inside (tax-inclusive)

``repair_invoice(data, text)`` applies a short list of conservative rules, in this order:

  0. an "invoice number" that is only a postal code / phone / SIRET read from the header -> null,
  1. drop "items" that are really a table header or a tax/total line,
  2. cut the OCR numbers that leaked into a description ("Bizou 3.006 85 255.006" -> "Bizou"),
  3. un-swap quantity and unit price when the quantity is not a whole number,
  3b. restore a lost quantity cell: "14 x 1 = 14" (or "14 x 14 = 14") when the OCR line only
      holds "14 14",
  4. if the line totals do not add up to the subtotal and exactly ONE line can explain the
     gap, correct that line's total (and quantity); a subtotal printed nowhere is replaced by
     the sum of the (consistent) lines,
  5. rebuild a unit price from total / quantity when they disagree,
  6. read the VAT recap printed on the ticket ("TVA 19,60 0,52 / HT TVA 19,60 2,68", ...): when it
     is internally consistent and matches the lines, it decides tax and total (tax-exclusive or
     tax-inclusive),
  7. fix a tax / total pair that does not satisfy the rules above (a tax-inclusive ticket, where
     subtotal = total and the tax is at most 30% of it, is left alone),
  8. when no usable item is left on a one-article receipt, rescue it from the OCR text.

Rules only touch values that already break a rule or that are printed nowhere on the document, so
correct output goes through unchanged (checked on 8,000 generated documents). Every change is
described in the returned notes, so the UI can show what was repaired and why. This is NOT a
replacement for reading the receipt: the repair can only be as good as the numbers that are still
correct, and it was tuned on a handful of real tickets (see the README, "Limitations").
"""
from __future__ import annotations

import copy
import re

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_NUMERIC_TOKEN = re.compile(r"^[\d.,]+\W{0,2}$|^[€$£]?[\d.,]+[€$£]?$")
_HEADER_WORDS = {"produit", "designation", "désignation", "article", "libellé", "libelle", "description",
                 "item", "product", "qté", "qte", "qt", "q", "pu", "p.u.", "p.u", "prix", "price",
                 "montant", "amount", "total", "qty", "quantity", "unit", "each", "ext", "pt", "u."}
_TAX_LINE = re.compile(r"^\W*(?:dont\s+|incl\.?\s+|(?:ht|net)\s+)?(?:tva|iva|t\.v\.a|tax[e]?|vat|sales\s+tax|ta)\b.*(?:\(|%|\d)", re.I)
_TOTAL_LINE = re.compile(r"^\W*(?:sous[- ]?total|sub[- ]?total|total|net\s+[àa]\s+payer|reste\s+[àa]\s+payer|montant\s+[àa]\s+payer|amount\s+due|balance\s+due)\b.*\d", re.I)
_PAYMENT_LINE = re.compile(r"^\W*(?:esp[eè]ces|cash|carte|cb|rendu|change|paiement|payment|mode\s+de)\b", re.I)
# Header lines whose digits are never amounts: postal code, phone, fax, SIRET, dates, times.
_ADDRESS_LINE = re.compile(r"^\W*(?:t[eé]l|fax|siret|siren|rc\b|if\b|ice\b|n[°º]\s*tva|vat\s+no|tva\s+intra)", re.I)
_POSTAL_LINE = re.compile(r"^(?:\S\s+){0,2}\d{5}\s+[A-Za-zÀ-ÿ'’\- ]{3,}\W*$")
_PHONE = re.compile(r"\b\d{2}[. ]\d{2}[. ]\d{2}[. ]\d{2}\b")
_DATE_IN_LINE = re.compile(r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}")
_ID_LINE = re.compile(r"^\W*(?:siret|siren|rc\b|if\b|ice\b|t[eé]l|fax)", re.I)
_LABELLED_NUMBER = re.compile(
    r"(?:ticket|tick\w{0,3}|facture|re[cç]u|bon|commande|receipt|order|check|trans|n[°º])\W{0,3}(?:n[°º]|no\.?|#)?\W{0,3}(\d{3,8})\b", re.I)
_NOT_A_NUMBER_LINE = re.compile(r"^\W*(?:\S\s+)?(?:table|couverts?|serveur|caisse|caissier|guests?|register|server|cashier|client)\b", re.I)
_VATKW = r"(?:t[vy]a|iva|t\.v\.a\.?|vat|taxe?|mwst)"
_VAT_WORD = re.compile(rf"\b{_VATKW}\b", re.I)
_NET_BEFORE_VAT = re.compile(rf"\b(?:h[t1il]|net|base)\b[^0-9]*\b{_VATKW}\b", re.I)
_NOT_A_RATE_LINE = re.compile(r"n[°º]|\bno\b|intra|num[eé]ro|siren|siret|\bice\b", re.I)
_CURRENCY_HINTS = [("EUR", r"€|\beur\b"), ("MAD", r"\bdhs?\b|\bmad\b"), ("USD", r"\$|\busd\b"), ("GBP", r"£|\bgbp\b")]


def _f(x):
    """float or None (bools and non numbers are None)."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = _NUM.fullmatch(x.strip())
        return float(m.group().replace(",", ".")) if m else None
    return None


def _is_int(x: float, eps: float = 0.01) -> bool:
    return abs(x - round(x)) <= eps


def _text_numbers(text: str) -> list[float]:
    return [float(m.group().replace(",", ".")) for m in _NUM.finditer(text or "")]


def _has(values: list[float], x: float, tol: float = 0.005) -> bool:
    return any(abs(v - x) <= tol for v in values)


def _is_header(desc: str) -> bool:
    toks = [t.lower().strip("|:.") for t in desc.split() if t.strip("|=:.")]
    toks = [t for t in toks if t]
    hits = sum(t in _HEADER_WORDS or t + "." in _HEADER_WORDS for t in toks)
    return hits >= 2 and hits >= 0.5 * len(toks)


def _strip_numbers(desc: str) -> str:
    """'Bizou 3.006 85 255.006' -> 'Bizou' (only when 2+ numeric tokens trail the text)."""
    toks = desc.split()
    k = len(toks)
    while k > 0 and _NUMERIC_TOKEN.match(toks[k - 1]):
        k -= 1
    return " ".join(toks[:k]) if 0 < k <= len(toks) - 2 else desc


_GROUPED = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00a0.,]\d{3})+(?:[.,]\d{1,2})?(?![\d])")


def _grouped_values(line: str) -> list[float]:
    """Amounts written with thousands separators ("4 300.80", "1.234,50", "1,234.50") as one number."""
    vals = []
    for m in _GROUPED.finditer(line):
        raw = m.group().replace("\u00a0", " ")
        dec = re.search(r"[.,](\d{1,2})$", raw)
        digits = re.sub(r"[ .,]", "", raw[:dec.start()] if dec else raw)
        vals.append(float(digits + ("." + dec.group(1) if dec else "")))
    return vals


def _amount_numbers(text: str | None) -> list[float]:
    """Numbers that can be money amounts: everything except the header lines (postal code, phone,
    SIRET) and the lines holding a date or a time. Thousands-grouped amounts are added next to the
    plain tokens ("1 230,00" gives 1, 230.0 AND 1230.0): more candidates can only make the repair
    more cautious, never bolder."""
    out: list[float] = []
    for line in (text or "").splitlines():
        if _ADDRESS_LINE.match(line) or _POSTAL_LINE.match(line) or _PHONE.search(line) or _DATE_IN_LINE.search(line):
            continue
        out.extend(_text_numbers(line))
        out.extend(_grouped_values(line))
    return out


def _fix_invoice_number(out: dict, text: str | None, notes: list[str]) -> None:
    """A "ticket number" that only exists as a postal code / phone / SIRET in the header is not one:
    use the one labelled number of the text if there is exactly one, else null."""
    num = out.get("invoice_number")
    if not num or not text:
        return
    digits = re.sub(r"\s", "", str(num))
    if not digits.isdigit():
        return
    pattern = re.compile(rf"(?<!\d){re.escape(digits)}(?!\d)")
    as_header, elsewhere = False, False
    for line in text.splitlines():
        squeezed = re.sub(r"(?<=\d) (?=\d)", "", line)
        if not pattern.search(squeezed):
            continue
        if _POSTAL_LINE.match(line) or _ID_LINE.match(line) or _PHONE.search(line):
            as_header = True
        else:
            elsewhere = True
    if as_header and not elsewhere:
        found = {m.group(1) for line in text.splitlines() if not _NOT_A_NUMBER_LINE.match(line)
                 for m in _LABELLED_NUMBER.finditer(line)}
        found.discard(digits)
        new = found.pop() if len(found) == 1 else None
        out["invoice_number"] = new
        notes.append(f"Invoice number {num!r} is a postal code / phone / SIRET from the header, not a ticket number"
                     + (f"; the labelled number {new!r} was used instead." if new else "; set to none."))


def _vat_recap(text: str | None, subtotal: float) -> tuple[float, str] | None:
    """Read the VAT recap of a ticket and return (total VAT, "inclusive" | "exclusive") when it is
    consistent with itself and with the sum of the lines; None otherwise."""
    if not text:
        return None
    vats: dict[float, float] = {}
    nets: dict[float, float] = {}
    for line in text.splitlines():
        if not _VAT_WORD.search(line) or _NOT_A_RATE_LINE.search(line):
            continue
        nums = _text_numbers(line[_VAT_WORD.search(line).end():])  # only what follows the VAT keyword
        if len(nums) < 2 or not 0 < nums[0] <= 30:
            continue
        if len(_VAT_WORD.findall(line)) >= 2 and len(nums) >= 3 and re.search(r"\b(?:h[t1il]|net|base)\b", line, re.I):
            vats.setdefault(nums[0], nums[2])  # "TVA 7%  HT 54.67  TVA 3.83"
            nets.setdefault(nums[0], nums[1])
        elif _NET_BEFORE_VAT.search(line):
            nets.setdefault(nums[0], nums[1])  # "HT TVA 19.60 2.68"
        else:
            vats.setdefault(nums[0], nums[1])  # "TVA 19.60 0.52"
    if not vats:
        return None
    for rate, vat in vats.items():  # each base must really produce its VAT
        if rate in nets and abs(nets[rate] * rate / 100 - vat) > 0.02 + 0.01 * vat:
            return None
    total_vat = round(sum(vats.values()), 2)
    tol = 0.02 + 0.01 * len(vats)
    if set(nets) == set(vats):
        base = round(sum(nets.values()), 2)
        if abs(base + total_vat - subtotal) <= tol:
            return total_vat, "inclusive"
        if abs(base - subtotal) <= tol:
            return total_vat, "exclusive"
        return None
    if len(vats) == 1:
        (rate, vat), = vats.items()
        if abs(vat - subtotal * rate / (100 + rate)) <= 0.02:
            return total_vat, "inclusive"
        if abs(vat - subtotal * rate / 100) <= 0.02:
            return total_vat, "exclusive"
    return None


def _fix_from_recap(out: dict, items: list[dict], subtotal: float | None, amounts: list[float],
                    text: str | None, notes: list[str]) -> None:
    """Tax and total from the printed VAT recap, once the lines and the subtotal agree."""
    ts = [_f(i.get("total")) for i in items]
    if subtotal is None or not items or None in ts or abs(sum(ts) - subtotal) > 0.02 * max(1, len(items)):
        return
    rec = _vat_recap(text, subtotal)
    if rec is None:
        return
    vat, convention = rec
    tax, total = _f(out.get("tax")), _f(out.get("total"))
    new_total = subtotal if convention == "inclusive" else round(subtotal + vat, 2)
    if tax is None or abs(tax - vat) > 0.02:
        out["tax"] = vat
        notes.append(f"Tax {tax if tax is not None else 'missing'} -> {vat:g} (the VAT recap printed on the ticket, "
                     f"{'already inside the total' if convention == 'inclusive' else 'added to the subtotal'}).")
    if total is None or (abs(total - new_total) > 0.02 and not _has(amounts, total)):
        out["total"] = new_total
        notes.append(f"Total {total if total is not None else 'missing'} -> {new_total:g} "
                     f"({'sum of the lines' if convention == 'inclusive' else 'subtotal + VAT'}; the previous value is not printed on the ticket).")


def _clean_items(items: list, notes: list[str]) -> list[dict]:
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        it = dict(it)
        desc = str(it.get("description") or "").strip()
        if not desc or not re.search(r"[A-Za-zÀ-ÿ]{2}", desc):
            notes.append(f"Removed an item without a readable description ({desc!r}).")
            continue
        if _is_header(desc):
            notes.append(f"Removed {desc!r}: it is the table header, not an item.")
            continue
        if _TAX_LINE.match(desc) or _TOTAL_LINE.match(desc):
            notes.append(f"Removed {desc!r}: it is a tax/total line, not an item.")
            continue
        new = _strip_numbers(desc)
        if new != desc:
            notes.append(f"Description {desc!r} -> {new!r} (numbers belong to the other columns).")
            it["description"] = new
        out.append(it)
    return out


def _fix_line(it: dict, idx: int, notes: list[str]) -> None:
    """Fill / un-swap one line using q x u = t (no subtotal needed)."""
    q, u, t = _f(it.get("quantity")), _f(it.get("unit_price")), _f(it.get("total"))
    if q is None:
        it["quantity"] = q = 1.0
    if t is None and u is not None:
        it["total"] = t = round(q * u, 2)
        notes.append(f"Item {idx}: total was missing, set to {q:g} x {u:g} = {t:g}.")
    if t is None:
        return
    if u is None and q:
        it["unit_price"] = round(t / q, 2)
        notes.append(f"Item {idx}: unit price was missing, set to {t:g} / {q:g}.")
    elif (u is not None and not _is_int(q, 1e-3) and _is_int(u, 1e-3) and u >= 1
          and 0.02 < abs(q * u - t) <= 0.01 * abs(t)):
        # Not an exact product but within 1%: the signature of an OCR-damaged decimal ("3.00 EUR" read
        # as "3.006") sitting in the quantity column. A real fractional quantity (2.5 x 40 = 100)
        # multiplies out exactly and is left alone.
        it["quantity"], it["unit_price"] = u, q
        notes.append(f"Item {idx}: quantity {q:g} and unit price {u:g} were swapped.")


def _fix_lost_quantity(items: list[dict], text: str | None, notes: list[str]) -> None:
    """Restore a quantity cell the OCR lost (the model then invents "quantity = price, unit = 1").

    Symptom: ``quantity == total`` and ``unit_price`` either 1 (14 x 1 = 14, arithmetically fine, so no
    other rule can see it) or equal to the total as well (14 x 14 = 14: the model copied the same
    number twice; without this rule the unit price would later be "rebuilt" as 14 / 14 = 1).
    Evidence from the text: an OCR line that has a description and exactly TWO numbers, both equal to
    that total ("The Menthe 14 | 14"). A genuine "14 x 1 DH" line would show the "1" (three numbers),
    so it is left alone. We only act when the number of such lines equals the number of suspicious
    items, which keeps the match unambiguous.
    """
    if not text:
        return
    suspects: dict[float, list[dict]] = {}
    for it in items:
        q, u, t = _f(it.get("quantity")), _f(it.get("unit_price")), _f(it.get("total"))
        if None in (q, u, t) or t <= 1 or abs(q - t) > 0.005 or not (abs(u - 1) <= 1e-6 or abs(u - t) <= 0.005):
            continue
        suspects.setdefault(round(t, 2), []).append(it)
    if not suspects:
        return
    for t, group in suspects.items():
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or _TOTAL_LINE.match(line) or _PAYMENT_LINE.match(line) or _TAX_LINE.match(line):
                continue
            first = _NUM.search(line)
            if not first or not re.search(r"[A-Za-zÀ-ÿ]{2}", line[:first.start()]):
                continue
            nums = _text_numbers(line[first.start():])
            if len(nums) == 2 and all(abs(n - t) <= 0.005 for n in nums):
                lines.append(line)
        if len(lines) != len(group):
            continue
        for it, line in zip(group, lines):
            old_q, old_u = _f(it.get("quantity")), _f(it.get("unit_price"))
            it["quantity"], it["unit_price"] = 1.0, t
            notes.append(f"Quantity {old_q:g} x unit price {old_u:g} -> 1 x {t:g}: the text line {line!r} holds only "
                         f"two numbers ({t:g} twice), so the quantity cell was lost.")


def _consistent(it: dict, tol: float = 0.02) -> bool:
    q, u, t = _f(it.get("quantity")), _f(it.get("unit_price")), _f(it.get("total"))
    if q is None or u is None or t is None:
        return True
    return abs(q * u - t) <= max(tol, 0.005 * abs(q * u))


def _reconcile_totals(items: list[dict], subtotal: float | None, numbers: list[float],
                      notes: list[str]) -> float | None:
    """Make line totals add up to the subtotal; returns the (possibly updated) subtotal."""
    ts = [_f(i.get("total")) for i in items]
    if subtotal is None or not items or any(t is None for t in ts):
        return subtotal
    tol = 0.02 * max(1, len(items))
    total_sum = round(sum(ts), 2)
    if abs(total_sum - subtotal) <= tol:
        return subtotal
    if numbers and _has(numbers, total_sum):  # the lines are right, the subtotal was misread
        notes.append(f"Subtotal {subtotal:g} -> {total_sum:g} (sum of the lines, found in the text).")
        return total_sum
    fixes = []
    for i, it in enumerate(items):
        new_t = round(subtotal - (total_sum - ts[i]), 2)
        u = _f(it.get("unit_price"))
        if new_t <= 0:
            continue
        if _consistent({**it, "total": new_t}) and u:
            fixes.append((i, new_t, _f(it.get("quantity")) or 1.0))
        elif u and u > 0 and _is_int(new_t / u) and round(new_t / u) >= 1:
            fixes.append((i, new_t, float(round(new_t / u))))
    if len(fixes) == 1:
        i, new_t, new_q = fixes[0]
        notes.append(f"Item {i + 1}: total {ts[i]:g} -> {new_t:g} (lines must add up to the subtotal {subtotal:g})"
                     + (f", quantity -> {new_q:g}." if new_q != _f(items[i].get('quantity')) else "."))
        items[i]["total"], items[i]["quantity"] = new_t, new_q
        return subtotal
    if numbers and not _has(numbers, subtotal) and all(_consistent(i) for i in items):
        # the lines are coherent and the subtotal is printed nowhere (e.g. copied from a postal code)
        notes.append(f"Subtotal {subtotal:g} -> {total_sum:g} (sum of the lines; {subtotal:g} is not printed as an amount).")
        return total_sum
    return subtotal


def _fix_units(items: list[dict], subtotal_ok: bool, notes: list[str]) -> None:
    """Rebuild unit prices from total / quantity once the totals are trusted."""
    if not subtotal_ok:
        return
    for idx, it in enumerate(items, start=1):
        q, u, t = _f(it.get("quantity")), _f(it.get("unit_price")), _f(it.get("total"))
        if None in (q, u, t) or q <= 0 or not _is_int(q, 1e-6):
            continue
        exact = round(t / q, 2)
        if abs(u - t / q) > 0.005 / q + 1e-6:
            notes.append(f"Item {idx}: unit price {u:g} -> {exact:g} ({t:g} / {q:g}).")
            it["unit_price"] = exact


def _fix_tax_total(data: dict, subtotal, numbers: list[float], notes: list[str], trusted: bool = False) -> None:
    tax, total = _f(data.get("tax")), _f(data.get("total"))
    if subtotal is None or total is None:
        return
    if abs(subtotal + (tax or 0) - total) <= 0.02:
        return
    if trusted and abs(total - subtotal) > 0.02 and not _has(numbers, total):
        # lines and subtotal agree, but this total is printed nowhere (e.g. the big bold total the OCR lost)
        data["total"] = subtotal
        notes.append(f"Total {total:g} -> {subtotal:g} (sum of the lines; {total:g} is not printed as an amount).")
        return
    if abs(total - subtotal) <= 0.02 and tax is not None and 0 < tax <= 0.3 * abs(total):
        return  # tax-inclusive ticket: the VAT is already inside the total
    if numbers:  # a (tax, total) pair that exists in the text and satisfies the equation
        given = {x for x in (tax, total) if x is not None}
        pairs = {(a, b) for a in numbers for b in numbers
                 if 0 < a <= 0.3 * subtotal and abs(subtotal + a - b) <= 0.02 and (b in given or a in given)}
        if len(pairs) == 1:
            a, b = next(iter(pairs))
            notes.append(f"Tax/total {tax}/{total} -> {a:g}/{b:g} (subtotal {subtotal:g} + tax = total, both found in the text).")
            data["tax"], data["total"] = a, b
            return
    if tax is not None and abs(total - subtotal) <= 0.02 and tax > subtotal and _has(numbers, tax):
        # "tax" is really the total incl. tax (and that amount is printed on the document)
        new_total, new_tax = tax, round(tax - subtotal, 2)
        notes.append(f"Tax/total {tax:g}/{total:g} -> {new_tax:g}/{new_total:g} (the 'tax' was the total incl. tax).")
        data["tax"], data["total"] = new_tax, new_total
    elif total > subtotal and (tax is None or tax <= 0):
        data["tax"] = round(total - subtotal, 2)
        notes.append(f"Tax set to {data['tax']:g} (total {total:g} - subtotal {subtotal:g}).")
    elif total > subtotal:
        data["tax"] = round(total - subtotal, 2)
        notes.append(f"Tax {tax:g} -> {data['tax']:g} (total {total:g} - subtotal {subtotal:g}).")


def _rescue_single_item(text: str, target: float | None, notes: list[str]) -> dict | None:
    """One-article receipt whose item was lost: rebuild it from the OCR line holding the total."""
    if not text or target is None:
        return None
    for raw in text.splitlines():
        toks = raw.split()
        while toks and (len(toks[0]) == 1 and not toks[0].isdigit()):  # junk prefix: "a", "|", "="
            toks = toks[1:]
        line = " ".join(toks)
        if not line or _TOTAL_LINE.match(line) or _PAYMENT_LINE.match(line) or _TAX_LINE.match(line):
            continue
        first_num = next((i for i, t in enumerate(toks) if _NUM.search(t)), None)
        nums = _text_numbers(" ".join(toks[first_num:])) if first_num is not None else []
        desc = " ".join(toks[:first_num]).strip(" |=:-") if first_num is not None else ""
        if len(desc) < 3 or len(nums) < 1 or len(nums) > 3 or abs(nums[-1] - target) > 0.005 or _is_header(desc):
            continue
        t = nums[-1]
        if len(nums) == 3 and _is_int(nums[1]) and abs(nums[0] * nums[1] - t) <= 0.02:
            q, u = nums[1], nums[0]
        elif len(nums) == 3 and _is_int(nums[0]) and abs(nums[0] * nums[1] - t) <= 0.02:
            q, u = nums[0], nums[1]
        else:
            q, u = 1.0, t
        notes.append(f"Item rebuilt from the text line {line!r}.")
        return {"description": desc, "quantity": q, "unit_price": u, "total": t}
    return None


def repair_invoice(data: dict, text: str | None = None) -> tuple[dict, list[str]]:
    """Return (repaired copy of ``data``, human-readable notes). ``text`` (the OCR text) is optional
    but lets the repair check its candidates against numbers that really appear on the document."""
    notes: list[str] = []
    if not isinstance(data, dict):
        return data, notes
    out = copy.deepcopy(data)
    numbers = _amount_numbers(text)

    _fix_invoice_number(out, text, notes)
    items = _clean_items(out.get("items") or [], notes)
    for idx, it in enumerate(items, start=1):
        _fix_line(it, idx, notes)
    _fix_lost_quantity(items, text, notes)

    subtotal = _f(out.get("subtotal"))
    subtotal = _reconcile_totals(items, subtotal, numbers, notes)
    ts = [_f(i.get("total")) for i in items]
    totals_ok = subtotal is not None and bool(items) and None not in ts \
        and abs(sum(ts) - subtotal) <= 0.02 * max(1, len(items))
    _fix_units(items, totals_ok, notes)

    if not items and (out.get("items") or text):
        rescued = _rescue_single_item(text or "", subtotal if subtotal is not None else _f(out.get("total")), notes)
        if rescued:
            items = [rescued]
    if subtotal is None and items and all(_f(i.get("total")) is not None for i in items) and _f(out.get("total")) is not None:
        subtotal = round(sum(_f(i["total"]) for i in items), 2)
        notes.append(f"Subtotal was missing, set to the sum of the lines ({subtotal:g}).")

    out["items"] = items
    if subtotal is not None:
        out["subtotal"] = subtotal
    _fix_from_recap(out, items, subtotal, numbers, text, notes)
    trusted = totals_ok and all(_consistent(i) for i in items)
    _fix_tax_total(out, subtotal, numbers, notes, trusted)

    if not out.get("currency") and text:
        for code, pattern in _CURRENCY_HINTS:
            if re.search(pattern, text, re.I):
                out["currency"] = code
                notes.append(f"Currency was missing, read {code} from the text.")
                break
    return out, notes