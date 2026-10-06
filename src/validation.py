"""Validation layer: Pydantic schema check + business-rule consistency checks.

Severities: "error" (a required field is missing), "warning" (the numbers do not add up) and
"info" (worth knowing, not wrong: e.g. no ticket number printed, or tax already included in the
total). ``ValidationReport.ok`` ignores "info" issues.

Tax conventions accepted (see src/generator.py):
  * tax-exclusive:  subtotal + tax = total
  * tax-inclusive:  subtotal = total and tax is the VAT already contained in it (typical of
                    shop/restaurant tickets: "TOTAL 61.70 ... dont TVA 4.35"). To avoid accepting a
                    swapped tax/total pair, the included tax must be at most 30% of the total.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from .schema import Invoice
from .utils import extract_json


@dataclass
class Issue:
    code: str
    message: str
    severity: str = "warning"  # "info" | "warning" | "error"


@dataclass
class ValidationReport:
    invoice: Invoice | None
    schema_errors: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Parsed, and no warning or error ("info" notes do not count)."""
        return self.invoice is not None and all(i.severity == "info" for i in self.issues)


def check_consistency(inv: Invoice, tol: float = 0.02) -> list[Issue]:
    """Arithmetic and date sanity checks. ``tol`` is the allowed absolute rounding error."""
    issues: list[Issue] = []

    for idx, it in enumerate(inv.items, start=1):
        if it.unit_price is not None:
            expected = it.quantity * it.unit_price
            if abs(expected - it.total) > max(tol, 0.005 * abs(expected)):
                issues.append(Issue(
                    "item_total_mismatch",
                    f"Item {idx} ('{it.description}'): {it.quantity:g} x {it.unit_price:.2f} "
                    f"= {expected:.2f}, but total is {it.total:.2f}.",
                ))

    if inv.items and inv.subtotal is not None:
        items_sum = sum(i.total for i in inv.items)
        if abs(items_sum - inv.subtotal) > tol * max(1, len(inv.items)):
            issues.append(Issue(
                "subtotal_mismatch",
                f"Sum of items is {items_sum:.2f}, but subtotal is {inv.subtotal:.2f}.",
            ))

    if inv.subtotal is not None and inv.total is not None:
        tax = inv.tax or 0.0
        if abs(inv.subtotal + tax - inv.total) > tol:
            tax_included = (0 < tax <= 0.3 * abs(inv.total)) and abs(inv.subtotal - inv.total) <= tol
            if tax_included:
                issues.append(Issue(
                    "tax_included",
                    f"Tax ({tax:.2f}) is already included in the total ({inv.total:.2f}).",
                    "info",
                ))
            else:
                issues.append(Issue(
                    "total_mismatch",
                    f"subtotal ({inv.subtotal:.2f}) + tax ({tax:.2f}) = {inv.subtotal + tax:.2f}, "
                    f"but total is {inv.total:.2f}.",
                ))

    if inv.invoice_date and inv.due_date and inv.due_date < inv.invoice_date:
        issues.append(Issue("due_before_invoice", "Due date is earlier than the invoice date."))

    for name in ("vendor", "total"):
        if getattr(inv, name) in (None, ""):
            issues.append(Issue("missing_field", f"Missing required field: {name}.", "error"))
    if inv.invoice_number in (None, ""):  # many tickets print no number: a note, not an error
        issues.append(Issue("missing_number", "No invoice / ticket number found on the document.", "info"))

    return issues


def validate_invoice(raw: str | dict | None) -> ValidationReport:
    """Parse raw LLM output (string or dict) into an Invoice and run consistency checks."""
    data = extract_json(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        return ValidationReport(None, ["No valid JSON object found in model output."])
    try:
        inv = Invoice.model_validate(data)
    except ValidationError as e:
        return ValidationReport(None, [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()])
    return ValidationReport(inv, [], check_consistency(inv))