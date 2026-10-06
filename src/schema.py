"""Pydantic schema for the structured invoice JSON the LLM must produce."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

# Keys in the order the model is trained to emit them.
FIELD_ORDER = [
    "invoice_number",
    "vendor",
    "invoice_date",
    "due_date",
    "currency",
    "items",
    "subtotal",
    "tax",
    "total",
]


class LineItem(BaseModel):
    description: str
    quantity: float = 1
    unit_price: float | None = None
    total: float


class Invoice(BaseModel):
    invoice_number: str | None = None
    vendor: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. USD")
    items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None