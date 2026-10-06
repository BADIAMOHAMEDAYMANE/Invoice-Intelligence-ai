"""Synthetic invoice generator.

Produces pairs of (OCR-like invoice text, gold structured JSON). Because the text is
rendered *from* the gold data, labels are exact. Variety comes from several layouts,
two languages (EN/FR), different currency/number formats, date formats, label wording
and light OCR-style noise.

Two document families are generated:
  * "invoice"  - office-style invoices (tables, "Invoice #", due dates, ...)
  * "receipt"  - POS tickets (cafe / shop): "Ticket N°" (or no number at all), server / table /
                 SIRET lines, several line layouts ("1 ENTRECOTE 25.00 25.00", PU/Qté/PT columns, ...),
                 DH/€ amounts, tax-exclusive or tax-inclusive totals (one or two VAT rates), textual
                 dates ("SAM 24 AOU 2013"), payment lines, and OCR-style noise (garbled labels and
                 product names, junk lines, stray symbols, lost accents, lost quantity cell).

Design decisions (document these in your README):
  * One date format per document. Numeric dates are DD/MM/YYYY, DD.MM.YYYY or ISO; the US
    MM/DD/YYYY format is only used when every date in the document has day > 12, so the
    text is never ambiguous.
  * All vendor/client names are invented combinations, not real companies.
  * OCR noise never changes a digit or a vendor name. Product names may be lightly garbled in the
    text while the gold keeps the clean name (the model learns to read through small OCR errors).
  * Tax conventions on receipts: no tax printed -> tax null and subtotal = total. Tax-EXCLUSIVE
    -> subtotal + tax = total. Tax-INCLUSIVE (prices already contain the VAT, the ticket only says
    how much of the total is VAT, maybe split per rate) -> subtotal = sum of the lines = total and
    tax = the VAT amount printed, so subtotal + tax != total on purpose.
  * A receipt that prints no ticket number has invoice_number = null (the model must not invent
    one from a postal code or a SIRET).
"""
from __future__ import annotations

import random
import re
import unicodedata
from datetime import date, timedelta

from .schema import FIELD_ORDER

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
_V_PREFIX = ["Acme", "Nova", "Atlas", "Zenith", "Orbit", "Cedar", "Blue Harbor", "Sahara",
             "Pioneer", "Lumen", "Vertex", "Northwind", "Maple", "Atlantic", "Kasbah",
             "Quantum", "Evergreen", "Summit", "Delta", "Horizon"]
_V_MIDDLE = ["Software", "Consulting", "Logistics", "Labs", "Digital", "Supplies",
             "Systems", "Media", "Energy", "Cloud", "Design", "Trading", "Services"]
_V_SUFFIX = ["Inc.", "LLC", "Ltd", "SARL", "GmbH", "Co.", "SA", ""]
_CLIENTS = ["Maria Lopez", "Karim Benali", "Sophie Martin", "John Carter", "Leila Haddad",
            "Omar Idrissi", "Emma Wilson", "Lucas Dubois", "Fatima Zahra", "Noah Johnson",
            "Studio Pixel SARL", "Dunmore Retail Ltd", "Greenfield Clinic", "Tangier Imports SA"]
_STREETS = ["12 Main Street", "48 Rue Victor Hugo", "221 Market Avenue", "7 Boulevard Zerktouni",
            "90 Oak Lane", "5 Avenue Mohammed V", "310 Harbor Road", "18 Park Drive"]
_CITIES = ["Casablanca", "Paris", "London", "New York", "Berlin", "Rabat", "Madrid",
           "Lyon", "Austin", "Marrakech"]

_ITEMS = [
    "Python Course", "Cloud Hosting", "Web Design", "Domain Renewal", "Technical Support",
    "Monthly Subscription", "Data Analysis Report", "Logo Design", "SEO Optimization",
    "Office Chair", "Laptop Stand", "USB-C Cable", "Wireless Mouse", "Printer Paper (A4)",
    "Consulting Hour", "API Integration", "Server Maintenance", "Training Workshop",
    "Mobile App Development", "Email Marketing Campaign", "Security Audit", "Cloud Storage 1TB",
    "Translation Service", "Photography Session", "Catering Service", "Delivery Fee",
    "Software License", "Database Backup", "Video Editing", "Standing Desk",
]
_ITEMS_FR = [
    "Formation Python", "Hébergement Cloud", "Création de site web", "Renouvellement de domaine",
    "Assistance technique", "Abonnement mensuel", "Rapport d'analyse", "Conception de logo",
    "Heure de conseil", "Maintenance serveur", "Atelier de formation", "Audit de sécurité",
    "Service de traduction", "Frais de livraison", "Licence logicielle",
]

# currency -> (symbol shown in text, position, number style)
_CURRENCIES = {
    "USD": [("$", "prefix", "us")],
    "GBP": [("£", "prefix", "us")],
    "EUR": [("€", "suffix", "eu"), ("€", "prefix", "us"), ("EUR", "suffix", "eu")],
    "MAD": [("MAD", "suffix", "us"), ("DH", "suffix", "eu")],
}

_MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
_MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
              "septembre", "octobre", "novembre", "décembre"]

_LABELS = {
    "en": {
        "title": ["INVOICE", "Invoice", "TAX INVOICE"],
        "number": ["Invoice No", "Invoice #", "Invoice Number", "Inv No."],
        "date": ["Date", "Invoice Date", "Issued"],
        "due": ["Due Date", "Payment Due", "Due"],
        "bill_to": ["Bill To", "Billed to", "Customer"],
        "desc": ["Description", "Item"],
        "qty": ["Qty", "Quantity"],
        "unit": ["Unit Price", "Price", "Rate"],
        "amount": ["Amount", "Total", "Line Total"],
        "subtotal": ["Subtotal", "Sub-total", "Sub Total"],
        "tax": ["Tax", "VAT", "Sales Tax"],
        "total": ["Total", "Total Due", "Amount Due", "Grand Total"],
    },
    "fr": {
        "title": ["FACTURE", "Facture"],
        "number": ["Facture N°", "N° de facture", "Numéro de facture", "Facture No"],
        "date": ["Date", "Date de facture", "Émise le"],
        "due": ["Échéance", "Date d'échéance", "À payer avant le"],
        "bill_to": ["Facturé à", "Client", "Destinataire"],
        "desc": ["Désignation", "Description"],
        "qty": ["Qté", "Quantité"],
        "unit": ["Prix unitaire", "P.U.", "Prix"],
        "amount": ["Montant", "Total", "Total ligne"],
        "subtotal": ["Sous-total", "Total HT", "Montant HT"],
        "tax": ["TVA", "Taxe"],
        "total": ["Total TTC", "Total à payer", "Total", "Net à payer"],
    },
}


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _format_number(x: float, style: str, rng: random.Random, decimals: int = 2) -> str:
    s = f"{x:,.{decimals}f}"  # 1,234.50
    if style == "eu":
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        if rng.random() < 0.3:  # thin-space thousands, common in FR
            s = s.replace(".", " ")
    return s


def _format_money(x: float, symbol: str, position: str, style: str, rng: random.Random) -> str:
    num = _format_number(x, style, rng)
    if position == "prefix":
        return f"{symbol}{num}" if len(symbol) == 1 else f"{symbol} {num}"
    return f"{num} {symbol}"


def _pick_date_kind(dates: list[date], rng: random.Random) -> str:
    """Choose ONE date format per document. MM/DD/YYYY is only allowed when every date
    in the document has day > 12, so the text is never ambiguous."""
    options = ["iso", "dmy_slash", "dmy_dot", "long"]
    if all(d.day > 12 for d in dates):
        options.append("mdy_slash")
    return rng.choice(options)


def _format_date(d: date, lang: str, kind: str) -> str:
    if kind == "iso":
        return d.isoformat()
    if kind == "dmy_slash":
        return d.strftime("%d/%m/%Y")
    if kind == "dmy_dot":
        return d.strftime("%d.%m.%Y")
    if kind == "mdy_slash":
        return d.strftime("%m/%d/%Y")
    if lang == "fr":
        return f"{d.day} {_MONTHS_FR[d.month - 1]} {d.year}"
    return f"{_MONTHS_EN[d.month - 1]} {d.day}, {d.year}"


def _invoice_number(rng: random.Random, d: date) -> str:
    kind = rng.randint(0, 5)
    n = rng.randint(1, 9999)
    if kind == 0:
        return f"INV-{d.year}-{n:04d}"
    if kind == 1:
        return f"F{d.year}/{n:04d}"
    if kind == 2:
        return f"INV/{d.month:02d}/{d.year}/{n:03d}"
    if kind == 3:
        return f"{rng.randint(10000, 999999)}"
    if kind == 4:
        return f"{rng.choice(['AB', 'CX', 'FR', 'MA', 'US'])}-{n:05d}"
    return f"{n:05d}"


def _vendor(rng: random.Random) -> str:
    name = f"{rng.choice(_V_PREFIX)} {rng.choice(_V_MIDDLE)}"
    suffix = rng.choice(_V_SUFFIX)
    return f"{name} {suffix}".strip()


# --------------------------------------------------------------------------- #
# Gold data
# --------------------------------------------------------------------------- #
def generate_gold(rng: random.Random, lang: str) -> dict:
    """Create one invoice as the target JSON (ISO dates, floats rounded to 2dp)."""
    invoice_date = date(2024, 1, 1) + timedelta(days=rng.randint(0, 3 * 365 - 1))
    has_due = rng.random() > 0.15
    due_date = invoice_date + timedelta(days=rng.choice([7, 14, 15, 30, 45, 60])) if has_due else None

    currency = rng.choice(list(_CURRENCIES))
    if lang == "fr":
        currency = rng.choice(["EUR", "EUR", "MAD", "USD"])

    pool = _ITEMS_FR if lang == "fr" else _ITEMS
    n_items = rng.choices([1, 2, 3, 4, 5, 6], weights=[2, 5, 5, 3, 2, 1])[0]
    items = []
    for desc in rng.sample(pool, n_items):
        qty = rng.choices([1, 2, 3, 5, 10], weights=[6, 3, 2, 1, 1])[0]
        unit = round(rng.choice([rng.uniform(5, 60), rng.uniform(60, 400), rng.uniform(400, 3000)]), 2)
        if rng.random() < 0.4:
            unit = float(round(unit))
        items.append({
            "description": desc,
            "quantity": qty,
            "unit_price": unit,
            "total": round(qty * unit, 2),
        })

    subtotal = round(sum(i["total"] for i in items), 2)
    rate = rng.choice([0, 5, 7, 10, 19, 20, 20])
    tax = round(subtotal * rate / 100, 2)
    gold = {
        "invoice_number": _invoice_number(rng, invoice_date),
        "vendor": _vendor(rng),
        "invoice_date": invoice_date.isoformat(),
        "due_date": due_date.isoformat() if due_date else None,
        "currency": currency,
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": round(subtotal + tax, 2),
    }
    assert list(gold) == FIELD_ORDER
    gold["_tax_rate"] = rate  # private render hint, stripped before returning from generate_example
    return gold


# --------------------------------------------------------------------------- #
# Rendering gold data into OCR-like text
# --------------------------------------------------------------------------- #
def _add_noise(text: str, rng: random.Random) -> str:
    """Light layout noise (extra spaces / blank lines). Never touches digits or letters."""
    lines = []
    for line in text.split("\n"):
        if rng.random() < 0.15:
            line = re.sub(r" {2,}", lambda m: " " * rng.randint(1, len(m.group()) + 2), line)
        if rng.random() < 0.05:
            lines.append("")
        lines.append(line.rstrip() if rng.random() < 0.5 else line)
    return "\n".join(lines)


def render_text(gold: dict, rng: random.Random, lang: str) -> str:
    L = {k: rng.choice(v) for k, v in _LABELS[lang].items()}
    symbol, position, style = rng.choice(_CURRENCIES[gold["currency"]])
    money = lambda x: _format_money(x, symbol, position, style, rng)  # noqa: E731
    num = lambda x: _format_number(x, style, rng)  # noqa: E731

    d_inv = date.fromisoformat(gold["invoice_date"])
    d_due = date.fromisoformat(gold["due_date"]) if gold["due_date"] else None
    date_kind = _pick_date_kind([d for d in (d_inv, d_due) if d], rng)
    inv_date = _format_date(d_inv, lang, date_kind)
    due = _format_date(d_due, lang, date_kind) if d_due else None

    vendor = gold["vendor"]
    if rng.random() < 0.4:
        vendor = vendor.upper()

    all_qty_one = all(i["quantity"] == 1 for i in gold["items"])
    layout = rng.choice(["table", "inline"] + (["simple"] if all_qty_one else []))

    sep = rng.choice([": ", ": ", ": ", " ", " - ", "\t", " : "])  # label/value separators

    out: list[str] = []
    if rng.random() < 0.25:  # banner noise before the vendor name
        out.append(rng.choice(["Page 1 of 1", "www.example.com", "Tel: +212 522 000 000", "ORIGINAL"]))
    out.append(vendor)
    if rng.random() < 0.7:
        out.append(f"{rng.choice(_STREETS)}, {rng.choice(_CITIES)}")
    if rng.random() < 0.5:
        out.append(L["title"])
    out.append("")

    header = [f"{L['number']}{sep}{gold['invoice_number']}", f"{L['date']}{sep}{inv_date}"]
    if due:
        header.append(f"{L['due']}{sep}{due}")
    if rng.random() < 0.6:
        header.append(f"{L['bill_to']}{sep}{rng.choice(_CLIENTS)}")
    if rng.random() < 0.35:
        rng.shuffle(header)
    out.extend(header)
    out.append("")

    if layout == "table":
        out.append(f"{L['desc']:<28}{L['qty']:>5}{L['unit']:>16}{L['amount']:>16}")
        for it in gold["items"]:
            out.append(f"{it['description']:<28}{it['quantity']:>5}{money(it['unit_price']):>16}{money(it['total']):>16}")
    elif layout == "inline":
        for it in gold["items"]:
            out.append(f"{it['description']} - {it['quantity']} x {money(it['unit_price'])} = {money(it['total'])}")
    else:  # simple: "Python Course        $120.00"
        for it in gold["items"]:
            out.append(f"{it['description']:<28}{money(it['total']):>14}")

    out.append("")
    out.append(f"{L['subtotal']}{sep}{money(gold['subtotal'])}")
    rate = gold["_tax_rate"]
    tax_label = f"{L['tax']} ({rate}%)" if rng.random() < 0.6 else L["tax"]
    out.append(f"{tax_label}{sep}{money(gold['tax'])}")
    out.append(f"{L['total']}{sep}{money(gold['total'])}")

    if rng.random() < 0.35:  # footer distractors containing digits
        out.append("")
        out.append(rng.choice([
            "Thank you for your business!",
            "Merci pour votre confiance.",
            "Payment by bank transfer. IBAN: FR76 3000 6000 0112 3456 7890 189",
            "Late payments incur a 5% monthly fee. Ref 2024-0001",
        ]))

    return _add_noise("\n".join(out), rng)


# --------------------------------------------------------------------------- #
# Receipts (tickets de caisse): POS layouts + OCR-style noise
# --------------------------------------------------------------------------- #
_R_KIND = ["Cafe", "Snack", "Restaurant", "Pizzeria", "Boulangerie", "Pharmacie", "Epicerie",
           "Librairie", "Boutique", "Superette", "Brasserie", "Salon de the", "Fast Food", "Kiosque"]
_R_NAME = ["Oasis", "Medina", "Palmier", "Soleil", "Riviera", "Zahra", "Lumiere", "Jasmin", "Marina",
           "Andalous", "Bahia", "Olivier", "Printemps", "Horizon", "Safran", "Mimosa", "Corniche",
           "Kenza", "Rialto", "Sirocco"]
_R_BRAND = ["Pizza", "Burger", "Market", "Store", "Shop", "Cafe", "Bakery", "Kebab", "Grill"]

_R_ITEMS_FR = [
    "Thé à la menthe", "Café noir", "Café crème", "Jus d'orange", "Croissant", "Pain au chocolat",
    "Eau minérale 50cl", "Pizza Margherita", "Sandwich poulet", "Tacos viande", "Burger maison",
    "Coca-Cola 33cl", "Salade césar", "Tajine poulet", "Msemen", "Harcha", "Espresso", "Cappuccino",
    "Baguette", "Gâteau chocolat", "Glace vanille", "Omelette", "Chicha", "Lait", "Yaourt nature",
    "Savon", "Cahier A4", "Stylo bleu", "Paracétamol 500mg", "Brosse à dents", "Crêpe sucrée",
    "Bizou", "Calin", "Boudin",
    "Entrecôte", "Magret de canard", "Assiette de fromage", "Vin du mois", "Plat du jour",
    "Menu enfant", "Salade niçoise", "Tarte tatin", "Crème brûlée", "Verre de vin rouge",
    "Demi pression", "Steak frites", "Moules frites", "Café gourmand", "Jambon beurre",
]
_R_ITEMS_EN = [
    "Latte", "Espresso", "Green tea", "Bagel", "Club sandwich", "Cheeseburger", "Fries",
    "Orange juice", "Mineral water", "Caesar salad", "Chocolate cake", "Notebook A5",
    "Ballpoint pen", "Phone charger", "Aspirin 500mg", "Toothbrush", "Muffin", "Cappuccino",
    "Pancakes", "Iced tea", "Grilled salmon", "House wine", "Draft beer", "Steak and fries",
    "Apple pie", "Daily special",
]
_R_STAFF = ["Youssef", "Hamza", "Salma", "Nadia", "Karim", "Sofia", "Amine", "Imane", "Julien",
            "Camille", "Thomas", "Lea", "Anas", "Meryem"]

_R_NUMBER_LABELS = {
    "fr": ["Ticket N°", "Ticket NO", "TICKET No", "Ticket :", "N° Ticket", "Reçu N°", "Facture N°",
           "Bon N°", "Commande N°", "Ticket"],
    "en": ["Receipt #", "Receipt No", "Ticket #", "Order #", "Check #", "Trans #", "Receipt"],
}
_R_COLS = {
    "fr": {"desc": ["Produit", "Désignation", "Article", "Libellé"],
           "unit": ["PU", "P.U.", "Prix U.", "P.Unit"],
           "qty": ["Qté", "Q", "Qte", "Qt"],
           "total": ["PT", "Montant", "Total", "Prix"]},
    "en": {"desc": ["Item", "Description", "Product"],
           "unit": ["Price", "Unit", "Each"],
           "qty": ["Qty", "Q", "Quantity"],
           "total": ["Amount", "Total", "Ext"]},
}
_R_TOTALS = {
    "fr": {"sub": ["Total HT", "Sous-total", "Montant HT"], "tax": ["TVA", "Taxe"],
           "tot_tax": ["Total TTC", "TOTAL TTC", "Net à payer", "Total à payer"],
           "tot": ["Total", "TOTAL", "Total à payer", "Montant à payer", "Net à payer", "Reste à payer"]},
    "en": {"sub": ["Subtotal", "Sub-total"], "tax": ["Tax", "VAT", "Sales Tax"],
           "tot_tax": ["Total", "TOTAL", "Amount Due", "Total Due"],
           "tot": ["Total", "TOTAL", "Amount Due", "Total Due", "Balance Due"]},
}
_R_FOOTERS = {
    "fr": ["Merci pour votre visite", "MERCI POUR VOTRE VISITE", "A BIENTOT", "Merci et à bientôt",
           "SIREN: 700 500 600", "ICE: 001234567000012", "N° TVA: FR12 345678901",
           "RC 12345 - IF 6789", "|||| 3 760123 456789 ||||", "Tel: 05 24 00 00 00",
           "Les articles ne sont ni repris ni échangés"],
    "en": ["Thank you!", "THANK YOU FOR YOUR VISIT", "See you soon", "Tel: 555 010 2030",
           "VAT No: GB123456789", "|||| 3 760123 456789 ||||", "No refunds without receipt"],
}
_R_JUNK_TOKENS = ["ee", "te", "a", "oe", "RS", "fi", "es", "ae", "RE", "Mee", "ie", "SSS", "iin",
                  "ete", "eT", "ese", "\\", "|", "=", "~", "—"]
_R_LINE_PREFIX = ["|", "=", "a", "A", "i", "!", "~", "-", "©", "\\", "{", "<"]
_R_LINE_SUFFIX = ["|", "À", "i", "{", "}", ".", "*", "\\"]
_CONFUSE = {"e": "c", "c": "e", "i": "l", "l": "i", "o": "0", "u": "v", "n": "m", "m": "n",
            "h": "b", "a": "o", "r": "n", "t": "f", "T": "I", "O": "0", "C": "G"}

_R_RATES = {"EUR": [5.5, 10, 20, 7, 19.6], "MAD": [20, 10, 14, 7], "USD": [5, 7, 8.25, 10], "GBP": [20, 5]}
_R_SYMBOLS = {
    "EUR": [("€", "tight"), ("€", "space"), ("€", "prefix"), ("EUR", "space")],
    "USD": [("$", "prefix"), ("$", "prefix"), ("USD", "space")],
    "GBP": [("£", "prefix"), ("£", "prefix"), ("GBP", "space")],
    "MAD": [("DH", "space"), ("DH", "space"), ("MAD", "space"), ("DHS", "space"), ("Dhs", "space")],
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _receipt_vendor(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.55:
        return f"{rng.choice(_R_KIND)} {rng.choice(_R_NAME)}"
    if r < 0.85:
        return f"{rng.choice(_R_NAME)} {rng.choice(_R_BRAND)}"
    return rng.choice(_R_NAME)


def _receipt_price(rng: random.Random, currency: str) -> float:
    if currency == "MAD":
        if rng.random() < 0.7:
            return float(rng.choice([5, 6, 7, 8, 10, 12, 14, 15, 18, 20, 25, 30, 35, 45, 60]))
        return round(rng.uniform(3, 150) * 2) / 2
    if rng.random() < 0.08:
        return round(rng.uniform(40, 400), 2)
    if rng.random() < 0.4:
        return rng.randint(1, 30) + rng.choice([0.0, 0.5, 0.9, 0.95])
    return round(rng.uniform(0.8, 45), 2)


def _receipt_formatters(rng: random.Random, currency: str):
    """Return (num, money): amount formatters sharing one style for the whole receipt."""
    dec = rng.choice([",", "."])
    int_if_whole = rng.random() < 0.35
    thousands = rng.choice(["", " ", "."] if dec == "," else ["", " ", ","])
    sym, pos = rng.choice(_R_SYMBOLS[currency])

    def num(x: float, force2: bool = False) -> str:
        s = str(int(round(x))) if (int_if_whole and not force2 and abs(x - round(x)) < 1e-9) else f"{x:.2f}"
        ip, _, fp = s.partition(".")
        if thousands and len(ip) > 3:
            ip = f"{int(ip):,}".replace(",", thousands)
        return ip + (dec + fp if fp else "")

    def money(x: float, force2: bool = False) -> str:
        n = num(x, force2)
        if pos == "tight":
            return f"{n}{sym}"
        if pos == "space":
            return f"{n} {sym}"
        return f"{sym}{n}" if len(sym) == 1 else f"{sym} {n}"

    return num, money


def _receipt_date(d: date, kind: str) -> str:
    return {"dmy_dash": d.strftime("%d-%m-%Y"), "dmy_slash": d.strftime("%d/%m/%Y"),
            "dmy_dot": d.strftime("%d.%m.%Y"), "dmy_short": d.strftime("%d/%m/%y"),
            "iso": d.isoformat(), "mdy_slash": d.strftime("%m/%d/%Y")}[kind]


_FR_DAYS = ["LUN", "MAR", "MER", "JEU", "VEN", "SAM", "DIM"]
_FR_MONTHS = ["JAN", "FEV", "MARS", "AVR", "MAI", "JUIN", "JUIL", "AOU", "SEPT", "OCT", "NOV", "DEC"]
_EN_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_EN_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _text_date(d: date, lang: str, rng: random.Random) -> str:
    """Textual dates as printed by POS machines: "SAM 24 AOU 2013", "Sat Aug 24, 2013"."""
    if lang == "fr":
        day, mon = _FR_DAYS[d.weekday()], _FR_MONTHS[d.month - 1]
        s = rng.choice([f"{day} {d.day:02d} {mon} {d.year}", f"{d.day:02d} {mon} {d.year}",
                        f"{day} {d.day} {mon} {d.year}"])
        return s if rng.random() < 0.8 else s.title()
    day, mon = _EN_DAYS[d.weekday()], _EN_MONTHS[d.month - 1]
    return rng.choice([f"{day} {mon} {d.day:02d}, {d.year}", f"{d.day:02d} {mon} {d.year}",
                       f"{mon} {d.day:02d}, {d.year}"])


def _garble_word(w: str, rng: random.Random) -> str:
    i, op = rng.randrange(len(w)), rng.random()
    if op < 0.5 and w[i] in _CONFUSE:
        return w[:i] + _CONFUSE[w[i]] + w[i + 1:]
    if op < 0.8 and len(w) >= 4:
        return w[:i] + w[i + 1:]  # dropped character
    return w


def _garble_line(line: str, rng: random.Random, p_word: float) -> str:
    def fix(m: re.Match) -> str:
        w = m.group()
        return _garble_word(w, rng) if len(w) >= 3 and rng.random() < p_word else w
    return re.sub(r"[A-Za-zÀ-ÿ]+", fix, line)


def _junk_line(rng: random.Random) -> str:
    if rng.random() < 0.4:
        return "".join(rng.choice("=-_.~*:|") for _ in range(rng.randint(6, 30)))
    return " ".join(rng.choice(_R_JUNK_TOKENS) for _ in range(rng.randint(2, 6)))


def generate_receipt(rng: random.Random) -> tuple[dict, str]:
    """One POS receipt: (gold JSON with private hints removed, OCR-like text)."""
    lang = rng.choices(["fr", "en"], weights=[0.65, 0.35])[0]
    currency = rng.choices(["EUR", "MAD", "USD", "GBP"],
                           weights=[0.45, 0.4, 0.15, 0.0] if lang == "fr" else [0.2, 0.1, 0.5, 0.2])[0]
    code = {"EUR": "EUR", "MAD": "DH", "USD": "USD", "GBP": "GBP"}[currency]
    noisy = rng.random() < 0.6
    strip_acc = noisy and rng.random() < 0.25
    d = date(2008, 1, 1) + timedelta(days=rng.randint(0, 6850))  # real tickets are often old

    # ---- gold ----
    pool = _R_ITEMS_FR if lang == "fr" else _R_ITEMS_EN
    n_items = rng.choices([1, 2, 3, 4, 5, 6], weights=[4, 4, 3, 2, 1, 1])[0]
    items = []
    for desc in rng.sample(pool, n_items):
        qty = rng.choices([1, 2, 3, 4, 5, 10, 25, 85], weights=[12, 6, 3, 2, 1, 1, 0.4, 0.2])[0]
        unit = round(_receipt_price(rng, currency), 2)
        items.append({"description": desc, "quantity": qty, "unit_price": unit,
                      "total": round(qty * unit, 2)})
    subtotal = round(sum(i["total"] for i in items), 2)
    has_tax = rng.random() < 0.55
    # Tax-INCLUSIVE ticket: prices already contain the VAT; only the VAT amount (maybe per rate) is
    # printed. Convention: subtotal = sum of the lines = total, tax = VAT amount.
    inclusive = has_tax and rng.random() < (0.5 if lang == "fr" else 0.15)
    two_rates = inclusive and n_items >= 2 and rng.random() < 0.4
    rates = rng.sample(_R_RATES[currency], 2 if two_rates else 1)
    rate = rates[0]
    groups: list[tuple[float, float, float, float]] = []  # (rate, gross, vat, net) per VAT rate
    if inclusive:
        slot = ([0, 1] + [rng.randrange(2) for _ in items[2:]]) if two_rates else [0] * n_items
        for k, r in enumerate(rates):
            gross = round(sum(it["total"] for it, sl in zip(items, slot) if sl == k), 2)
            vat = round(gross - gross / (1 + r / 100), 2)
            groups.append((r, gross, vat, round(gross - vat, 2)))
        tax = round(sum(g[2] for g in groups), 2)
        total = subtotal
    else:
        tax = round(subtotal * rate / 100, 2) if has_tax else None
        total = round(subtotal + (tax or 0), 2)
    vendor = _receipt_vendor(rng)
    width = rng.choice([4, 5, 5, 6])
    number = None if rng.random() < 0.15 else str(rng.randint(1, 999999) % 10 ** width or 1).zfill(width)

    roll = rng.random()
    show_sym_items, show_sym_totals = (False, False) if roll < 0.08 else ((True, True) if roll < 0.45 else (False, True))
    cur_in_label = rng.random() < 0.2  # "TOTAL EUR 61,70"
    gold = {
        "invoice_number": number,
        "vendor": vendor,
        "invoice_date": d.isoformat(),
        "due_date": None,
        "currency": currency if (show_sym_items or show_sym_totals or cur_in_label) else None,
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }
    assert list(gold) == FIELD_ORDER

    # ---- text ----
    num, money = _receipt_formatters(rng, currency)
    m_item = money if show_sym_items else num
    m_tot = money if show_sym_totals else num
    sep = lambda: rng.choice([" : ", " ", "  ", ": "])  # noqa: E731
    # kinds: "crit" never garbled, "semi" lightly, "soft" more; digits are never touched
    lines: list[tuple[str, str]] = []
    add = lambda t, k="soft": lines.append((t, k))  # noqa: E731

    if rng.random() < 0.15:
        add(rng.choice(["***", "*** BIENVENUE ***" if lang == "fr" else "*** WELCOME ***", "www.example.com", "Page 1/1"]))
    vendor_text = vendor.upper() if rng.random() < 0.55 else vendor
    add(f"*** {vendor_text} ***" if rng.random() < 0.1 else vendor_text, "crit")
    city = rng.choice(_CITIES)
    add(f"{rng.randint(1, 200)}, {rng.choice(_STREETS).split(' ', 1)[1]}")
    add(f"{rng.randint(10000, 99999)} {city.upper()}" if rng.random() < 0.7 else city)
    if rng.random() < 0.5:
        add(f"Tel: 0{rng.randint(5, 7)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}")
    if rng.random() < 0.2:
        add(f"Fax: 0{rng.randint(5, 7)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}")
    if lang == "fr" and rng.random() < 0.25:  # a long number that is NOT a ticket number
        digits = " ".join([f"{rng.randint(0, 999):03d}" for _ in range(4)] + [f"{rng.randint(0, 99):02d}"])
        add(f"{rng.choice(['SIRET', 'SIRET:', 'Siret n°'])} {digits}")
    if rng.random() < 0.12:
        add("null")

    kinds = ["dmy_dash", "dmy_slash", "dmy_dot", "dmy_short", "iso", "txt"] + (["mdy_slash"] if d.day > 12 and lang == "en" else [])
    kind = rng.choice(kinds)
    stamp = _text_date(d, lang, rng) if kind == "txt" else _receipt_date(d, kind)
    if rng.random() < 0.8:
        stamp += f" {rng.randint(7, 23):02d}:{rng.randint(0, 59):02d}" + (f":{rng.randint(0, 59):02d}" if rng.random() < 0.2 else "")
    head: list[tuple[str, str]] = []
    nlabel = rng.choice(_R_NUMBER_LABELS[lang])
    if number is not None:
        head.append((f"{nlabel} {number}", "soft"))
    head.append((stamp, "crit"))
    staff = (lambda: rng.choice(_R_STAFF).upper() if rng.random() < 0.6 else rng.choice(_R_STAFF))  # noqa: E731
    pool_dist = ([f"Serveur{sep()}{staff()}", f"Table N°{sep()}{rng.randint(1, 200)}", f"Caissier{sep()}{staff()}",
                  f"Caisse {rng.randint(1, 9):02d}", f"Couverts{sep()}{rng.randint(1, 8)}", f"Client{sep()}{rng.choice(_CLIENTS)}",
                  f"TABLE {rng.randint(1, 80)}  COUVERTS {rng.randint(1, 8)}"]
                 if lang == "fr" else
                 [f"Server{sep()}{staff()}", f"Table {rng.randint(1, 80)}", f"Cashier{sep()}{staff()}",
                  f"Register {rng.randint(1, 9)}", f"Guests{sep()}{rng.randint(1, 8)}"])
    head += [(t, "soft") for t in rng.sample(pool_dist, rng.choice([0, 1, 2, 2, 3]))]
    if rng.random() < 0.35:
        rng.shuffle(head)
    if rng.random() < 0.3:
        lines.append(("", "soft"))
    lines.extend(head)
    if rng.random() < 0.5:
        lines.append(("", "soft"))

    # items
    def dtxt(desc: str) -> str:
        desc = _strip_accents(desc) if strip_acc else desc
        if noisy:
            desc = _garble_line(desc, rng, 0.1)  # the gold keeps the clean name
        return desc.upper() if upper_desc else desc
    upper_desc = rng.random() < 0.3
    all_one = all(i["quantity"] == 1 for i in items)
    layout = rng.choices(["cols", "pos", "simple", "qtyfirst"],
                         weights=[0.35, 0.15, 0.25, 0.25] if all_one else [0.5, 0.25, 0, 0.25])[0]
    if layout == "cols":
        order = rng.choices([["desc", "unit", "qty", "total"], ["desc", "qty", "unit", "total"],
                             ["qty", "desc", "unit", "total"]], weights=[0.4, 0.4, 0.2])[0]
        cl = {k: rng.choice(v) for k, v in _R_COLS[lang].items()}
        style = rng.choice(["pad", "single", "pipe"])
        if rng.random() < 0.9:
            add(" ".join(cl[k] for k in order) if style != "pipe" else " | ".join(cl[k] for k in order))
        for it in items:
            cells = {"desc": dtxt(it["description"]), "unit": m_item(it["unit_price"]),
                     "qty": str(it["quantity"]), "total": m_item(it["total"])}
            if noisy and it["quantity"] == 1 and rng.random() < 0.25:
                order_i = [k for k in order if k != "qty"]  # lost quantity cell (still recoverable: qty=1)
            else:
                order_i = order
            if style == "pad":  # always keep one space between cells so columns never fuse
                row = " ".join((cells[k].ljust(24) if k == "desc" else cells[k].rjust(10)) for k in order_i)
            else:
                row = (" | " if style == "pipe" else " ").join(cells[k] for k in order_i)
            add(row, "crit")
    elif layout == "pos":
        with_unit = rng.random() < 0.5
        for it in items:
            tail = f"  {m_item(it['unit_price'])}  {m_item(it['total'])}" if with_unit else f"   {m_item(it['total'])}"
            add(f"{it['quantity']} x {dtxt(it['description'])}{tail}", "crit")
    elif layout == "qtyfirst":  # "1 ENTRECOTE 25.00 25.00": bare quantity, name, unit price, line total
        code_tok = rng.random() < 0.15  # some POS print a one-letter code ("1 V VIN DU MOIS")
        with_unit = not all_one or rng.random() < 0.75
        for it in items:
            letter = f"{rng.choice('VAPDSC')} " if code_tok else ""
            tail = f" {m_item(it['unit_price'])} {m_item(it['total'])}" if with_unit else f" {m_item(it['total'])}"
            add(f"{it['quantity']} {letter}{dtxt(it['description'])}{tail}", "crit")
    else:  # simple: "Cafe noir      12.00"
        for it in items:
            add(f"{dtxt(it['description']).ljust(26)} {m_item(it['total']).rjust(10)}", "crit")
    if rng.random() < 0.3:
        lab = rng.choice(["Nbre de produits", "Nb articles", "Nombre d'articles"] if lang == "fr" else ["Items", "Item count"])
        add(f"{lab}{sep()}{sum(i['quantity'] for i in items)}")

    # totals
    T = _R_TOTALS[lang]
    fmt2 = rng.random() < 0.4
    comma = lang == "fr" and rng.random() < 0.5
    def rtxt(r: float) -> str:
        s = f"{r:.2f}" if fmt2 else f"{r:g}"
        return s.replace(".", ",") if comma else s
    def tline(label: str, value: float) -> str:
        if rng.random() < 0.3:
            return label.ljust(20) + " " + m_tot(value).rjust(12)
        return f"{label}{sep()}{m_tot(value)}"
    def tot_label(labels: list[str]) -> str:
        lab = rng.choice(labels)
        return f"{lab} {code}" if cur_in_label else lab
    vat_word = "TVA" if lang == "fr" else rng.choice(["VAT", "Tax"])
    net_word = "HT" if lang == "fr" else "Net"
    if inclusive:
        recap: list[str] = []
        if len(groups) == 1:
            r, _, vat, _ = groups[0]
            labels = ([f"TVA {rtxt(r)}%", f"Dont TVA {rtxt(r)}%", f"TVA incluse ({rtxt(r)}%)", f"TVA {rtxt(r)}"]
                      if lang == "fr" else
                      [f"Incl. {rtxt(r)}% VAT", f"VAT {rtxt(r)}% incl.", f"Tax included ({rtxt(r)}%)"])
            recap.append(tline(rng.choice(labels), vat))
        elif rng.random() < 0.6:  # "TVA 19.60 0.52" / "HT TVA 19.60 2.68", one pair per rate
            for r, _, vat, net in groups:
                recap.append(tline(f"{vat_word} {rtxt(r)}", vat))
                recap.append(tline(f"{net_word} {vat_word} {rtxt(r)}", net))
        else:  # one line per rate
            for r, _, vat, net in groups:
                recap.append(f"{vat_word} {rtxt(r)}%  {net_word} {m_tot(net)}  {vat_word} {m_tot(vat)}")
        total_line = tline(tot_label(T["tot"]), total)
        block = [(total_line, "semi")] + [(x, "soft") for x in recap]
        if rng.random() < 0.5:
            block = block[1:] + block[:1]
        for text_, k in block:
            add(text_, k)
        if rng.random() < 0.3:  # the same amount repeated ("RESTE A PAYER 61,70")
            add(tline(tot_label(["Reste à payer", "Net à payer"] if lang == "fr" else ["Balance Due", "Amount Due"]), total), "semi")
    elif has_tax:
        tax_label = rng.choice([f"{rng.choice(T['tax'])} ({rtxt(rate)}%)", f"{rng.choice(T['tax'])} {rtxt(rate)}%",
                                rng.choice(T["tax"])])
        parts = {"sub": (tline(rng.choice(T["sub"]), subtotal), "semi"), "tax": (tline(tax_label, tax), "soft"),
                 "tot": (tline(tot_label(T["tot_tax"]), total), "semi")}
        for k in rng.choice([("sub", "tax", "tot"), ("sub", "tax", "tot"), ("tax", "sub", "tot")]):
            add(*parts[k])
    else:
        add(tline(tot_label(T["tot"]), total), "semi")

    # payment
    if rng.random() < 0.75:
        if rng.random() < 0.55:
            cash = rng.choice(["Espèces", "Especes", "Cash"] if lang == "fr" else ["Cash"])
            tender = total if rng.random() < 0.4 else next((t for t in (5, 10, 20, 50, 100, 200, 500) if t >= total), total)
            add(f"{cash}{sep()}{num(float(tender), True)}")
            if tender - total > 0.004:
                add(f"{'Rendu' if lang == 'fr' else 'Change'}{sep()}{num(round(tender - total, 2), True)}")
        else:
            card = rng.choice(["Carte bancaire", "CB", "Carte"] if lang == "fr" else ["Card", "Visa", "Credit card"])
            add(f"{'Mode de paiement' if lang == 'fr' else 'Payment'}{sep()}{card}")
    for t in rng.sample(_R_FOOTERS[lang], rng.choice([0, 1, 2, 2, 3])):
        add(t)

    # ---- OCR-style noise (never touches digits or the vendor line) ----
    out_lines: list[str] = []
    for text, kind_ in lines:
        if noisy and kind_ == "soft":
            text = _garble_line(text, rng, 0.15)
        elif noisy and kind_ == "semi":
            text = _garble_line(text, rng, 0.06)
        if noisy and text.strip() and rng.random() < 0.25:
            text = (f"{rng.choice(_R_LINE_PREFIX)} " if rng.random() < 0.7 else "") + text \
                   + (f" {rng.choice(_R_LINE_SUFFIX)}" if rng.random() < 0.4 else "")
        out_lines.append(text)
    if noisy:
        for _ in range(rng.choice([0, 0, 1, 2, 3])):
            out_lines.insert(rng.randint(0, len(out_lines)), _junk_line(rng))
    return gold, "\n".join(out_lines)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
INSTRUCTION = "Extract invoice information as JSON."
DEFAULT_RECEIPT_RATIO = 0.35  # share of receipts in generate_dataset(); 0 gives invoices only


def generate_example(rng: random.Random) -> dict:
    """One invoice record: {"instruction", "input", "output", "kind"}."""
    lang = rng.choices(["en", "fr"], weights=[0.7, 0.3])[0]
    gold = generate_gold(rng, lang)
    text = render_text(gold, rng, lang)
    gold = {k: v for k, v in gold.items() if not k.startswith("_")}
    return {"instruction": INSTRUCTION, "input": text, "output": gold, "kind": "invoice"}


def generate_receipt_example(rng: random.Random) -> dict:
    """One receipt record, same format as generate_example()."""
    gold, text = generate_receipt(rng)
    return {"instruction": INSTRUCTION, "input": text, "output": gold, "kind": "receipt"}


def generate_dataset(n: int, seed: int = 42, receipt_ratio: float = DEFAULT_RECEIPT_RATIO) -> list[dict]:
    """Mixed dataset of invoices and receipts. ``receipt_ratio=0`` reproduces the invoices-only
    dataset of the first version (invoices use their own random stream, untouched by receipts)."""
    rng = random.Random(seed)              # invoices
    mix = random.Random(seed + 10_007)     # invoice vs receipt decision
    rrng = random.Random(seed + 20_011)    # receipts
    out = []
    for _ in range(n):
        if receipt_ratio > 0 and mix.random() < receipt_ratio:
            out.append(generate_receipt_example(rrng))
        else:
            out.append(generate_example(rng))
    return out