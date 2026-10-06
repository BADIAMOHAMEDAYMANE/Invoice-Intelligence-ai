Evaluation on 40 held-out test invoices

| Metric | Regex baseline | Base model | Fine-tuned |
|---|---:|---:|---:|
| field_accuracy | 0.53 | 0.54 | 0.99 |
| exact_match | 0.00 | 0.07 | 0.93 |
| json_validity | 1.00 | 1.00 | 1.00 |
| invoice_number_accuracy | 0.45 | 0.65 | 1.00 |
| vendor_accuracy | 0.75 | 0.72 | 1.00 |
| invoice_date_accuracy | 0.42 | 0.62 | 1.00 |
| due_date_accuracy | 0.65 | 0.42 | 1.00 |
| currency_accuracy | 0.95 | 0.42 | 0.97 |
| subtotal_accuracy | 0.47 | 0.45 | 1.00 |
| tax_accuracy | 0.55 | 0.70 | 0.97 |
| total_accuracy | 0.50 | 0.47 | 1.00 |
| items_accuracy | 0.00 | 0.40 | 0.95 |
| validation_pass | 0.50 | 0.28 | 0.97 |
