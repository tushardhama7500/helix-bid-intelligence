"""Which Phase 5 fields addenda can modify, and how change language names them."""

from dataclasses import dataclass

DATE = r"\d{1,2}-[A-Za-z]{3}-\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"


@dataclass(frozen=True)
class ReconField:
    key: str  # Phase 5 field key (app.extraction.fields)
    terms: tuple[str, ...]  # words that name this field in addendum change language
    kind: str = "scalar"  # "scalar" | "list"
    # Regexes (group "v") that read the ORIGINAL value straight from original-document text.
    # Only used for PDF originals; a listing page such as BidNet already mirrors later changes.
    locators: tuple[str, ...] = ()


RECONCILABLE: tuple[ReconField, ...] = (
    ReconField(
        "due_date",
        ("due date", "closing date", "submission deadline", "proposal due date", "bid due date",
         "solicitation due", "submission date", "deadline", "opening date"),
        locators=(rf"Solicitation Due\s*[:\-]?\s*(?P<v>{DATE})",),
    ),
    ReconField(
        "pre_bid_meeting",
        ("pre-bid meeting", "pre-proposal meeting", "pre-bid conference", "pre-proposal conference"),
        locators=(rf"Pre-?\s*Proposal Meeting\s*[:\-]?\s*(?P<v>{DATE})",),
    ),
    ReconField("term_of_bid", ("term of the contract", "contract term", "initial term", "renewal term", "term")),
    ReconField("delivery_date", ("delivery date", "delivery period", "delivery deadline", "delivery time", "delivery")),
    ReconField("bid_bond", ("bid bond", "bid security")),
    ReconField("payment_terms", ("payment terms", "payment term", "payment")),
    ReconField("installation", ("installation", "deployment", "setup")),
    ReconField(
        "additional_documentation",
        ("additional documentation", "required documentation", "required documents", "documentation", "affidavit"),
        kind="list",
    ),
    ReconField("model_no", ("model number", "model no", "model")),
    ReconField("part_no", ("part number", "part no", "sku")),
    ReconField("product", ("product", "equipment")),
)
RECONCILABLE_BY_KEY = {f.key: f for f in RECONCILABLE}

# Composite / generated fields are not reconciled directly: their parts are.
SKIPPED_FIELDS = {
    "Product Specification": "composite field; reconcile the underlying requirements (model, part, product) instead",
    "Bid Summary": "generated text; regenerate it from the reconciled fields",
}
