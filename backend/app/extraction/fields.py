"""The fields to extract, how to retrieve evidence for each, and how they are grouped.

`kind` controls how a field's value is shaped and validated:
  scalar - one short value; several distinct supported values count as a conflict
  list   - several items that are all valid together (merged, never a conflict)
  text   - free text generated from evidence (the summary)
  spec   - {"product": str, "specifications": {name: value}}; one candidate per product line, so a bid
           with several products yields a list of such objects (a single product stays one object)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    key: str  # stable snake_case id used in prompts and LLM output
    name: str  # name used in the API response (matches the assessment)
    description: str  # what to extract; goes into the prompt
    queries: tuple[str, ...]  # field-oriented hybrid-retrieval queries
    kind: str = "scalar"
    addenda: bool = True  # also fetch addendum evidence (amendments may change the value)
    group: str = "requirements"
    evidence_limit: int | None = None  # chunks of evidence for this field; default: EXTRACTION_EVIDENCE_PER_FIELD


FIELDS: tuple[FieldSpec, ...] = (
    # ---------------------------------------------------------------- identity
    FieldSpec(
        "bid_number", "Bid Number",
        "The solicitation's own number, exactly as written: the identifier the documents label as the bid, "
        "solicitation, RFP or PORFP number. List the number printed on the solicitation document itself first. "
        "Do NOT put other kinds of identifier in the value: a portal 'Sourcing #', a 'Reference Number', a "
        "project number, or a number that is not labelled as the solicitation's number. If such identifiers "
        "appear, mention them with their label in `notes` instead. Return a list only when the evidence labels "
        "more than one identifier as the solicitation number.",
        ("bid number solicitation number RFP number PORFP number", "project number reference number"),
        kind="list", addenda=False, group="identity",
    ),
    FieldSpec(
        "title", "Title",
        "The official title of the solicitation.",
        ("title of the solicitation request for proposal",),
        addenda=False, group="identity",
    ),
    FieldSpec(
        "company_name", "company_name",
        "The issuing organization (agency, district or company that issued the solicitation).",
        ("issuing organization agency school district department",),
        addenda=False, group="identity",
    ),
    FieldSpec(
        "contact_info", "contact_info",
        "Procurement contact(s): name, email and phone, exactly as written. Return a list of strings.",
        ("contact buyer procurement officer email phone",
         "contact information procurement services telephone email"),
        kind="list", addenda=False, group="identity",
    ),
    FieldSpec(
        "contract_cooperative", "Contract or Cooperative to use",
        "A referenced contract, master contract or cooperative purchasing vehicle that bidders must use or are "
        "eligible under, including its number if given.",
        ("cooperative contract master contract purchasing vehicle contract number",),
        addenda=False, group="identity",
    ),
    # ---------------------------------------------------------------- schedule
    FieldSpec(
        "due_date", "Due Date",
        "The proposal/bid submission due (closing) date and time, exactly as the evidence states it. "
        "Do NOT reconcile addenda: if different documents give different due dates, list each distinct date "
        "as its own candidate with its own citations.",
        ("submission due date closing date solicitation due proposals due",),
        group="schedule",
    ),
    FieldSpec(
        "pre_bid_meeting", "Pre Bid Meeting",
        "The pre-bid / pre-proposal meeting: date, time, location or call details if stated.",
        ("pre-bid pre-proposal meeting conference date time",),
        group="schedule",
    ),
    FieldSpec(
        "delivery_date", "Delivery Date",
        "When the purchased goods must be delivered: a delivery date or period (e.g. within N days of award). "
        "Not warranty repair, support response or replacement times: if only those appear, return no candidates.",
        ("delivery deadline delivery within days of award delivery requirements",),
        group="schedule",
    ),
    FieldSpec(
        "term_of_bid", "Term of Bid",
        "The contract or offer term: length, renewal/extension options, and how long quotes must stay valid.",
        ("contract term period years renewal extension",
         "proposals quotes must be valid for at least days after the due date"),
        group="schedule",
    ),
    # ------------------------------------------------------------ requirements
    FieldSpec(
        "bid_submission_type", "Bid Submission Type",
        "How bids may be submitted (e.g. electronic/online system, email, physical delivery). Several permitted "
        "methods are all valid: return a list with one string per method. Only return what the evidence states.",
        ("how to submit proposal electronically online email hand delivery submission instructions",),
        kind="list", group="requirements",
    ),
    FieldSpec(
        "bid_bond", "Bid Bond Requirement",
        "Whether a bid bond (or bid security) is required, and the amount or percentage if explicitly stated.",
        ("bid bond bid security surety performance bond",),
        group="requirements",
    ),
    FieldSpec(
        "payment_terms", "Payment Terms",
        "Payment terms: when and how the vendor is paid, invoicing requirements, net days.",
        ("payment terms invoice net days payment",),
        group="requirements",
    ),
    FieldSpec(
        "installation", "Installation",
        "Installation, deployment or setup requirements (e.g. white-glove services, asset tagging, software "
        "installation). These stack rather than contradict: return a list with one string per requirement.",
        ("installation setup deployment services asset tagging",),
        kind="list", group="requirements",
    ),
    FieldSpec(
        "additional_documentation", "Any Additional Documentation Required",
        "Documents the bidder must submit besides the proposal itself: affidavits, certifications, forms, "
        "signed addenda. Return a list of strings, one per required document.",
        ("required affidavit certification forms documentation must submit",),
        kind="list", group="requirements",
    ),
    # ----------------------------------------------------------------- product
    FieldSpec(
        "mfg_for_registration", "MFG for Registration",
        "The manufacturer named for the product or for registration/authorization (e.g. 'must be an authorized "
        "reseller for X').",
        ("manufacturer authorized reseller manufacturer name OEM",),
        group="product",
    ),
    FieldSpec(
        "model_no", "Model_no",
        "The exact model number(s) or model name(s) of the product(s) being procured: a list with one string "
        "per model. Return nothing if the documents specify no particular model.",
        ("model number model name product",),
        kind="list", group="product",
    ),
    FieldSpec(
        "part_no", "Part_no",
        "The exact part number(s) / SKU(s) / catalog number(s) of the product(s): a list with one string per "
        "part. Do not confuse with model number.",
        ("part number SKU catalog number",),
        kind="list", group="product",
    ),
    FieldSpec(
        "product", "Product",
        "The product(s) or equipment being procured. A solicitation may buy several items (e.g. a laptop and a "
        "docking station): return a list with one string per item.",
        ("product name equipment devices laptops being procured",
         "what goods or services does this request for proposal solicit"),
        kind="list", group="product",
    ),
    FieldSpec(
        "product_specification", "Product Specification",
        "Technical specifications of the product(s). Return one candidate per distinct product line the "
        "evidence gives separate specifications for (for example each laptop tier, desktop, tablet or monitor), "
        'each shaped {"product": "<name as written>", "specifications": {"<attribute>": "<value>", ...}} '
        "using attribute names such as processor, memory, storage, display, ports, battery, warranty. Cite only "
        "the chunk(s) that contain that product's specifications. Include only attributes present in that "
        "evidence, with the exact wording of the values, even if it looks like a typo. Never merge two "
        "products' specifications into one object. At most 8 products.",
        ("specifications processor memory storage display requirements",
         "minimum requirements ports battery wireless operating system"),
        kind="spec", group="product", evidence_limit=8,  # a bid can list many product lines
    ),
    # ----------------------------------------------------------------- summary
    FieldSpec(
        "bid_summary", "Bid Summary",
        "A concise summary (3-5 sentences) written ONLY from the evidence: what is being procured, quantity "
        "if explicitly stated, key requirements, submission information, and notable documentation "
        "requirements. Do not state which of several conflicting dates is final.",
        ("scope of work description purpose of this solicitation",),
        kind="text", addenda=False, group="summary",
    ),
)

FIELDS_BY_KEY = {f.key: f for f in FIELDS}
GROUPS: tuple[str, ...] = ("identity", "schedule", "requirements", "product", "summary")

assert len(FIELDS) == 20 and len(FIELDS_BY_KEY) == 20
