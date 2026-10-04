# Sample Q&A log

Real answers from `python -m app.agents.qa_eval` against the real Bid1 and Bid2 documents, taken unchanged from [`outputs/qa_eval_gpt-oss-20b.json`](../outputs/qa_eval_gpt-oss-20b.json). The questions and the hand-written expectations are in [`backend/eval/qa_eval.json`](../backend/eval/qa_eval.json).

**Which model produced these answers:** `openai/gpt-oss-20b`, **not** the configured `openai/gpt-oss-120b`. The 120b run could not be completed within the Groq free-tier daily token limit, so the 14-question set was run on 20b. These answers predate the latest Bid Number / Product Specification extraction changes (which do not affect the Q&A path). Each answer is produced from retrieved chunks of the asked bid only, and every cited source is re-checked against the database.

Per-question checks come from the evaluator: *answer correct* (expected text appears), *isolation* (no chunk from the other bid), *citation correct* (a cited chunk has the expected file and text), *grounded* (the answer's figures and key words appear in the cited chunks). Where the answer is a refusal the standard sentence is shown.

| # | id | bid | category | result |
| - | - | - | - | - |
| 1 | `b1-final-deadline` | Bid1 | deadline (addendum-aware) | answered; answer correct, isolation, citation correct, grounded OK |
| 2 | `b1-original-deadline` | Bid1 | deadline (original) | answered; answer correct, isolation, citation correct, grounded OK |
| 3 | `b1-addendum2-change` | Bid1 | addendum | answered; answer correct, isolation, citation correct, grounded OK |
| 4 | `b1-prebid` | Bid1 | requirement / schedule | answered; answer correct, isolation, citation correct, grounded OK |
| 5 | `b1-product` | Bid1 | product | answered; answer correct, isolation, citation correct, grounded OK |
| 6 | `b1-documents` | Bid1 | requirements | answered; answer correct, isolation, citation correct, grounded OK |
| 7 | `b2-model` | Bid2 | product | answered; answer correct, isolation, citation correct, grounded OK |
| 8 | `b2-processor` | Bid2 | product specification | answered; answer correct, isolation, citation correct, grounded OK |
| 9 | `b2-warranty` | Bid2 | requirements | answered; answer correct, isolation, citation correct, grounded OK |
| 10 | `b2-affidavits` | Bid2 | requirements | answered; answer correct, isolation, citation correct, grounded OK |
| 11 | `b2-deadline` | Bid2 | deadline | **refused although the documents contain the answer** (safe, but not a correct answer) |
| 12 | `b1-award` | Bid1 | unsupported question | **correct refusal** (no support in this bid's documents) |
| 13 | `b2-award` | Bid2 | unsupported question | **correct refusal** (no support in this bid's documents) |
| 14 | `b1-mercury` | Bid1 | cross-bid trap (unsupported) | **correct refusal** (no support in this bid's documents) |

Summary of this run: 10 answered questions, all correct, isolated, correctly cited and grounded; 3 correct refusals; 1 safe refusal on `b2-deadline` (its answer was not scored). Token and latency figures per question are in the JSON file.

### Q1. What is the final submission deadline?

Bid: `Bid1` | category: deadline (addendum-aware)

Answer:
> July 9, 2024 at 2:00 PM CST

Sources:
- Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf p.1
- Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf p.4

### Q2. What was the original submission deadline?

Bid: `Bid1` | category: deadline (original)

Answer:
> 27-JUN-2024 14:00:00

Sources:
- JA-207652 Student and Staff Computing Devices FINAL.pdf p.2

### Q3. What changed in Addendum 2?

Bid: `Bid1` | category: addendum

Answer:
> Addendum 2 extends the RFP’s due date to July 9, 2024 at 2:00 PM CST.

Sources:
- Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf p.1

### Q4. Is there a pre-bid meeting?

Bid: `Bid1` | category: requirement / schedule

Answer:
> Yes, a pre‑bid meeting is scheduled for 10‑JUN‑2024 at 2:00 PM CST via TEAMS video conference.

Sources:
- JA-207652 Student and Staff Computing Devices FINAL.pdf p.2
- Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html (HTML page, no page number)

### Q5. What product is being procured?

Bid: `Bid1` | category: product

Answer:
> Student and staff computing devices, including laptops, desktops, tablet devices, and display monitors.

Sources:
- JA-207652 Student and Staff Computing Devices FINAL.pdf p.3

### Q6. What documentation must be submitted?

Bid: `Bid1` | category: requirements

Answer:
> The offer must include pertinent literature/documentation for the proposed products or services, any required credit application, factory‑authorized repair and maintenance certifications for resellers, a certificate of insurance/bond, Form 1295 (Certificate of Interested Parties) attached to the response, and any required subcontractor change forms with written justification and supporting documentation. For manual submissions, a single original hard copy is required. During contract performance, the prime must submit an M/WBE Pay Activity Report with proof of payments.

Sources:
- JA-207652 Student and Staff Computing Devices FINAL.pdf p.9
- JA-207652 Student and Staff Computing Devices FINAL.pdf p.9
- JA-207652 Student and Staff Computing Devices FINAL.pdf p.3
- JA-207652 Student and Staff Computing Devices FINAL.pdf p.17
- Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf p.4
- JA-207652 Student and Staff Computing Devices FINAL.pdf p.51

### Q7. What laptop model is required?

Bid: `Bid2` | category: product

Answer:
> Dell Latitude 5550

Sources:
- PORFP_-_Dell_Laptop_Final.pdf p.3
- Dell Laptops w_Extended Warranty - Bid Information - {3} _ BidNet Direct.html (HTML page, no page number)
- Dell_Laptop_Specs.pdf p.1

### Q8. What processor is required?

Bid: `Bid2` | category: product specification

Answer:
> Intel Core Ultra 5 125U processor

Sources:
- Dell_Laptop_Specs.pdf p.1

### Q9. What warranty is required?

Bid: `Bid2` | category: requirements

Answer:
> A 3‑year Dell Limited Hardware Warranty, extended for all machines purchased, commencing on the date of delivery and ending three years thereafter. A warranty certificate or affidavit must be presented upon award.

Sources:
- PORFP_-_Dell_Laptop_Final.pdf p.3
- PORFP_-_Dell_Laptop_Final.pdf p.3

### Q10. What affidavits are mentioned?

Bid: `Bid2` | category: requirements

Answer:
> The documents refer to a Contract Affidavit, a Mercury Affidavit, and a Proposal Affidavit.

Sources:
- Contract_Affidavit.pdf p.3
- Mercury_Affidavit.pdf p.1
- Mercury_Affidavit.pdf p.1
- Contract_Affidavit.pdf p.1
- Mercury_Affidavit.pdf p.1
- Contract_Affidavit.pdf p.3

### Q11. What is the submission deadline?

Bid: `Bid2` | category: deadline

Answer (**REFUSAL**: the system states the evidence is insufficient):
> I could not find sufficient evidence to answer this question.

Sources: none (nothing is cited for a refusal)

Note recorded by the system: The model found the evidence insufficient.

Expected: an answer exists (RFP text matching `PROPOSAL DUE`). The model refused instead, which is the safe failure; the evaluator scores the citation check as failed.

### Q12. Which vendor was awarded the contract?

Bid: `Bid1` | category: unsupported question

Answer (**REFUSAL**: the system states the evidence is insufficient):
> I could not find sufficient evidence to answer this question.

Sources: none (nothing is cited for a refusal)

Note recorded by the system: The model found the evidence insufficient.

Expected: refusal. The documents of this bid contain no support for the question.

### Q13. Which vendor was awarded the contract?

Bid: `Bid2` | category: unsupported question

Answer (**REFUSAL**: the system states the evidence is insufficient):
> I could not find sufficient evidence to answer this question.

Sources: none (nothing is cited for a refusal)

Note recorded by the system: The model found the evidence insufficient.

Expected: refusal. The documents of this bid contain no support for the question.

### Q14. Is a Mercury Affidavit required?

Bid: `Bid1` | category: cross-bid trap (unsupported)

Answer (**REFUSAL**: the system states the evidence is insufficient):
> I could not find sufficient evidence to answer this question.

Sources: none (nothing is cited for a refusal)

Note recorded by the system: The model found the evidence insufficient.

Expected: refusal. The documents of this bid contain no support for the question.

