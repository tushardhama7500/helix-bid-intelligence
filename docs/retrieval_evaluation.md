# Retrieval evaluation report

Everything here is deterministic: no Groq call is involved. The numbers below were produced by re-running the evaluator on the real documents while preparing the submission, and they equal the Phase 4 numbers in the README.

Reproduce:

```bash
cd backend
python -m app.retrieval                 # with metadata filters
python -m app.retrieval --no-filters    # no filters
python -m app.retrieval -v              # per-question top-5
```

Raw results: [`outputs/retrieval_eval_filters.json`](../outputs/retrieval_eval_filters.json), [`outputs/retrieval_eval_nofilters.json`](../outputs/retrieval_eval_nofilters.json). Question set and relevance labels: [`backend/eval/retrieval_eval.json`](../backend/eval/retrieval_eval.json) (unchanged since it was written; it was not edited to improve scores). Evaluator: `backend/app/retrieval/evaluation.py`.

## Method

- **Corpus:** the 9 real documents (Bid1: 4, Bid2: 5), 285 chunks, each with a 384-d `BAAI/bge-small-en-v1.5` embedding.
- **Questions:** 19 (8 Bid1, 7 Bid2, 4 exact-term queries such as a bid number or a model number).
- **Relevance** is judged at chunk level: a retrieved chunk is relevant when it comes from a file whose name contains the expected text **and** its (whitespace-normalised) text matches the expected evidence pattern. A hit in the top K means at least one relevant chunk is among the first K results.
- **Recall@K** here is therefore the share of questions with at least one relevant chunk in the top K (each question has one or a few acceptable chunks; it is not a share of all relevant chunks). **MRR@5** is the mean over questions of 1/rank of the first relevant chunk within the top 5 (0 when none).
- **Configurations tested (6):** vector, keyword and hybrid, each with and without metadata filters. With filters, the Bid1/Bid2 questions are restricted to their `bid_id` (the exact-term queries use no filter). No reranker was tested because none is implemented.

## Results

### With metadata filters

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| Vector only (pgvector cosine) | 0.632 | 0.895 | 1.000 | 0.769 |
| Keyword only (PostgreSQL full-text) | 0.579 | 0.684 | 0.684 | 0.614 |
| Hybrid (RRF, k=60) | 0.684 | 0.895 | 1.000 | 0.775 |

### Without filters (every query searches all 285 chunks)

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| Vector only (pgvector cosine) | 0.474 | 0.632 | 0.789 | 0.578 |
| Keyword only (PostgreSQL full-text) | 0.526 | 0.684 | 0.684 | 0.579 |
| Hybrid (RRF, k=60) | 0.526 | 0.737 | 0.842 | 0.632 |

## Interpretation

- With filters, vector search already reaches Recall@5 = 1.000 because only one bid's chunks compete; hybrid ties it at Recall@3/@5 and is slightly ahead at Recall@1 (0.684 vs 0.632) and MRR@5 (0.775 vs 0.769), a one-question difference.
- Without filters hybrid is best at Recall@3 (0.737), Recall@5 (0.842) and MRR@5 (0.632). Keyword search is strong for exact terms (bid numbers, model names) but misses paraphrased questions; vector search is the reverse. Fusion keeps the benefit of both.
- Filtering by bid never hurts and usually helps (keyword Recall@5 is unchanged at 0.684); the application always filters by `bid_id`, which is also what keeps bids isolated.
- 19 questions is small: one question is 0.053, so differences of that size should not be over-read. The labels were checked by hand against the same documents the system searches.

## Question set and first-hit rank per configuration

Rank of the first relevant chunk in the top 5 (`-` = not in the top 5). F = with filters, U = without filters.

| id | question | filter | vec F | kw F | hyb F | vec U | kw U | hyb U |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `b1-final-deadline` | What is the final submission deadline for Bid1? | Bid1 | 3 | - | 3 | 3 | - | 3 |
| `b1-original-deadline` | What was the original submission deadline? | Bid1 | 4 | - | 5 | 5 | - | - |
| `b1-addendum2-change` | What changed in Addendum 2? | Bid1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `b1-usb` | What USB specification is required? | Bid1 | 1 | 3 | 1 | 1 | 3 | 1 |
| `b1-accidental-damage` | Is accidental damage protection required? | Bid1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `b1-warranty` | What minimum warranty is required for student Chromebooks? | Bid1 | 2 | 3 | 3 | 2 | 3 | 3 |
| `b1-prebid-meeting` | When is the pre-proposal meeting? | Bid1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `b1-bidnet-closing` | What is the closing date listed on the BidNet page? | Bid1 | 2 | - | 3 | 2 | - | 4 |
| `b2-model` | What laptop model is required? | Bid2 | 1 | 1 | 1 | 5 | 1 | 1 |
| `b2-processor` | What processor is specified? | Bid2 | 1 | - | 1 | 4 | - | 4 |
| `b2-ram` | How much RAM is required? | Bid2 | 5 | - | 5 | - | - | - |
| `b2-warranty` | What warranty period is required? | Bid2 | 1 | 1 | 1 | - | 3 | 3 |
| `b2-mercury` | Is a Mercury Affidavit required? | Bid2 | 1 | 1 | 1 | 1 | 1 | 1 |
| `b2-delivery` | What is the delivery requirement? | Bid2 | 2 | - | 3 | - | - | - |
| `b2-manufacturer` | What manufacturer is specified? | Bid2 | 3 | 1 | 1 | - | 1 | 2 |
| `x-bid-number-ja` | JA-207652 | none | 1 | 1 | 1 | 1 | 1 | 1 |
| `x-bid-number-bpm` | BPM044557 | none | 1 | 1 | 1 | 1 | 1 | 1 |
| `x-model-name` | Dell Latitude 5550 | none | 1 | 1 | 1 | 1 | 1 | 1 |
| `x-dock-model` | WD22TB4 | none | 1 | 1 | 1 | 1 | 1 | 1 |

Expected evidence per question (file name contains / text matches):

| id | file contains | text pattern |
| --- | --- | --- |
| `b1-final-deadline` | `Addendum 2` | `July 9, 2024` |
| `b1-original-deadline` | `FINAL.pdf` | `Solicitation Due` |
| `b1-addendum2-change` | `Addendum 2` | `extend the due date` |
| `b1-usb` | `Addendum 1` | `USB 3\.1 is a minimum` |
| `b1-accidental-damage` | `Addendum 1` | `will not ask for Accidental Damage Protection` |
| `b1-warranty` | `Addendum 1` | `one-year warranty` |
| `b1-prebid-meeting` | `FINAL.pdf` | `Pre-Proposal Meeting` |
| `b1-bidnet-closing` | `BidNet` | `Closing Date` |
| `b2-model` | `PORFP` | `Latitude\s+5550` |
| `b2-model` | `Dell_Laptop_Specs` | `Latitude\s+5550` |
| `b2-processor` | `Dell_Laptop_Specs` | `Core Ultra 5 125U` |
| `b2-ram` | `Dell_Laptop_Specs` | `16 GB` |
| `b2-warranty` | `PORFP` | `Extended for all machines\s+purchased - 3 Years` |
| `b2-mercury` | `PORFP` | `must provide a Mercury Affidavit` |
| `b2-mercury` | `Mercury_Affidavit` | `MERCURY AFFIDAVIT` |
| `b2-delivery` | `PORFP` | `Delivery within 45 days` |
| `b2-manufacturer` | `PORFP` | `Manufacturer Name` |
| `b2-manufacturer` | `PORFP` | `authorized reseller for Dell` |
| `x-bid-number-ja` | `` | `JA-207652` |
| `x-bid-number-bpm` | `` | `BPM044557` |
| `x-model-name` | `` | `Latitude\s+5550` |
| `x-dock-model` | `` | `WD22TB4` |
