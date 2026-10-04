# Helix

Helix is a bid / procurement document intelligence system. It ingests the PDF and HTML documents of a bid (RFP, addenda, specifications, affidavits, a BidNet listing), indexes them for hybrid search, extracts 20 structured bid fields with citations and confidence scores, reconciles addenda that change earlier requirements (for example a moved due date), validates every citation against the database, and answers questions about a bid with cited, grounded answers. It has a FastAPI backend, a React UI and a CLI.

> **Evidence status in one paragraph.** Retrieval is evaluated deterministically (Recall@1/3/5 and MRR@5, no LLM). Extraction and reconciliation were validated with real Groq runs on `openai/gpt-oss-120b` in earlier phases; Q&A was validated on `openai/gpt-oss-20b` (not the configured model). The latest targeted extraction changes (Bid Number scoping, multi-product Product Specification) are covered by deterministic tests only and have **not** been re-validated with a live Groq run. A full end-to-end 120b run and a 120b Q&A run could not be completed within the Groq free-tier daily token limit. Details are under [Evaluation results](#evaluation-results) and [Limitations](#limitations).

## Contents

[Features](#features) · [Architecture](#architecture) · [Tech stack](#tech-stack) · [Setup](#setup) · [Running](#running-the-application) · [API](#api) · [CLI](#cli) · [Pipeline modes](#pipeline-modes) · [Design: ingestion](#document-ingestion-design) · [chunking](#chunking-strategy) · [embeddings](#embedding-model) · [retrieval](#retrieval-strategy) · [extraction](#extraction) · [reconciliation](#addendum-reconciliation) · [agents](#multi-agent-system) · [framework](#agent-framework) · [prompts](#prompt-design) · [Q&A](#qa) · [Evaluation](#evaluation-results) · [Artifacts](#output-artifacts) · [Limitations](#limitations) · [Appendix: phase notes](#appendix-detailed-phase-notes)

## Features

- **PDF and HTML ingestion** (PyMuPDF, BeautifulSoup), idempotent by SHA-256 of the file bytes. Text is normalised and each document is typed (`addendum`, `affidavit`, `specification`, `rfp`, `bid_document`) from its file name.
- **Text-only document processing.** PDFs are read page by page; HTML is split into sections at `<h1>`/`<h2>` headings. HTML tables are flattened to text. **There is no table-structure extraction and no OCR**: a scanned, textless PDF is reported as having no extractable text.
- **Chunking** with paragraph/sentence/word boundaries, 1000 characters with 150 overlap by default; chunks never span pages.
- **Embeddings** with FastEmbed `BAAI/bge-small-en-v1.5` (384-d, local, no embedding API), stored in PostgreSQL with **pgvector** (HNSW, cosine).
- **PostgreSQL full-text keyword search** (generated `tsvector` + GIN index) and **hybrid retrieval** by Reciprocal Rank Fusion, with **metadata filtering** (`bid_id`, document type, file, document id, page).
- **Structured bid extraction**: 20 fields, JSON from Groq, validated against Pydantic models, with citations, grounding checks and confidence scores. Missing values stay `null`.
- **Addendum reconciliation** with deterministic rules first and a Groq fallback only for ambiguous wording; the original value and the change history are always kept.
- **Multi-agent orchestration** (a fixed sequence of thin agents; no agent framework).
- **Grounded Q&A** that refuses when the evidence does not support an answer.
- **Citation validation**: every citation is re-read from the database and checked (exists, same bid, same file/page, contains the value).
- **Confidence scores**: a documented heuristic, not a calibrated probability.
- **Bid isolation**: every retrieval is filtered by `bid_id` and the code re-checks it; nothing from Bid2 can be used for Bid1.
- **React UI**, **FastAPI API** and **CLI** over the same services.

## Architecture

```
Documents (PDF / HTML)
  -> Ingestion (discover, hash, parse, normalise, type)
  -> Chunking
  -> Embeddings (FastEmbed)
  -> PostgreSQL + pgvector (+ full-text index)
  -> Hybrid retrieval (vector + keyword, Reciprocal Rank Fusion, bid filter)
  -> Extraction (Groq, grounded, cited)
  -> Reconciliation (addendum timeline, rules, Groq only if ambiguous)
  -> Validation (re-check citations in the database)
  -> Report / Q&A
  -> API / React UI / CLI
```

The `Orchestrator` (`backend/app/agents/orchestrator.py`) runs five stages in a fixed order and builds the report: **ingestion, retrieval, extraction, reconciliation, validation**. Agents do not call each other; each stage reads and writes one shared `BidPipelineState`. The `ReportQAAgent` builds the report from the validated state and answers questions separately, using the same retrieval agent.

### Architecture diagram

The same diagram is in [`docs/architecture.md`](docs/architecture.md) and [`docs/architecture.mmd`](docs/architecture.mmd). (No PNG is included; GitHub and VS Code render the Mermaid source.)

```mermaid
flowchart TB
    DOCS["Bid documents<br/>data/Bid1, data/Bid2<br/>PDF + HTML"]

    subgraph ING["Ingestion (app/ingestion, IngestionAgent)"]
        direction TB
        DISC["Discovery<br/>size + path checks, SHA-256 idempotency"]
        PARSE["Parsing<br/>PyMuPDF (PDF pages)<br/>BeautifulSoup (HTML sections)"]
        NORM["Normalization + document typing<br/>addendum / affidavit / specification / rfp / bid_document"]
        CHUNK["Chunking<br/>1000 chars, 150 overlap, never across pages"]
        EMB["Embeddings<br/>FastEmbed BAAI/bge-small-en-v1.5 (384-d, local)"]
        DISC --> PARSE --> NORM --> CHUNK --> EMB
    end

    subgraph PG["PostgreSQL"]
        direction TB
        TBL[("documents, document_chunks<br/>+ metadata, page / section")]
        VEC["pgvector column<br/>HNSW cosine index"]
        FTS["tsvector column<br/>GIN full-text index"]
        TBL --- VEC
        TBL --- FTS
    end

    subgraph RET["Retrieval (app/retrieval, RetrievalAgent)"]
        direction TB
        QV["Vector search"]
        QK["Keyword search"]
        RRF["Reciprocal Rank Fusion (k=60)<br/>bid_id / metadata filters<br/>query expansion for weak results"]
        QV --> RRF
        QK --> RRF
    end

    subgraph ORCH["Orchestrator: plain Python, shared BidPipelineState<br/>stages in order: ingestion, retrieval, extraction, reconciliation, validation"]
        direction TB
        EXT["ExtractionAgent<br/>20 fields, 5 grouped calls"]
        REC["ReconciliationAgent<br/>addendum timeline + rules"]
        VAL["ValidationAgent<br/>re-checks citations in the database"]
        REP["Report"]
        QA["ReportQAAgent<br/>grounded Q&A"]
        EXT --> REC --> VAL --> REP
        REP -.-> QA
    end

    GROQ(["Groq API<br/>JSON mode, openai/gpt-oss-120b configured"])

    API["FastAPI /api/v1"]
    UI["React + Vite UI"]
    CLI["CLI<br/>python -m app.agents ..."]

    DOCS --> DISC
    EMB --> TBL
    VEC --> QV
    FTS --> QK
    RRF -->|"evidence chunks per field"| EXT
    RRF -->|"evidence for a question"| QA
    EXT <-->|"evidence in, JSON out"| GROQ
    REC <-.->|"only ambiguous wording"| GROQ
    QA <-->|"evidence in, cited answer out"| GROQ
    VAL -->|"chunk lookup"| TBL
    REP --> API
    QA --> API
    API --> UI
    CLI --> ORCH
    API --> ORCH
```

## Tech stack

| Area | What is used |
| --- | --- |
| Language / API | Python 3.11, FastAPI, uvicorn, Pydantic 2 / pydantic-settings |
| Database | PostgreSQL with the `pgvector` extension, SQLAlchemy 2 (async) with psycopg 3 |
| Parsing | PyMuPDF (PDF), BeautifulSoup 4 (HTML) |
| Embeddings | FastEmbed, `BAAI/bge-small-en-v1.5` (local ONNX model) |
| LLM | Groq SDK (JSON mode); model chosen by `GROQ_MODEL` (configured: `openai/gpt-oss-120b`) |
| Frontend | React 18, TypeScript, Vite 6, React Router, axios |
| Tests | pytest, pytest-asyncio, httpx |
| Packaging | Docker and Docker Compose (optional; `pgvector/pgvector:pg17`, backend, frontend) |

No agent framework (LangChain, LlamaIndex, AutoGen, CrewAI), no separate vector database and no reranker model are used.

## Setup

**Prerequisites:** Python 3.11+, Node.js 18+, and a PostgreSQL server with the `pgvector` extension available (PostgreSQL 17 in Docker and 18 locally were exercised). A Groq API key is needed only for extraction, reconciliation fallback and Q&A; ingestion, embeddings, search, deterministic reconciliation and the tests work without it.

**1. Database.** Create an empty database (the app creates the `vector` extension, tables and indexes on startup; the database role needs permission to `CREATE EXTENSION vector`):

```sql
CREATE DATABASE helix;
```

**2. Backend environment and dependencies.**

```bash
cd backend
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit backend/.env (never commit it)
```

Set at least these in `backend/.env` (the full list is in [Environment variables](#environment-variables)):

```
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/helix
GROQ_API_KEY=<your Groq key>          # leave empty to run without Groq
GROQ_MODEL=openai/gpt-oss-120b        # any Groq chat model that supports JSON mode
```

`DATA_DIR` (default `data`, relative to the repository root) holds one folder per bid; the repository ships `data/Bid1` and `data/Bid2`. Check the Groq setup with `python -m app.llm --smoke-test`.

**3. Frontend.**

```bash
cd frontend
npm install
```

**Docker alternative (optional).** `cp .env.example .env`, set `GROQ_*`, optionally `POSTGRES_PORT` if 5432 is taken, then `docker compose up --build`. The frontend is on http://localhost:5173 and the API on http://localhost:8000. This was validated from fresh volumes in Phase 9 (see the appendix).

### Environment variables

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy URL, e.g. `postgresql+psycopg://user:pass@host:5432/db`. Set automatically by Compose. |
| `GROQ_API_KEY` | Groq API key. Never committed or returned by the API. |
| `GROQ_MODEL` | Groq model id. Intentionally has no default; the client fails with a clear error if unset. |
| `DATA_DIR` | Folder holding the bid folders (default `data`, relative to the repo root; Compose mounts it at `/data`). |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunk length and overlap in characters (default `1000` / `150`). |
| `EMBEDDING_MODEL` | FastEmbed model name (default `BAAI/bge-small-en-v1.5`). |
| `EMBEDDING_BATCH_SIZE` | Chunks per embedding batch / commit (default `32`). |
| `VECTOR_SEARCH_LIMIT` / `KEYWORD_SEARCH_LIMIT` | Candidates fetched per method before fusion (default `20` / `20`). |
| `RRF_K` | RRF constant (default `60`). |
| `DEFAULT_TOP_K` / `MAX_TOP_K` | Results returned by default / allowed maximum (default `5` / `20`). |
| `KEYWORD_MIN_TERM_COVERAGE` | Share of meaningful query terms a keyword hit must contain (default `0.6`). |
| `EXTRACTION_EVIDENCE_PER_FIELD` / `EXTRACTION_ADDENDUM_EVIDENCE` | Chunks retrieved per field / extra addendum chunks (default `4` / `2`). |
| `EXTRACTION_MAX_CHUNKS_PER_CALL` / `EXTRACTION_CHUNK_CHAR_LIMIT` / `EXTRACTION_MAX_OUTPUT_TOKENS` | Context limits per Groq call (default `14` / `1200` / `3000`). |
| `RECONCILIATION_CHUNKS_PER_ADDENDUM` / `RECONCILIATION_LLM_MAX_CHUNKS` | Chunks examined per addendum per field / sent to Groq per ambiguous field (default `6` / `8`). |
| `APP_ENV` | `development` (default). |
| `LOG_LEVEL` | `INFO` (default). |
| `FRONTEND_ORIGIN` | Origin allowed by CORS (default `http://localhost:5173`). |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Compose-only database credentials (default `helix`). |
| `POSTGRES_PORT` | Compose-only host port for PostgreSQL (default `5432`; change it if that port is taken). |
| `MAX_INGEST_FILE_MB` | Files above this size are refused and reported during ingestion (default `50`). |
| `VITE_API_URL` | API base URL used by the browser (default `http://localhost:8000`). |

## Running the application

Backend (from `backend/`, with the virtual environment active):

```bash
python run_dev.py            # API on http://localhost:8000, docs on /docs
```

`run_dev.py` sets the Windows event loop that psycopg's async mode needs; plain `uvicorn app.main:app` fails to reach PostgreSQL on Windows (on macOS/Linux `uvicorn app.main:app --reload` also works).

Frontend (from `frontend/`):

```bash
npm run dev                  # http://localhost:5173
```

First run on an empty database (the UI has buttons for these, or use the API/CLI):

```bash
cd backend
python -m app.ingestion                # parse + chunk every document under DATA_DIR
python -m app.embeddings               # embed the chunks (downloads the ~67 MB model once)
python -m app.agents Bid1              # full pipeline for one bid
```

Tests (use a separate `helix_test` database on the same server): `cd backend && pytest -q`. Frontend build check: `cd frontend && npm run build`.

## API

All endpoints are under `/api/v1`. Interactive docs: `http://localhost:8000/docs`.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Database + pgvector check; 200 healthy, 503 otherwise. |
| GET | `/health/groq` | Groq configuration check (`?probe=true` makes a 1-token live request). |
| GET | `/health/status` | Per-component health: database, pgvector, embedding model, Groq, available features. |
| POST | `/ingestion/scan` | List supported files under `DATA_DIR`; stores nothing. |
| POST | `/ingestion/run` | Ingest new or changed documents. |
| GET | `/ingestion/documents` | List ingested documents (`?bid_id=`). |
| GET | `/ingestion/documents/{document_id}` | One document with page and chunk counts. |
| POST | `/embeddings/generate` | Embed every chunk that has no embedding. |
| GET | `/embeddings/status` | Embedded / pending / failed chunk counts, model, dimension. |
| POST | `/search` | Hybrid retrieval with filters; returns ranked chunks with citations (no generated answer). |
| GET | `/search/filters` | Bids, document types and file names available as filters. |
| POST | `/extraction/{bid_id}` | Structured extraction of the 20 fields via Groq (stateless). |
| POST | `/reconciliation/{bid_id}` | Addendum reconciliation (`?use_extraction=true` also runs extraction). Works without Groq for explicit wording. |
| POST | `/pipeline/{bid_id}` | The full workflow (`?ingest=false` skips ingestion): status, per-stage states, validation report, final report, `run_id`. |
| POST | `/qa/{bid_id}` | Body `{"question": "..."}`: cited answer or the standard refusal. |

Status codes: `404` unknown bid; `422` invalid input (bad bid id, empty question, nothing to analyse); `503` database, embedding model or (for Groq-only endpoints) Groq not configured; `502` Groq failure or unusable model output; `500` unexpected error or failed validation (generic body, no traceback); a `partial` pipeline result is still `200`. Request bodies over 64 KB get `413`. The per-endpoint details are in the appendix.

## CLI

Run from `backend/` with the virtual environment active.

| Command | Purpose | Needs Groq |
| --- | --- | --- |
| `python -m app.ingestion [--force]` | Ingest everything under `DATA_DIR`. | no |
| `python -m app.embeddings` | Embed chunks that lack an embedding. | no |
| `python -m app.retrieval [--no-filters] [--json FILE] [-v]` | Retrieval evaluation (Recall@1/3/5, MRR@5). | no |
| `python -m app.extraction Bid1 Bid2 [--json FILE]` | Extract with Groq and score against ground truth. | yes |
| `python -m app.extraction --evidence-only` | Is the expected text in the retrieved evidence? | no |
| `python -m app.reconciliation Bid1 [--json FILE] [--use-extraction]` | Addendum reconciliation. | no (`--use-extraction`: yes) |
| `python -m app.reconciliation --eval` | Score reconciliation against `eval/reconciliation_eval.json`. | no |
| `python -m app.reconciliation --llm-eval` | Groq fallback on controlled ambiguous cases. | yes |
| `python -m app.agents Bid1 [--no-ingest] [--json FILE] [--ask "..."]` | Full agent pipeline; `--ask` also answers a question. | extraction and `--ask`: yes |
| `python -m app.agents.qa_eval [--evidence-only] [-v] [--json FILE]` | Q&A evaluation (14 questions). | yes (`--evidence-only`: no) |
| `python -m app.agents.regression` | 25 behavioural regression cases on the real data. | no |
| `python -m app.agents.acceptance --from-json FILE --bid Bid1 [--require-extraction]` | Acceptance checks on a saved pipeline result. | no |
| `python -m app.agents.real_run --patient --bids Bid1 Bid2 --qa` | Full live run with the configured model, waiting out rate limits. | yes |
| `python -m app.agents.export_submission --extraction Bid1=FILE ...` | Writes `outputs/<Bid>.json` and the agent trace from a stored extraction (no Groq). | no |
| `python -m app.diagnostics` | 14 database / index checks. | no |
| `python -m app.llm --smoke-test` | Tiny Groq request to check the configuration. | yes |

## Pipeline modes

1. **Ingestion:** `python -m app.ingestion` or `POST /ingestion/run`.
2. **Embedding generation:** `python -m app.embeddings` or `POST /embeddings/generate`.
3. **Retrieval / search:** `POST /search` (body `{"query": "...", "bid_id": "Bid1"}`), or the Search page. Evaluate with `python -m app.retrieval`.
4. **Extraction:** `python -m app.extraction Bid1` or `POST /extraction/Bid1` (needs Groq).
5. **Reconciliation:** `python -m app.reconciliation Bid1` or `POST /reconciliation/Bid1` (deterministic without Groq).
6. **Full agent pipeline:** `python -m app.agents Bid1`, `POST /pipeline/Bid1`, or **Run Analysis** on the Analysis page. Without Groq the status is `partial` (extraction skipped; reconciliation, validation and report still run).
7. **Q&A:** `python -m app.agents Bid1 --ask "What is the final submission deadline?"`, `POST /qa/Bid1`, or the question box on the Analysis page.
8. **Evaluation, regression, acceptance:** `python -m app.retrieval`, `python -m app.agents.qa_eval`, `python -m app.reconciliation --eval`, `python -m app.agents.regression`, `python -m app.agents.acceptance ...`, `pytest -q`.

## Document ingestion design

```
discover -> hash -> parse (PyMuPDF / BeautifulSoup) -> normalise -> classify -> store pages -> chunk -> store chunks
```

- **Discovery:** each folder directly under `DATA_DIR` is a bid. Supported: `.pdf`, `.html`, `.htm`. Files above `MAX_INGEST_FILE_MB`, symlinks that resolve outside `DATA_DIR`, and PDFs over 2000 pages are refused and reported.
- **PDF parser:** PyMuPDF, text per page (1-based page numbers). Password-protected or unreadable files fail per file without stopping the run. Pages without extractable text stay empty and are dropped with a warning; a document with no text at all is rejected as "scanned or empty". No OCR.
- **HTML parser:** BeautifulSoup; removes script/style/navigation/footer noise and splits the page into sections at `<h1>`/`<h2>`. A section becomes a "page" with `page_number = NULL` and the heading in `metadata.section`. Tables are flattened to text.
- **Normalisation:** Unicode NFC, line-ending and whitespace cleanup, removal of control and zero-width characters, rejoining of words hyphenated across lines.
- **Document classification:** rule-based on the file name (`addendum`, `affidavit`, `specification`, `rfp`, else `bid_document`); addenda are additionally recognised later from an `ADDENDUM No. N` header in their content.
- **Idempotency:** a file is identified by its path under `DATA_DIR` plus the SHA-256 of its bytes. Unchanged and completed means skipped; changed, failed or interrupted means its old pages and chunks are replaced.
- **Tables:** `documents`, `document_pages`, `document_chunks` (page and chunk `metadata` is JSONB). Every chunk carries `bid_id`, `file_name`, `file_path`, `document_type`, `page_number`, `chunk_index` (plus `section` for HTML).

## Chunking strategy

- Text is split into paragraphs, then sentences (only for paragraphs that do not fit), then words (only for oversized sentences), and packed greedily up to `CHUNK_SIZE` (default 1000 characters). Each new chunk begins with the last `CHUNK_OVERLAP` (default 150) characters of the previous one, snapped to a word boundary.
- **Chunks never cross pages**, so each chunk has exactly one page number (PDF) or one section heading (HTML), which is what makes page-level citations exact. Chunk indexes are continuous across the document.
- A real corpus of 9 documents produced 285 chunks.

## Embedding model

- **Model:** `BAAI/bge-small-en-v1.5` through FastEmbed (local ONNX on CPU, about 67 MB, downloaded on first use; no embedding API or key). Output is **384-dimensional**; the dimension is read from the model metadata and re-checked when the model loads.
- **Storage:** `document_chunks.embedding vector(384)` in PostgreSQL (pgvector), with `embedding_model` and `embedding_error` columns. Only chunks with `embedding IS NULL` are embedded, so re-running is a no-op; failed chunks are retried individually on the next run.
- **Similarity metric:** cosine distance (`<=>`).
- **Index:** HNSW with `vector_cosine_ops` (`ix_document_chunks_embedding_hnsw`). At this corpus size (285 rows) the planner prefers a sequential scan; the index exists and was verified usable (`python -m app.diagnostics`).

## Retrieval strategy

- **Vector search:** the query is embedded with the same model and ranked by cosine distance in pgvector.
- **Keyword search:** PostgreSQL full-text search over a generated `tsvector` column (GIN index). A hit must contain a minimum share of the query's meaningful terms (`KEYWORD_MIN_TERM_COVERAGE`, default 0.6), so one common word cannot match everything. Exact identifiers (bid numbers, model numbers) are found this way.
- **Hybrid:** both result lists are fused with **Reciprocal Rank Fusion**, `score = sum 1/(k + rank)`, `k = 60` (`RRF_K`). Each result records which method(s) found it.
- **Metadata filtering:** `bid_id`, `document_type`, `file_name`, `document_id`, `page_number`, applied inside both searches. The agents always filter by `bid_id`, and the code additionally drops (and counts) any chunk of another bid.
- **Query expansion:** when a question is generic ("What product is being procured?") and neither method confirms any result, one deterministic, solicitation-phrased variant query is also run. Evidence for Q&A improved from 10/11 to 11/11 with this.
- **Citations:** every result carries file, document type, page (or HTML section), chunk index and chunk id.
- **Reranking: not implemented.** There is no cross-encoder or LLM reranker; ranking is RRF only.
- **Evaluation:** see [Retrieval evaluation](#retrieval-evaluation).

## Extraction

```
bid -> per-field queries -> hybrid retrieval (bid-scoped) -> evidence per field -> Groq (JSON mode) -> Pydantic -> citation + grounding checks -> conflicts -> JSON
```

- **Field schema:** the 20 assessment fields are defined in `backend/app/extraction/fields.py` (name, description, retrieval queries, kind: `scalar`, `list`, `text` or `spec`). Each `ExtractedField` has `value`, `sources`, `confidence`, `notes`, `status` (`found`, `not_found`, `conflict`, `unsupported`, `error`) and `conflicting_values`; a Pydantic validator enforces the invariants (a value needs a source; no value means confidence 0 and no sources).
- **Evidence per field:** each field has its own queries (amendable fields also fetch addendum chunks). Fields are sent to Groq in **5 grouped calls** (identity, schedule, requirements, product, summary) with only the deduplicated evidence for those fields.
- **Structured output:** Groq JSON mode at temperature 0, parsed and validated against Pydantic models; one bounded retry feeds the validation error back to the model (a provider-rejected JSON is retried the same way), and one bounded follow-up asks for missing citations.
- **Citation handling:** the model returns only chunk ids. The backend rebuilds `sources` (file, page, type, chunk index) from the stored chunks and drops any id that was not in the evidence sent for that call, so a model cannot cite a chunk it was not shown or invent a page.
- **Grounding:** every number in a value must occur in its cited chunks, and at least 60% of its words must too. A value that fails is not returned (`unsupported`).
- **Confidence:** `min(model estimate, 0.95 x grounding ratio)`, minus 0.2 if a citation had to be dropped; summaries are capped at 0.8. It is a heuristic, not a calibrated probability.
- **Missing values:** nothing supported means `value: null`, `sources: []`, `confidence: 0.0` and a "not found" note; the model is never forced to fill a field.
- **Conflicts:** for single-value fields, different supported values become `status: "conflict"` with every candidate and its citations; extraction never picks a winner (that is reconciliation's job). The same value in a different notation counts once. List fields (bid number, product, model, part, installation, submission type, documentation, contact) hold several values together without conflict.
- **Product Specification:** one candidate per product line, each cited and grounded on its own chunks. One product returns a single `{product, specifications}` object; several return a list of them (at most 8).

## Addendum reconciliation

- **Detection:** documents of type `addendum`, plus any document whose first page starts with an `ADDENDUM No. N` header. Numbers and issue dates come from the document content first and the file name only as a fallback.
- **Ordering:** explicit numbers, else unique dates, else file-name numbers (flagged unreliable), else unordered with a note. Changes are applied in sequence only when the order is established.
- **Explicit change detection (deterministic):** rules parse wording such as "The new due date ... will be X", "changed / extended to X", "replace X with Y", "deleted / waived / no longer required", "must also submit X". A sentence must name the field **and** use change wording; "please refer to the original" does not count.
- **Original values:** deterministic locators read the RFP's `Solicitation Due` and `Pre-Proposal Meeting`; other fields take the original from extraction (`--use-extraction`).
- **Output per field:** `original_value` + `original_sources`, every change (addendum, type, previous and new value, quoted sentence, sources, `rule` or `llm`), and the final value + sources. Originals are preserved even when replaced.
- **Conflict handling:** contradictory originals with no superseding addendum, addenda that disagree when their order is unknown, or an addendum contradicting itself yield `conflict` with `final_value: null`; uninterpretable evidence yields `insufficient_evidence`. Both are listed as unresolved, never guessed.
- **Groq fallback:** only for ambiguous wording, and only if Groq is configured. Cited chunks must be addendum chunks that were supplied, the proposed value must be contained in the cited text, and a model answer of "none" cannot clear explicit change wording.
- **Real example (Bid1):** original due date `27-JUN-2024 14:00:00` (RFP, page 2) is replaced by `July 9, 2024 at 2:00 PM CST` (Addendum 2, page 1); status `resolved`, method `explicit_addendum_change`, confidence 0.95. Bid2 has no addenda, so nothing is changed.

## Multi-agent system

The agents are thin classes in `backend/app/agents/`; the heavy logic stays in services they wrap.

| Agent (class, file) | Responsibility | Input | Output | Consumer |
| --- | --- | --- | --- | --- |
| `Orchestrator` (`orchestrator.py`) | Runs the stages in order, classifies failures (fatal vs recoverable), sets the status, assigns a `run_id`. | `bid_id` | `BidPipelineState` / `PipelineResponse` | API, CLI |
| `IngestionAgent` (`ingestion_agent.py`) | Ingest the bid's files and embed new chunks; skip unchanged files. | `bid_id` | `IngestionResult` (documents, counts) | Retrieval agent |
| `RetrievalAgent` (`retrieval_agent.py`) | Gather bid-scoped evidence per field via hybrid search; discard foreign-bid chunks. | `bid_id` | Evidence per field + `RetrievalStageResult` | Extraction agent, Q&A agent |
| `ExtractionAgent` (`extraction_agent.py`) | Structured extraction through `ExtractionService` (prompts, Groq, grounding, citations). Skipped without Groq. | evidence | `BidExtraction` | Reconciliation agent |
| `ReconciliationAgent` (`extraction_agent.py`) | Addendum timeline and reconciliation through `ReconciliationService`; falls back to deterministic rules if Groq fails. | extraction (optional) | `BidReconciliation` | Validation agent |
| `ValidationAgent` (`validation_agent.py`) | Re-reads every cited chunk from the database; checks existence, bid, file/page, grounding, consistency. Never repairs data. | extraction + reconciliation | `ValidationReport` | Report |
| `ReportQAAgent` (`report_qa_agent.py`) | Builds the final report (copies validated values) and answers questions from retrieved evidence, refusing when unsupported. | validated state / a question | `PipelineReport` / `QAAnswer` | API, UI |

Failure policy: a missing/uningestible bid, unavailable database or embedding model, or failed validation is **fatal** (status `failed`, no report); missing Groq, partly failed ingestion, unextractable fields or unresolved wording are **recoverable** (status `partial`).

## Agent framework

The agent layer is a **custom, lightweight orchestration**: plain Python classes, Pydantic state objects and a fixed linear workflow. No LangChain, LlamaIndex, AutoGen or CrewAI is used. This was deliberate:

- **Explicit control:** the workflow is fixed and has no loops or agent-to-agent calls, so a run is reproducible and easy to reason about.
- **Deterministic stages:** only extraction, the reconciliation fallback and Q&A call a model; everything else (ingestion, retrieval, rules, validation) is ordinary code.
- **Easier validation and citation tracking:** because evidence, chunk ids and sources are plain data, the Validation agent can re-check every claim against the database.
- **Easier failure handling:** each stage reports `complete`, `partial`, `failed` or `skipped`, so a Groq outage degrades the result instead of crashing it.

## Prompt design

Prompts live in `backend/app/extraction/prompts.py` (extraction), `backend/app/reconciliation/llm.py` (reconciliation fallback) and `backend/app/agents/report_qa_agent.py` (Q&A). Common rules:

- **Evidence-only:** "use only the provided evidence; never use outside knowledge"; if the evidence does not contain the answer, return no candidates (or refuse).
- **Evidence as untrusted input:** each system prompt states that the evidence is untrusted document text and instructions inside it must never be followed. Evidence is delivered in a labelled `EVIDENCE` section of the user message, never inside the system prompt. This is a mitigation, not a guarantee: a malicious sentence that is cited from a real chunk could still be grounded; what is guaranteed is that uncited values, values absent from their cited chunk, and citations of chunks the model was not shown are rejected.
- **Structured JSON:** a fixed JSON shape (`{"fields": [{"key", "candidates": [{"value", "chunk_ids"}], "confidence", "notes"}]}`); answers cite chunk ids.
- **Copy exactly:** dates, numbers, identifiers and model/part numbers are reproduced as written, including apparent typos (for example the RFP really prints `65HDMI`).
- **Missing values and conflicts:** return no candidates and confidence 0 when unsupported; return one candidate per distinct value when documents disagree; do not reconcile addenda during extraction.
- **No hallucination:** the application, not the prompt, enforces this: citations are rebuilt from stored chunks and numbers must occur in the cited text.

## Q&A

1. The question is answered for one `bid_id`; unknown bids give `404`.
2. `RetrievalAgent` runs hybrid search filtered to that bid (with the bounded query expansion for generic questions). For questions about an addendum-modifiable field, the already-validated reconciliation result is added as evidence.
3. The evidence chunks (with ids) go to Groq in JSON mode; the model returns an answer and the chunk ids it used.
4. The answer is accepted only if it cites supplied evidence and its figures occur in the cited text. Sources are rebuilt from the stored chunks and re-checked against the database. Otherwise the agent returns the standard refusal: "I could not find sufficient evidence to answer this question."
5. **Bid isolation:** evidence is bid-filtered at the query and re-checked in code; an evaluation case asks Bid1 about the Mercury Affidavit (which exists only in Bid2) and expects a refusal.
6. Confidence is the same heuristic as in extraction.

## Evaluation results

Clear separation between kinds of evidence:

| Kind | What | Result | Model |
| --- | --- | --- | --- |
| Automated tests | `cd backend && pytest -q` | **417 passed, 0 skipped** | none (scripted models) |
| Deterministic evaluation | Retrieval, 19 questions (`python -m app.retrieval`) | Hybrid with filters: Recall@1/3/5 = 0.684 / 0.895 / 1.000, MRR@5 0.775; without filters 0.526 / 0.737 / 0.842, MRR@5 0.632. Full table in [Retrieval evaluation](#retrieval-evaluation) and [`docs/retrieval_evaluation.md`](docs/retrieval_evaluation.md) | none |
| Deterministic evaluation | Reconciliation, 4 hand-checked real-data cases (`python -m app.reconciliation --eval`) | resolution 1/1, unchanged-stays-unchanged 1/1, original preserved 2/2, citations 2/2, timeline order 2/2 (very small set; see the appendix) | none |
| Deterministic evaluation | Regression cases (`python -m app.agents.regression`) | **25/25** hold on the real Bid1/Bid2 data | none |
| Deterministic evaluation | Acceptance, Bid1 and Bid2 (`app/agents/acceptance.py`, run in the test suite without Groq) | pass | none |
| Live Groq | Extraction on real documents (Phase 8, before the latest extraction changes) | Bid1 7/8 and Bid2 16/16 gold fields, 0 validation errors | `openai/gpt-oss-120b` |
| Live Groq | Reconciliation fallback, 6 controlled ambiguous cases | 6/6 in two complete runs | `openai/gpt-oss-120b` |
| Live Groq | Q&A, 14 questions | 10/10 answers correct, 3/3 refusals, 0 cross-bid citations, 10/10 grounded; 1 safe refusal on the Bid2 deadline | `openai/gpt-oss-20b` (**not** the configured model) |
| **Not validated live** | Latest Bid Number scoping and multi-product Product Specification | deterministic tests only | **no live Groq run** |
| **Not completed** | Full end-to-end 120b run for Bid1/Bid2 and the 120b Q&A subset | blocked by the Groq free-tier daily token limit | |

To finish the missing live validation later: `cd backend && python -m app.agents.real_run --patient --bids Bid1 Bid2 --qa --out eval/runs/final`, then `python -m app.agents.acceptance --from-json eval/runs/final/pipeline_Bid1.json --bid Bid1 --require-extraction` (and Bid2).

### Retrieval evaluation

19 questions over the 9 real documents (8 Bid1, 7 Bid2, 4 exact-term). A chunk is relevant when it comes from the expected file **and** its text matches the expected evidence pattern, so this is chunk-level. Recall@K is the share of questions with at least one relevant chunk in the top K; MRR@5 is the mean reciprocal rank of the first relevant chunk within the top 5. Numbers below were re-run for the submission and equal the Phase 4 results.

With metadata filters (`bid_id` for the Bid1/Bid2 questions):

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| Vector only | 0.632 | 0.895 | 1.000 | 0.769 |
| Keyword only | 0.579 | 0.684 | 0.684 | 0.614 |
| **Hybrid (RRF)** | 0.684 | 0.895 | 1.000 | 0.775 |

Without filters (every query searches all 285 chunks):

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| Vector only | 0.474 | 0.632 | 0.789 | 0.578 |
| Keyword only | 0.526 | 0.684 | 0.684 | 0.579 |
| **Hybrid (RRF)** | 0.526 | 0.737 | 0.842 | 0.632 |

Hybrid matches or beats the better single method at every cutoff, with the clearest gain in the harder unfiltered setting. With only 19 questions one question is 0.053, so small differences should not be over-read. The full report with the question set, relevance labels and per-question ranks is [`docs/retrieval_evaluation.md`](docs/retrieval_evaluation.md); raw results are in `outputs/retrieval_eval_*.json`.

## Output artifacts

| Deliverable | Location |
| --- | --- |
| Per-bid JSON output (values, confidence, citations, reconciliation, validation) | [`outputs/Bid1.json`](outputs/Bid1.json), [`outputs/Bid2.json`](outputs/Bid2.json) |
| Stored real extraction runs these were built from, and their run logs | `outputs/stored_extraction/` |
| Retrieval evaluation report | [`docs/retrieval_evaluation.md`](docs/retrieval_evaluation.md), raw: `outputs/retrieval_eval_filters.json`, `outputs/retrieval_eval_nofilters.json` |
| Sample Q&A log (14 questions) | [`docs/sample_qa.md`](docs/sample_qa.md), raw: `outputs/qa_eval_gpt-oss-20b.json` |
| Agent trace (one complete run) | [`docs/agent_trace.md`](docs/agent_trace.md) |
| Architecture diagram | [`docs/architecture.md`](docs/architecture.md), [`docs/architecture.mmd`](docs/architecture.mmd) |
| Evaluation sets and regression cases | `backend/eval/` |

**Provenance of `outputs/Bid*.json` and the trace (read before relying on them).** They were produced by `python -m app.agents.export_submission`, which runs the real pipeline against the real database but **replays the stored extraction** from an earlier live `openai/gpt-oss-120b` run instead of calling Groq. Ingestion check, retrieval, reconciliation (deterministic rules only), validation and the report were executed live. That stored extraction predates the Bid Number and multi-product changes, so `Bid Number` still lists three identifiers for Bid1 and `Product Specification` is a single object there. Bid2's status is `partial` because its Due Date extraction conflict (several differently written dates, no addenda) is reported as unresolved rather than guessed. The Q&A log is from `gpt-oss-20b`.

## Limitations

- The latest extraction changes and a full end-to-end 120b run are not validated against live Groq (see above); the Q&A numbers are from `gpt-oss-20b`.
- Extraction output is non-deterministic across runs; in the best real Bid1 run one gold field (of 8) was still wrong.
- The evaluation sets are small (2 bids; 19 retrieval questions; 14 Q&A questions) and were checked by hand against the same documents the system uses, so they say nothing about other bid layouts.
- No OCR (scanned PDFs are not read) and no table-structure extraction; HTML tables are flattened to text.
- No reranker; confidence is a heuristic.
- Prompt-injection protection is a prompt rule plus grounding checks, not a classifier.
- No authentication, per-client rate limiting or migrations (tables are created at startup); not a multi-tenant service.
- The Groq free tier limits real-model runs to roughly 200k tokens per day.

## Appendix: detailed phase notes

The sections below are the original phase-by-phase engineering notes, kept for their detail (design reasons, measured results, real-run findings). Where they differ from the sections above, the sections above describe the current state.


### Ingestion pipeline (Phase 2)

```
discover -> hash -> parse (PyMuPDF / BeautifulSoup) -> normalize -> classify -> store pages -> chunk -> store chunks
```

- Any folder directly under `data/` is a bid (`Bid1`, `Bid2`, `Bid3`, ...). Supported: `.pdf`, `.html`, `.htm`; everything else is ignored.
- Tables: `documents`, `document_pages` (a PDF page, or an HTML section with `page_number` NULL), `document_chunks`. Page and chunk `metadata` is JSONB.
- Every chunk carries `bid_id`, `file_name`, `file_path`, `document_type`, `page_number`, `chunk_index` (plus `section` for HTML).
- Idempotent: a file is identified by its path under `data/` and a SHA-256 of its bytes. Unchanged + completed -> skipped. Changed, failed or interrupted -> old pages/chunks are replaced.
- Document type is rule-based on the file name: `addendum`, `affidavit`, `specification`, `rfp`, else `bid_document`.

```bash
cd backend && python -m app.ingestion
```

### Phase 3 - Embeddings

Chunks are embedded locally and stored in PostgreSQL.

- **Why FastEmbed:** it runs ONNX models on CPU inside the backend process, so there is no external embedding API, no key and no per-call cost. The model is loaded once per process and texts are embedded in batches.
- **Model:** `BAAI/bge-small-en-v1.5` (`EMBEDDING_MODEL`), a small, strong English retrieval model, about 67 MB. Its output is **384-dimensional**. The dimension comes from FastEmbed's model metadata, and the real output size is checked when the model loads, so a mismatch stops the run instead of writing bad vectors.
- **Why pgvector:** vectors live next to the chunks they describe (same transaction, same backups, joins and metadata filters in SQL), so there is no second datastore to keep in sync.
- **Cosine similarity:** BGE embeddings are compared by angle, not magnitude, and cosine is the standard metric for semantic text search. The index uses `vector_cosine_ops`, so Phase 4 queries should use the `<=>` operator.
- **Index:** `HNSW` (`ix_document_chunks_embedding_hnsw`). Unlike IVFFlat it needs no training data or minimum row count and stays valid as chunks are added. On a corpus this small (hundreds of chunks) Postgres will often choose an exact sequential scan instead, which is fine; the index matters as data grows.
- **Columns added to `document_chunks`:** `embedding vector(384)`, `embedding_model`, `embedding_error`. They are added automatically at startup, including to databases created by Phase 2.
- **Idempotent:** only rows with `embedding IS NULL` are embedded, so a second run skips everything. Re-ingesting a changed document replaces its chunks with new rows that start with no embedding, so stale vectors never stay attached to new text. If `EMBEDDING_MODEL` changes, vectors from the old model are cleared at startup and regenerated.
- **Failures:** if a batch fails it is retried chunk by chunk, so one bad chunk does not block the rest. Failed chunks keep `embedding IS NULL`, record `embedding_error`, are counted as `failed`, and are retried on the next run.

Generate embeddings (after ingestion):

```bash
cd backend && python -m app.embeddings          # CLI; exits non-zero if any chunk failed
curl -X POST http://localhost:8000/api/v1/embeddings/generate
curl http://localhost:8000/api/v1/embeddings/status
```

The first run downloads the model (set `FASTEMBED_CACHE_PATH` to control where it is cached). The Embeddings page in the UI shows the same status and has a Generate Embeddings button.

### Phase 4 - Hybrid retrieval

```
query → preprocess → ┬─ query embedding (same model, BGE query prefix) → pgvector cosine search ─┐
                     └─ keyword terms → PostgreSQL full-text search (tsvector + GIN) ────────────┴→ RRF → top-K
```

- **Why vector search:** it matches meaning, so "final submission deadline" finds a chunk that says "the new due date". It struggles with identifiers.
- **Why keyword search:** procurement text is full of exact tokens (`JA-207652`, `BPM044557`, `Dell Latitude 5550`, `WD22TB4`, dates) that embeddings blur together and that a user expects to match literally.
- **Query embedding:** the same `BAAI/bge-small-en-v1.5` model as the chunks. BGE is trained so that a *query* carries the instruction `Represent this sentence for searching relevant passages: `; chunks are embedded without it (Phase 3 stores them that way). 384 dimensions, cosine distance (`<=>`).
- **Keyword search:** `document_chunks.search_vector` is a generated `tsvector` (`to_tsvector('english', content)`), so it stays in sync with the text automatically and re-ingested chunks are searchable immediately. It is indexed with **GIN** (`ix_document_chunks_search_vector`), the standard index for `@@` lookups. Each query term is parsed with `websearch_to_tsquery` so identifiers are tokenised exactly as in the index. A chunk must contain at least `KEYWORD_MIN_TERM_COVERAGE` (default 0.6) of the query's meaningful terms (after PostgreSQL drops stop-words like "what", "is", "for"); results are ordered by terms matched, then `ts_rank_cd`. Returning nothing is a normal outcome.
- **RRF:** `RRF(d) = Σ 1 / (RRF_K + rank_in_list(d))` over the vector and keyword lists (`RRF_K=60`). Only ranks are used, so cosine distance and `ts_rank` never need normalising. A chunk found by both methods adds both terms. **The RRF score is a ranking score, not a probability or confidence**; with k=60 the maximum for two lists is 2/61 ≈ 0.033. Each chunk id appears at most once.
- **Scores stay separate:** every result carries `scores.rrf`, `scores.vector` (rank, cosine distance, cosine similarity) and `scores.keyword` (rank, `ts_rank`). `score` at the top level is the RRF score.
- **Metadata filters:** `bid_id`, `document_type`, `file_name`, `document_id`, `page_number` are applied inside both SQL queries (`WHERE`), not in Python. With filters the vector query enables pgvector's `hnsw.iterative_scan` (0.8+) so a filtered HNSW scan still returns enough rows. Unknown `bid_id` / `document_type` values are rejected with `422`.
- **Citations:** every result has `source` = `{document_id, bid_id, file_name, document_type, page_number, chunk_index, metadata}`. PDF chunks have a real `page_number`; HTML chunks have `page_number: null` and `metadata.section` (page numbers are never invented).
- **Bounds:** `VECTOR_SEARCH_LIMIT` / `KEYWORD_SEARCH_LIMIT` (20) candidates per method, `DEFAULT_TOP_K` 5, `MAX_TOP_K` 20, queries up to 500 characters. All work happens in PostgreSQL (`ORDER BY embedding <=> $1 LIMIT n` and `@@` over the GIN index); nothing is scored in Python, so the design does not depend on the corpus being small.

```bash
curl -X POST http://localhost:8000/api/v1/search -H "Content-Type: application/json" \
  -d '{"query": "What is the final submission deadline for Bid1?", "bid_id": "Bid1", "top_k": 5}'
```

**Example** (real documents, real run). Query: `What is the final submission deadline for Bid1?` with `bid_id=Bid1`:

| # | Source | Page | Chunk | Retrieved by | RRF |
| - | ------ | ---- | ----- | ------------ | --- |
| 1 | `Student and Staff Computing Devices __SOURCING #168884__ ... BidNet Direct.html` | - (section) | 4 | vector | 0.0164 |
| 2 | `JA-207652 Student and Staff Computing Devices FINAL.pdf` | 32 | 146 | vector | 0.0161 |
| 3 | `Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf` | 1 | 0 | vector | 0.0159 |
| 4 | `JA-207652 ... FINAL.pdf` | 9 | 27 | vector | 0.0156 |
| 5 | `JA-207652 ... FINAL.pdf` | 2 | 1 | vector | 0.0154 |

Result 3 is Addendum 2 (type `addendum`, page 1) and contains: *"The Purpose of this Addendum is to extend the due date of this RFP. The new due date for this RFP will be July 9, 2024 at 2:00 PM CST."* The original RFP's schedule (result 5, page 2) lists "Solicitation Due 27-JUN-2024 14:00:00". Here the keyword search returned nothing (no chunk contains enough of "final / submission / deadline / bid1"), so every result is vector-only. Deciding which date is *final* is reconciliation, which is a later phase. Retrieval only has to surface the Addendum 2 evidence.

#### Retrieval evaluation

19 questions over the 9 real documents (`backend/eval/retrieval_eval.json`: 8 Bid1, 7 Bid2, 4 exact-term). A retrieved chunk is relevant if it comes from the expected file *and* its text matches the expected evidence pattern, so this is chunk-level, not just document-level. Recall@K is the share of questions with at least one relevant chunk in the top K. Reproduce with `cd backend && python -m app.retrieval [--no-filters] [-v]`.

With metadata filters from the dataset (`bid_id` for the Bid1/Bid2 questions, none for exact-term queries):

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| ------ | -------- | -------- | -------- | ----- |
| Vector only | 0.632 | 0.895 | 1.000 | 0.769 |
| Keyword only | 0.579 | 0.684 | 0.684 | 0.614 |
| **Hybrid (RRF)** | 0.684 | 0.895 | 1.000 | 0.775 |

Without any filters (every query searches all 285 chunks):

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| ------ | -------- | -------- | -------- | ----- |
| Vector only | 0.474 | 0.632 | 0.789 | 0.578 |
| Keyword only | 0.526 | 0.684 | 0.684 | 0.579 |
| **Hybrid (RRF)** | 0.526 | 0.737 | 0.842 | 0.632 |

What this does and does not show:

- Hybrid is at least as good as the better single method at every cutoff in both tables (it ties at several), and its clearest gain is in the harder unfiltered setting (Recall@3 0.737 vs 0.684, Recall@5 0.842 vs 0.789 for the best single method). With filters the candidate pool per query is so small that vector search already reaches Recall@5 1.000, so hybrid only matches it there (its Recall@1 and MRR edge is a one-question difference).
- With only 19 questions, one question is 0.053, so these differences are small and should not be over-read.
- Keyword-only is strong on exact identifiers (all four exact-term queries rank 1) and weak on paraphrased questions, which is the reason for combining them.
- Honest development note: my first keyword search matched *any* query term. Hybrid then scored Recall@5 0.842 against 1.000 for vector alone, because words like "required" matched nearly every chunk and crowded out correct vector hits (including Addendum 2). I fixed it by requiring term coverage (above). The questions were used to find that problem, and there is no separate held-out set.
- Unfiltered hybrid still misses 3 of 19 within the top 5: `What was the original submission deadline?` (original RFP schedule chunk), `How much RAM is required?` (the spec says `16 GB: 2 x 8 GB, DDR5`, no word "RAM"), and `What is the delivery requirement?` (`Delivery within 45 days of Award`). These are paraphrase gaps neither method bridges; a reranker or query expansion would be the next step, but is out of scope here.
- Phase 2 bug found by this work: the BidNet HTML pages had been ingested as navigation only, because the parser removed every `<form>` and BidNet wraps the whole page in one. Fixed (`python -m app.ingestion --force`), after which the closing date, contacts and description became searchable. The corpus is now 285 chunks.

### Phase 5 - Structured bid extraction

```
bid_id → field queries → hybrid retrieval (Phase 4) → evidence per field → Groq (JSON) → Pydantic → citation + grounding checks → conflicts → JSON
```

`POST /api/v1/extraction/{bid_id}` returns the 20 assessment fields (`Bid Number`, `Title`, `Due Date`, ..., `Product Specification`, `Bid Summary`). Each field is:

```json
{"value": "...", "sources": [{"chunk_id": 302, "file_name": "...", "document_type": "addendum", "page_number": 1, "chunk_index": 0, "section": null}],
 "confidence": 0.9, "notes": null, "status": "found", "conflicting_values": []}
```

`status` is `found`, `not_found`, `conflict`, `unsupported` (the model proposed something the evidence did not support) or `error` (the LLM output stayed invalid after one retry).

- **Why field-specific retrieval:** one generic "extract everything" query would return the same few chunks for every field. Each field has its own retrieval queries (`app/extraction/fields.py`), run through the existing `RetrievalService` scoped to the bid. Fields that amendments can change (due date, pre-bid meeting, delivery, term, submission, ...) also run a second query restricted to `document_type=addendum`, so amendments are in the evidence even when the original RFP outranks them. That only gathers evidence; it decides nothing.
- **How Groq is used:** through the existing `GroqClient` (model from `GROQ_MODEL`, nothing hardcoded; a new `generate_json` helper requests Groq's JSON mode at temperature 0). Fields are sent in 5 grouped calls (identity, schedule, requirements, product, summary), not 20, and only the deduplicated evidence for those fields is included. Limits: `EXTRACTION_EVIDENCE_PER_FIELD` (4) + `EXTRACTION_ADDENDUM_EVIDENCE` (2) chunks per field, at most `EXTRACTION_MAX_CHUNKS_PER_CALL` (14) chunks per call chosen round-robin by rank, each cut to `EXTRACTION_CHUNK_CHAR_LIMIT` (1200) characters. The system prompt forbids outside knowledge and guessing and forbids reconciling addenda.
- **Structured output:** JSON mode works on every Groq model, whereas strict JSON-schema output does not, so the reply is parsed and validated against Pydantic models (`LLMGroupOutput`); the final `ExtractedField` model enforces invariants (a value needs a source, no value means confidence 0 and no sources, a conflict needs two values). One bounded retry feeds the validation error back to the model; after that the group's fields are returned with `status: "error"` and `complete: false`, never partial guesses. Groq rate limits are waited out once; other provider errors give `502`.
- **Citation validation:** the model only returns chunk ids. The backend rebuilds `sources` (file, page, type, chunk index) from the stored chunk, so it cannot invent a file name or page. `validate_citations` drops any id that was not in the evidence sent for *that field* (wrong chunk, other bid, made up). If no valid citation remains the field becomes `unsupported`.
- **Grounding:** every number in a value must occur in the cited chunks (leading zeros ignored), and at least 60% of its words must too (prefix match, so `June` matches `JUN`). So a model cannot "calculate" `2:00 PM` from `14:00` or invent `5% bid bond`. The summary can only be checked for numbers.
- **Missing values:** nothing supported means `value: null`, `sources: []`, `confidence: 0.0`, note "Not found in the retrieved evidence".
- **Conflicts:** for single-value fields, distinct supported values become `status: "conflict"`: `value` stays `null`, `confidence` 0, `conflicting_values` lists every value with its own sources, and the field is listed in `reconciliation_required`. The same value in different wording counts once. Nothing picks a winner (Phase 5 never prefers the "latest" document).
- **Why reconciliation is deferred:** deciding that Addendum 2 supersedes the RFP due date is a reasoning task with its own rules (which addendum changes what, precedence, partial amendments). Phase 6 will build it on top of these conflict records and their citations.
- **Confidence:** `min(model estimate, 0.95 x grounding ratio)`, minus 0.2 if a citation had to be dropped (summary capped at 0.8). It is an extraction-quality heuristic, **not** a calibrated probability: it has not been validated against outcomes.
- **Stateless by design:** results are not stored. They are cheap to recompute, would go stale whenever documents are re-ingested, and a snapshot table would add a schema and invalidation rules without a requirement for it. The UI re-runs extraction on demand.

Run it (needs `GROQ_API_KEY` and `GROQ_MODEL` in `backend/.env`):

```bash
cd backend
python -m app.extraction --evidence-only      # no LLM: is the expected text in the retrieved evidence?
python -m app.extraction Bid1 Bid2            # extract with Groq and score against ground truth
curl -X POST http://localhost:8000/api/v1/extraction/Bid1
```

#### Extraction evaluation

Ground truth (`backend/eval/extraction_eval.json`) was checked by hand against the ingested text of the real documents: 8 scored Bid1 checks and 14 scored Bid2 checks, plus unscored info checks and two "must be null" checks (Bid2 has no pre-bid meeting or bid bond).

- **Evidence coverage (measured, no LLM):** for 22 of 22 scored checks the evidence retrieved for the right field contains the expected text (Bid1 8/8, Bid2 14/14). That includes the Bid1 due date, where the evidence holds both `27-JUN-2024 14:00:00` (RFP, page 2) and `The new due date for this RFP will be July 9, 2024 at 2:00 PM CST` (Addendum 2, page 1), plus the BidNet page's `Closing Date 07/09/2024 03:00 PM EDT`. One Bid2 check (quotes valid 90 days) only passed after I added a second "offer validity" query to the Term of Bid field. The Contract Affidavit info check is not in the evidence (the PORFP never lists it, it is just a separate file in the bid package), so a correct extraction should not claim it as required.
- **End-to-end extraction with Groq: not yet measured.** `GROQ_API_KEY` and `GROQ_MODEL` are empty in this environment, so no real extraction has been run and no extraction accuracy is claimed. Retrieval being able to supply the evidence says nothing about whether the model reads it correctly; run `python -m app.extraction Bid1 Bid2` once Groq is configured to get that number.
- Format example of a conflict (from the test suite with a scripted model, **not** a Groq result):

```json
"Due Date": {"value": null, "confidence": 0.0, "status": "conflict",
  "notes": "Conflicting values found in the retrieved documents; addendum reconciliation is required.",
  "conflicting_values": [
    {"value": "27-JUN-2024 14:00:00", "sources": [{"file_name": "JA-207652 FINAL.pdf", "document_type": "bid_document", "page_number": 2}]},
    {"value": "July 9, 2024 at 2:00 PM CST", "sources": [{"file_name": "Addendum 2 RFP JA-207652.pdf", "document_type": "addendum", "page_number": 1}]}]}
```

### Phase 6 - Addendum reconciliation

Phase 5 reports that two documents disagree (RFP due date vs Addendum 2) but deliberately picks no winner. Phase 6 decides which value is authoritative, using what the addenda *explicitly say*, and keeps the whole history. It is one focused service (`app/reconciliation/`); Phase 7 will introduce the multi-agent orchestration layer, and this service is what the Addendum agent will wrap.

```
bid → addendum timeline → per field: original value + addendum evidence → change-language rules → (Groq only if ambiguous) → ordered application → ReconciledField
```

- **Why it is needed:** a requirement stated in the RFP can be replaced, clarified, extended or removed by any addendum. Taking the first or the "most relevant" chunk gives the wrong answer for exactly the fields bidders are disqualified on (dates, bonds, deliverables).
- **How addenda are identified:** documents with `document_type = addendum`, and also any document whose first page starts with an `ADDENDUM No. N` header (so a mis-named file is still found). The number comes from the document's own header first (`Addendum 1`, `ADDENDUM NO. 1`, `ADDENDUM #2`, `Addendum Two`) and the file name only as a fallback. Dates are read only when written next to an issue-date label; the blank `Date:` signature lines in the real addenda yield `null`. Nothing is invented: an unknown number stays `null`.
- **How they are ordered:** explicit numbers (from content), else unique dates, else numbers from file names (flagged `reliable: false`), else left unordered with a note. Dates that contradict the numbers also mark the timeline unreliable. Multiple conflicting changes are only applied in sequence when the order is established.
- **Deterministic reconciliation first:** explicit wording is parsed with rules, no LLM: *"The new due date ... will be X"*, *"changed/extended/revised ... to X"*, *"changed from X to Y"*, *"replace X with Y"*, *"X is replaced by Y"*, *"instead of / in lieu of"*, *"deleted / waived / no longer required"*, *"must also submit X"*. A sentence must name the field **and** use change wording. *"Please refer to the original due date"* and *"does not anticipate extending the deadline"* are not changes. *"For clarification ..."* is a recorded clarification that leaves the value alone. Wording that sounds like a change but yields no usable value (*"replace Section 3.2 with the following"*, *"extended to allow vendors to ..."*) is `unknown`, never guessed. Detected values must pass the Phase 5 grounding check against the cited chunk.
- **When Groq is used:** only for those `unknown`/ambiguous cases, and only if configured. The prompt contains just the field, the relevant original chunks and the relevant chunks of the addendum in question. Its answer is validated like Phase 5 (JSON mode, Pydantic, one retry via the shared helper): cited chunk ids must be addendum chunks that were supplied, and a proposed value must be contained in the cited text, otherwise it is rejected and the ambiguity stays visible. Groq-derived results are labelled `llm_reconciliation` with confidence capped at 0.8. Without Groq those fields are reported unresolved and listed in `errors`; the rest of the bid still reconciles.
- **Original values:** deterministic locators read them for the highest-value fields (RFP `Solicitation Due`, `Pre-Proposal Meeting`; PDFs only, because a listing page such as BidNet already mirrors later changes). For other fields pass `use_extraction=true` (CLI `--use-extraction`) to take them from Phase 5 extraction, which needs Groq.
- **Conflict handling:** contradictory originals with no superseding addendum, addenda that disagree when their order is unknown, or one addendum contradicting itself give `status: conflict`, `final_value: null`, with every value and its sources listed. Evidence that cannot be interpreted gives `insufficient_evidence`. Both appear in `reconciliation_required`. A genuine addendum override of disagreeing originals resolves, and the originals are kept in `conflicting_values`.
- **Provenance:** a `ReconciledField` carries `original_value` + `original_sources`, every `change` (addendum number and file, type, previous and new value, quoted sentence, sources, `rule`/`llm`) and the `final_value` + `sources`. Sources are built from retrieved chunks (file, type, page, chunk index), never from model output. Retrieval is scoped to the bid and, per addendum, to that addendum's document id; results from any other bid are discarded even if retrieval returned them.
- **Fields:** due date, term, pre-bid meeting, installation, bid bond, delivery, payment terms, additional documentation, model no, part no, product. Product Specification and Bid Summary are not reconciled directly (composite / generated), and the response says so.

```bash
cd backend
python -m app.reconciliation Bid1 [Bid2] [--json out.json] [--use-extraction]
python -m app.reconciliation --eval        # score against eval/reconciliation_eval.json
curl -X POST http://localhost:8000/api/v1/reconciliation/Bid1
```

**Bid1 due date (real documents, deterministic, no Groq):**

| | Value | Source |
| - | ----- | ------ |
| Original | `27-JUN-2024 14:00:00` | `JA-207652 Student and Staff Computing Devices FINAL.pdf`, page 2 |
| Addendum 2 (`replacement`, rule) | `July 9, 2024 at 2:00 PM CST` | `Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf`, page 1 |
| **Final** | `July 9, 2024 at 2:00 PM CST` | status `resolved`, method `explicit_addendum_change`, confidence 0.95 |

Addendum 1 (Q&A) contains no change to the due date (it says Dallas ISD "does not anticipate extending the submission deadline"). The pre-bid meeting (`10-JUN-2024 14:00:00`) is reported `unchanged`. Bid2 has no addenda, so nothing is reconciled there.

#### Reconciliation evaluation

`python -m app.reconciliation --eval` on the real documents (4 hand-checked cases in `backend/eval/reconciliation_eval.json`; deterministic path only, **real Groq was not configured and is not covered**):

| Measure | Result |
| ------- | ------ |
| Resolution accuracy (expected-resolved cases) | 1/1 (Bid1 due date) |
| Unchanged fields stay unchanged | 1/1 (Bid1 pre-bid meeting) |
| Original-value preservation (value + history + citation) | 2/2 |
| Citation correctness (chunk exists, expected file, contains the value) | 2/2 |
| Timeline order (Bid1 Addendum 1 -> 2; Bid2 none) / Bid2 nothing changed | 2/2, 1/1 |
| Conflict preservation | 0/0: the real corpus has no genuinely unresolvable conflict; covered by unit tests instead |

This is a very small real-data set. The unit tests exercise the spec scenarios (explicit replacement, from/to, clarification, multiple addenda in order, later clarification, unresolved conflict) with controlled text, not real documents.

**Limitations:** Groq-based reconciliation (ambiguous wording, `--use-extraction`) is implemented and tested only against a scripted model, not real Groq. Rules cover common English phrasings and one change per sentence; changes spread over several sentences or tables fall to Groq. Q&A-style addenda that restate a changed requirement without change wording ("Answer: USB 3.1 is a minimum") are not treated as changes. Only the due date and pre-bid meeting have built-in original locators. Confidence is a heuristic, not a calibrated probability.

### Phase 7 - Multi-agent orchestration

Plain Python classes (`app/agents/`), no agent framework. The agents are thin: each one coordinates an existing service and adds a guarantee; none re-implements parsing, retrieval, extraction or reconciliation.

```
                       Orchestrator  (fixed linear workflow, no loops, no agent-to-agent calls)
                            │
   ┌──────────┬─────────────┼───────────────┬──────────────────┬─────────────────┐
   ▼          ▼             ▼               ▼                  ▼                 ▼
Ingestion  Retrieval    Extraction     Reconciliation      Validation      Report / Q&A
 Agent      Agent         Agent           Agent              Agent            Agent
   │          │             │               │                  │                │
 Phase 2/3  Phase 4/5   Phase 5          Phase 6           DB checks     Phase 4 retrieval
 services   retrieval   service          service           (new)         + Groq (answers)
```

| Agent | Wraps | What it adds |
| ----- | ----- | ------------ |
| Orchestrator | the agents | Order, failure classification, one status. It never parses, searches, extracts, reconciles or calls Groq. |
| Ingestion | `IngestionService`, `EmbeddingService` | Per-bid discovery, reports ingested/skipped/failed documents and ids; embeds only when chunks are pending (unchanged files are skipped by hash). |
| Retrieval | `RetrievalService`, Phase 5 evidence gathering | Field-specific requests; discards any chunk not from the requested bid. Its evidence is handed to extraction so nothing is retrieved twice. |
| Extraction | `ExtractionService` | Runs it (prompts, Groq, grounding, citation checks and retries stay in the service). Skipped when Groq is not configured. |
| Reconciliation | `ReconciliationService` | Runs it on the extraction; if Groq fails mid-way it re-runs deterministically instead of failing. |
| Validation | database | **New.** Re-checks the whole output against stored data (below). Reports, never repairs. |
| Report / Q&A | `RetrievalService`, shared Groq helper | Builds the merged report (copies values, never invents any) and answers questions from cited evidence. |

**Pipeline** (`POST /api/v1/pipeline/{bid_id}`, CLI `python -m app.agents Bid1`): ingestion → retrieval → extraction → reconciliation → validation → report. State is a plain Pydantic model (`BidPipelineState`) holding every stage's output, errors and warnings.

**Failure handling.** `status` is `complete`, `partial` or `failed`; `errors` and `warnings` are separate.
- *Fatal (`failed`, later stages skipped):* unknown bid (404), no document of the bid could be ingested, database or embedding model unavailable, no retrievable evidence, or validation finds an error (the report is then withheld: `result: null`).
- *Recoverable (`partial`):* some files failed to ingest, Groq not configured (extraction skipped, deterministic reconciliation still runs), Groq failing during extraction, fields that could not be extracted, ambiguous addendum wording, unresolved conflicts, validation warnings.
- Stage errors are reported by category only (e.g. "database error"), never with connection details.

**Validation agent** checks, against the database:
schema validity (every field re-validated, all 20 extraction fields present) · every cited chunk exists · it belongs to the requested bid · its stored file/type/page/chunk match the citation · found values and final values are grounded in the cited text (Phase 5 grounding) · a resolved field's final value equals its last addendum change, cites that change, the change comes from a timeline addendum, and the history starts from the cited original · `conflict`/`insufficient_evidence` have `final_value = null` and confidence 0 · missing values have no value, no sources and confidence 0. Unresolved conflicts and uninterpretable addendum wording are warnings, not errors.

**Q&A** (`POST /api/v1/qa/{bid_id}`, body `{"question": "..."}`): the agent retrieves evidence for the question through `RetrievalService` (plus an addendum-only query so amendments are visible) and, if the question concerns an addendum-modifiable field (due date, pre-bid meeting, ...), adds the validated reconciled value and its cited chunks as extra evidence. It never reads whole PDFs. Groq answers in JSON; the answer is accepted only if (1) it cites chunk ids that were supplied for this bid, and (2) every figure in it occurs in the cited text (or the question / file name). Otherwise, or if Groq says the evidence is insufficient, the response is the fixed refusal *"I could not find sufficient evidence to answer this question."* with no sources. Wrong-bid chunks are dropped before the prompt and again at citation time. Needs Groq (503 without it).

**Guarantees:** values come only from retrieved evidence plus validated model output; citations are rebuilt from stored chunks, never taken from model text; evidence cannot cross bids; the orchestrator never calls Groq (only the Extraction and Reconciliation services and the Q&A agent do, through the one shared `GroqClient` and its JSON/retry helper). Each agent execution logs `agent, bid, stage, ok, duration` (no prompts, documents or keys).

**Bid1, real documents, no Groq** (`python -m app.agents Bid1`): ingestion complete (4 files, 0 re-ingested, 4 skipped), retrieval complete (20 fields, 69 chunks), extraction skipped, reconciliation complete (2 addenda, ordered by their own headers), validation complete (5 citations checked, 0 errors). Status `partial` only because Groq is not configured. Report: **Due Date = `July 9, 2024 at 2:00 PM CST`**, original `27-JUN-2024 14:00:00`, cited to `Addendum 2 ... .pdf` page 1 and `JA-207652 ... FINAL.pdf` page 2. Bid2 (no addenda, nothing to extract without Groq) runs to `partial` with no values, rather than inventing any.

#### Q&A evaluation (`backend/eval/qa_eval.json`, 14 questions)

`python -m app.agents.qa_eval --evidence-only` (no Groq) on the real documents measures whether the expected evidence reaches the model: **10/11** answerable questions, and **11/11** with only the asked bid's chunks. The miss is "What product is being procured?" for Bid1: the question is too generic and retrieves boilerplate instead of the product description. I broadened one expectation after seeing results ("What affidavits are mentioned?" also accepts the Mercury/Contract Affidavit documents, which the retrieval returned and which do answer it).

**Not measured (needs Groq, which is not configured here):** answer correctness, answer grounding as scored on real model output, and refusal behaviour (including the cross-bid trap "Is a Mercury Affidavit required?" asked of Bid1). The same evaluator computes these with `python -m app.agents.qa_eval` once Groq is set; the logic is unit-tested with a scripted model only.

**Limitations:** extraction and Q&A have not run against real Groq. Extraction and reconciliation are recomputed on each run (no cache); only ingestion and embeddings are skipped when unchanged. Q&A uses one retrieval round, so open-ended questions can miss evidence (see the miss above). Ambiguous evidence is surfaced rather than resolved. The Q&A evaluation set is small and its expectations were checked by hand against the same documents the system uses.

### Phase 8 - Real Groq validation and hardening

Everything before this phase had only been tested against scripted models. This phase ran the real Groq API, fixed what it exposed, and added regression tests (deterministic, wording-independent) for each problem.

**What was validated with real Groq, and with which model** (read this before trusting any number):

| Area | Real Groq? | Model |
| ---- | ---------- | ----- |
| Smoke test (JSON mode) | yes | configured `openai/gpt-oss-120b` |
| Structured extraction, Bid1 and Bid2 | yes | configured `openai/gpt-oss-120b` |
| Reconciliation fallback, controlled ambiguous cases | yes | configured `openai/gpt-oss-120b` |
| Q&A evaluation (14 questions, 2 runs) | yes | **`openai/gpt-oss-20b`, not the configured model** (see below) |
| Q&A on the configured 120b model | **no** | quota exhausted |
| Reconciliation with real-model extraction (`--use-extraction`) | **no** | quota exhausted |
| Everything else (tests, deterministic paths, retrieval, API semantics) | n/a | verified deterministically |

**Why Q&A used a different model.** The key is on Groq's on-demand tier: 8,000 tokens/minute and 200,000 tokens/day per model, and a request's reserved `max_tokens` counts against the daily bucket, which refills continuously (about 2.3 tokens/s). The reasoning model spends many hidden tokens, so repeated extraction runs used up the 120b budget (a full extraction is roughly 17-22k tokens in and 7-9k out per bid). Q&A then needed about 12 hours of refill. It was run with `GROQ_MODEL=openai/gpt-oss-20b` as a process-local override (`.env` untouched). Results show the Q&A *design* works with a real model; they say nothing precise about 120b. To run it on the configured model once quota allows: `python -m app.agents.qa_eval --patient -v` (`--patient` waits out rate limits; evaluation only).

**Configuration.** `GROQ_API_KEY` and `GROQ_MODEL` (no default, nothing model-specific in code) live in `backend/.env`, which is git-ignored; `.env.example` documents them. Missing key/model give a clear `ConfigurationError`. The key is a `SecretStr`: printing or logging `Settings` shows asterisks, provider errors surface only the exception type, and tests assert the key appears in no response, log, repr or prompt. Check the setup without sending documents: `python -m app.llm --smoke-test` prints `Groq configuration: OK / Model: configured / JSON generation: OK` (never the key). The client has an explicit 60 s timeout and at most 2 SDK retries; `GroqClient.calls` records per-request seconds, tokens and `finish_reason` (no content).

#### Real extraction (`openai/gpt-oss-120b`; `python -m app.extraction Bid1 Bid2`)

Scored against `eval/extraction_eval.json`, which was not edited to make the model pass. "Before" is the first real run; "after" is the final run after the fixes below. Both are single runs, and the fixes were chosen by diagnosing these same runs, so the improvement is **not** a held-out result.

| | Scored checks correct | Null checks correct | Citations valid (database check) | Groq calls / retries |
| - | - | - | - | - |
| Bid1 before | 6/8 | n/a (none defined) | 47/47 | 6 calls (1 rate-limited then waited) |
| Bid1 after | 7/8 | n/a | 37/37 | 5 calls, 0 malformed, 0 citation follow-ups |
| Bid2 before | 10/16 | 2/2 | 41/41 | 5 calls |
| Bid2 after | 16/16 | 2/2 | 36/36 | 6 calls, 0 malformed, 1 citation follow-up |

"Citations valid" is the Validation agent re-checking every cited chunk against the database (exists, right bid, matching file/page/chunk) and re-running grounding on every found value: 0 errors in every run. Every group reply was valid JSON on the first attempt in the final runs.

- **Critical Bid1 due-date test: passed in every run.** The model returned three distinct supported values with their sources and no chosen value: `July 9, 2024 at 2:00 PM CST` (Addendum 2, page 1), `07/09/2024 03:00 PM EDT` (BidNet listing) and `27-JUN-2024 14:00:00` (RFP, page 2). Phase 5 reported a conflict and `reconciliation_required`; reconciliation then resolved it to July 9.
- **Remaining miss (Bid1):** the pre-bid meeting. The same event appears as `10-JUN-2024 14:00:00 / June 10, 2024 at 2:00 PM CST` in the RFP and `06/10/2024 03:00 PM EDT` on the listing; the model merged them in one run and split them (a false conflict) in two. The failure direction is the safe one, but it is non-deterministic.
- **Imprecision not covered by the ground truth:** Bid1 `Bid Number` also lists the BidNet reference number `00004079100`; Bid2 `Due Date` offers `06/01/2024 2:00 PM EDT`, which is the *questions* deadline in the PORFP, next to the listing's date (a wrong candidate, but surfaced as a conflict); Bid1 `Bid Bond Requirement` was "required" in one run and absent in the next (not verified against the document). Unverified items are listed as limitations, not hidden.

**What real runs exposed, and the fix** (each has regression tests in `tests/test_phase8_regressions.py`):

| Real problem | Cause | Fix |
| ------------ | ----- | --- |
| `contact_info` / `company_name` rejected as unsupported | The model cited a chunk that was in the call but not in that field's own evidence list; correct values were rejected | A citation may name any chunk sent in that call. Grounding still checks the value against the cited text, so no safety is lost. Plus one bounded citation follow-up per group (only for rejected fields; never repaired by the backend) and a second `contact_info` query using the "Contact Information" heading |
| Wrong "Delivery Date" (a repair-time clause) | Field description too loose | Description excludes repair/response times |
| False conflicts: PORFP number vs project number, laptop vs dock, installation aspects | Scalar fields used for things that are legitimately plural | `bid_number`, `model_no`, `part_no`, `product`, `installation`, `bid_submission_type` are now lists. Dates stay scalar so real conflicts still surface |
| `product_specification` rejected in one run | Model cited wrong chunks (run-to-run variation) | Same citation follow-up |
| HTTP 400 `json_validate_failed` treated as a provider outage | Groq's JSON mode returns 400 when the model's output is not valid JSON | Treated as malformed output: one retry, then a structured invalid-output error (Q&A: 502). Other 400s stay fatal |

#### Real reconciliation fallback (`openai/gpt-oss-120b`; `python -m app.reconciliation --llm-eval`)

The real documents contain no ambiguous addendum wording, so six **controlled** cases (synthetic evidence, real model, real fallback code) were used; `eval/reconciliation_llm_cases.json`. Result: **6/6 as expected** on two complete runs (a third was cut short by the daily token cap and is not counted). The model resolved a genuine amendment (`10% of the bid amount`, grounded and cited), recognised a clarification as unchanged, resolved "replace ... with the following: ...", and never invented a value when none was given. Model self-confidence 1.0 was capped to 0.8 by the system. About 1.1 s and 0.6k tokens in per call. The deterministic path on the real Bid1 documents still resolves the due date 27-JUN → July 9 (1/1).
*Weakness found and fixed:* for "The submission deadline has been extended. A new date will be announced." the model said the addendum had no effect, which made the old deadline look final. A model answer of "none" can no longer clear explicit change wording the rules saw (it stays unresolved).

#### Real Q&A (`openai/gpt-oss-20b`, not the configured model; `eval/qa_eval.json`, 14 questions)

Second run (all 14 evaluated), with the fixes in place:

| Measure | Result |
| ------- | ------ |
| Answerable questions scored for correctness | 10 correct / 10 (0 incorrect answers) |
| Correct refusals (no evidence in that bid) | 3/3, including the cross-bid trap "Is a Mercury Affidavit required?" asked of Bid1 |
| Incorrect refusals | 1: Bid2 "submission deadline" was refused because the PORFP layout is ambiguous (no answer is scored for it; a refusal is the safe outcome) |
| Cited chunks from another bid | 0 of 14 questions |
| Answers whose figures are grounded in the cited text | 10/10 |
| Citation correct (expected file and text cited) | 10 of the 10 answered questions |

Run 1 gave the same facts, citations and refusals; wording differed on at least the "What changed in Addendum 2?" answer. Its two non-passes were an evaluator bug (the model wrote non-breaking hyphens in `10‑JUN‑2024`, so a regex missed it; the evaluator now normalises typographic dashes) and the provider 400 above on one question (fixed; that question is now answered). Two runs are not enough to measure variability statistically.
Latency and size (run 2): 0.5 / 12.6 / 44.3 s min/avg/max per question, 15 Groq calls (the 15th was the JSON-failure retry), about 1.9k tokens in and 0.25k out per question. A full pipeline per bid is 5-6 Groq calls, 130-220 s, because the reasoning model takes 3-46 s per call.

**Retrieval weakness (Phase 7's one evidence miss).** "What product is being procured?" (Bid1): the answering chunks (RFP scope text and the BidNet description) ranked 45th and 31st by vector similarity and were not found by keyword search at all. It is not top-K, filtering or chunking: the question says "product / procured" while the documents say "computing devices, including laptops...", and boilerplate about "produced equipment" ranked first. Fix (`retrieval/expansion.py`): when no top result is confirmed by both vector and keyword search, and the question matches a generic "what is being bought" intent, add one bid-scoped query phrased in solicitation vocabulary ("what goods or services does this request for proposal solicit", checked to put the chunk at vector rank 2 / keyword rank 4). Bounded (at most 2 variants), deterministic, no extra Groq call. Evidence available for the expected answer went from **10/11 to 11/11**, nothing else changed, and the real model then answered that question correctly from page 3 of the RFP. Only one intent has been verified against real retrieval; more should be added only the same way.

**Other decisions**
- *Pipeline HTTP status.* A fatal pipeline failure used to be HTTP 200 with `status: "failed"`, unlike every other endpoint. It is now 503 (database / embedding model), 422 (nothing to analyse), 500 (output failed validation or internal error), always with the structured body; `partial` stays 200. The React page shows the explanation from that body.
- *Caching.* Not added. Reuse would need a fingerprint over document hashes, chunks, embeddings, prompts, field definitions and model; real runs also showed extraction is non-deterministic, so silently reusing one output would hide variability. Ingestion and embeddings are still skipped when unchanged.
- *Diagnostics.* Per-stage `duration_ms` and, for extraction and reconciliation, Groq calls/tokens/latency/truncation are in the pipeline state and API `timings_ms`; the UI does not show them. No prices are computed (they depend on the model's pricing).
- *Failure paths seen live:* rate limit (one bounded wait, then a structured error; the pipeline degraded to deterministic reconciliation and still resolved the due date), and the 400 above. Timeouts, invalid JSON and missing key are covered by unit tests only.

**Limitations (not validated):** Q&A and reconciliation-with-extraction on the configured 120b model; more than one real extraction run per configuration (the pre-bid result flipped between runs); Bid1 `Bid Bond Requirement` correctness; real-model behaviour on documents other than these nine; token/day quota means a full evaluation of all stages on one model is not possible in a single day on this tier.

### Phase 9 - Production hardening, evaluation and finalization

No new features and no new infrastructure. This phase hardened what exists, added regression/acceptance structure, validated Docker from scratch, and documented what is and is not proven.

#### Evaluation: deterministic vs real-model (read this first)

| What | Deterministic (no Groq) | Real model |
| --- | --- | --- |
| Unit / API / PostgreSQL tests (`cd backend && pytest`) | **417 passed, 0 skipped** (366 before Phase 9, +34 in Phase 9, +17 for the extraction fixes below). Scripted models only. | - |
| Regression cases (`python -m app.agents.regression`) | **25/25** hold on the real Bid1/Bid2 data: evidence reaches the model, reconciliation result, bid isolation, absent values stay absent. | - |
| Acceptance (`python -m app.agents.acceptance`) | Bid1 and Bid2 pass without Groq (due date resolved to July 9 with the original preserved and both cited; Bid2 has no invented timeline or changes). | See "Real-model gap" below. |
| DB / index checks (`python -m app.diagnostics`) | **14/14** on the local PostgreSQL (pgvector 0.8.6) and on the fresh Docker database (pgvector 0.8.7). | - |
| Extraction | - | `openai/gpt-oss-120b` (Phase 8): Bid1 7/8 and Bid2 16/16 gold fields after fixes, 0 validation errors. |
| Reconciliation LLM fallback | - | `openai/gpt-oss-120b`: 6/6 controlled cases, in two complete runs. |
| Q&A | - | `openai/gpt-oss-20b` (**not the configured model**): 10/10 answers correct, 3/3 refusals, 0 cross-bid citations, 10/10 grounded; 1 safe refusal on the ambiguous Bid2 deadline. |

The PostgreSQL tests use the `helix_test` database and the real-data tests use the ingested `helix` database; none were skipped in the reported run. The suite never calls Groq.

**Real-model gap (not validated in this phase).** The full end-to-end run (ingestion, retrieval, extraction, reconciliation, validation, report) on Bid1 and Bid2 with the configured `openai/gpt-oss-120b`, and the Q&A subset on 120b, were **not completed**: the Groq free tier allows about 200k tokens per day and a run costs roughly 28k tokens per bid plus about 35k for the Q&A subset. A patient runner was started and was later stopped without results, so **no Phase 9 end-to-end 120b result exists**. The per-stage 120b results above are real but are not a full pipeline run. To finish it later (hours on the free tier, minutes on a paid one):

```bash
cd backend
python -m app.agents.real_run --patient --bids Bid1 Bid2 --qa --out eval/runs/final
python -m app.agents.acceptance --from-json eval/runs/final/pipeline_Bid1.json --bid Bid1 --require-extraction
python -m app.agents.acceptance --from-json eval/runs/final/pipeline_Bid2.json --bid Bid2 --require-extraction
```

`--patient` waits out rate limits (evaluation only; the API never does). Outputs are git-ignored. The acceptance command lists each check (extraction, reconciliation, validation, report) separately and exits non-zero on any failure. Quote results from the JSON files, not from this README.

#### Regression structure (`backend/eval/regression/*.json`)

Seven files (extraction, reconciliation, qa, grounding, isolation, missing_values, conflicts), 25 cases. Each case has `id, bid_id, question_or_field, expected_behavior, expected_sources, notes, check, params`. They check behaviour (evidence, grounding, isolation, absence), never one model's wording. Every "absent" claim was verified against the stored chunks; one pattern was too broad ("notice of award" appears in a contract-precedence clause), so the pattern was narrowed, not the claim. Existing evaluation sets were not changed.

```bash
python -m app.agents.regression          # no Groq needed; needs the ingested and embedded documents
```

#### Targeted extraction fixes (after the first real Bid1 run)

- **Bid Number** now means the solicitation's own number (the identifier the documents label as the bid, solicitation, RFP or PORFP number). Portal "Sourcing #" ids, "Reference Number" and project numbers stay out of the value; the model is told to mention them, with their label, in the field's `notes`. This is a field-definition/prompt rule: grounding and citation checks are unchanged, so it cannot be verified without a live model run (not run here). In the Bid1 documents the labels are `Solicitation Number JA-207652`, `SOURCING #168884` and `Reference Number 00004079100`.
- **`65HDMI`** is printed in the Bid1 RFP itself (FINAL.pdf, student laptop ports). Extraction preserves source wording; the prompt now says never to correct apparent typos. A regression case pins that the source contains it.
- **Product Specification** can now describe several products: one candidate per product line, each cited to and grounded in its own chunk(s) (an item whose figures are not in its own cited text is dropped). One product still returns a single object as before; several return a list of `{product, specifications}` objects (at most 8). The UI shows one block per product. Bid1 does list separate requirements for several product lines (Chromebooks, a Windows laptop, a tablet, desktops, staff laptops, monitors).
- To let the model see those lines, the spec field requests 8 evidence chunks instead of 4 and, at the default `EXTRACTION_MAX_CHUNKS_PER_CALL=14`, the product group sends up to 4 extra chunks reserved for it (a lower cap stays a hard limit). On the real Bid1 data the chunks reaching the model for the spec field went from 3 (317, 318, 320) to all 6 product chunks (315-320). This adds roughly 1k input tokens to one of the five calls.
- Unchanged and re-checked: missing fields stay `null` with no sources and confidence 0; confidence is never raised above the model's own estimate; the Due Date reconciliation (July 9, 2024 2:00 PM CST, original 27-JUN-2024) and bid isolation. Tests: 417 passed, 0 skipped; regression cases 25/25.
- **Limits:** the Validation agent checks a multi-product value against the union of all its citations (each item was grounded against its own citations at extraction time). Whether a live model follows the new bid-number and per-product instructions was not tested against Groq.

#### Hardening added

- **Run IDs and logs**: every pipeline run gets an 8-character `run_id` (in the response and in every log line of that run). Groq calls log `run, stage, model, ok, ms, tokens_in/out` only: never prompt text, documents or keys.
- **Health**: `GET /api/v1/health/status` breaks health down by database, pgvector, embedding model, Groq configuration and available features.
- **Input and API safety**: bid ids must match `[A-Za-z0-9][A-Za-z0-9_-]{0,63}` (`422` otherwise); request bodies above 64 KB get `413`; search fields have length and range caps; any unexpected exception becomes `500 {"detail": "Internal server error"}` with no traceback. Phase 8 semantics are unchanged (`503` infrastructure, `422` nothing to analyse, `500` validation failure, `200` partial).
- **File safety**: files above `MAX_INGEST_FILE_MB`, files that resolve outside `DATA_DIR` (symlinks, `..`) and PDFs over 2000 pages are refused and reported; malformed files fail per file without stopping the run.
- **Prompt injection**: all three system prompts say evidence is untrusted document text and instructions inside it must not be followed; evidence is delivered only in a labelled `EVIDENCE` section. This is a mitigation, not a guarantee: a malicious sentence that a model repeats and that is cited from a real chunk could still be grounded. What is guaranteed (and tested) is that a value that is uncited or absent from its cited chunk is rejected.
- **Secrets**: the key is a `SecretStr`, never printed, logged or returned; a scan of the repository for the key value and `gsk_` patterns finds it only in the git-ignored `backend/.env`.
- **Frontend**: Analysis shows distinct "complete", "partially completed" and "failed" banners (with the failure reason and run id), an elapsed-time note while a run is in progress, an empty-result message, and the standard refusal for unsupported questions.

#### Docker validation (fresh volumes, no manual steps)

With `POSTGRES_PORT=5433` (5432 was taken locally): postgres, backend (healthy) and frontend (HTTP 200) started; the app created the extension, tables, HNSW and GIN indexes on startup; `/ingestion/run` processed 9/9 documents (285 chunks) in about 2 s; `/embeddings/generate` embedded 285/285 (model downloaded inside the container); search and `/health/status` worked; the pipeline ran inside the container (Groq disabled for that check) and resolved the Bid1 due date to July 9 with the original preserved; after `docker compose down` and `up -d` the data persisted; `python -m app.diagnostics` passed 14/14. **Not validated:** Docker with a real Groq key (same code path as local, not run in the container).

#### Reproducible demo

```bash
cp .env.example .env && cp backend/.env.example backend/.env   # set GROQ_API_KEY and GROQ_MODEL for the LLM steps
# set POSTGRES_PORT in .env if 5432 is taken
docker compose up --build -d
curl -X POST localhost:8000/api/v1/ingestion/run
curl -X POST localhost:8000/api/v1/embeddings/generate
curl localhost:8000/api/v1/health/status
docker compose exec backend python -m app.agents Bid1                  # works without Groq (status partial)
docker compose exec backend python -m app.agents Bid1 --ask "What is the final submission deadline?"   # needs Groq
docker compose exec backend python -m app.diagnostics
```

Expected: the Bid1 Due Date is `July 9, 2024 at 2:00 PM CST` (Addendum 2, page 1) with the original `27-JUN-2024 14:00:00` (RFP, page 2) shown alongside; Bid2 has no addenda and no changes. In the UI open **Analysis**, pick Bid1, Run Analysis, then ask the question above. Tables are created on startup (`create_all` plus idempotent index DDL); there are no migrations.

#### Measured performance

Ingestion of 9 documents / 285 chunks: about 2 s (local and Docker). The deterministic pipeline for an ingested bid (no Groq) took well under a second in the orchestrator tests. Full test suite: about 70 s. Real-model latency is dominated by Groq rate limits and was not measured end to end in this phase.

#### Limitations

- Real 120b end-to-end and 120b Q&A are unvalidated (see above); Q&A was measured on gpt-oss-20b.
- Extraction is non-deterministic; one Bid1 gold field was still wrong in the best real run (7/8), and output varies between runs.
- The evaluation sets are small (2 bids, 14 Q&A questions) and were checked by hand against the same documents; they say nothing about other bid layouts.
- Scanned or image-only PDFs are not OCRed.
- Confidence is a heuristic. The planner does not use HNSW at this corpus size (285 rows); the index exists and was verified usable.
- No authentication, per-client rate limiting or migrations; this is not a multi-tenant service.
- Prompt-injection protection is a prompt rule plus grounding checks, not a classifier.


### API reference (detailed, per endpoint)

| Endpoint | Description |
| --- | --- |
| `GET /api/v1/health` | Runs real queries against PostgreSQL and checks pgvector. `200` when healthy, `503` otherwise. |
| `GET /api/v1/health/status` | Per-component health: database + pgvector, embedding model (name, dimension, loaded), Groq (configuration only) and which features are available. `503` only if the database, pgvector or embedding model is unusable; a missing Groq key does not make the system unhealthy. `?probe_embeddings=true` loads the model. |
| `GET /api/v1/health/groq` | Checks Groq configuration. Add `?probe=true` to make a 1-token live request. |
| `POST /api/v1/ingestion/scan` | Lists supported files under `DATA_DIR`; stores nothing. |
| `POST /api/v1/ingestion/run` | Ingests new/changed documents and returns `{total, processed, skipped, failed, results}`. |
| `GET /api/v1/ingestion/documents` | Lists ingested documents (optional `?bid_id=`). |
| `GET /api/v1/ingestion/documents/{id}` | Document metadata plus page/chunk counts. |
| `POST /api/v1/embeddings/generate` | Embeds all chunks without an embedding; returns `{total_chunks, embedded, skipped, failed, errors}`. |
| `GET /api/v1/embeddings/status` | Counts of embedded / pending / failed chunks, model and dimension. |
| `POST /api/v1/search` | Hybrid retrieval. Body: `{query, bid_id?, document_type?, file_name?, document_id?, page_number?, top_k?}`. Returns ranked evidence with citations, no generated answer. `422` for bad input, `503` if the database or embedding model is unavailable. |
| `GET /api/v1/search/filters` | Available bids, document types and file names for the filters. |
| `POST /api/v1/pipeline/{bid_id}` | Runs the full workflow; returns `status` (`complete`/`partial`/`failed`), `failure`, per-stage states, `errors`, `warnings`, the validation report, the merged report (`result`, null if validation failed), `timings_ms` and a `run_id` that also appears in the server logs. Works without Groq (`partial`, HTTP 200). `failed` uses `503` (database/embedding), `422` (nothing to analyse) or `500` (invalid output) with the same body. `?ingest=false` skips ingestion. `404` unknown bid. |
| `POST /api/v1/qa/{bid_id}` | Body `{"question": "..."}`. Answer with `sources` (`file`, `page`, `chunk_id`, ...) and `confidence`, or the standard refusal. `404` unknown bid, `422` bad question, `502` Groq failure / invalid model output, `503` Groq not configured / database / embedding model. |
| `POST /api/v1/reconciliation/{bid_id}` | Resolves fields changed by addenda, with original value, change history and citations. Works without Groq (ambiguous wording is then reported unresolved). `?use_extraction=true` also runs Phase 5 extraction (needs Groq, else `503`). `404` unknown bid, `502` Groq failure, `503` database / embedding model. |
| `POST /api/v1/extraction/{bid_id}` | Structured extraction of the 20 bid fields with citations, via Groq. `404` unknown bid, `502` Groq failure, `503` Groq not configured / database / embedding model unavailable. Stateless. |
