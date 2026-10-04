# Helix architecture

The diagram below is the same source as [`architecture.mmd`](architecture.mmd) (Mermaid; GitHub and VS Code render it). A pre-rendered PNG is not included because no Mermaid renderer could be run in the environment this was prepared in.

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

## How the pieces talk to each other

- **Everything is in-process Python.** The agents are plain classes called in order by `Orchestrator` (`backend/app/agents/orchestrator.py`); they share one `BidPipelineState` object. There is no message bus, queue or agent framework.
- **PostgreSQL is the only store.** Ingestion writes `documents` and `document_chunks` (text, page or HTML section, metadata, a 384-d `vector` column with an HNSW cosine index, and a generated `tsvector` column with a GIN index). Retrieval and the Validation agent read from it.
- **Retrieval** runs vector and keyword search separately, fuses them with Reciprocal Rank Fusion (k=60), and always filters by `bid_id`, which is what keeps Bid1 and Bid2 isolated. No reranker is implemented.
- **Groq is called only by three components**, always through the single `GroqClient` (`backend/app/llm/groq_client.py`): extraction (5 grouped calls per bid), reconciliation (only for ambiguous change wording; explicit wording is handled by deterministic rules) and Q&A. The model receives evidence chunks and returns JSON (extraction) or a cited answer (Q&A); the application rebuilds citations from the stored chunks and never trusts model-written citations.
- **The Validation agent** re-reads every cited chunk from PostgreSQL and checks that it exists, belongs to the bid, matches its file and page, and contains the value. A report is only returned when validation passes.
- **Entry points:** the FastAPI app (`/api/v1/...`, used by the React UI) and the CLI (`python -m app.agents BID`, plus per-stage modules). Both call the same orchestrator and services.
