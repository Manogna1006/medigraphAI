# MediGraph AI

**A GraphRAG framework for hallucination-resistant medical decision support.**

MediGraph AI answers clinical questions by combining a Neo4j knowledge graph
(clinical symptom → disease → treatment triplets) with a ChromaDB semantic
vector store, gated by a **semantic firewall** that blocks the LLM from
answering when there isn't enough grounded evidence. Every generated answer is
scored by a **hallucination metrics** pipeline that reports exactly how much
of the answer was actually backed by evidence.

## Architecture

```
                    ┌─────────────────────┐
   User Query  ───► │  HybridRetriever     │
                    │  (semantic firewall) │
                    └──────────┬──────────┘
                     ┌─────────┴─────────┐
                     ▼                   ▼
             ┌───────────────┐   ┌───────────────┐
             │  Neo4j (KG)   │   │  ChromaDB      │
             │  clinical     │   │  vector search │
             │  triplets     │   │  + distance    │
             └───────────────┘   │  thresholding  │
                                  └───────────────┘
                     │                   │
                     └─────────┬─────────┘
                                ▼
                     Evidence passes firewall?
                     ── No ──► Abstain (no LLM call)
                     ── Yes ─► LLMSynthesisEngine (Groq / Llama 3.1)
                                │  every claim must cite an evidence line
                                ▼
                     HallucinationScorer
                     ── extract claims
                     ── verify each claim against evidence (LLM-judge,
                        with an offline lexical-overlap fallback)
                     ── hallucination_rate / groundedness_score /
                        contradiction_rate
                                │
                                ▼
                          QueryResponse (answer + hallucination_report)
```

## Why this reduces hallucination (not just measures it)

1. **Semantic firewall** — if the KG returns no relevant triplets *and* no
   vector chunk is within the distance threshold, the system **abstains
   before calling the LLM at all**. No evidence → no answer, by construction.
2. **Forced citations** — the synthesis prompt requires every claim to cite
   an evidence line (`[E3]`); uncitable claims aren't supposed to be made.
3. **Independent verification** — the hallucination scorer doesn't trust the
   model's own citations. It re-extracts every claim and re-checks it against
   the evidence set with a separate judge call, so a model that fabricates a
   citation number doesn't get a free pass.
4. **Deterministic vitals flagging** — patient vitals are flagged against
   reference ranges in code (`app/clinical/vitals_analyzer.py`), not left for
   the LLM to interpret from raw numbers. "132 bpm is ABOVE normal:
   tachycardia" is a lookup, not a generation — one less thing the model can
   get wrong.
5. **Fail-safe degradation everywhere** — a Neo4j outage, an empty vector
   collection, a corrupted/encrypted PDF, or a Groq API failure all degrade
   to "not enough evidence, won't guess" rather than to a stack trace or a
   confident wrong answer.

## Multi-source synthesis

A `/query` request can include `patient_context` (age, symptoms, vitals,
notes) alongside the question:

```json
{
  "question": "What should be considered given these symptoms?",
  "patient_context": {
    "age": 68,
    "symptoms": ["chest pain", "shortness of breath"],
    "vitals": {"heart_rate": 132, "spo2": 89, "systolic_bp": 100},
    "notes": "History of hypertension."
  }
}
```

Under the hood:
- Each reported **symptom** is also traversed against the KG (`extra_kg_terms`
  in `HybridRetriever.retrieve`), so triplets relevant to *all* symptoms are
  pulled in, not just whatever the free-text question happens to match.
- **Vitals** are flagged against reference ranges and rendered as explicit
  prose before reaching the LLM (see point 4 above).
- **Clinical PDF records** (discharge summaries, lab reports) can be ingested
  via `app/clinical/pdf_record_ingestor.py` and indexed into ChromaDB tagged
  `patient_record:<id>`, so they're retrievable alongside the literature
  corpus:
  ```python
  from app.clinical.pdf_record_ingestor import ingest_pdf_record
  ingest_pdf_record("discharge_summary.pdf", patient_id="patient-42")
  ```

## Hallucination metrics (milestone 2)

| Metric | Definition | Good direction |
|---|---|---|
| `hallucination_rate` | unsupported claims / total claims | lower |
| `contradiction_rate` | claims that conflict with evidence / total claims | lower |
| `groundedness_score` | supported claims / total claims | higher |
| `abstention_rate` | % of benchmark queries the firewall refused | context-dependent — high isn't automatically bad if the KB is genuinely sparse |
| `p50` / `p95` latency | end-to-end response time | lower |

Metrics are computed per-query (`HallucinationReport`) and aggregated across a
benchmark set (`BenchmarkResult`). Run the benchmark with:

```bash
python -m scripts.run_benchmark
```

This writes `reports/latest_benchmark.json` and prints the aggregate rates.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in GROQ_API_KEY and Neo4j credentials
```

### 2. Start Neo4j
```bash
docker compose up -d       # local Neo4j on bolt://localhost:7687
```
(Or point `NEO4J_URI` at Neo4j Aura / any managed instance.)

### 3. Seed the knowledge graph

**Option A — demo data (fast, offline):**
```bash
python -m scripts.seed_neo4j --wipe
```

**Option B — real literature via OpenAlex (matches the CURO AI reference):**
```bash
python -m scripts.ingest_openalex --topics "bacterial pneumonia" "type 2 diabetes" "myocardial infarction"
```
This fetches works from OpenAlex, scores each by a custom **authority score**
(citation count, recency, open-access status, abstract completeness — see
`app/ingestion/authority_scorer.py`), extracts clinical triplets from
above-threshold abstracts via the LLM, and writes both the triplets (Neo4j)
and the source abstracts (ChromaDB) — with each triplet's `confidence`
capped by its source's authority score, so a claim sourced from a
low-authority paper never outranks one from a well-cited review. Tune
thresholds via `OPENALEX_MIN_AUTHORITY_SCORE` / `OPENALEX_MAX_WORKS_PER_TOPIC`
in `.env`, or `--max-works` / `--min-authority` on the CLI.

### 4. (Optional) seed the vector store directly
The `ChromaStore` collection starts empty — add reference-corpus chunks (e.g.
OpenAlex abstracts) via `ChromaStore().add_documents(texts, sources)` before
querying, or the firewall will rely on KG triplets alone.

> **Note:** ChromaDB's default embedding function downloads a small ONNX
> model from the internet the first time it runs. If that download fails
> (offline environment, restricted network), `add_documents`/`query` fail
> **safely** — they log the error and return `0`/`[]` rather than crashing,
> and the firewall correctly falls back to KG-only evidence.

### 5. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```
Then:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What treats bacterial pneumonia?"}'
```

### 6. Run the tests
```bash
pytest tests/ -v
```
The full test suite (51 tests) runs **fully offline** — no live Neo4j, Groq
key, or OpenAlex access required, since every external dependency is mocked
at the boundary. This is what runs in CI (`.github/workflows/ci.yml`) on
every push/PR.

### Docker (full stack)
```bash
docker compose up --build
```
Starts Neo4j and the API together; the API waits for Neo4j's healthcheck
before starting. Remember to seed the graph afterward (step 3 above) — the
compose file doesn't auto-seed.

## Project layout

```
app/
  config.py                    # env-driven settings, validated at import
  main.py                      # FastAPI app: /health, /query, /benchmark
  models/schemas.py            # Pydantic contracts used everywhere
  graph/neo4j_client.py        # KG queries, retried, fail-safe
  vector/chroma_store.py       # vector search, fail-safe
  retrieval/hybrid_retriever.py # combines both + the semantic firewall
  synthesis/llm_engine.py      # Groq/Llama synthesis, forced citations
  ingestion/
    openalex_client.py         # OpenAlex Works API client, retried
    authority_scorer.py        # citation/recency/OA/abstract-quality scoring
    triplet_extractor.py       # LLM-based clinical triplet extraction
    pipeline.py                # orchestrates fetch -> score -> extract -> write
  clinical/
    vitals_analyzer.py         # deterministic reference-range flagging
    pdf_record_ingestor.py     # PDF text extraction, chunking, indexing
  evaluation/
    claim_extractor.py         # splits an answer into checkable claims
    verifier.py                # LLM-judge + offline lexical fallback
    hallucination_metrics.py   # per-answer scoring
    benchmark_runner.py        # aggregate scoring over a test set
scripts/
  seed_neo4j.py                # load data/sample_clinical_triplets.json
  ingest_openalex.py           # CLI: real-literature ingestion
  run_benchmark.py             # CLI benchmark entrypoint
data/
  sample_clinical_triplets.json
  eval_questions.json
tests/                         # fully mocked, offline test suite
```

## Known limitations / next steps

- Claim splitting is regex-based, not a clinical NLP model — fine for v1,
  but a scispaCy-based segmenter would improve claim boundary accuracy.
- The LLM-judge verifier makes one Groq call per claim, which is the main
  cost/latency driver of the benchmark; batching claims into fewer calls is
  the natural next optimization once accuracy is validated.
- OpenAlex triplet extraction is constrained to a fixed 6-predicate
  vocabulary (`INDICATES`, `TREATED_BY`, `REQUIRES`, `CONTRAINDICATED_WITH`,
  `CAUSES`, `PREVENTS`) — expand `_ALLOWED_PREDICATES` in
  `triplet_extractor.py` as the graph's needs grow, keeping `seed_neo4j.py`
  and any Cypher queries in sync.
- The authority-scoring weights are fixed constants, not learned — reasonable
  for an auditable v1, but worth validating against expert judgments on a
  sample of works before trusting it at scale.
