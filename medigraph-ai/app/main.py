"""
MediGraph AI — FastAPI entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.clinical.vitals_analyzer import vitals_to_evidence_text
from app.evaluation.benchmark_runner import run_benchmark
from app.evaluation.hallucination_metrics import HallucinationScorer
from app.models.schemas import QueryRequest, QueryResponse
from app.retrieval.hybrid_retriever import HybridRetriever
from app.synthesis.llm_engine import LLMSynthesisEngine
from app.utils.logger import logger

_retriever: HybridRetriever | None = None
_engine: LLMSynthesisEngine | None = None
_scorer: HallucinationScorer | None = None


def _build_patient_context_text(patient_context) -> str:
    """
    Renders patient context as explicit prose for the synthesis prompt,
    rather than a raw JSON dump — vitals are pre-flagged against reference
    ranges (see app.clinical.vitals_analyzer) so the LLM is told "132 bpm is
    ABOVE normal: tachycardia" instead of being asked to infer that itself.
    """
    if patient_context is None:
        return ""
    lines = []
    if patient_context.age is not None:
        lines.append(f"Age: {patient_context.age}")
    if patient_context.symptoms:
        lines.append(f"Reported symptoms: {', '.join(patient_context.symptoms)}")
    vitals_text = vitals_to_evidence_text(patient_context.vitals)
    if vitals_text:
        lines.append("Vitals:\n" + vitals_text)
    if patient_context.notes:
        lines.append(f"Clinical notes: {patient_context.notes}")
    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _engine, _scorer
    _retriever = HybridRetriever()
    _engine = LLMSynthesisEngine()
    _scorer = HallucinationScorer()
    if not _retriever._graph.verify_connectivity():
        logger.warning("Starting up without a verified Neo4j connection — KG retrieval will return empty results.")
    yield
    _retriever.close()


app = FastAPI(
    title="MediGraph AI",
    description="A GraphRAG framework for hallucination-resistant medical decision support.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # Last line of defense: never leak a stack trace to the client. Everything
    # module-specific should already be caught closer to the source.
    logger.error("Unhandled exception on {}: {}", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal error. Please try again."})


@app.get("/")
def root() -> dict:
    return {
        "name": "MediGraph AI API",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "POST /query",
            "benchmark": "POST /benchmark",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
def health() -> dict:
    neo4j_ok = _retriever._graph.verify_connectivity() if _retriever else False
    return {"status": "ok", "neo4j_connected": neo4j_ok}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, include_hallucination_report: bool = True) -> QueryResponse:
    if _retriever is None or _engine is None or _scorer is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")

    context = _retriever.retrieve(
        request.question,
        extra_kg_terms=request.patient_context.symptoms if request.patient_context else None,
    )
    patient_text = _build_patient_context_text(request.patient_context)
    answer = _engine.synthesize(context, patient_context_text=patient_text)

    report = None
    if include_hallucination_report:
        try:
            report = _scorer.score(answer)
        except Exception as exc:
            logger.error("Hallucination scoring failed (answer still returned): {}", exc)

    return QueryResponse(answer=answer, hallucination_report=report)


@app.post("/benchmark")
def benchmark(questions: list[str]) -> dict:
    if not questions:
        raise HTTPException(status_code=400, detail="Provide at least one question.")
    if len(questions) > 50:
        raise HTTPException(status_code=400, detail="Max 50 questions per benchmark run — batch larger sets offline.")
    result = run_benchmark(questions, retriever=_retriever, engine=_engine, scorer=_scorer)
    return result.to_dict()
