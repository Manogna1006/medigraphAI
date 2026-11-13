"""
Runs the full pipeline (retrieve -> synthesize -> score) over a batch of test
questions and produces an aggregate hallucination report — the artifact you'd
put in a README, paper, or dashboard ("system-wide hallucination rate: 6.2%").
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.hallucination_metrics import HallucinationScorer
from app.models.schemas import HallucinationReport, SynthesizedAnswer
from app.retrieval.hybrid_retriever import HybridRetriever
from app.synthesis.llm_engine import LLMSynthesisEngine
from app.utils.logger import logger


@dataclass
class BenchmarkResult:
    per_query_reports: list[HallucinationReport] = field(default_factory=list)
    aggregate_hallucination_rate: float = 0.0
    aggregate_groundedness_score: float = 0.0
    aggregate_contradiction_rate: float = 0.0
    abstention_rate: float = 0.0
    p50_latency_seconds: float = 0.0
    p95_latency_seconds: float = 0.0
    total_queries: int = 0

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "aggregate_hallucination_rate": self.aggregate_hallucination_rate,
            "aggregate_groundedness_score": self.aggregate_groundedness_score,
            "aggregate_contradiction_rate": self.aggregate_contradiction_rate,
            "abstention_rate": self.abstention_rate,
            "p50_latency_seconds": self.p50_latency_seconds,
            "p95_latency_seconds": self.p95_latency_seconds,
            "per_query_reports": [r.model_dump() for r in self.per_query_reports],
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * pct), len(sorted_vals) - 1)
    return round(sorted_vals[idx], 3)


def run_benchmark(questions: list[str], retriever: HybridRetriever | None = None,
                   engine: LLMSynthesisEngine | None = None, scorer: HallucinationScorer | None = None) -> BenchmarkResult:
    retriever = retriever or HybridRetriever()
    engine = engine or LLMSynthesisEngine()
    scorer = scorer or HallucinationScorer()

    reports: list[HallucinationReport] = []
    latencies: list[float] = []
    abstentions = 0

    for question in questions:
        try:
            context = retriever.retrieve(question)
            answer: SynthesizedAnswer = engine.synthesize(context)
            report = scorer.score(answer)
            reports.append(report)
            latencies.append(answer.latency_seconds)
            if answer.abstained:
                abstentions += 1
        except Exception as exc:
            logger.error("Benchmark query {!r} failed end-to-end: {}. Skipping.", question, exc)

    total = len(reports)
    if total == 0:
        logger.warning("Benchmark produced zero successful reports.")
        return BenchmarkResult()

    non_abstained = [r for r in reports if not r.abstained]
    hallucination_rates = [r.hallucination_rate for r in non_abstained]
    groundedness_scores = [r.groundedness_score for r in non_abstained]
    contradiction_rates = [r.contradiction_rate for r in non_abstained]

    return BenchmarkResult(
        per_query_reports=reports,
        aggregate_hallucination_rate=round(statistics.mean(hallucination_rates), 4) if hallucination_rates else 0.0,
        aggregate_groundedness_score=round(statistics.mean(groundedness_scores), 4) if groundedness_scores else 1.0,
        aggregate_contradiction_rate=round(statistics.mean(contradiction_rates), 4) if contradiction_rates else 0.0,
        abstention_rate=round(abstentions / total, 4),
        p50_latency_seconds=_percentile(latencies, 0.50),
        p95_latency_seconds=_percentile(latencies, 0.95),
        total_queries=total,
    )


def save_report(result: BenchmarkResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Benchmark report saved to {}", path)
