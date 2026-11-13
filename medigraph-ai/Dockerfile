FROM python:3.12-slim

WORKDIR /app

# System deps for chromadb (sqlite) and pypdf are covered by the slim image already;
# no extra apt packages required at this pin.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# No secrets baked into the image — GROQ_API_KEY / NEO4J_* are supplied at
# runtime via `docker run --env-file .env` or your orchestrator's secrets.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=3).raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
