FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LightGBM links against the GNU OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN pip install . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p data/raw data/processed models reports/figures \
    && chown -R appuser:appuser /app

USER appuser

# Training is not part of the image: mount a trained models/ directory at runtime.
# docker run -v "$(pwd)/models:/app/models:ro" -p 8000:8000 credit-risk:latest
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "credit_risk.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
