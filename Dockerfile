FROM python:3.11-slim AS base
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.pkl \
    FEATURES_PATH=/app/models/features.json
WORKDIR /app


# ---------- API ----------
FROM base AS api
COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-api.txt
COPY src ./src
COPY configs ./configs
COPY models/model.pkl ./models/model.pkl
COPY models/features.json ./models/features.json
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ---------- Monitoring ----------
FROM base AS monitor
COPY requirements-monitor.txt requirements-api.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-monitor.txt
COPY src ./src
COPY configs ./configs
COPY models/model.pkl ./models/model.pkl
COPY models/features.json ./models/features.json
CMD ["python", "-m", "src.monitoring"]