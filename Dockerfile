FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
ARG MODEL_FILE=models/model.pkl
ARG FEATURES_FILE=models/features.json
COPY ${MODEL_FILE} ./models/model.pkl
COPY ${FEATURES_FILE} ./models/features.json
ENV MODEL_PATH=/app/models/model.pkl
ENV FEATURES_PATH=/app/models/features.json
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]