FROM python:3.11-slim

WORKDIR /app

# libgomp1: requerido en runtime por scikit-learn/TensorFlow (OpenMP) en imágenes slim
# curl: usado por el HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY tests/ tests/
COPY data/ data/
COPY pyproject.toml .
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TF_CPP_MIN_LOG_LEVEL=3

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["app"]
