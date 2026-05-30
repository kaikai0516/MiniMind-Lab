# ── Base: PyTorch with CUDA ───────────────────────────────────────
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

# ── Dev: editable install + all extras ────────────────────────────
FROM base AS dev
COPY . .
RUN pip install --no-cache-dir -e ".[all]"
EXPOSE 8000
CMD ["minimind-serve"]

# ── Prod: minimal dependencies, only serve ────────────────────────
FROM base AS prod
COPY . .
RUN pip install --no-cache-dir -e ".[serve]"
EXPOSE 8000
CMD ["minimind-serve"]
