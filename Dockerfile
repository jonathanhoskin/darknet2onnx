ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim AS base
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
WORKDIR /src
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install --system .

FROM base AS final
ARG PYTHON_VERSION
WORKDIR /d2o
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY . .
ENTRYPOINT [ "python", "darknet2onnx.py" ]
