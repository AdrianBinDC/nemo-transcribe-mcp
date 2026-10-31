# MCP client image only. Does not include model weights or vLLM.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 1000 mcp \
    && mkdir -p /data/audio /data/work \
    && chown -R mcp:mcp /data /app

USER mcp

# Settings come from Compose env_file / environment at start time — not baked in.
CMD ["nemo-transcribe-mcp"]
