FROM python:3.12-slim

WORKDIR /opt/futuresim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap ca-certificates curl git socat \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY futuresim.py pathing.py server.py ./
COPY agents ./agents
COPY environment ./environment
COPY inference ./inference
COPY integrations ./integrations

RUN pip install --no-cache-dir ".[hosted-hybrid]"

CMD ["tail", "-f", "/dev/null"]
