FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY futuresim.py server.py ./
COPY agents ./agents
COPY environment ./environment
COPY inference ./inference
COPY integrations ./integrations

RUN pip install --no-cache-dir ".[openreward]"

EXPOSE 8080

CMD ["python", "server.py"]
