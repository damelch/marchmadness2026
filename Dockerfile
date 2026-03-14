FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null; true

COPY . .
RUN pip install --no-cache-dir ".[dev]"

ENTRYPOINT ["marchmadness"]
