FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null; true

COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

ENTRYPOINT ["marchmadness"]
