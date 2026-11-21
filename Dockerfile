FROM docker:28-cli AS dockercli
FROM python:3.12-slim-bookworm
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv==0.11.28 \
    && uv sync --frozen --no-dev --no-install-project --extra embedding
COPY patchpilot ./patchpilot
COPY migrations ./migrations
COPY alembic.ini ./
CMD ["uvicorn", "patchpilot.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
