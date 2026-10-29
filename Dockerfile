# Build stage: install the dependencies into one folder
FROM python:3.12-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# Runtime stage: Python, the dependencies and the app, nothing else
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY alembic.ini ./
COPY alembic ./alembic
COPY frontend ./frontend
COPY src ./src
COPY scripts/seed.py ./scripts/

# The app writes sessions here, so the non-root user must own it
RUN mkdir sessions && chown app:app sessions
USER app

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
