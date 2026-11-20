set shell := ["bash", "-euo", "pipefail", "-c"]

default: setup

# Install the producer's dependencies.
setup:
    uv sync --locked

# Start Postgres (with pgvector) and RabbitMQ, and wait until both are healthy.
up:
    docker compose up -d --wait

# Stop everything and delete the volumes.
down:
    docker compose down -v

# Replay the registry feed onto the company-events queue.
publish:
    uv run python -m producer.publish

# Create the schema explicitly (the read-only API never changes it).
init-db:
    uv run python -m onecredit.cli init-db

# Consume indefinitely; bounded prefetch and ack after database commit.
consume:
    uv run python -m onecredit.consumer

# Drain a published feed, then exit after five idle seconds.
drain:
    uv run python -m onecredit.consumer --idle-seconds 5

serve:
    uv run uvicorn onecredit.api:app --host 127.0.0.1 --port 8000

test:
    uv run pytest -q

# Explicitly opt into a disposable test database and the local broker.
test-integration:
    TEST_DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:${POSTGRES_PORT:-5432}/companies}" TEST_RABBITMQ_URL="${RABBITMQ_URL:-amqp://guest:guest@localhost:${RABBITMQ_PORT:-5672}/}" uv run pytest -q

lint:
    uv run ruff check onecredit tests evaluation scripts
    uv run ruff format --check onecredit tests evaluation scripts

audit:
    uv run python -m onecredit.cli audit

evaluate:
    uv run python -m evaluation.run --output evaluation/results/evaluation.json

# Full official feed, model/index validation, live HTTP load, quality and replay.
pipeline: up
    uv run python -m scripts.pipeline

# Run the complete ingest/audit path and a temporary HTTP server for the demo.
demo: up init-db publish drain audit
    uv run python -m scripts.demo
