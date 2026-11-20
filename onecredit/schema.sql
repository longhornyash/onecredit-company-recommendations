CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS companies (
    registration_number TEXT PRIMARY KEY,
    company_name TEXT,
    nature_of_business TEXT,
    state TEXT,
    industry TEXT,
    is_active BOOLEAN NOT NULL,
    source_updated_at TIMESTAMPTZ NOT NULL,
    event_priority SMALLINT NOT NULL,
    event_hash TEXT NOT NULL,
    embedding vector,
    embedding_key TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    search_document tsvector GENERATED ALWAYS AS
        (to_tsvector('simple'::regconfig, coalesce(nature_of_business, ''))) STORED,
    CONSTRAINT companies_embedding_dimension_valid CHECK (embedding IS NULL OR vector_dims(embedding) IN (384,1024)),
    CHECK ((embedding IS NULL) = (embedding_key IS NULL)),
    CHECK (embedding IS NULL OR (is_active AND nature_of_business IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS processed_messages (
    message_id TEXT PRIMARY KEY,
    body_hash TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('applied', 'stale', 'rejected')),
    processed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS rejected_events (
    message_id TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    reason TEXT NOT NULL,
    body BYTEA NOT NULL,
    rejected_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (message_id, body_hash)
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    embedding_key TEXT NOT NULL,
    description_hash TEXT NOT NULL,
    embedding vector NOT NULL CHECK (vector_dims(embedding) IN (384,1024)),
    PRIMARY KEY (embedding_key, description_hash)
);

CREATE INDEX IF NOT EXISTS companies_search_document_idx ON companies USING gin(search_document);

-- Explicit init-db upgrades the original 384-d schema without dropping data.
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid='companies'::regclass
             AND attname='embedding' AND atttypmod <> -1) THEN
    ALTER TABLE companies ALTER COLUMN embedding TYPE vector;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid='embedding_cache'::regclass
             AND attname='embedding' AND atttypmod <> -1) THEN
    ALTER TABLE embedding_cache ALTER COLUMN embedding TYPE vector;
  END IF;
END $$;
