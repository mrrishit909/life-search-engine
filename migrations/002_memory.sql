-- Domain model. Names follow the blueprint's schema; additions are marked (+).
-- The blueprint's owner_id is tenant_id here: one tenant is one person's (or one household's) archive, and row-level
-- security keeps every archive to its owner. Vectors are 128-dimensional (a local LSA model), not the blueprint's 1536.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE artifacts (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, kind text NOT NULL, source text NOT NULL, uri text NOT NULL, captured_at timestamptz NOT NULL,
  checksum text NOT NULL, metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  ends_at timestamptz, body text NOT NULL,                                                                                   -- (+) the text stands in for object storage
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED,                                                    -- (+)
  UNIQUE (tenant_id, checksum));
CREATE INDEX artifacts_when ON artifacts (tenant_id, captured_at);
CREATE INDEX artifacts_tsv ON artifacts USING gin (tsv);
SELECT enable_tenant_rls('artifacts');

CREATE TABLE entities (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, entity_type text NOT NULL, canonical_name text NOT NULL, attributes jsonb NOT NULL DEFAULT '{}'::jsonb);
CREATE INDEX entities_name ON entities (tenant_id, entity_type, canonical_name);
SELECT enable_tenant_rls('entities');

CREATE TABLE mentions (artifact_id uuid REFERENCES artifacts ON DELETE CASCADE, entity_id uuid REFERENCES entities ON DELETE CASCADE, span jsonb, confidence real, PRIMARY KEY (artifact_id, entity_id),
  tenant_id uuid NOT NULL REFERENCES tenants);                                                                               -- (+)
CREATE INDEX mentions_entity ON mentions (entity_id);
SELECT enable_tenant_rls('mentions');

CREATE TABLE events (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, event_type text NOT NULL, starts_at timestamptz NOT NULL, ends_at timestamptz NOT NULL,
  place_id uuid REFERENCES entities ON DELETE SET NULL, summary text,
  parent_id uuid REFERENCES events ON DELETE SET NULL);                                                                      -- (+) the trip a moment belongs to
CREATE INDEX events_when ON events (tenant_id, event_type, starts_at);
SELECT enable_tenant_rls('events');

CREATE TABLE event_artifacts (event_id uuid REFERENCES events ON DELETE CASCADE, artifact_id uuid REFERENCES artifacts ON DELETE CASCADE, relevance real, PRIMARY KEY (event_id, artifact_id),
  tenant_id uuid NOT NULL REFERENCES tenants);                                                                               -- (+)
CREATE INDEX event_artifacts_artifact ON event_artifacts (artifact_id);
SELECT enable_tenant_rls('event_artifacts');

CREATE TABLE embeddings (artifact_id uuid PRIMARY KEY REFERENCES artifacts ON DELETE CASCADE, vector vector(128) NOT NULL,
  tenant_id uuid NOT NULL REFERENCES tenants);                                                                               -- (+)
CREATE INDEX embeddings_hnsw ON embeddings USING hnsw (vector vector_cosine_ops);
SELECT enable_tenant_rls('embeddings');

CREATE TABLE indexes (tenant_id uuid PRIMARY KEY REFERENCES tenants, embedder bytea NOT NULL, fitted_on int NOT NULL, version text NOT NULL, fitted_at timestamptz NOT NULL DEFAULT now());   -- (+) each archive's own embedding model
SELECT enable_tenant_rls('indexes');

-- (+) A purge must survive the next sync: the fingerprint (never the content) of a purged record is kept so an import skips it.
CREATE TABLE tombstones (tenant_id uuid NOT NULL REFERENCES tenants, checksum text NOT NULL, purged_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (tenant_id, checksum));
SELECT enable_tenant_rls('tombstones');
