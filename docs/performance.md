# Performance

Docker stack (one uvicorn process, one worker, Postgres 16 + pgvector) after the demo: 2,548 records, 1,032 moments, 12 trips,
2,849 mentions. Reproduce with `make demo && make load-test`.

## Reads: moment timeline (50 per page), entity lookup, trip timeline (in rotation), closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 1192 | 119.1 req/s | 9.8 ms | 11.0 ms | 12.3 ms | 0.00% |
| 8 | 3838 | 383.2 req/s | 19.6 ms | 27.8 ms | 36.1 ms | 0.00% |
| 32 | 2943 | 292.7 req/s | 85.3 ms | 102.4 ms | 128.7 ms | 0.00% |

## Search

200 sequential `POST /v1/search` calls rotating four queries (two multi-hop questions, two plain searches): p50 11.8 ms, p95 15.7 ms.
Each call embeds the query locally, runs the hybrid query, a plain vector query for comparison and, when the question allows, the chain.

Target from the blueprint: p95 under 400 ms for reads: met.

## Compute

| Work | Time |
|---|---|
| Import job: 2,549 records stored, LSA model fitted, 2,549 vectors, entities, 1,032 moments, 12 trips | about 1.6 s |
| Evaluation: six archives imported and indexed, 291 questions through the database | about 20 s |

Vector search with a narrow date filter uses pgvector's iterative index scan (`hnsw.iterative_scan = strict_order`); without it an
HNSW index can return fewer rows than asked for. A test compares the filtered result with a brute-force scan.

Not measured: a real archive. Three years here is 2,549 records; a real photo library alone is tens of thousands. The whole-archive
reindex on every import would have to become incremental, and the LSA model would need refitting on a schedule rather than per import.
