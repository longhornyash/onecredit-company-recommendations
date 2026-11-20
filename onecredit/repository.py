"""Database effects are atomic even if a message is delivered concurrently."""

import asyncio
import hashlib

from psycopg import sql

from onecredit.embeddings import validate_vector
from onecredit.events import InvalidEvent, parse_event


class Repository:
    def __init__(self, pool, embedding_key: str, dimensions: int = 384, search_mode="exact"):
        self.pool = pool
        self.embedding_key = embedding_key
        self.dimensions = dimensions
        if dimensions not in (384, 1024):
            raise ValueError("unsupported vector dimensions")
        self.search_mode = search_mode

    async def process(self, message_id: str | None, body: bytes, embedder) -> str:
        body_hash = hashlib.sha256(body).hexdigest()
        missing_id = (
            not isinstance(message_id, str) or not message_id.strip() or len(message_id) > 255 or "\x00" in message_id
        )
        key = f"missing:{body_hash}" if missing_id else message_id
        async with self.pool.connection() as connection:
            prior = await (
                await connection.execute("SELECT body_hash FROM processed_messages WHERE message_id=%s", (key,))
            ).fetchone()
        if prior and prior["body_hash"] == body_hash:
            return "duplicate"

        reason = "missing or unusable AMQP message_id" if missing_id else None
        event = None
        try:
            if not reason:
                event = parse_event(body)
        except InvalidEvent as exc:
            reason = str(exc)

        vector = None
        description_hash = None
        if event and event.eligible and not prior:
            async with self.pool.connection() as connection:
                current = await (
                    await connection.execute(
                        "SELECT source_updated_at, event_priority, event_hash FROM companies WHERE registration_number=%s",
                        (event.registration_number,),
                    )
                ).fetchone()
                stale = current and event.version <= (
                    current["source_updated_at"],
                    current["event_priority"],
                    current["event_hash"],
                )
                if not stale:
                    description_hash = hashlib.sha256(event.nature_of_business.encode()).hexdigest()
                    cached = await (
                        await connection.execute(
                            "SELECT embedding FROM embedding_cache WHERE embedding_key=%s AND description_hash=%s",
                            (self.embedding_key, description_hash),
                        )
                    ).fetchone()
                else:
                    cached = None
            if not stale:
                vector = validate_vector(
                    cached["embedding"]
                    if cached
                    else await asyncio.to_thread(embedder.embed, event.nature_of_business),
                    self.dimensions,
                )

        async with self.pool.connection() as connection:
            async with connection.transaction():
                claimed = await (
                    await connection.execute(
                        "INSERT INTO processed_messages(message_id,body_hash,outcome) VALUES (%s,%s,'rejected') "
                        "ON CONFLICT DO NOTHING RETURNING message_id",
                        (key, body_hash),
                    )
                ).fetchone()
                if not claimed:
                    existing = await (
                        await connection.execute("SELECT body_hash FROM processed_messages WHERE message_id=%s", (key,))
                    ).fetchone()
                    if existing["body_hash"] == body_hash:
                        return "duplicate"
                    reason = "AMQP message_id reused for a different body"

                if reason:
                    await connection.execute(
                        "INSERT INTO rejected_events(message_id,body_hash,reason,body) VALUES (%s,%s,%s,%s) "
                        "ON CONFLICT DO NOTHING",
                        (key, body_hash, reason, body),
                    )
                    return "rejected"

                # If preflight found this event stale, a monotonic row cannot become older.
                import numpy as np

                result = await (
                    await connection.execute(
                        """
                    INSERT INTO companies AS c
                      (registration_number, company_name, nature_of_business, state, industry,
                       is_active, source_updated_at, event_priority, event_hash, embedding, embedding_key)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (registration_number) DO UPDATE SET
                      company_name=excluded.company_name, nature_of_business=excluded.nature_of_business,
                      state=excluded.state, industry=excluded.industry, is_active=excluded.is_active,
                      source_updated_at=excluded.source_updated_at, event_priority=excluded.event_priority,
                      event_hash=excluded.event_hash, embedding=excluded.embedding,
                      embedding_key=excluded.embedding_key, ingested_at=clock_timestamp()
                    WHERE (excluded.source_updated_at,excluded.event_priority,excluded.event_hash)
                        > (c.source_updated_at,c.event_priority,c.event_hash)
                    RETURNING registration_number
                    """,
                        (
                            event.registration_number,
                            event.company_name,
                            event.nature_of_business,
                            event.state,
                            event.industry,
                            event.active,
                            event.updated_at,
                            event.priority,
                            event.event_hash,
                            np.array(vector, dtype=np.float32) if vector is not None else None,
                            self.embedding_key if vector is not None else None,
                        ),
                    )
                ).fetchone()
                outcome = "applied" if result else "stale"
                if result and vector is not None:
                    await connection.execute(
                        "INSERT INTO embedding_cache VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                        (self.embedding_key, description_hash, np.array(vector, dtype=np.float32)),
                    )
                await connection.execute("UPDATE processed_messages SET outcome=%s WHERE message_id=%s", (outcome, key))
                return outcome

    async def company(self, reg: str):
        async with self.pool.connection() as connection:
            return await (
                await connection.execute(
                    "SELECT registration_number,company_name,nature_of_business,state,industry,is_active,"
                    "source_updated_at,embedding IS NOT NULL AS has_embedding,embedding_key FROM companies "
                    "WHERE registration_number=%s",
                    (reg,),
                )
            ).fetchone()

    async def similar(self, reg: str, limit: int, method: str):
        async with self.pool.connection() as connection:
            async with connection.transaction():
                # Source and candidates use the same consistent snapshot.
                await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                source = await (
                    await connection.execute("SELECT * FROM companies WHERE registration_number=%s", (reg,))
                ).fetchone()
                if source is None or not source["is_active"]:
                    return "missing", []
                if source["embedding"] is None or source["embedding_key"] != self.embedding_key:
                    return "unavailable", []
                import numpy as np

                params = {
                    "reg": reg,
                    "key": self.embedding_key,
                    "limit": limit,
                    "vector": np.array(validate_vector(source["embedding"], self.dimensions), dtype=np.float32),
                    "description": source["nature_of_business"],
                }
                if method == "dense":
                    if self.search_mode == "hnsw" and limit <= 4:
                        await connection.execute("SET LOCAL hnsw.ef_search = 200")
                        await connection.execute("SET LOCAL enable_seqscan = off")
                        query = sql.SQL("""WITH candidates AS MATERIALIZED (
                            SELECT registration_number,company_name,industry,state,
                              embedding::vector({dim}) <=> %(vector)s AS distance
                            FROM companies WHERE is_active AND company_name IS NOT NULL
                              AND embedding_key=%(key)s AND registration_number<>%(reg)s
                            ORDER BY embedding::vector({dim}) <=> %(vector)s LIMIT 64)
                            SELECT registration_number,company_name,industry,state,1-distance AS score
                            FROM candidates ORDER BY distance,registration_number LIMIT %(limit)s
                            """).format(dim=sql.SQL(str(self.dimensions)))
                        rows = await (await connection.execute(query, params)).fetchall()
                        if len(rows) >= limit:
                            return "ok", rows
                    rows = await (
                        await connection.execute(
                            """
                        SELECT registration_number,company_name,industry,state,
                               1 - (embedding <=> %(vector)s) AS score
                        FROM companies WHERE is_active AND company_name IS NOT NULL
                          AND embedding_key=%(key)s AND registration_number<>%(reg)s
                        ORDER BY embedding <=> %(vector)s, registration_number LIMIT %(limit)s
                        """,
                            params,
                        )
                    ).fetchall()
                else:
                    rows = await (
                        await connection.execute(
                            """
                        WITH eligible AS MATERIALIZED (
                          SELECT * FROM companies WHERE is_active AND company_name IS NOT NULL
                            AND embedding_key=%(key)s AND registration_number<>%(reg)s
                        ), dense_top AS (
                          SELECT registration_number,embedding <=> %(vector)s AS distance FROM eligible
                          ORDER BY distance,registration_number LIMIT 30
                        ), dense AS (
                          SELECT registration_number, row_number() OVER (ORDER BY distance,registration_number) AS rank
                          FROM dense_top
                        ), query AS (
                          SELECT websearch_to_tsquery('simple',array_to_string(
                            tsvector_to_array(to_tsvector('simple',%(description)s)), ' OR ')) AS q
                        ), lexical_top AS (
                          SELECT registration_number,ts_rank_cd(search_document,q) AS relevance
                          FROM eligible,query WHERE search_document @@ q
                          ORDER BY relevance DESC,registration_number LIMIT 30
                        ), lexical AS (
                          SELECT registration_number,row_number() OVER (ORDER BY relevance DESC,registration_number) AS rank
                          FROM lexical_top
                        ), fused AS (
                          SELECT registration_number,sum(contribution) AS score FROM (
                            SELECT registration_number,1.0/(60+rank) AS contribution FROM dense
                            UNION ALL SELECT registration_number,1.0/(60+rank) FROM lexical
                          ) votes GROUP BY registration_number
                        )
                        SELECT c.registration_number,c.company_name,c.industry,c.state,f.score::float8 AS score
                        FROM fused f JOIN eligible c USING(registration_number)
                        ORDER BY score DESC,c.registration_number LIMIT %(limit)s
                        """,
                            params,
                        )
                    ).fetchall()
                return "ok", rows
