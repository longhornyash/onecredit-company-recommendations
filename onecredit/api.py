"""Read-only serving: no model loading or inference in HTTP requests."""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Literal

import psycopg
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as ApiPath
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field

from onecredit.config import Settings
from onecredit.db import make_pool
from onecredit.observability import LATENCY, REQUESTS
from onecredit.repository import Repository


class Card(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    registration_number: str = Field(alias="registrationNumber")
    company_name: str = Field(alias="companyName")
    industry: str | None
    state: str | None
    score: float


class Recommendations(BaseModel):
    registrationNumber: str
    method: Literal["dense", "hybrid"]
    scoreType: Literal["cosine_similarity", "reciprocal_rank_fusion"]
    modelVersion: str
    results: list[Card]


class Company(BaseModel):
    registrationNumber: str
    companyName: str | None
    natureOfBusiness: str | None
    industry: str | None
    state: str | None
    isActive: bool
    recommendationEligible: bool
    sourceUpdatedAt: datetime


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        async with make_pool(settings) as pool:
            await pool.wait()
            async with pool.connection() as connection:
                index = await (
                    await connection.execute(
                        "SELECT 1 FROM pg_indexes WHERE schemaname=%s AND indexname=%s",
                        (settings.database_schema, settings.search_index_name),
                    )
                ).fetchone()
            if settings.search_mode not in ("exact", "auto", "hnsw"):
                raise ValueError("SEARCH_MODE must be exact, auto or hnsw")
            if settings.search_mode == "hnsw" and not index:
                raise RuntimeError("Run scripts.index_benchmark to build and verify this model's HNSW index")
            mode = "hnsw" if index and settings.search_mode != "exact" else "exact"
            app.state.repository = Repository(pool, settings.embedding_key, settings.embedding_dimensions, mode)
            yield

    app = FastAPI(title="OneCredit company recommendations", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        label = route.path if route else "unmatched"
        REQUESTS.labels(route=label, status=str(response.status_code)).inc()
        LATENCY.labels(route=label).observe(time.perf_counter() - started)
        return response

    @app.exception_handler(psycopg.Error)
    @app.exception_handler(PoolTimeout)
    async def database_failure(request, exc):
        return JSONResponse(
            status_code=503, content={"detail": "Database temporarily unavailable"}, headers={"Retry-After": "5"}
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        async with request.app.state.repository.pool.connection() as connection:
            await connection.execute("SELECT registration_number FROM companies LIMIT 1")
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/companies/{registration_number}", response_model=Company)
    async def company(registration_number: Annotated[str, ApiPath(min_length=1, max_length=128)], request: Request):
        row = await request.app.state.repository.company(registration_number)
        if row is None:
            raise HTTPException(404, "Company not found")
        return Company(
            registrationNumber=row["registration_number"],
            companyName=row["company_name"],
            natureOfBusiness=row["nature_of_business"],
            industry=row["industry"],
            state=row["state"],
            isActive=row["is_active"],
            sourceUpdatedAt=row["source_updated_at"],
            recommendationEligible=bool(
                row["is_active"]
                and row["company_name"]
                and row["has_embedding"]
                and row["embedding_key"] == settings.embedding_key
            ),
        )

    @app.get("/companies/{registration_number}/similar", response_model=Recommendations)
    async def similar(
        registration_number: Annotated[str, ApiPath(min_length=1, max_length=128)],
        request: Request,
        limit: int = Query(default=4, ge=1, le=20),
        method: Literal["dense", "hybrid"] = "dense",
    ):
        status, rows = await request.app.state.repository.similar(registration_number, limit, method)
        if status == "missing":
            raise HTTPException(404, "Company not found or inactive")
        if status == "unavailable":
            raise HTTPException(409, "Company has no usable embedding for the configured model")
        return Recommendations(
            registrationNumber=registration_number,
            method=method,
            scoreType="cosine_similarity" if method == "dense" else "reciprocal_rank_fusion",
            modelVersion=settings.embedding_key,
            results=[Card(**row) for row in rows],
        )

    return app


app = create_app()
