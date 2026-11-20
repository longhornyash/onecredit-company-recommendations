"""Short, explicit transactions; all connections have bounded statement duration."""

from importlib.resources import files

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from onecredit.config import Settings


async def initialize(settings: Settings) -> None:
    async with await AsyncConnection.connect(settings.database_url, connect_timeout=5) as connection:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        await connection.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(settings.database_schema))
        )
        await connection.execute(files("onecredit").joinpath("schema.sql").read_text())


def make_pool(settings: Settings, max_size: int = 8) -> AsyncConnectionPool:
    async def configure(connection):
        await connection.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(settings.database_schema))
        )
        await register_vector_async(connection)
        await connection.execute(
            "SELECT set_config('statement_timeout', %s, false)", (str(settings.statement_timeout_ms),)
        )
        await connection.commit()

    return AsyncConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=max_size,
        timeout=5,
        open=False,
        configure=configure,
        kwargs={"row_factory": dict_row, "connect_timeout": 5, "prepare_threshold": None},
    )
