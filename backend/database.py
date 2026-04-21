"""Database connection and initialization."""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

from config import settings
from models import Base

# Convert postgresql:// to postgresql+asyncpg://
async_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(async_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create tables, enable pgvector, add tsvector for hybrid search (idempotent)."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        # Phase 2 migration: tsvector column + GIN index for hybrid keyword search.
        # Idempotent — uses IF NOT EXISTS guards so safe to run on every startup.
        await conn.execute(text("""
            ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (content_tsv)"
        ))


async def get_session() -> AsyncSession:
    """Dependency that yields a database session."""
    async with async_session() as session:
        yield session
