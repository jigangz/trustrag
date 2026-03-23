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
    """Create tables and enable pgvector extension."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Dependency that yields a database session."""
    async with async_session() as session:
        yield session
