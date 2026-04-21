"""Add tsvector column and GIN index for keyword search.

Revision ID: 20260420_tsvector
Revises: None
Create Date: 2026-04-20
"""

from alembic import op

# revision identifiers
revision = "20260420_tsvector"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    """)
    op.execute("CREATE INDEX idx_chunks_tsv ON chunks USING GIN (content_tsv)")


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_chunks_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
