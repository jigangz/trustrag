"""Tests for the tsvector migration (P2-1).

These tests verify the migration SQL is syntactically correct and produces
the expected schema changes. They do NOT require a live database — they
validate the migration script structure and SQL statements.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def migration_module():
    """Load the migration module directly."""
    migration_path = (
        Path(__file__).resolve().parent.parent
        / "alembic"
        / "versions"
        / "20260420_add_tsvector.py"
    )
    spec = importlib.util.spec_from_file_location("migration_tsvector", migration_path)
    module = importlib.util.module_from_spec(spec)
    # Provide mock alembic.op
    mock_op = MagicMock()
    sys.modules["alembic"] = MagicMock()
    sys.modules["alembic.op"] = mock_op
    spec.loader.exec_module(module)
    return module, mock_op


class TestTsvectorMigration:
    def test_revision_id_set(self, migration_module):
        module, _ = migration_module
        assert module.revision == "20260420_tsvector"

    def test_down_revision_is_none(self, migration_module):
        module, _ = migration_module
        assert module.down_revision is None

    def test_upgrade_adds_column_and_index(self, migration_module):
        module, _ = migration_module
        mock_op = MagicMock()

        # Patch op at module level
        module.op = mock_op
        module.upgrade()

        assert mock_op.execute.call_count == 2
        upgrade_sql_1 = mock_op.execute.call_args_list[0][0][0]
        upgrade_sql_2 = mock_op.execute.call_args_list[1][0][0]

        # First statement adds the tsvector column
        assert "content_tsv" in upgrade_sql_1
        assert "tsvector" in upgrade_sql_1
        assert "GENERATED ALWAYS AS" in upgrade_sql_1
        assert "to_tsvector" in upgrade_sql_1
        assert "'english'" in upgrade_sql_1
        assert "STORED" in upgrade_sql_1

        # Second statement creates GIN index
        assert "idx_chunks_tsv" in upgrade_sql_2
        assert "GIN" in upgrade_sql_2

    def test_downgrade_drops_index_and_column(self, migration_module):
        module, _ = migration_module
        mock_op = MagicMock()

        module.op = mock_op
        module.downgrade()

        assert mock_op.execute.call_count == 2
        downgrade_sql_1 = mock_op.execute.call_args_list[0][0][0]
        downgrade_sql_2 = mock_op.execute.call_args_list[1][0][0]

        # First drops index
        assert "DROP INDEX" in downgrade_sql_1
        assert "idx_chunks_tsv" in downgrade_sql_1

        # Second drops column
        assert "DROP COLUMN" in downgrade_sql_2
        assert "content_tsv" in downgrade_sql_2

    def test_upgrade_downgrade_are_inverse(self, migration_module):
        """Verify upgrade adds what downgrade removes."""
        module, _ = migration_module
        mock_op = MagicMock()
        module.op = mock_op

        module.upgrade()
        module.downgrade()

        # 4 total calls: 2 upgrade + 2 downgrade
        assert mock_op.execute.call_count == 4
