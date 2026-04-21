"""Shared test fixtures for backend tests."""

import sys
from pathlib import Path

# Ensure backend module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
