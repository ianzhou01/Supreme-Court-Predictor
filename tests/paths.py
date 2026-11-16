"""File locations shared by the tests.

Paths are built from this file's location, not the current working directory,
so the tests find the data no matter which folder they are run from.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "SCDB_2026_01_caseCentered_Citation.csv"
