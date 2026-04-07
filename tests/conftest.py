from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def feast_repo_path(project_root: Path) -> Path:
    return project_root / "feature_repo"


@pytest.fixture(scope="session")
def feast_data_dir(project_root: Path) -> Path:
    return project_root / "data" / "feast"
