# tests/conftest.py
"""Shared fixtures available to all test modules."""
import os
import pytest

# Run tests from the project root so relative data paths resolve correctly.
@pytest.fixture(autouse=True)
def _chdir_to_project_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prev = os.getcwd()
    os.chdir(root)
    yield
    os.chdir(prev)
