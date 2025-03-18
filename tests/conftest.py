import pytest
import os

def pytest_ignore_collect(path, config):
    """Ignore test files in the data directory."""
    if str(path).startswith(os.path.join('tests', 'data')):
        return True 