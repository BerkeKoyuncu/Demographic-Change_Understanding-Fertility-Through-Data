import numpy as np
import pandas as pd


def test_imports() -> None:
    """Simple smoke test for imports."""
    assert pd is not None
    assert np is not None
