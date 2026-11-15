"""Shared fixtures of the outlier detector tests."""

import numpy as np
import pandas as pd
import pytest


def make_prices(
    days: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic price frame with business-day dates.

    Parameters
    ----------
    days : int
        Number of trading days.
    seed : int, optional
        Seed of the random generator. The default is 0.

    Returns
    -------
    pd.DataFrame
        Frame with the columns ``Close`` and ``Volume``.
    """
    generator = np.random.default_rng(seed)
    returns = generator.normal(0.0005, 0.01, size=days)
    close = 20 * np.exp(np.cumsum(returns))
    volume = generator.integers(1_000, 5_000, size=days).astype(float)
    index = pd.bdate_range("2023-01-02", periods=days)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=index)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Provide 400 business days of synthetic prices.

    Returns
    -------
    pd.DataFrame
        Frame with the columns ``Close`` and ``Volume``.
    """
    return make_prices(400)
