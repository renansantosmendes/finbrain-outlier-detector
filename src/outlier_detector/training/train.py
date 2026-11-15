"""Train one Isolation Forest detector per ticker and store it.

Run the command from the project root, for example::

    uv run python -m outlier_detector.training.train --tickers PETR4.SA,AAPL

For each ticker the command downloads the prices, builds the features,
evaluates the detector with walk-forward scores and stores the detector
fitted on all the features, together with a JSON file of metadata.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from outlier_detector.data.market_data import load_prices
from outlier_detector.features.price_features import (
    FEATURE_COLUMNS,
    build_features,
)
from outlier_detector.models.artifact_store import save_model
from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)
from outlier_detector.models.walk_forward import (
    REFIT_EVERY,
    TRAIN_WINDOW,
    walk_forward_scores,
)

DEFAULT_MODEL_DIRECTORY = Path("models")
DEFAULT_PERIOD = "3y"
DEFAULT_CONTAMINATION = 0.02
DEFAULT_RANDOM_SEED = 42
MODEL_FAMILY = "isolation_forest"

logger = logging.getLogger(__name__)


def train_ticker(
    ticker: str,
    period: str,
    model_directory: Path,
    contamination: float,
    random_seed: int,
) -> dict[str, Any]:
    """Fit the detector of one ticker, evaluate it and store it.

    The stored detector is fitted on every feature row of the period. The
    walk-forward evaluation only reports how many days it would have
    flagged without looking at the future, so the metric is not a test
    score.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol to train on.
    period : str
        Period of price history, for example ``3y``.
    model_directory : Path
        Folder where the detector and its metadata are stored.
    contamination : float
        Expected share of atypical days.
    random_seed : int
        Seed used by the detectors.

    Returns
    -------
    dict[str, Any]
        Metadata saved next to the detector.

    Example
    -------
    >>> from pathlib import Path
    >>> metadata = train_ticker(
    ...     "AAPL", "3y", Path("models"), 0.02, 42
    ... )  # doctest: +SKIP
    >>> metadata["ticker"]  # doctest: +SKIP
    'AAPL'
    """
    prices = load_prices(ticker, period=period)
    features = build_features(prices)
    detector = IsolationForestDetector(
        contamination=contamination,
        random_seed=random_seed,
    ).fit(features)
    walk_forward = walk_forward_scores(
        features,
        contamination=contamination,
        random_seed=random_seed,
    )
    metadata = {
        "ticker": ticker,
        "model_family": MODEL_FAMILY,
        "period": period,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(features),
        "feature_columns": FEATURE_COLUMNS,
        "contamination": contamination,
        "n_estimators": detector.n_estimators,
        "random_seed": random_seed,
        "train_window": TRAIN_WINDOW,
        "refit_every": REFIT_EVERY,
        "walk_forward_rows": len(walk_forward),
        "walk_forward_flagged_share": _flagged_share(walk_forward),
    }
    save_model(model_directory, ticker, detector, metadata)
    logger.info("Stored the model of %s with %d rows.", ticker, len(features))
    return metadata


def _flagged_share(scores: pd.Series) -> float | None:
    """Return the share of negative scores, or None when there are none.

    Parameters
    ----------
    scores : pd.Series
        Walk-forward scores. A negative score marks an atypical day.

    Returns
    -------
    float | None
        Share between 0 and 1 rounded to 4 decimals, or None when the
        series is empty.
    """
    if len(scores) == 0:
        return None
    return round(float((scores < 0).mean()), 4)


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Read the command line options of the training command.

    Parameters
    ----------
    arguments : Sequence[str] | None, optional
        Arguments without the program name. ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options, with ``tickers`` already split into a list.
    """
    parser = argparse.ArgumentParser(
        description="Train the outlier detector of one or more tickers.",
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated symbols, for example AAPL,PETR4.SA",
    )
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument(
        "--contamination",
        type=float,
        default=DEFAULT_CONTAMINATION,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY,
    )
    options = parser.parse_args(arguments)
    symbols = (symbol.strip() for symbol in options.tickers.split(","))
    options.tickers = [symbol for symbol in symbols if symbol]
    if not options.tickers:
        parser.error("--tickers needs at least one symbol.")
    return options


def main(arguments: Sequence[str] | None = None) -> int:
    """Train every ticker of the command line and report the outcome.

    Parameters
    ----------
    arguments : Sequence[str] | None, optional
        Command line arguments. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 when every ticker was trained, 1 when at least one failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    options = parse_arguments(arguments)
    failures = 0
    for ticker in options.tickers:
        try:
            train_ticker(
                ticker=ticker,
                period=options.period,
                model_directory=options.model_directory,
                contamination=options.contamination,
                random_seed=options.random_seed,
            )
        except ValueError as error:
            failures += 1
            logger.error("Could not train %s: %s", ticker, error)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
