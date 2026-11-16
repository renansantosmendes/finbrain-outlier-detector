"""Train one Isolation Forest per ticker and register it in Weights & Biases.

Run the command from the project root, for example::

    uv run python -m outlier_detector.training.train --tickers PETR4.SA,AAPL

Each ticker is one wandb run. The run logs the walk-forward evaluation as
metrics and registers the detector, fitted on all the features, as a model
artifact. The API downloads that artifact when it starts.
"""

import argparse
import logging
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from outlier_detector.data.market_data import (
    DEFAULT_BENCHMARK,
    load_benchmark_log_returns,
    load_earnings_days,
    load_prices,
)
from outlier_detector.features.price_features import (
    FEATURE_COLUMNS,
    build_features,
)
from outlier_detector.models.isolation_forest_detector import (
    DEFAULT_CONTAMINATION,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_SEED,
    IsolationForestDetector,
)
from outlier_detector.models.walk_forward import (
    STEP,
    TRAIN_MIN,
    walk_forward_scores,
)
from outlier_detector.registry.wandb_model_store import log_detector
from outlier_detector.tracking.wandb_tracker import WandbExperimentTracker

DEFAULT_PROJECT_NAME = "outlier-detector"
DEFAULT_START = "2020-01-01"
MODEL_FAMILY = "isolation_forest"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingSettings:
    """Hyperparameters and data settings of one training command.

    Attributes
    ----------
    start : str
        First date of the price history, in ``YYYY-MM-DD`` format.
    benchmark : str
        Benchmark index used for the excess return.
    contamination : float
        Expected share of outliers in each training window.
    n_estimators : int
        Number of isolation trees.
    random_seed : int
        Seed of the detectors.
    train_min : int
        Number of rows in the first training window.
    step : int
        Number of rows scored before the detector is refitted.
    """

    start: str = DEFAULT_START
    benchmark: str = DEFAULT_BENCHMARK
    contamination: float = DEFAULT_CONTAMINATION
    n_estimators: int = DEFAULT_N_ESTIMATORS
    random_seed: int = DEFAULT_RANDOM_SEED
    train_min: int = TRAIN_MIN
    step: int = STEP


def train_ticker(
    ticker: str,
    settings: TrainingSettings,
    project_name: str,
    work_directory: Path,
) -> dict[str, object]:
    """Train the detector of one ticker and register it in wandb.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol to train on.
    settings : TrainingSettings
        Hyperparameters and data settings of the run.
    project_name : str
        wandb project where the run is logged.
    work_directory : Path
        Local folder where the detector file is written before upload.

    Returns
    -------
    dict[str, object]
        Metadata stored with the registered artifact.

    Raises
    ------
    ValueError
        If there are not enough feature rows to train and evaluate.
    """
    config = {
        "ticker": ticker,
        "model_family": MODEL_FAMILY,
        "feature_columns": FEATURE_COLUMNS,
        **asdict(settings),
    }
    tracker = WandbExperimentTracker(project_name=project_name, config=config)
    tracker.start_run()
    try:
        return _train_and_register(ticker, settings, tracker, work_directory)
    finally:
        tracker.finish_run()


def _train_and_register(
    ticker: str,
    settings: TrainingSettings,
    tracker: WandbExperimentTracker,
    work_directory: Path,
) -> dict[str, object]:
    """Fit, evaluate and register the detector of a ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol to train on.
    settings : TrainingSettings
        Hyperparameters and data settings of the run.
    tracker : WandbExperimentTracker
        Tracker whose run is active.
    work_directory : Path
        Local folder where the detector file is written before upload.

    Returns
    -------
    dict[str, object]
        Metadata stored with the registered artifact.

    Raises
    ------
    ValueError
        If there are not enough feature rows to train and evaluate.
    """
    prices = load_prices(ticker, settings.start)
    benchmark = load_benchmark_log_returns(settings.benchmark, settings.start)
    earnings = load_earnings_days(ticker)
    features = build_features(prices, benchmark, earnings)
    if len(features) <= settings.train_min:
        raise ValueError(
            f"Not enough history for {ticker!r}: {len(features)} feature "
            f"rows, {settings.train_min} needed before the first score."
        )

    walk_forward = walk_forward_scores(
        features,
        train_min=settings.train_min,
        step=settings.step,
        contamination=settings.contamination,
        n_estimators=settings.n_estimators,
        random_seed=settings.random_seed,
    )
    detector = IsolationForestDetector(
        contamination=settings.contamination,
        n_estimators=settings.n_estimators,
        random_seed=settings.random_seed,
    ).fit(features)

    metrics = {
        "training_rows": len(features),
        "walk_forward_rows": len(walk_forward),
        "walk_forward_flagged_share": round(
            float(walk_forward["is_atypical"].mean()), 4
        ),
    }
    tracker.log_metrics(metrics)
    metadata = {
        **metrics,
        "ticker": ticker,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
    }
    log_detector(tracker, detector, ticker, metadata, work_directory)
    logger.info("Registered the detector of %s.", ticker)
    return metadata


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
        description="Train and register the outlier detector of tickers.",
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated symbols, for example AAPL,PETR4.SA",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--contamination",
        type=float,
        default=DEFAULT_CONTAMINATION,
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    parser.add_argument("--train-min", type=int, default=TRAIN_MIN)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_PROJECT_NAME,
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
        0 when every ticker was registered, 1 when at least one failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    options = parse_arguments(arguments)
    settings = TrainingSettings(
        start=options.start,
        benchmark=options.benchmark,
        contamination=options.contamination,
        n_estimators=options.n_estimators,
        random_seed=options.random_seed,
        train_min=options.train_min,
        step=options.step,
    )
    failures = 0
    with tempfile.TemporaryDirectory() as work_directory:
        for ticker in options.tickers:
            try:
                train_ticker(
                    ticker=ticker,
                    settings=settings,
                    project_name=options.wandb_project,
                    work_directory=Path(work_directory),
                )
            except (ValueError, RuntimeError) as error:
                failures += 1
                logger.error("Could not train %s: %s", ticker, error)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
