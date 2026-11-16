"""Save and download Isolation Forest detectors as wandb model artifacts."""

import re
from pathlib import Path

import wandb

from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)
from outlier_detector.tracking.wandb_tracker import WandbExperimentTracker

ARTIFACT_PREFIX = "outlier-detector-"
UNSAFE_CHARACTERS = re.compile(r"[^\w.-]")


def build_ticker_slug(ticker: str) -> str:
    """Convert a ticker into a slug that is safe in artifact names.

    Parameters
    ----------
    ticker : str
        Ticker symbol, for example ``PETR4.SA`` or ``^BVSP``.

    Returns
    -------
    str
        Lowercase ticker in which every character other than letters,
        digits, ``_``, ``.`` and ``-`` is replaced by ``-``.

    Example
    -------
    >>> build_ticker_slug("PETR4.SA")
    'petr4.sa'
    """
    return UNSAFE_CHARACTERS.sub("-", ticker.lower())


def build_artifact_name(ticker: str) -> str:
    """Return the wandb artifact name of the detector of a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    str
        Name such as ``outlier-detector-aapl``.

    Example
    -------
    >>> build_artifact_name("AAPL")
    'outlier-detector-aapl'
    """
    return f"{ARTIFACT_PREFIX}{build_ticker_slug(ticker)}"


def log_detector(
    tracker: WandbExperimentTracker,
    detector: IsolationForestDetector,
    ticker: str,
    metadata: dict[str, object],
    work_directory: Path,
) -> None:
    """Serialize a fitted detector and register it as a wandb artifact.

    Parameters
    ----------
    tracker : WandbExperimentTracker
        Tracker whose run is active.
    detector : IsolationForestDetector
        Fitted detector to register.
    ticker : str
        Ticker the detector belongs to.
    metadata : dict[str, object]
        Facts about the training run, stored with the artifact.
    work_directory : Path
        Local folder where the joblib file is written before upload.
    """
    work_directory.mkdir(parents=True, exist_ok=True)
    model_file = work_directory / f"{build_ticker_slug(ticker)}.joblib"
    detector.save(model_file)
    tracker.log_model(
        model_file_path=model_file,
        model_name=build_artifact_name(ticker),
        description=f"Isolation Forest outlier detector of {ticker}.",
        metadata=metadata,
    )


def download_detector(
    api: wandb.Api,
    project_path: str,
    ticker: str,
    download_directory: Path,
) -> tuple[IsolationForestDetector, dict[str, object]]:
    """Download the latest detector of a ticker from wandb.

    Parameters
    ----------
    api : wandb.Api
        Authenticated wandb client.
    project_path : str
        Project path in the form ``<entity>/<project>``.
    ticker : str
        Ticker symbol.
    download_directory : Path
        Local folder where the artifact files are stored.

    Returns
    -------
    tuple[IsolationForestDetector, dict[str, object]]
        The fitted detector and the metadata stored with its artifact.

    Raises
    ------
    wandb.errors.CommError
        If the artifact does not exist or wandb is unreachable.
    """
    artifact_path = f"{project_path}/{build_artifact_name(ticker)}:latest"
    artifact = api.artifact(artifact_path, type="model")
    directory = artifact.download(root=str(download_directory))
    model_file = next(Path(directory).glob("*.joblib"))
    return IsolationForestDetector.load(model_file), dict(artifact.metadata)
