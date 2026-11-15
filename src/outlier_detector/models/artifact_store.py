"""Store and read the detectors trained for each ticker.

Each ticker produces two files in the model directory: a joblib file with
the fitted detector and a JSON file with its metadata. The file names come
from the ticker, which can be user input, so they are sanitized before use.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)

UNSAFE_FILE_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]")


class ModelNotFoundError(LookupError):
    """Raised when no trained detector exists for a ticker."""


def safe_file_stem(ticker: str) -> str:
    """Turn a ticker into a file name that cannot escape the model folder.

    Every character other than letters, digits, ``_`` and ``-`` becomes
    ``_``, so ``PETR4.SA`` becomes ``PETR4_SA`` and ``../x`` becomes
    ``___x``.

    Parameters
    ----------
    ticker : str
        Ticker symbol as given by the user or by the training command.

    Returns
    -------
    str
        File name without extension.

    Example
    -------
    >>> safe_file_stem("PETR4.SA")
    'PETR4_SA'
    """
    return UNSAFE_FILE_CHARACTERS.sub("_", ticker.strip())


def model_path(model_directory: Path, ticker: str) -> Path:
    """Return the joblib path of the detector of a ticker.

    Parameters
    ----------
    model_directory : Path
        Folder that holds the trained models.
    ticker : str
        Ticker symbol.

    Returns
    -------
    Path
        Path of the ``.joblib`` file, which may not exist yet.
    """
    return model_directory / f"{safe_file_stem(ticker)}.joblib"


def metadata_path(model_directory: Path, ticker: str) -> Path:
    """Return the JSON metadata path of the detector of a ticker.

    Parameters
    ----------
    model_directory : Path
        Folder that holds the trained models.
    ticker : str
        Ticker symbol.

    Returns
    -------
    Path
        Path of the ``.json`` file, which may not exist yet.
    """
    return model_directory / f"{safe_file_stem(ticker)}.json"


def save_model(
    model_directory: Path,
    ticker: str,
    detector: IsolationForestDetector,
    metadata: dict[str, Any],
) -> None:
    """Write the detector and its metadata of a ticker.

    The folder is created when it does not exist. Existing files of the
    same ticker are replaced.

    Parameters
    ----------
    model_directory : Path
        Folder that holds the trained models.
    ticker : str
        Ticker symbol the detector belongs to.
    detector : IsolationForestDetector
        Fitted detector to store.
    metadata : dict[str, Any]
        JSON-serializable facts about the training run, such as the period,
        the number of rows and the training date.
    """
    model_directory.mkdir(parents=True, exist_ok=True)
    detector.save(model_path(model_directory, ticker))
    with metadata_path(model_directory, ticker).open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
    load_model.cache_clear()


@lru_cache(maxsize=32)
def load_model(
    model_directory: Path,
    ticker: str,
) -> IsolationForestDetector:
    """Read the detector of a ticker and keep it in memory.

    Parameters
    ----------
    model_directory : Path
        Folder that holds the trained models.
    ticker : str
        Ticker symbol.

    Returns
    -------
    IsolationForestDetector
        The stored detector.

    Raises
    ------
    ModelNotFoundError
        If the ticker has no stored detector.
    """
    path = model_path(model_directory, ticker)
    if not path.is_file():
        raise ModelNotFoundError(f"No trained model for ticker {ticker!r}.")
    return IsolationForestDetector.load(path)


def load_metadata(model_directory: Path, ticker: str) -> dict[str, Any]:
    """Read the metadata written with the detector of a ticker.

    Parameters
    ----------
    model_directory : Path
        Folder that holds the trained models.
    ticker : str
        Ticker symbol.

    Returns
    -------
    dict[str, Any]
        Metadata saved by ``save_model``.

    Raises
    ------
    ModelNotFoundError
        If the ticker has no metadata file.
    """
    path = metadata_path(model_directory, ticker)
    if not path.is_file():
        raise ModelNotFoundError(f"No metadata for ticker {ticker!r}.")
    with path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def list_trained_tickers(model_directory: Path) -> list[str]:
    """List the tickers that have a stored detector.

    Parameters
    ----------
    model_directory : Path
        Folder that holds the trained models. A missing folder means no
        trained tickers.

    Returns
    -------
    list[str]
        Tickers in alphabetical order, taken from the metadata files.
    """
    if not model_directory.is_dir():
        return []
    tickers = []
    for metadata_file in sorted(model_directory.glob("*.json")):
        with metadata_file.open(encoding="utf-8") as stream:
            tickers.append(json.load(stream)["ticker"])
    return sorted(tickers)
