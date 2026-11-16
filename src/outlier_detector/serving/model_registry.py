"""In-memory registry of the detectors downloaded from Weights & Biases."""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

import wandb
from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)
from outlier_detector.registry.wandb_model_store import (
    ARTIFACT_PREFIX,
    download_detector,
)

DEFAULT_PROJECT_NAME = "outlier-detector"
WANDB_DIRECTORY_VARIABLES = {
    "WANDB_CONFIG_DIR": "config",
    "WANDB_CACHE_DIR": "cache",
    "WANDB_DATA_DIR": "data",
}

logger = logging.getLogger(__name__)


def use_writable_wandb_directories(base_directory: Path | None = None) -> None:
    """Point the W&B client to writable folders, unless they are already set.

    Hosts such as Vercel make the home folder read-only, and the W&B client
    writes its configuration there when it starts. Setting these variables
    before the client exists moves that writing to a temporary folder.

    Parameters
    ----------
    base_directory : Path | None, optional
        Folder that receives the ``config``, ``cache`` and ``data``
        subfolders. ``None`` uses ``wandb`` inside the temporary folder.
    """
    root = base_directory or Path(tempfile.gettempdir()) / "wandb"
    for variable, subfolder in WANDB_DIRECTORY_VARIABLES.items():
        os.environ.setdefault(variable, str(root / subfolder))


class ModelNotFoundError(LookupError):
    """Raised when no detector is loaded in memory for a ticker."""


@dataclass
class LoadedModel:
    """Detector kept in memory together with the metadata of its artifact.

    Attributes
    ----------
    ticker : str
        Ticker the detector was trained on.
    detector : IsolationForestDetector
        Fitted detector.
    metadata : dict[str, object]
        Facts stored with the artifact, such as the training date.
    """

    ticker: str
    detector: IsolationForestDetector
    metadata: dict[str, object] = field(default_factory=dict)


class WandbModelRegistry:
    """Load the latest detector of each ticker from wandb into memory.

    The registry downloads an artifact once and keeps the detector in a
    dictionary, so requests are answered without calling wandb. The API key
    is read from ``WANDB_API_KEY`` in the environment or the ``.env`` file.

    Parameters
    ----------
    project_name : str, optional
        wandb project that holds the artifacts. The default is
        ``outlier-detector``.
    entity : str | None, optional
        wandb entity (user or team). ``None`` uses the default entity of the
        account.
    download_directory : Path | None, optional
        Local folder for the downloaded files. ``None`` uses a temporary
        folder, which works on read-only hosts such as Vercel.
    env_file_path : str, optional
        Path to the ``.env`` file. The default is ``.env``.

    Attributes
    ----------
    models : dict[str, LoadedModel]
        Detectors kept in memory, indexed by ticker.

    Example
    -------
    >>> registry = WandbModelRegistry(entity="my-team")  # doctest: +SKIP
    >>> registry.load_all(["AAPL"])  # doctest: +SKIP
    ['AAPL']
    """

    def __init__(
        self,
        project_name: str = DEFAULT_PROJECT_NAME,
        entity: str | None = None,
        download_directory: Path | None = None,
        env_file_path: str = ".env",
    ) -> None:
        self.project_name = project_name
        self.entity = entity
        self.download_directory = download_directory or Path(
            tempfile.gettempdir()
        ) / "outlier-detector-artifacts"
        self.env_file_path = env_file_path
        self.models: dict[str, LoadedModel] = {}
        self._api: wandb.Api | None = None

    @property
    def api(self) -> wandb.Api:
        """Return the wandb client, creating it on first use.

        Returns
        -------
        wandb.Api
            Client shared by every call of this registry.
        """
        if self._api is None:
            load_dotenv(dotenv_path=self.env_file_path)
            use_writable_wandb_directories()
            self._api = wandb.Api()
        return self._api

    @property
    def project_path(self) -> str:
        """Return the project path in the form ``<entity>/<project>``.

        Returns
        -------
        str
            Project path. The default entity of the account is used when
            no entity was configured.
        """
        entity = self.entity or self.api.default_entity
        return f"{entity}/{self.project_name}"

    def load_model(self, ticker: str) -> LoadedModel:
        """Download the latest detector of a ticker and keep it in memory.

        Parameters
        ----------
        ticker : str
            Ticker symbol, for example ``AAPL``.

        Returns
        -------
        LoadedModel
            The detector kept in memory for the ticker.

        Raises
        ------
        wandb.errors.CommError
            If the artifact does not exist or wandb is unreachable.
        """
        detector, metadata = download_detector(
            api=self.api,
            project_path=self.project_path,
            ticker=ticker,
            download_directory=self.download_directory,
        )
        loaded = LoadedModel(
            ticker=ticker,
            detector=detector,
            metadata=metadata,
        )
        self.models[ticker] = loaded
        return loaded

    def list_trained_tickers(self) -> list[str]:
        """Discover the tickers that have a model in the W&B project.

        Each collection whose name starts with the detector prefix is read
        at its latest version, and the ticker comes from the metadata stored
        with the artifact.

        Returns
        -------
        list[str]
            Tickers found in the project, in alphabetical order.
        """
        tickers = []
        collections = self.api.artifact_collections(
            project_name=self.project_path,
            type_name="model",
        )
        for collection in collections:
            if not collection.name.startswith(ARTIFACT_PREFIX):
                continue
            artifact = self.api.artifact(
                f"{self.project_path}/{collection.name}:latest",
                type="model",
            )
            ticker = artifact.metadata.get("ticker")
            if ticker:
                tickers.append(ticker)
        return sorted(tickers)

    def load_all(self, tickers: list[str]) -> list[str]:
        """Load the detectors of several tickers, skipping the failures.

        Parameters
        ----------
        tickers : list[str]
            Tickers to load.

        Returns
        -------
        list[str]
            Tickers that were loaded. A ticker that fails is logged and left
            out, so the API still starts when one artifact is missing.
        """
        loaded_tickers = []
        for ticker in tickers:
            try:
                self.load_model(ticker)
            except (wandb.errors.CommError, ValueError, OSError) as error:
                logger.error(
                    "Could not load the model of %s: %s", ticker, error
                )
                continue
            loaded_tickers.append(ticker)
        return loaded_tickers

    def get_model(self, ticker: str) -> LoadedModel:
        """Return the detector of a ticker that is already in memory.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        LoadedModel
            The detector kept in memory for the ticker.

        Raises
        ------
        ModelNotFoundError
            If no detector is loaded for the ticker. wandb is not called.
        """
        if ticker not in self.models:
            raise ModelNotFoundError(
                f"No model loaded for ticker {ticker!r}."
            )
        return self.models[ticker]

    def loaded_tickers(self) -> list[str]:
        """List the tickers whose detectors are in memory.

        Returns
        -------
        list[str]
            Tickers in alphabetical order.
        """
        return sorted(self.models)
