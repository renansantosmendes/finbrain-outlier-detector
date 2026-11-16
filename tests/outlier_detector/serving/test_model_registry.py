"""Tests of the in-memory model registry, with wandb mocked."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)
from outlier_detector.serving import model_registry
from outlier_detector.serving.model_registry import (
    ModelNotFoundError,
    WandbModelRegistry,
)


def make_detector() -> IsolationForestDetector:
    """Fit a small detector on random rows.

    Returns
    -------
    IsolationForestDetector
        Fitted detector with the columns ``a`` and ``b``.
    """
    generator = np.random.default_rng(5)
    training = pd.DataFrame(
        generator.normal(size=(100, 2)),
        columns=["a", "b"],
    )
    return IsolationForestDetector().fit(training)


@pytest.fixture
def registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> WandbModelRegistry:
    """Build a registry whose downloads are replaced by a fake.

    Parameters
    ----------
    tmp_path : Path
        Folder used as the download directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest helper used to replace the download function.

    Returns
    -------
    WandbModelRegistry
        Registry for the team ``team`` that downloads only ``AAPL``.
    """

    def fake_download(
        api: object,
        project_path: str,
        ticker: str,
        download_directory: Path,
    ) -> tuple[IsolationForestDetector, dict[str, object]]:
        """Return a fitted detector for AAPL and fail for other tickers.

        Parameters
        ----------
        api : object
            Ignored client.
        project_path : str
            Ignored project path.
        ticker : str
            Ticker requested.
        download_directory : Path
            Ignored folder.

        Returns
        -------
        tuple[IsolationForestDetector, dict[str, object]]
            Detector and metadata for AAPL.

        Raises
        ------
        ValueError
            For any ticker other than AAPL, to simulate a missing artifact.
        """
        if ticker != "AAPL":
            raise ValueError(f"no artifact for {ticker}")
        return make_detector(), {"trained_at": "2026-01-01T00:00:00+00:00"}

    monkeypatch.setattr(model_registry, "download_detector", fake_download)
    monkeypatch.setattr(
        WandbModelRegistry,
        "api",
        property(lambda self: object()),
    )
    return WandbModelRegistry(
        entity="team",
        download_directory=tmp_path,
        env_file_path=str(tmp_path / "missing.env"),
    )


def test_load_model_keeps_the_detector_in_memory(
    registry: WandbModelRegistry,
) -> None:
    registry.load_model("AAPL")
    loaded = registry.get_model("AAPL")
    assert loaded.ticker == "AAPL"
    assert loaded.metadata["trained_at"] == "2026-01-01T00:00:00+00:00"


def test_load_all_skips_the_tickers_that_fail(
    registry: WandbModelRegistry,
) -> None:
    assert registry.load_all(["AAPL", "NOPE"]) == ["AAPL"]
    assert registry.loaded_tickers() == ["AAPL"]


def test_get_model_raises_for_a_ticker_not_in_memory(
    registry: WandbModelRegistry,
) -> None:
    with pytest.raises(ModelNotFoundError):
        registry.get_model("PETR4.SA")


def test_project_path_uses_the_configured_entity(
    registry: WandbModelRegistry,
) -> None:
    assert registry.project_path == "team/outlier-detector"


def test_default_download_directory_is_a_temporary_folder() -> None:
    default = WandbModelRegistry(entity="team")
    assert default.download_directory.name == "outlier-detector-artifacts"


class FakeProjectApi:
    """Stand-in for ``wandb.Api`` that lists one project's models.

    Parameters
    ----------
    collections : list[SimpleNamespace]
        Collections returned for the model type.
    metadata_by_name : dict[str, dict[str, object]]
        Metadata of each collection, looked up by name.
    """

    def __init__(
        self,
        collections: list[SimpleNamespace],
        metadata_by_name: dict[str, dict[str, object]],
    ) -> None:
        self.collections = collections
        self.metadata_by_name = metadata_by_name

    def artifact_collections(
        self,
        project_name: str,
        type_name: str,
    ) -> list[SimpleNamespace]:
        """Return the fake collections of the project.

        Parameters
        ----------
        project_name : str
            Ignored project path.
        type_name : str
            Ignored artifact type.

        Returns
        -------
        list[SimpleNamespace]
            The fake collections.
        """
        return self.collections

    def artifact(self, path: str, **options: str) -> SimpleNamespace:
        """Return the metadata of the collection named in the path.

        Parameters
        ----------
        path : str
            Path ending in ``<collection>:latest``.
        options : str
            Ignored keyword options, including the artifact type.

        Returns
        -------
        SimpleNamespace
            Artifact with the collection's metadata.
        """
        name = path.split("/")[-1].split(":")[0]
        return SimpleNamespace(metadata=self.metadata_by_name[name])


def test_list_trained_tickers_reads_the_ticker_from_each_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = WandbModelRegistry(entity="team", download_directory=tmp_path)
    fake_api = FakeProjectApi(
        collections=[
            SimpleNamespace(name="outlier-detector-aapl"),
            SimpleNamespace(name="unrelated-model"),
            SimpleNamespace(name="outlier-detector-petr4.sa"),
        ],
        metadata_by_name={
            "outlier-detector-aapl": {"ticker": "AAPL"},
            "outlier-detector-petr4.sa": {"ticker": "PETR4.SA"},
        },
    )
    monkeypatch.setattr(
        WandbModelRegistry,
        "api",
        property(lambda self: fake_api),
    )
    assert registry.list_trained_tickers() == ["AAPL", "PETR4.SA"]


def test_use_writable_wandb_directories_keeps_existing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_CONFIG_DIR", "/already/set")
    model_registry.use_writable_wandb_directories(tmp_path)
    assert os.environ["WANDB_CONFIG_DIR"] == "/already/set"
    assert os.environ["WANDB_CACHE_DIR"] == str(tmp_path / "cache")
    assert os.environ["WANDB_DATA_DIR"] == str(tmp_path / "data")
