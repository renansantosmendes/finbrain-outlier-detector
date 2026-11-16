"""Tests of the wandb model store, with the wandb client mocked."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)
from outlier_detector.registry import wandb_model_store
from outlier_detector.registry.wandb_model_store import (
    build_artifact_name,
    build_ticker_slug,
    download_detector,
)


class FakeDownloadedArtifact:
    """Stand-in for a wandb artifact that was downloaded to a folder.

    Parameters
    ----------
    metadata : dict[str, object]
        Metadata stored with the artifact.
    model_file : Path
        Joblib file that the download folder contains.
    """

    def __init__(self, metadata: dict[str, object], model_file: Path) -> None:
        self.metadata = metadata
        self.model_file = model_file

    def download(self, root: str) -> str:
        """Copy the model file into the download folder.

        Parameters
        ----------
        root : str
            Folder where the artifact is downloaded.

        Returns
        -------
        str
            The download folder.
        """
        target = Path(root)
        target.mkdir(parents=True, exist_ok=True)
        content = self.model_file.read_bytes()
        (target / self.model_file.name).write_bytes(content)
        return str(target)


class FakeApi:
    """Stand-in for ``wandb.Api`` that serves one artifact.

    Parameters
    ----------
    artifact : FakeDownloadedArtifact
        Artifact returned by every ``artifact`` call.
    """

    def __init__(self, artifact: FakeDownloadedArtifact) -> None:
        self.artifact_paths: list[str] = []
        self.artifact_types: list[str] = []
        self._artifact = artifact

    def artifact(self, path: str, **options: str) -> FakeDownloadedArtifact:
        """Record the request and return the fake artifact.

        Parameters
        ----------
        path : str
            Artifact path requested.
        options : str
            Keyword options, including the artifact ``type``.

        Returns
        -------
        FakeDownloadedArtifact
            The fake artifact.
        """
        self.artifact_paths.append(path)
        self.artifact_types.append(options["type"])
        return self._artifact


def make_saved_detector(directory: Path) -> Path:
    """Fit a small detector and save it as a joblib file.

    Parameters
    ----------
    directory : Path
        Folder where the joblib file is written.

    Returns
    -------
    Path
        Path of the joblib file.
    """
    generator = np.random.default_rng(3)
    training = pd.DataFrame(
        generator.normal(size=(120, 2)),
        columns=["a", "b"],
    )
    model_file = directory / "source.joblib"
    IsolationForestDetector().fit(training).save(model_file)
    return model_file


def test_build_ticker_slug_replaces_unsafe_characters() -> None:
    assert build_ticker_slug("PETR4.SA") == "petr4.sa"
    assert build_ticker_slug("^BVSP") == "-bvsp"


def test_build_artifact_name_uses_the_prefix() -> None:
    assert build_artifact_name("AAPL") == "outlier-detector-aapl"


def test_download_detector_reads_latest_artifact(tmp_path: Path) -> None:
    model_file = make_saved_detector(tmp_path)
    artifact = FakeDownloadedArtifact({"trained_at": "2026-01-01"}, model_file)
    api = FakeApi(artifact)
    detector, metadata = download_detector(
        api=api,
        project_path="team/outlier-detector",
        ticker="AAPL",
        download_directory=tmp_path / "downloads",
    )
    assert api.artifact_paths == [
        "team/outlier-detector/outlier-detector-aapl:latest"
    ]
    assert api.artifact_types == ["model"]
    assert metadata == {"trained_at": "2026-01-01"}
    assert detector.feature_columns == ["a", "b"]


def test_log_detector_writes_the_file_and_calls_the_tracker(
    tmp_path: Path,
) -> None:
    recorded = {}

    class FakeTracker:
        """Tracker that records the call to ``log_model``."""

        def log_model(self, **kwargs: object) -> None:
            """Record the keyword arguments of the call.

            Parameters
            ----------
            **kwargs : object
                Arguments of ``log_model``.
            """
            recorded.update(kwargs)

    generator = np.random.default_rng(4)
    training = pd.DataFrame(
        generator.normal(size=(120, 2)),
        columns=["a", "b"],
    )
    detector = IsolationForestDetector().fit(training)
    wandb_model_store.log_detector(
        tracker=FakeTracker(),
        detector=detector,
        ticker="PETR4.SA",
        metadata={"training_rows": 120},
        work_directory=tmp_path,
    )
    assert (tmp_path / "petr4.sa.joblib").is_file()
    assert recorded["model_name"] == "outlier-detector-petr4.sa"
    assert recorded["metadata"] == {"training_rows": 120}


@pytest.mark.parametrize(
    "ticker, expected",
    [("AAPL", "outlier-detector-aapl"), ("^BVSP", "outlier-detector--bvsp")],
)
def test_artifact_names_for_several_tickers(
    ticker: str,
    expected: str,
) -> None:
    assert build_artifact_name(ticker) == expected
