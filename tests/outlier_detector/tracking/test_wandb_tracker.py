"""Tests of the Weights & Biases tracker, with wandb mocked."""

from pathlib import Path

import pytest

from outlier_detector.tracking import wandb_tracker
from outlier_detector.tracking.wandb_tracker import WandbExperimentTracker


class FakeRun:
    """Stand-in for a wandb run that records what is logged to it.

    Attributes
    ----------
    logged_metrics : list[dict[str, object]]
        Metric dictionaries passed to ``log``.
    artifacts : list[object]
        Artifacts passed to ``log_artifact``.
    finished : bool
        True after ``finish`` is called.
    """

    def __init__(self) -> None:
        self.logged_metrics: list[dict[str, object]] = []
        self.artifacts: list[object] = []
        self.finished = False

    def log(self, metrics: dict[str, object]) -> None:
        """Record a metric dictionary.

        Parameters
        ----------
        metrics : dict[str, object]
            Metrics logged in this step.
        """
        self.logged_metrics.append(metrics)

    def log_artifact(self, artifact: object) -> None:
        """Record an artifact.

        Parameters
        ----------
        artifact : object
            Artifact registered in the run.
        """
        self.artifacts.append(artifact)

    def finish(self) -> None:
        """Mark the run as finished."""
        self.finished = True


class FakeArtifact:
    """Stand-in for ``wandb.Artifact`` that records its construction.

    Parameters
    ----------
    name : str
        Artifact name.
    type : str
        Artifact type.
    description : str | None
        Artifact description.
    metadata : dict[str, object]
        Artifact metadata.
    """

    def __init__(
        self,
        name: str,
        description: str | None,
        metadata: dict[str, object],
        **options: str,
    ) -> None:
        self.name = name
        self.type = options["type"]
        self.description = description
        self.metadata = metadata
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        """Record a file added to the artifact.

        Parameters
        ----------
        path : str
            Path of the file.
        """
        self.files.append(path)


def make_tracker(tmp_path: Path) -> WandbExperimentTracker:
    """Create a tracker that reads no ``.env`` file.

    Parameters
    ----------
    tmp_path : Path
        Folder where the missing ``.env`` file is looked for.

    Returns
    -------
    WandbExperimentTracker
        Tracker with a fixed project and configuration.
    """
    return WandbExperimentTracker(
        project_name="outlier-detector",
        config={"ticker": "AAPL", "random_seed": 42},
        env_file_path=str(tmp_path / "missing.env"),
    )


def test_start_run_raises_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        make_tracker(tmp_path).start_run()


def test_start_run_logs_in_with_the_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(
        wandb_tracker.wandb,
        "login",
        lambda key, timeout: calls.update(key=key, timeout=timeout),
    )
    monkeypatch.setattr(
        wandb_tracker.wandb,
        "init",
        lambda project, mode, config: FakeRun(),
    )
    make_tracker(tmp_path).start_run()
    assert calls == {"key": "test-key", "timeout": 10}


def test_disabled_mode_starts_a_run_without_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        wandb_tracker.wandb,
        "login",
        lambda **kwargs: pytest.fail("wandb.login must not be called"),
    )
    init_modes = []
    monkeypatch.setattr(
        wandb_tracker.wandb,
        "init",
        lambda project, mode, config: init_modes.append(mode) or FakeRun(),
    )
    make_tracker(tmp_path).start_run()
    assert init_modes == ["disabled"]


def test_log_model_registers_the_file_with_config_in_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wandb_tracker.wandb, "Artifact", FakeArtifact)
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"model")
    tracker = make_tracker(tmp_path)
    tracker.wandb_run = FakeRun()
    tracker.log_model(
        model_file_path=model_file,
        model_name="outlier-detector-aapl",
        description="Detector of AAPL.",
        metadata={"training_rows": 500},
    )
    artifact = tracker.wandb_run.artifacts[0]
    assert artifact.type == "model"
    assert artifact.files == [str(model_file)]
    assert artifact.metadata == {
        "ticker": "AAPL",
        "random_seed": 42,
        "training_rows": 500,
    }


def test_finish_run_finishes_the_active_run(tmp_path: Path) -> None:
    tracker = make_tracker(tmp_path)
    run = FakeRun()
    tracker.wandb_run = run
    tracker.finish_run()
    assert run.finished


def test_log_metrics_without_a_run_only_prints(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_tracker(tmp_path).log_metrics({"training_rows": 10})
    assert "skipping metric logging" in capsys.readouterr().out
