"""Tests of the training command, with market data and wandb mocked."""

from pathlib import Path

import pytest

from outlier_detector.training import train
from outlier_detector.training.train import TrainingSettings
from tests.outlier_detector.conftest import make_market_inputs


class FakeTracker:
    """Tracker that records metrics and the registered model.

    Parameters
    ----------
    project_name : str
        Project name, ignored.
    config : dict[str, object]
        Run configuration, recorded in ``configs``.
    """

    instances: list["FakeTracker"] = []

    def __init__(self, project_name: str, config: dict[str, object]) -> None:
        self.project_name = project_name
        self.config = config
        self.started = False
        self.finished = False
        self.metrics: list[dict[str, object]] = []
        self.models: list[dict[str, object]] = []
        FakeTracker.instances.append(self)

    def start_run(self) -> None:
        """Mark the run as started."""
        self.started = True

    def log_metrics(self, metrics: dict[str, object]) -> None:
        """Record a metrics dictionary.

        Parameters
        ----------
        metrics : dict[str, object]
            Metrics of the run.
        """
        self.metrics.append(metrics)

    def log_model(self, **kwargs: object) -> None:
        """Record the registered model.

        Parameters
        ----------
        **kwargs : object
            Arguments of the model registration.
        """
        self.models.append(kwargs)

    def finish_run(self) -> None:
        """Mark the run as finished."""
        self.finished = True


@pytest.fixture
def patched_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the market data functions and the tracker with fakes.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest helper used to replace the functions.
    """
    FakeTracker.instances = []
    prices, benchmark, earnings = make_market_inputs(500)
    monkeypatch.setattr(train, "load_prices", lambda ticker, start: prices)
    monkeypatch.setattr(
        train,
        "load_benchmark_log_returns",
        lambda symbol, start: benchmark,
    )
    monkeypatch.setattr(
        train,
        "load_earnings_days",
        lambda ticker: earnings,
    )
    monkeypatch.setattr(train, "WandbExperimentTracker", FakeTracker)


def test_train_ticker_registers_the_model_and_finishes_the_run(
    tmp_path: Path,
    patched_training: None,
) -> None:
    settings = TrainingSettings(train_min=252, step=63)
    metadata = train.train_ticker(
        ticker="AAPL",
        settings=settings,
        project_name="outlier-detector",
        work_directory=tmp_path,
    )
    tracker = FakeTracker.instances[0]
    assert tracker.started and tracker.finished
    assert tracker.config["ticker"] == "AAPL"
    assert tracker.models[0]["model_name"] == "outlier-detector-aapl"
    assert metadata["ticker"] == "AAPL"
    assert metadata["feature_columns"][0] == "ret"


def test_train_ticker_logs_the_walk_forward_metrics(
    tmp_path: Path,
    patched_training: None,
) -> None:
    train.train_ticker(
        ticker="AAPL",
        settings=TrainingSettings(train_min=252, step=63),
        project_name="outlier-detector",
        work_directory=tmp_path,
    )
    metrics = FakeTracker.instances[0].metrics[0]
    assert metrics["walk_forward_rows"] > 0
    assert 0.0 <= metrics["walk_forward_flagged_share"] <= 1.0


def test_train_ticker_raises_when_history_is_too_short(
    tmp_path: Path,
    patched_training: None,
) -> None:
    with pytest.raises(ValueError, match="Not enough history"):
        train.train_ticker(
            ticker="AAPL",
            settings=TrainingSettings(train_min=1000),
            project_name="outlier-detector",
            work_directory=tmp_path,
        )
    assert FakeTracker.instances[0].finished


def test_main_returns_1_when_a_ticker_fails(
    tmp_path: Path,
    patched_training: None,
) -> None:
    exit_code = train.main(
        ["--tickers", "AAPL", "--train-min", "1000", "--step", "63"]
    )
    assert exit_code == 1


def test_main_returns_0_when_every_ticker_is_registered(
    patched_training: None,
) -> None:
    exit_code = train.main(
        ["--tickers", "AAPL, MSFT", "--train-min", "252", "--step", "63"]
    )
    assert exit_code == 0
    assert len(FakeTracker.instances) == 2


def test_parse_arguments_splits_the_tickers() -> None:
    options = train.parse_arguments(["--tickers", "AAPL,PETR4.SA,,"])
    assert options.tickers == ["AAPL", "PETR4.SA"]
    assert options.benchmark == "^BVSP"


def test_parse_arguments_rejects_an_empty_ticker_list() -> None:
    with pytest.raises(SystemExit):
        train.parse_arguments(["--tickers", " , "])
