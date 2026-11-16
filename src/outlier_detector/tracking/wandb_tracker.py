"""Experiment tracking utilities backed by Weights & Biases."""

import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm

import wandb


class WandbExperimentTracker:
    """Track one training run and register its model in wandb.

    The tracker follows the pattern of the course repository: it logs in
    with the ``WANDB_API_KEY`` read from the environment or from a ``.env``
    file, starts a run with a reproducible ``config``, logs metrics to the
    run and to the console, registers the model file as a wandb artifact,
    and finishes the run. Setting ``WANDB_MODE=disabled`` runs without
    logging in, which is useful for local tests.

    Parameters
    ----------
    project_name : str
        Name of the wandb project where the run is logged.
    config : dict[str, object]
        Hyperparameters and dataset settings stored with the run. The same
        dictionary is merged into the metadata of every logged artifact.
    env_file_path : str, optional
        Path to the ``.env`` file with ``WANDB_API_KEY``. The default is
        ``.env`` in the current directory.
    login_timeout_seconds : int, optional
        Maximum time, in seconds, allowed for ``wandb.login``. The default
        is 10.

    Raises
    ------
    RuntimeError
        When logging in is needed and no ``WANDB_API_KEY`` is available.

    Example
    -------
    >>> tracker = WandbExperimentTracker(  # doctest: +SKIP
    ...     project_name="outlier-detector",
    ...     config={"ticker": "AAPL", "random_seed": 42},
    ... )
    >>> tracker.start_run()  # doctest: +SKIP
    >>> tracker.finish_run()  # doctest: +SKIP
    """

    def __init__(
        self,
        project_name: str,
        config: dict[str, object],
        env_file_path: str = ".env",
        login_timeout_seconds: int = 10,
    ) -> None:
        load_dotenv(dotenv_path=env_file_path)
        self.project_name = project_name
        self.config = config
        self.login_timeout_seconds = login_timeout_seconds
        self.wandb_run = None

    def _login(self) -> None:
        """Log in to wandb with the API key, unless tracking is disabled.

        Raises
        ------
        RuntimeError
            If ``WANDB_API_KEY`` is missing and ``WANDB_MODE`` is not
            ``disabled``.
        """
        if os.getenv("WANDB_MODE") == "disabled":
            return
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise RuntimeError(
                "WANDB_API_KEY is not set. Put it in the .env file or in "
                "the environment, or set WANDB_MODE=disabled."
            )
        wandb.login(key=api_key, timeout=self.login_timeout_seconds)

    def start_run(self) -> None:
        """Log in to wandb and start a run with the stored ``config``."""
        self._login()
        mode = "online"
        if os.getenv("WANDB_MODE") == "disabled":
            mode = "disabled"
        self.wandb_run = wandb.init(
            project=self.project_name,
            mode=mode,
            config=self.config,
        )
        tqdm.write(
            f"[wandb] Run started in '{mode}' mode "
            f"(project='{self.project_name}')."
        )

    def log_metrics(self, metrics: dict[str, object]) -> None:
        """Log metrics to the active run and print them to the console.

        Parameters
        ----------
        metrics : dict[str, object]
            Metric names and their values, for example
            ``{"walk_forward_flagged_share": 0.07}``.
        """
        formatted_metrics = " | ".join(
            f"{name}: {_format_metric_value(value)}"
            for name, value in metrics.items()
        )
        tqdm.write(formatted_metrics)
        if self.wandb_run is None:
            print("[wandb] No active run; skipping metric logging.")
            return
        self.wandb_run.log(metrics)

    def log_model(
        self,
        model_file_path: Path,
        model_name: str,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Register a model file as a wandb artifact of the active run.

        The artifact metadata always contains the run ``config``, merged
        with the extra ``metadata`` given here, so the hyperparameters are
        never left out of the registry.

        Parameters
        ----------
        model_file_path : Path
            Local file with the serialized model.
        model_name : str
            Name of the artifact collection in wandb.
        description : str | None, optional
            Human-readable description of the model. The default is None.
        metadata : dict[str, object] | None, optional
            Extra facts about the run, such as the training date. The
            default is None.
        """
        if self.wandb_run is None:
            print("[wandb] No active run; skipping model registration.")
            return
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=description,
            metadata={**self.config, **(metadata or {})},
        )
        artifact.add_file(str(model_file_path))
        self.wandb_run.log_artifact(artifact)
        tqdm.write(f"[wandb] Logged model artifact '{model_name}'.")

    def finish_run(self) -> None:
        """Finish the active run, if there is one."""
        if self.wandb_run is not None:
            self.wandb_run.finish()


def _format_metric_value(metric_value: object) -> str:
    """Format a metric value for the console.

    Parameters
    ----------
    metric_value : object
        Value to format. Floats get six decimal places.

    Returns
    -------
    str
        Readable form of ``metric_value``.
    """
    if isinstance(metric_value, float):
        return f"{metric_value:.6f}"
    return str(metric_value)
