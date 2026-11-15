"""FastAPI application that serves the trained outlier detectors.

Run it locally from the project root::

    uv run uvicorn api.index:app --reload

The model folder is read from the ``OUTLIER_MODEL_DIRECTORY`` variable and
defaults to the ``models`` folder at the project root.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from outlier_detector.models.artifact_store import (
    ModelNotFoundError,
    list_trained_tickers,
)
from outlier_detector.serving.schemas import (
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)
from outlier_detector.serving.service import detect_atypical_days

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIRECTORY_VARIABLE = "OUTLIER_MODEL_DIRECTORY"


def default_model_directory() -> Path:
    """Return the folder with the trained models.

    Returns
    -------
    Path
        The value of ``OUTLIER_MODEL_DIRECTORY`` when it is set, otherwise
        the ``models`` folder at the project root.
    """
    configured = os.environ.get(MODEL_DIRECTORY_VARIABLE)
    if configured:
        return Path(configured)
    return PROJECT_ROOT / "models"


def create_app(model_directory: Path | None = None) -> FastAPI:
    """Build the FastAPI application with its routes.

    Parameters
    ----------
    model_directory : Path | None, optional
        Folder with the trained models. ``None`` uses
        ``default_model_directory``.

    Returns
    -------
    FastAPI
        Application with the routes ``GET /health``, ``GET /models`` and
        ``POST /predict``.

    Example
    -------
    >>> from pathlib import Path
    >>> app = create_app(Path("models"))  # doctest: +SKIP
    >>> app.title  # doctest: +SKIP
    'Outlier Detector API'
    """
    directory = model_directory or default_model_directory()
    application = FastAPI(
        title="Outlier Detector API",
        description="Scores market days with Isolation Forest detectors.",
        version="0.1.0",
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        """Report that the service is running.

        Returns
        -------
        dict[str, str]
            Constant status ``{"status": "ok"}``.
        """
        return {"status": "ok"}

    @application.get("/models", response_model=ModelsResponse)
    def models() -> ModelsResponse:
        """List the tickers that have a trained detector.

        Returns
        -------
        ModelsResponse
            Tickers in alphabetical order.
        """
        return ModelsResponse(tickers=list_trained_tickers(directory))

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        """Score the most recent days of the prices in the request.

        Parameters
        ----------
        request : PredictRequest
            Ticker, daily prices and number of days to return.

        Returns
        -------
        PredictResponse
            Scores of the requested days.

        Raises
        ------
        HTTPException
            404 when the ticker has no trained detector, 422 when the
            prices cannot be scored with the stored detector.
        """
        try:
            return detect_atypical_days(directory, request)
        except ModelNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return application
