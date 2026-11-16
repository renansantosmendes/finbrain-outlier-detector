"""FastAPI application that serves the outlier detectors held in memory.

When the application is created, it downloads from Weights & Biases the
detectors of the tickers listed in ``OUTLIER_TICKERS`` and keeps them in
memory. Requests then download only the market data.

Run it locally from the project root::

    uv run uvicorn api.index:app --reload
"""

import os

from fastapi import FastAPI, HTTPException

from outlier_detector.serving.model_registry import (
    DEFAULT_PROJECT_NAME,
    ModelNotFoundError,
    WandbModelRegistry,
)
from outlier_detector.serving.schemas import (
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)
from outlier_detector.serving.service import detect_atypical_days

TICKERS_VARIABLE = "OUTLIER_TICKERS"
PROJECT_VARIABLE = "OUTLIER_WANDB_PROJECT"
ENTITY_VARIABLE = "WANDB_ENTITY"


def configured_tickers() -> list[str]:
    """Read the tickers to load at startup from the environment.

    Returns
    -------
    list[str]
        Tickers of the ``OUTLIER_TICKERS`` variable, separated by commas.
        Empty when the variable is not set.

    Example
    -------
    >>> os.environ["OUTLIER_TICKERS"] = "AAPL, PETR4.SA"
    >>> configured_tickers()
    ['AAPL', 'PETR4.SA']
    """
    raw_value = os.environ.get(TICKERS_VARIABLE, "")
    symbols = (symbol.strip() for symbol in raw_value.split(","))
    return [symbol for symbol in symbols if symbol]


def build_default_registry() -> WandbModelRegistry:
    """Create the registry from the project and entity in the environment.

    Returns
    -------
    WandbModelRegistry
        Registry for ``OUTLIER_WANDB_PROJECT`` (default ``outlier-detector``)
        and the entity in ``WANDB_ENTITY``, when set. No network call is made.
    """
    return WandbModelRegistry(
        project_name=os.environ.get(PROJECT_VARIABLE, DEFAULT_PROJECT_NAME),
        entity=os.environ.get(ENTITY_VARIABLE) or None,
    )


def create_app(
    registry: WandbModelRegistry | None = None,
    tickers: list[str] | None = None,
) -> FastAPI:
    """Build the FastAPI application and load its detectors into memory.

    Parameters
    ----------
    registry : WandbModelRegistry | None, optional
        Registry that holds the detectors. ``None`` builds it from the
        environment.
    tickers : list[str] | None, optional
        Tickers to load at startup. ``None`` reads ``OUTLIER_TICKERS``.

    Returns
    -------
    FastAPI
        Application with the routes ``GET /health``, ``GET /models`` and
        ``POST /predict``.

    Example
    -------
    >>> app = create_app(tickers=[])  # doctest: +SKIP
    """
    active_registry = registry or build_default_registry()
    startup_tickers = configured_tickers() if tickers is None else tickers
    if startup_tickers:
        active_registry.load_all(startup_tickers)

    application = FastAPI(
        title="Outlier Detector API",
        description="Scores trading days with Isolation Forest detectors.",
        version="0.2.0",
    )

    @application.get("/health")
    def health() -> dict[str, object]:
        """Report that the service is running and how many models are loaded.

        Returns
        -------
        dict[str, object]
            Status and the number of detectors in memory.
        """
        return {
            "status": "ok",
            "models_loaded": len(active_registry.loaded_tickers()),
        }

    @application.get("/models", response_model=ModelsResponse)
    def models() -> ModelsResponse:
        """List the tickers whose detectors are in memory.

        Returns
        -------
        ModelsResponse
            Tickers in alphabetical order.
        """
        return ModelsResponse(tickers=active_registry.loaded_tickers())

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        """Score the most recent days of a ticker.

        Parameters
        ----------
        request : PredictRequest
            Ticker and number of days to return.

        Returns
        -------
        PredictResponse
            Scores of the requested days.

        Raises
        ------
        HTTPException
            404 when no detector is in memory for the ticker, 422 when the
            market data does not allow building the features.
        """
        try:
            return detect_atypical_days(request, active_registry)
        except ModelNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return application
