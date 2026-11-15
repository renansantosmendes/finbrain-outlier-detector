"""Score the recent days of a request with the stored detector of a ticker."""

from pathlib import Path

import pandas as pd

from outlier_detector.features.price_features import build_features
from outlier_detector.models.artifact_store import (
    load_metadata,
    load_model,
)
from outlier_detector.serving.schemas import (
    OutlierDay,
    PredictRequest,
    PredictResponse,
)


def detect_atypical_days(
    model_directory: Path,
    request: PredictRequest,
) -> PredictResponse:
    """Score the most recent days of a request with its ticker's detector.

    The features are built from the prices of the request only, so the
    detector never sees data that is not in the request.

    Parameters
    ----------
    model_directory : Path
        Folder that holds the trained models.
    request : PredictRequest
        Validated request with the prices and the number of days to return.

    Returns
    -------
    PredictResponse
        Scores of the last ``request.last_n_days`` days that have features.

    Raises
    ------
    ModelNotFoundError
        If the ticker has no trained detector.
    ValueError
        If the features of the prices do not match the stored detector.

    Example
    -------
    >>> from pathlib import Path
    >>> detect_atypical_days(Path("models"), request)  # doctest: +SKIP
    """
    detector = load_model(model_directory, request.ticker)
    metadata = load_metadata(model_directory, request.ticker)
    prices = pd.DataFrame(
        {
            "Close": [record.close for record in request.prices],
            "Volume": [record.volume for record in request.prices],
        },
        index=pd.DatetimeIndex([record.day for record in request.prices]),
    )
    features = build_features(prices)
    if features.empty:
        raise ValueError("Not enough prices to build the features.")
    scores = detector.decision_scores(features)
    atypical = detector.is_atypical(scores)
    days = [
        OutlierDay(
            day=timestamp.date(),
            score=round(float(score), 6),
            is_atypical=bool(atypical.loc[timestamp]),
        )
        for timestamp, score in scores.tail(request.last_n_days).items()
    ]
    return PredictResponse(
        ticker=request.ticker,
        model_trained_at=metadata.get("trained_at"),
        days=days,
    )
