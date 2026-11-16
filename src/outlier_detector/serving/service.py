"""Score the recent days of a ticker with the detector kept in memory."""

from datetime import date, timedelta

from outlier_detector.data.market_data import (
    DEFAULT_BENCHMARK,
    load_benchmark_log_returns,
    load_earnings_days,
    load_prices,
)
from outlier_detector.features.price_features import build_features
from outlier_detector.serving.model_registry import WandbModelRegistry
from outlier_detector.serving.schemas import (
    OutlierDay,
    PredictRequest,
    PredictResponse,
)

LOOKBACK_DAYS = 365


def detect_atypical_days(
    request: PredictRequest,
    registry: WandbModelRegistry,
    today: date | None = None,
) -> PredictResponse:
    """Score the most recent days of a ticker with its in-memory detector.

    The prices, the benchmark and the earnings dates are downloaded for the
    last ``LOOKBACK_DAYS`` days. The features use only the current day and
    earlier days, so each score is the one the detector gives on that day.

    Parameters
    ----------
    request : PredictRequest
        Ticker and number of days to return.
    registry : WandbModelRegistry
        Registry that holds the detector of the ticker.
    today : date | None, optional
        Reference date of the lookback window. ``None`` uses today.

    Returns
    -------
    PredictResponse
        Scores of the last ``request.last_n_days`` days that have features.

    Raises
    ------
    ModelNotFoundError
        If no detector is in memory for the ticker.
    ValueError
        If the market data does not give any feature row.
    """
    loaded = registry.get_model(request.ticker)
    reference_day = today or date.today()
    start = (reference_day - timedelta(days=LOOKBACK_DAYS)).isoformat()
    features = build_features(
        load_prices(request.ticker, start),
        load_benchmark_log_returns(DEFAULT_BENCHMARK, start),
        load_earnings_days(request.ticker),
    )
    if features.empty:
        raise ValueError(
            "Not enough market data to build the features of "
            f"{request.ticker!r}."
        )
    scored = loaded.detector.score_days(features).tail(request.last_n_days)
    days = [
        OutlierDay(
            day=timestamp.date(),
            score=round(float(row["score"]), 6),
            is_atypical=bool(row["is_atypical"]),
        )
        for timestamp, row in scored.iterrows()
    ]
    trained_at = loaded.metadata.get("trained_at")
    return PredictResponse(
        ticker=request.ticker,
        model_trained_at=str(trained_at) if trained_at else None,
        days=days,
    )
