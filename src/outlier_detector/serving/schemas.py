"""Request and response bodies of the outlier detector API."""

from datetime import date

from pydantic import BaseModel, Field

TICKER_PATTERN = r"^[A-Za-z0-9^=._-]{1,20}$"
MAXIMUM_LAST_DAYS = 60


class PredictRequest(BaseModel):
    """Ticker and number of recent days to score.

    The API downloads the prices, the benchmark and the earnings dates
    itself, so the request only names the ticker and the period.

    Attributes
    ----------
    ticker : str
        Ticker whose detector is in memory, for example ``AAPL``.
    last_n_days : int
        How many of the most recent days to return. The default is 5.
    """

    ticker: str = Field(pattern=TICKER_PATTERN)
    last_n_days: int = Field(default=5, ge=1, le=MAXIMUM_LAST_DAYS)


class OutlierDay(BaseModel):
    """Score of one trading day.

    Attributes
    ----------
    day : date
        Trading day.
    score : float
        Anomaly score. Higher means more atypical.
    is_atypical : bool
        True when the Isolation Forest classifies the day as an outlier.
    """

    day: date
    score: float
    is_atypical: bool


class PredictResponse(BaseModel):
    """Scores of the most recent days of a ticker.

    Attributes
    ----------
    ticker : str
        Ticker that was scored.
    model_trained_at : str | None
        Training date of the detector, in ISO 8601, when known.
    days : list[OutlierDay]
        Scores of the requested days, oldest first.
    """

    ticker: str
    model_trained_at: str | None
    days: list[OutlierDay]


class ModelsResponse(BaseModel):
    """Tickers whose detectors are in memory.

    Attributes
    ----------
    tickers : list[str]
        Ticker symbols in alphabetical order.
    """

    tickers: list[str]
