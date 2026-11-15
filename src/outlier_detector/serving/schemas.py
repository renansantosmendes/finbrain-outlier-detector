"""Request and response bodies of the outlier detector API."""

from datetime import date

from pydantic import BaseModel, Field, model_validator

from outlier_detector.features.price_features import MINIMUM_PRICE_ROWS

TICKER_PATTERN = r"^[A-Za-z0-9^=._-]{1,20}$"
MAXIMUM_LAST_DAYS = 60


class PriceRecord(BaseModel):
    """Daily close and volume of one trading day.

    Attributes
    ----------
    day : date
        Trading day.
    close : float
        Closing price, greater than zero.
    volume : float
        Traded volume, zero or greater.
    """

    day: date
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class PredictRequest(BaseModel):
    """Recent prices of a ticker to score with its trained detector.

    Attributes
    ----------
    ticker : str
        Ticker whose trained detector is used.
    prices : list[PriceRecord]
        Consecutive trading days, oldest first. At least
        ``MINIMUM_PRICE_ROWS`` days are needed to build the features.
    last_n_days : int
        How many of the most recent days to return. The default is 5.
    """

    ticker: str = Field(pattern=TICKER_PATTERN)
    prices: list[PriceRecord] = Field(min_length=MINIMUM_PRICE_ROWS)
    last_n_days: int = Field(default=5, ge=1, le=MAXIMUM_LAST_DAYS)

    @model_validator(mode="after")
    def check_dates_are_increasing(self) -> "PredictRequest":
        """Reject price lists whose dates are not strictly increasing.

        Returns
        -------
        PredictRequest
            The validated request.

        Raises
        ------
        ValueError
            If a day is not after the day before it.
        """
        days = [record.day for record in self.prices]
        if any(later <= earlier for earlier, later in zip(days, days[1:])):
            raise ValueError(
                "The prices must be sorted by strictly increasing date."
            )
        return self


class OutlierDay(BaseModel):
    """Score of one trading day.

    Attributes
    ----------
    day : date
        Trading day.
    score : float
        Isolation score. Negative values mean the day is atypical.
    is_atypical : bool
        True when the score is below the anomaly threshold.
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
    """Tickers that have a trained detector.

    Attributes
    ----------
    tickers : list[str]
        Ticker symbols in alphabetical order.
    """

    tickers: list[str]
