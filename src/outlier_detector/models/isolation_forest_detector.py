"""Isolation Forest detector of atypical trading days."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

DEFAULT_CONTAMINATION = 0.1
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RANDOM_SEED = 42


class IsolationForestDetector:
    """Score how isolated each trading day is from the training days.

    A higher score means a more atypical day. The label ``is_atypical`` is
    True when the Isolation Forest classifies the day as an outlier, using
    the ``contamination`` share of the training data as the cut-off.

    Parameters
    ----------
    contamination : float, optional
        Expected share of outliers in the training data. The default is 0.1.
    n_estimators : int, optional
        Number of isolation trees. The default is 300.
    random_seed : int, optional
        Seed that makes the trees reproducible. The default is 42.

    Attributes
    ----------
    feature_columns : list[str]
        Names of the columns seen during ``fit``, in order.

    Example
    -------
    >>> detector = IsolationForestDetector()  # doctest: +SKIP
    >>> detector.fit(training_features)  # doctest: +SKIP
    >>> detector.score_days(new_features)  # doctest: +SKIP
    """

    def __init__(
        self,
        contamination: float = DEFAULT_CONTAMINATION,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_seed,
        )
        self.feature_columns: list[str] = []

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        """Learn the typical shape of the rows in ``features``.

        Parameters
        ----------
        features : pd.DataFrame
            Numeric training rows without missing values.

        Returns
        -------
        IsolationForestDetector
            The same detector, now fitted.
        """
        self.feature_columns = list(features.columns)
        self.forest.fit(features)
        return self

    def score_days(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score rows against the fitted model.

        Parameters
        ----------
        features : pd.DataFrame
            Rows with the same columns, in the same order, as the
            training rows.

        Returns
        -------
        pd.DataFrame
            Frame indexed like ``features`` with the columns ``score``
            (higher means more atypical) and ``is_atypical`` (boolean).

        Raises
        ------
        ValueError
            If the columns of ``features`` differ from the training columns.
        """
        if list(features.columns) != self.feature_columns:
            raise ValueError(
                "The feature columns differ from the training columns: "
                f"expected {self.feature_columns}, "
                f"got {list(features.columns)}."
            )
        return pd.DataFrame(
            {
                "score": -self.forest.score_samples(features),
                "is_atypical": self.forest.predict(features) == -1,
            },
            index=features.index,
        )

    def save(self, path: Path) -> None:
        """Write the fitted detector to a joblib file.

        Parameters
        ----------
        path : Path
            Destination file. Its parent directory must exist.
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "IsolationForestDetector":
        """Read a detector written by ``save``.

        Parameters
        ----------
        path : Path
            Joblib file written by ``save``.

        Returns
        -------
        IsolationForestDetector
            The stored detector.
        """
        return joblib.load(path)
