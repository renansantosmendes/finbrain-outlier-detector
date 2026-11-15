"""Isolation Forest detector of atypical feature rows."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

ANOMALY_THRESHOLD = 0.0


class IsolationForestDetector:
    """Score how isolated each row of features is from the training rows.

    The features are scaled with a ``RobustScaler``, which uses medians
    and quartiles and is therefore less sensitive to the extreme days the
    detector is looking for. A score below ``ANOMALY_THRESHOLD`` (0) marks
    the row as atypical.

    Parameters
    ----------
    contamination : float, optional
        Expected share of atypical rows in the training data, between 0
        and 0.5. It sets the offset of the scores. The default is 0.02.
    n_estimators : int, optional
        Number of isolation trees. The default is 200.
    random_seed : int, optional
        Seed that makes the trees reproducible. The default is 42.

    Attributes
    ----------
    feature_columns : list[str]
        Names of the columns seen during ``fit``, in order.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> training = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    >>> detector = IsolationForestDetector().fit(training)
    >>> scores = detector.decision_scores(training.tail(3))
    >>> scores.shape
    (3,)
    """

    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 200,
        random_seed: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.scaler = RobustScaler()
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
        scaled_features = self.scaler.fit_transform(features)
        self.forest.fit(scaled_features)
        return self

    def decision_scores(self, features: pd.DataFrame) -> pd.Series:
        """Score rows against the fitted model.

        Parameters
        ----------
        features : pd.DataFrame
            Rows with the same columns, in the same order, as the
            training rows.

        Returns
        -------
        pd.Series
            Score per row, indexed like ``features``. Negative values mean
            the row is atypical.

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
        scaled_features = self.scaler.transform(features)
        scores = self.forest.decision_function(scaled_features)
        return pd.Series(scores, index=features.index)

    def is_atypical(self, scores: pd.Series) -> pd.Series:
        """Flag the scores that fall below the anomaly threshold.

        Parameters
        ----------
        scores : pd.Series
            Scores returned by ``decision_scores``.

        Returns
        -------
        pd.Series
            Boolean series on the same index, True for atypical rows.
        """
        return scores < ANOMALY_THRESHOLD

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
