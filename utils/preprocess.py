"""Preprocessing utilities for EV Adoption Likelihood Classification.

This module provides reusable functions and classes for loading the dataset,
detecting column types, encoding categorical variables, scaling numerical
features, and splitting data for training/testing.
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR = "dataset"

TARGET_COLUMN = "ev_adoption_likelihood"

# Columns that are already binary (0/1) and need no encoding
BINARY_COLUMNS = ["home_charging_available", "previous_ev_experience"]

# Ordinal categorical columns (have a natural order) — will use Label Encoding
ORDINAL_CATEGORICAL = ["education_level"]

# Nominal categorical columns (no natural order) — will use OneHot Encoding
NOMINAL_CATEGORICAL = ["city_type", "current_vehicle_type"]


# ---------------------------------------------------------------------------
# Helper: automatically detect the CSV in the dataset directory
# ---------------------------------------------------------------------------

def _find_csv_path() -> str:
    """Return the path to the single CSV file inside ``DATASET_DIR``.

    Raises
    ------
    FileNotFoundError
        If no CSV file is found or more than one CSV exists.
    """
    pattern = os.path.join(DATASET_DIR, "*.csv")
    files = glob.glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No CSV file found in '{DATASET_DIR}'."
        )
    if len(files) > 1:
        raise FileNotFoundError(
            f"Multiple CSV files found in '{DATASET_DIR}'. "
            "Expected exactly one."
        )
    return files[0]


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load the EV adoption dataset from a CSV file.

    Parameters
    ----------
    path : str or None
        Full path to the CSV.  If ``None`` the file is auto-detected inside
        the ``dataset/`` folder.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = _find_csv_path()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. Separating features & target
# ---------------------------------------------------------------------------

def separate_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the DataFrame into feature matrix ``X`` and target series ``y``.

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")
    return df.drop(columns=[target_column]), df[target_column]


# ---------------------------------------------------------------------------
# 3. Detecting column types
# ---------------------------------------------------------------------------

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return dictionary mapping type groups to lists of column names.

    Groups
    ------
    - ``numerical`` : int / float columns (excluding binary).
    - ``binary``    : columns that already hold 0/1 values.
    - ``ordinal``   : categorical columns with a natural order.
    - ``nominal``   : categorical columns without a natural order.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
    """
    all_cols = set(df.columns)
    binary_set = {c for c in BINARY_COLUMNS if c in all_cols}
    ordinal_set = {c for c in ORDINAL_CATEGORICAL if c in all_cols}
    nominal_set = {c for c in NOMINAL_CATEGORICAL if c in all_cols}

    numerical = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in binary_set
    ]

    return {
        "numerical": numerical,
        "binary": list(binary_set),
        "ordinal": list(ordinal_set),
        "nominal": list(nominal_set),
    }


# ---------------------------------------------------------------------------
# 4. Encoding strategy
# ---------------------------------------------------------------------------

def get_encoding_strategy() -> Dict[str, str]:
    """Return the encoding strategy for each categorical feature.

    Returns
    -------
    dict
        Keys are column names, values are ``"label"`` or ``"onehot"``.

    Decision rationale
    ------------------
    - ``education_level`` : label encoding — the levels have a clear ordinal
      relationship (High School < Bachelor < Master < PhD).
    - ``city_type``       : one-hot encoding — no ordinal relationship
      (Urban, Suburban, Rural).
    - ``current_vehicle_type`` : one-hot encoding — no ordinal relationship
      (Hatchback, Sedan, SUV, Truck).
    - Binary columns (``home_charging_available``,
      ``previous_ev_experience``) are kept as-is.
    """
    strategy = {}
    for col in ORDINAL_CATEGORICAL:
        strategy[col] = "label"
    for col in NOMINAL_CATEGORICAL:
        strategy[col] = "onehot"
    return strategy


# ---------------------------------------------------------------------------
# 5. Numerical transformer (wraps StandardScaler)
# ---------------------------------------------------------------------------

class NumericalTransformer:
    """Scale numerical features using StandardScaler.

    The scaler is **not** fitted at initialisation — call ``fit`` first.
    """

    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False
        self._columns: List[str] = []

    def fit(self, X: pd.DataFrame, columns: List[str]) -> "NumericalTransformer":
        """Fit the scaler on the given columns.

        Parameters
        ----------
        X : pd.DataFrame
        columns : list of str
            Numerical column names to scale.

        Returns
        -------
        self
        """
        self._columns = columns
        self._scaler.fit(X[columns])
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            A copy of ``X`` with scaled numerical columns.
        """
        if not self._fitted:
            raise RuntimeError("NumericalTransformer is not fitted yet. Call .fit() first.")
        X = X.copy()
        X[self._columns] = self._scaler.transform(X[self._columns])
        return X

    def fit_transform(self, X: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit the scaler and transform in one step."""
        return self.fit(X, columns).transform(X)


# ---------------------------------------------------------------------------
# 6. Categorical encoder (handles both label and one-hot)
# ---------------------------------------------------------------------------

class CategoricalEncoder:
    """Encode categorical features based on a provided strategy.

    The encoder is **not** fitted at initialisation — call ``fit`` first.
    """

    def __init__(self, strategy: Dict[str, str]):
        """
        Parameters
        ----------
        strategy : dict
            Mapping from column name to ``"label"`` or ``"onehot"``.
        """
        self._strategy = strategy
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._onehot_encoder: Optional[OneHotEncoder] = None
        self._onehot_columns: List[str] = []
        self._label_columns: List[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "CategoricalEncoder":
        """Fit encoders on the columns specified in the strategy.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        self
        """
        for col, method in self._strategy.items():
            if col not in X.columns:
                continue
            if method == "label":
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self._label_encoders[col] = le
                self._label_columns.append(col)
            elif method == "onehot":
                self._onehot_columns.append(col)

        if self._onehot_columns:
            oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            oh.fit(X[self._onehot_columns].astype(str))
            self._onehot_encoder = oh

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted encoders.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        if not self._fitted:
            raise RuntimeError("CategoricalEncoder is not fitted yet. Call .fit() first.")
        X = X.copy()

        # Label encoding
        for col in self._label_columns:
            le = self._label_encoders[col]
            X[col] = le.transform(X[col].astype(str))

        # One-hot encoding
        if self._onehot_columns:
            oh_array = self._onehot_encoder.transform(
                X[self._onehot_columns].astype(str)
            )
            oh_cols = self._onehot_encoder.get_feature_names_out(self._onehot_columns)
            oh_df = pd.DataFrame(
                oh_array, columns=oh_cols, index=X.index
            ).astype(int)
            X = X.drop(columns=self._onehot_columns).join(oh_df)

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders and transform in one step."""
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# 7. PreprocessingPipeline — orchestrates everything
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """End-to-end preprocessing pipeline.

    Steps
    -----
    1. Load dataset
    2. Separate X and y
    3. Detect column types
    4. Encode categorical features
    5. Scale numerical features
    6. Label-encode the target
    7. Train-test split
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

        self.num_transformer = NumericalTransformer()
        self.cat_encoder = CategoricalEncoder(get_encoding_strategy())
        self.target_encoder = LabelEncoder()

        self._column_types: Optional[Dict[str, List[str]]] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """Fit all transformers on the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        self
        """
        X, y = separate_features_target(df)
        self._column_types = detect_column_types(X)

        # Fit categorical encoder
        cat_cols = self._column_types["ordinal"] + self._column_types["nominal"]
        cat_strategy = {c: get_encoding_strategy()[c] for c in cat_cols if c in get_encoding_strategy()}
        self.cat_encoder = CategoricalEncoder(cat_strategy)
        self.cat_encoder.fit(X)

        # Fit numerical scaler
        self.num_transformer.fit(X, self._column_types["numerical"])

        # Fit target encoder
        self.target_encoder.fit(y)

        self._fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, fit: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Transform the DataFrame and return train/test splits.

        Parameters
        ----------
        df : pd.DataFrame
        fit : bool
            If True, fit the pipeline on ``df`` before transforming.

        Returns
        -------
        X_train, X_test, y_train, y_test : tuple of (pd.DataFrame, pd.Series, pd.Series, pd.Series)
        """
        if fit:
            self.fit(df)

        if not self._fitted:
            raise RuntimeError("PreprocessingPipeline is not fitted. Call .fit() first.")

        X, y = separate_features_target(df)

        # Encode categorical features
        X = self.cat_encoder.transform(X)

        # Scale numerical features
        X = self.num_transformer.transform(X)

        # Encode target
        y_encoded = pd.Series(
            self.target_encoder.transform(y), index=y.index, name=TARGET_COLUMN
        )

        # Train-test split
        stratify_param = y_encoded if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param,
        )

        return X_train, X_test, y_train, y_test

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Fit the pipeline and return train/test splits."""
        return self.transform(df, fit=True)
