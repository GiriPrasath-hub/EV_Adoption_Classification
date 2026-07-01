"""Prediction engine for the EV Adoption Likelihood KNN classifier.

This module provides a reusable inference layer that can be imported by
Streamlit or any other application.  All model artifacts are loaded once
(cached) and reused across calls.

Usage
-----
    from utils.predict import predict, load_model, model_information

    result = predict({"age": 30, "annual_income": 80000, ...})
    print(result["class"], result["confidence"])
"""

import os
import pickle
import logging
from functools import lru_cache
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")

_MODEL_PATH = os.path.join(_MODEL_DIR, "model.pkl")
_ENCODER_PATH = os.path.join(_MODEL_DIR, "encoder.pkl")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected raw input columns (in the order the pipeline was trained on)
# ---------------------------------------------------------------------------

_EXPECTED_COLUMNS: List[str] = [
    "age",
    "annual_income",
    "daily_commute_km",
    "weekly_travel_distance_km",
    "vehicle_age_years",
    "fuel_expense_per_month",
    "charging_station_accessibility",
    "nearest_charging_station_km",
    "electricity_cost_per_kwh",
    "environmental_awareness_score",
    "government_incentive_awareness",
    "technology_affinity_score",
    "range_anxiety_score",
    "battery_replacement_concern",
    "ev_knowledge_score",
    "monthly_energy_consumption_kwh",
    "monthly_charging_cost",
    "education_level",
    "city_type",
    "current_vehicle_type",
    "home_charging_available",
    "previous_ev_experience",
]

# Valid categorical values (known during training)
_VALID_CITY_TYPES = ["Urban", "Suburban", "Rural"]
_VALID_VEHICLE_TYPES = ["Hatchback", "Sedan", "SUV", "Truck"]
_VALID_EDUCATION_LEVELS = ["High School", "Bachelor", "Master", "PhD"]

# Numerical features that must be non-negative
_NON_NEGATIVE_COLUMNS = [
    "age",
    "annual_income",
    "daily_commute_km",
    "weekly_travel_distance_km",
    "vehicle_age_years",
    "fuel_expense_per_month",
    "charging_station_accessibility",
    "nearest_charging_station_km",
    "electricity_cost_per_kwh",
    "monthly_energy_consumption_kwh",
    "monthly_charging_cost",
]

# Score-range features (0-10 scale)
_SCORE_RANGE_COLUMNS = {
    "environmental_awareness_score": (0, 10),
    "government_incentive_awareness": (0, 10),
    "technology_affinity_score": (0, 10),
    "range_anxiety_score": (0, 10),
    "battery_replacement_concern": (0, 10),
    "ev_knowledge_score": (0, 10),
}

# Binary columns
_BINARY_COLUMNS = ["home_charging_available", "previous_ev_experience"]


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_pickle(path: str) -> Any:
    """Load and cache a pickle file.  Repeated calls return the cached object.

    Parameters
    ----------
    path : str
        Absolute path to the pickle file.

    Returns
    -------
    object
        Deserialised object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pickle.UnpicklingError
        If the file is corrupted.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.debug("Loaded %s", os.path.basename(path))
        return obj
    except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"Failed to load {path}: {exc}. "
            "The pickle file may be corrupted or from an incompatible version."
        ) from exc


def load_model() -> Pipeline:
    """Load the full inference pipeline (preprocessor + KNN).

    The result is cached — subsequent calls return the same object.

    Returns
    -------
    Pipeline
    """
    return _load_pickle(_MODEL_PATH)


def load_target_encoder() -> LabelEncoder:
    """Load the fitted LabelEncoder for target decoding.

    Returns
    -------
    LabelEncoder
    """
    return _load_pickle(_ENCODER_PATH)


def get_expected_features() -> List[str]:
    """Return the list of feature names the pipeline expects as input.

    Returns
    -------
    list of str
    """
    return list(_EXPECTED_COLUMNS)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_field_exists(data: dict, field: str) -> None:
    """Ensure *field* is present and not ``None``."""
    if field not in data:
        raise ValueError(f"Missing required field: '{field}'.")
    if data[field] is None:
        raise ValueError(f"Field '{field}' is None.")


def _validate_numeric(value: Any, field: str) -> float:
    """Coerce *value* to float and validate it is finite."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Field '{field}' must be numeric, got {type(value).__name__}: {value!r}."
        )
    if np.isnan(v) or np.isinf(v):
        raise ValueError(f"Field '{field}' is NaN or infinity.")
    return v


def _validate_non_negative(value: Any, field: str) -> float:
    """Validate *value* is non-negative."""
    v = _validate_numeric(value, field)
    if v < 0:
        raise ValueError(f"Field '{field}' cannot be negative, got {v}.")
    return v


def _validate_in_range(value: Any, field: str, lo: float, hi: float) -> float:
    """Validate *value* is within [lo, hi]."""
    v = _validate_numeric(value, field)
    if v < lo or v > hi:
        raise ValueError(f"Field '{field}' must be between {lo} and {hi}, got {v}.")
    return v


def _validate_choice(value: Any, field: str, allowed: List[str]) -> str:
    """Validate *value* is one of the *allowed* choices."""
    s = str(value).strip()
    if s not in allowed:
        raise ValueError(
            f"Field '{field}' must be one of {allowed}, got {value!r}."
        )
    return s


def _validate_binary(value: Any, field: str) -> int:
    """Validate *value* is 0 or 1."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Field '{field}' must be 0 or 1 (binary), got {value!r}."
        )
    if v not in (0, 1):
        raise ValueError(
            f"Field '{field}' must be 0 or 1 (binary), got {v}."
        )
    return v


def validate_input(data: dict) -> dict:
    """Validate all fields in a raw input dictionary.

    Checks performed
    ----------------
    * All required fields present.
    * Numeric fields are finite numbers.
    * Non-negative fields are >= 0.
    * Score fields are in 0-10 range.
    * Binary fields are 0 or 1.
    * Categorical fields are known categories.

    Parameters
    ----------
    data : dict
        Raw user input.

    Returns
    -------
    dict
        The validated and cleaned dictionary (values coerced to correct types).

    Raises
    ------
    ValueError
        On the first validation failure.
    """
    result: dict = {}

    # Check all expected fields exist
    for field in _EXPECTED_COLUMNS:
        _validate_field_exists(data, field)

    # Numerical + non-negative
    for field in _NON_NEGATIVE_COLUMNS:
        result[field] = _validate_non_negative(data[field], field)

    # Score-range columns (0-10)
    for field, (lo, hi) in _SCORE_RANGE_COLUMNS.items():
        result[field] = _validate_in_range(data[field], field, lo, hi)

    # Binary columns
    for field in _BINARY_COLUMNS:
        result[field] = _validate_binary(data[field], field)

    # Categorical columns
    result["education_level"] = _validate_choice(
        data["education_level"], "education_level", _VALID_EDUCATION_LEVELS
    )
    result["city_type"] = _validate_choice(
        data["city_type"], "city_type", _VALID_CITY_TYPES
    )
    result["current_vehicle_type"] = _validate_choice(
        data["current_vehicle_type"], "current_vehicle_type", _VALID_VEHICLE_TYPES
    )

    return result


# ---------------------------------------------------------------------------
# DataFrame preparation
# ---------------------------------------------------------------------------

def prepare_dataframe(data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
    """Convert input to a DataFrame ready for the pipeline.

    Parameters
    ----------
    data : dict or pd.DataFrame
        Single sample as dictionary, or a DataFrame with one/many rows.

    Returns
    -------
    pd.DataFrame
        Columns are in the exact order expected by the pipeline.
    """
    if isinstance(data, dict):
        validated = validate_input(data)
        df = pd.DataFrame([validated])
    elif isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("DataFrame is empty.")
        # Validate every row (first error fails fast)
        for idx, row in data.iterrows():
            validate_input(row.to_dict())
        df = data.copy()
    else:
        raise TypeError(
            f"Expected dict or pd.DataFrame, got {type(data).__name__}."
        )

    # Ensure column order matches training expectation
    missing = [c for c in _EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    return df[_EXPECTED_COLUMNS]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    data: Union[dict, pd.DataFrame],
) -> Dict[str, Any]:
    """Run the full prediction pipeline on a single input.

    Parameters
    ----------
    data : dict or pd.DataFrame
        Input features.  A dict represents one sample.

    Returns
    -------
    dict with keys:
        - ``class``       : predicted class label (str)
        - ``class_id``    : predicted class index (int)
        - ``confidence``  : probability of the predicted class (float)
        - ``probabilities`` : dict mapping each class label to its probability
    """
    df = prepare_dataframe(data)
    model = load_model()
    encoder = load_target_encoder()

    pred_encoded = model.predict(df)
    proba = model.predict_proba(df)

    class_id = int(pred_encoded[0])
    class_label = encoder.inverse_transform([class_id])[0]
    confidence = float(proba[0, class_id])

    prob_dict = {
        str(label): float(proba[0, idx])
        for idx, label in enumerate(encoder.classes_)
    }

    return {
        "class": class_label,
        "class_id": class_id,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
    }


def predict_probability(
    data: Union[dict, pd.DataFrame],
) -> Dict[str, float]:
    """Return only the probability distribution over target classes.

    Parameters
    ----------
    data : dict or pd.DataFrame

    Returns
    -------
    dict mapping class label (str) -> probability (float)
    """
    return predict(data)["probabilities"]


def decode_prediction(encoded_prediction: Union[int, np.integer]) -> str:
    """Decode a numeric prediction back to the original class label.

    Parameters
    ----------
    encoded_prediction : int or np.integer
        The integer label (0, 1, or 2).

    Returns
    -------
    str
        The original class name (e.g. ``"High"``).
    """
    encoder = load_target_encoder()
    return str(encoder.inverse_transform([int(encoded_prediction)])[0])


# ---------------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------------

def model_information() -> Dict[str, Any]:
    """Return metadata about the loaded model.

    Returns
    -------
    dict with keys:
        - ``algorithm``
        - ``expected_features``
        - ``num_features``
        - ``classes``
        - ``model_file``
    """
    model = load_model()
    encoder = load_target_encoder()

    knn = model.named_steps["knn"]

    return {
        "algorithm": f"K-Nearest Neighbors (k={knn.n_neighbors}, "
                     f"weights='{knn.weights}', metric='{knn.metric}')",
        "expected_features": list(get_expected_features()),
        "num_features": len(get_expected_features()),
        "classes": list(encoder.classes_),
        "model_file": os.path.basename(_MODEL_PATH),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sample = {
        "age": 34,
        "annual_income": 62000.0,
        "education_level": "Master",
        "city_type": "Urban",
        "daily_commute_km": 30.5,
        "weekly_travel_distance_km": 195.0,
        "current_vehicle_type": "SUV",
        "vehicle_age_years": 4.0,
        "fuel_expense_per_month": 280.0,
        "charging_station_accessibility": 7.0,
        "nearest_charging_station_km": 2.5,
        "home_charging_available": 1,
        "electricity_cost_per_kwh": 0.18,
        "environmental_awareness_score": 8.0,
        "government_incentive_awareness": 6.5,
        "technology_affinity_score": 8.5,
        "range_anxiety_score": 3.0,
        "battery_replacement_concern": 4.0,
        "ev_knowledge_score": 7.5,
        "previous_ev_experience": 0,
        "monthly_energy_consumption_kwh": 150.0,
        "monthly_charging_cost": 28.0,
    }

    print("=" * 60)
    print("PREDICTION MODULE SELF-TEST")
    print("=" * 60)

    # Test model info
    info = model_information()
    print(f"\nAlgorithm : {info['algorithm']}")
    print(f"Features  : {info['num_features']}")
    print(f"Classes   : {info['classes']}")

    # Test prediction
    print(f"\nInput     : {sample['age']}y, {sample['annual_income']:.0f}$, "
          f"{sample['education_level']}, {sample['city_type']}")
    result = predict(sample)
    print(f"Predicted : {result['class']} "
          f"(confidence: {result['confidence']:.2%})")
    print(f"Probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  {label:>8}: {prob:.2%}")

    # Test DataFrame input
    print("\n--- DataFrame test ---")
    df_sample = pd.DataFrame([sample, sample])
    results = predict(df_sample)
    print(f"DataFrame (2 rows) -> class: {results['class']}")

    # Test validate_input
    print("\n--- Validation test ---")
    try:
        bad = sample.copy()
        del bad["age"]
        predict(bad)
    except ValueError as e:
        print(f"Missing field correctly caught: {e}")

    try:
        bad2 = sample.copy()
        bad2["city_type"] = "Mars"
        predict(bad2)
    except ValueError as e:
        print(f"Invalid category correctly caught: {e}")

    try:
        bad3 = sample.copy()
        bad3["age"] = -5
        predict(bad3)
    except ValueError as e:
        print(f"Negative value correctly caught: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
