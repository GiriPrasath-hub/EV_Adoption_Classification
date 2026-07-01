"""KNN training pipeline for EV Adoption Likelihood classification.

Reuses Phase 1 utilities for dataset loading and column type detection.
Builds a proper sklearn Pipeline with imputation, encoding, scaling,
hyperparameter tuning, evaluation, and model serialisation.
"""

import sys
import os
import pickle
import logging
import warnings
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import load_dataset, separate_features_target, detect_column_types

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror Phase 1 decisions)
# ---------------------------------------------------------------------------

TARGET_COLUMN = "ev_adoption_likelihood"
BINARY_COLUMNS = ["home_charging_available", "previous_ev_experience"]
ORDINAL_CATEGORICAL = ["education_level"]
NOMINAL_CATEGORICAL = ["city_type", "current_vehicle_type"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__))

# Ordered categories for ordinal encoding (Phase 1 decision: High School < Bachelor < Master < PhD)
EDUCATION_LEVELS = ["High School", "Bachelor", "Master", "PhD"]


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_negative_fuel_expense(df: pd.DataFrame) -> pd.DataFrame:
    """Clip negative ``fuel_expense_per_month`` values to 0.

    Rationale
    ---------
    271 rows in the dataset contain negative fuel expense values, which are
    data-entry errors (monthly fuel cost cannot be negative).  Capping at 0
    is the most conservative fix that avoids introducing bias.
    """
    col = "fuel_expense_per_month"
    if col not in df.columns:
        return df
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        logger.info("Clipping %d negative %s values to 0", neg_count, col)
    df = df.copy()
    df[col] = df[col].clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Build the preprocessing ColumnTransformer
# ---------------------------------------------------------------------------

def build_preprocessor(col_types: dict) -> ColumnTransformer:
    """Build a ``ColumnTransformer`` with imputation, encoding and scaling.

    Parameters
    ----------
    col_types : dict
        Output of ``detect_column_types()``.

    Returns
    -------
    ColumnTransformer
    """
    # Numerical pipeline: median imputation → standard scaling
    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Ordinal categorical pipeline: most-frequent imputation → ordinal encoding
    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[EDUCATION_LEVELS])),
    ])

    # Nominal categorical pipeline: most-frequent imputation → one-hot encoding
    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Binary columns: most-frequent imputation only (already 0/1)
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    transformers = [
        ("numerical", numerical_pipe, col_types["numerical"]),
        ("ordinal", ordinal_pipe, col_types["ordinal"]),
        ("nominal", nominal_pipe, col_types["nominal"]),
        ("binary", binary_pipe, col_types["binary"]),
    ]

    return ColumnTransformer(transformers, verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# Hyperparameter grid for KNN
# ---------------------------------------------------------------------------

def get_param_grid() -> dict:
    """Return the hyperparameter search space for KNN."""
    return {
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list,
) -> dict:
    """Compute and log all required evaluation metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1_weighted, f1_macro, conf_matrix
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "conf_matrix": confusion_matrix(y_true, y_pred),
    }

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("-" * 70)
    print(f"{'Accuracy':<30} {metrics['accuracy']:.4f}")
    print(f"{'Precision (weighted)':<30} {metrics['precision']:.4f}")
    print(f"{'Recall (weighted)':<30} {metrics['recall']:.4f}")
    print(f"{'F1 Score (weighted)':<30} {metrics['f1_weighted']:.4f}")
    print(f"{'F1 Score (macro)':<30} {metrics['f1_macro']:.4f}")
    print("-" * 70)

    print("CONFUSION MATRIX")
    cm = metrics["conf_matrix"]
    print(f"{'':>12}", end="")
    for name in target_names:
        print(f"{name:>12}", end="")
    print()
    for i, name in enumerate(target_names):
        print(f"{name:<12}", end="")
        for j in range(len(target_names)):
            print(f"{cm[i, j]:>12}", end="")
        print()
    print("=" * 70)

    return metrics


def cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> tuple:
    """Perform stratified cross-validation on the *best* pipeline.

    Returns (mean_cv_score, std_cv_score).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
    mean_score = scores.mean()
    std_score = scores.std()

    print("\n" + "-" * 70)
    print("CROSS-VALIDATION SCORES (Stratified 5-Fold, F1 Weighted)")
    print("-" * 70)
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean CV Score: {mean_score:.4f} +/- {std_score:.4f}")
    print("-" * 70)

    return mean_score, std_score


# ---------------------------------------------------------------------------
# Feature importance explanation
# ---------------------------------------------------------------------------

def print_feature_importance_analysis() -> None:
    """Print an explanation of why KNN has no built-in feature importance."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    print("""
K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning
algorithm.  It does NOT provide a built-in feature importance mechanism
because:

1. KNN does not learn an explicit decision function or coefficient vector
   (unlike linear models or tree-based models).

2. Predictions are based purely on distance/similarity in the full feature
   space — every feature contributes equally to the distance computation
   (unless explicitly weighted).

3. There is no internal model parameter that can be inspected to determine
   which features are "more important."

HOW FEATURE IMPORTANCE COULD BE ESTIMATED IN FUTURE
-----------------------------------------------------
| Method              | Description                                       |
|---------------------|---------------------------------------------------|
| Permutation         | Shuffle each feature and measure the drop in      |
| Importance          | performance.  The larger the drop, the more        |
|                     | important the feature.                             |
|                     |                                                   |
| Recursive Feature   | Iteratively remove the least important feature     |
| Elimination (RFE)   | and re-fit.                                       |
|                     |                                                   |
| Mutual Information  | Compute mutual information between each feature    |
|                     | and the target as a univariate importance score.   |
|                     |                                                   |
| Model-Agnostic      | Use SHAP or LIME to explain individual predictions |
| Explanations        | and aggregate feature contributions globally.      |
-----------------------------------------------------------------------
""" + "=" * 70)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_artifacts(
    best_estimator: Pipeline,
    preprocessor: ColumnTransformer,
    scaler: StandardScaler,
    ordinal_encoder: OrdinalEncoder,
    onehot_encoder: OneHotEncoder,
    target_encoder: LabelEncoder,
    feature_names: np.ndarray,
    directory: str,
) -> None:
    """Save all required pickle files to ``directory``.

    Files
    -----
    - model.pkl            : full pipeline (preprocessor + KNN)
    - preprocessor.pkl     : ColumnTransformer (imputers, scalers, encoders)
    - scaler.pkl           : fitted StandardScaler
    - encoder.pkl          : fitted LabelEncoder for the target
    - feature_columns.pkl  : list of feature column names after preprocessing
    - ordinal_categories.pkl : fitted OrdinalEncoder for education_level
    - onehot_encoder.pkl   : fitted OneHotEncoder for nominal columns
    """
    os.makedirs(directory, exist_ok=True)

    artifacts = {
        "model.pkl": best_estimator,
        "preprocessor.pkl": preprocessor,
        "scaler.pkl": scaler,
        "encoder.pkl": target_encoder,
        "feature_columns.pkl": feature_names,
        "ordinal_encoder.pkl": ordinal_encoder,
        "onehot_encoder.pkl": onehot_encoder,
    }

    for filename, obj in artifacts.items():
        path = os.path.join(directory, filename)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Saved %s", path)

    logger.info("All %d artifacts saved to '%s'", len(artifacts), directory)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete training pipeline."""
    start_time = time.time()

    # ---- 1. Load dataset ----
    logger.info("Loading dataset ...")
    df = load_dataset()
    logger.info("Dataset loaded: %d rows, %d columns", df.shape[0], df.shape[1])

    # ---- 2. Data cleaning ----
    df = clean_negative_fuel_expense(df)

    # ---- 3. Separate features and target ----
    X, y = separate_features_target(df)
    logger.info("Features: %s, Target: %s", X.shape, y.shape)

    # ---- 4. Detect column types ----
    col_types = detect_column_types(X)
    logger.info(
        "Column types — numerical: %d, binary: %d, ordinal: %d, nominal: %d",
        len(col_types["numerical"]),
        len(col_types["binary"]),
        len(col_types["ordinal"]),
        len(col_types["nominal"]),
    )

    # ---- 5. Encode target ----
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    logger.info("Target classes: %s", list(target_encoder.classes_))

    # ---- 6. Train-test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    logger.info(
        "Train: %d samples, Test: %d samples (stratified 80/20)",
        X_train.shape[0],
        X_test.shape[0],
    )
    train_dist = pd.Series(target_encoder.inverse_transform(y_train)).value_counts()
    test_dist = pd.Series(target_encoder.inverse_transform(y_test)).value_counts()
    logger.info("Train target distribution: %s", train_dist.to_dict())
    logger.info("Test target distribution:  %s", test_dist.to_dict())

    # ---- 7. Build preprocessor ----
    preprocessor = build_preprocessor(col_types)

    # ---- 8. Full pipeline ----
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier()),
    ])

    # ---- 9. Hyperparameter search ----
    param_grid = get_param_grid()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    logger.info("Starting hyperparameter search ...")
    logger.info("Parameter grid: %s", param_grid)

    grid = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_cv_score = grid.best_score_

    logger.info("Hyperparameter search completed.")
    logger.info("Best parameters: %s", best_params)
    logger.info("Best CV f1_weighted: %.4f", best_cv_score)

    best_estimator = grid.best_estimator_

    # ---- 10. Test-set evaluation ----
    logger.info("Evaluating best model on test set ...")
    y_pred = best_estimator.predict(X_test)
    target_names = list(target_encoder.classes_)

    metrics = evaluate_model(y_test, y_pred, target_names)

    # ---- 11. Cross-validation on best pipeline ----
    cv_mean, cv_std = cross_validate_model(best_estimator, X_train, y_train)

    # ---- 12. Feature importance analysis ----
    print_feature_importance_analysis()

    # ---- 13. Extract fitted components for serialisation ----
    fitted_preprocessor = best_estimator.named_steps["preprocessor"]
    fitted_knn = best_estimator.named_steps["knn"]

    # Extract sub-components from the ColumnTransformer
    numerical_transformer = fitted_preprocessor.named_transformers_["numerical"]
    scaler = numerical_transformer.named_steps["scaler"]

    ordinal_transformer = fitted_preprocessor.named_transformers_["ordinal"]
    ordinal_encoder_transformer = ordinal_transformer.named_steps["encoder"]

    nominal_transformer = fitted_preprocessor.named_transformers_["nominal"]
    onehot_encoder_transformer = nominal_transformer.named_steps["encoder"]

    # Feature names after transformation
    feature_names = fitted_preprocessor.get_feature_names_out()

    # ---- 14. Save artifacts ----
    save_artifacts(
        best_estimator=best_estimator,
        preprocessor=fitted_preprocessor,
        scaler=scaler,
        ordinal_encoder=ordinal_encoder_transformer,
        onehot_encoder=onehot_encoder_transformer,
        target_encoder=target_encoder,
        feature_names=feature_names,
        directory=ARTIFACTS_DIR,
    )

    # ---- 15. Final summary ----
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Total time':<30} {elapsed:.1f}s")
    print(f"{'Best k (n_neighbors)':<30} {_param(best_params, 'knn__n_neighbors')}")
    print(f"{'Weight strategy':<30} {_param(best_params, 'knn__weights')}")
    print(f"{'Distance metric':<30} {_param(best_params, 'knn__metric')}")
    print(f"{'Cross-validation F1 (weighted)':<30} {best_cv_score:.4f} +/- {cv_std:.4f}")
    print(f"{'Test accuracy':<30} {metrics['accuracy']:.4f}")
    print(f"{'Test precision (weighted)':<30} {metrics['precision']:.4f}")
    print(f"{'Test recall (weighted)':<30} {metrics['recall']:.4f}")
    print(f"{'Test F1 (weighted)':<30} {metrics['f1_weighted']:.4f}")
    print(f"{'Test F1 (macro)':<30} {metrics['f1_macro']:.4f}")
    print("-" * 70)
    cm = metrics["conf_matrix"]
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>12}", end="")
    for name in target_names:
        print(f"{name:>12}", end="")
    print()
    for i, name in enumerate(target_names):
        print(f"{name:<12}", end="")
        for j in range(len(target_names)):
            print(f"{cm[i, j]:>12}", end="")
        print()
    print("=" * 70)
    print(f"Artifacts saved to: {os.path.abspath(ARTIFACTS_DIR)}")
    print("=" * 70)


def _param(params: dict, key: str):
    """Safely extract a parameter value from the best_params_ dict."""
    return params.get(key, "N/A")


if __name__ == "__main__":
    main()
