# EV Adoption Likelihood Classification

A machine learning system that predicts EV adoption likelihood (**High** / **Medium** / **Low**) using a K-Nearest Neighbors classifier, deployed as an interactive Streamlit web application.

## Features

- **Interactive prediction form** — 22 input features across demographics, travel behaviour, charging infrastructure, and attitudinal scores
- **Real-time inference** — KNN model (k=15, distance-weighted, euclidean) with full probability breakdown
- **Visual results** — Confidence metrics, progress bars, bar chart of class probabilities
- **Input validation** — All fields validated before prediction with user-friendly error messages
- **Production-ready** — Modular prediction engine (`utils.predict`) usable independently of the UI

## Dataset

The model was trained on **50,000 records** from the **Global EV Adoption Behavior (2026)** dataset with **23 columns**:

| Category | Features |
|----------|----------|
| Demographics | age, annual income, education level, city type |
| Travel behaviour | daily commute, weekly travel distance, current vehicle type, vehicle age |
| Financial | fuel expense, electricity cost, monthly charging cost |
| Infrastructure | charging station accessibility, nearest charging station distance, home charging availability |
| Attitude / Awareness | environmental awareness, govt incentive awareness, technology affinity, range anxiety, battery replacement concern, EV knowledge, prior EV experience |
| **Target** | `ev_adoption_likelihood` (High / Medium / Low) |

## Project Structure

```
├── app.py                      # Streamlit application (entry point)
├── utils/
│   ├── __init__.py             # Package exports
│   ├── predict.py              # Prediction engine (reusable, no Streamlit dependency)
│   └── preprocess.py           # Data loading, column detection, encoding, scaling
├── model/
│   ├── model.pkl               # Full inference pipeline (preprocessor + KNN)
│   ├── encoder.pkl             # LabelEncoder for target decoding
│   ├── feature_columns.pkl     # Post-transform feature names
│   ├── preprocessor.pkl        # ColumnTransformer (fitted)
│   ├── scaler.pkl              # StandardScaler (fitted)
│   ├── ordinal_encoder.pkl     # OrdinalEncoder (fitted)
│   ├── onehot_encoder.pkl      # OneHotEncoder (fitted)
│   └── train_model.py          # Training script (Phase 2)
├── dataset/
│   └── global_ev_adoption_behavior_2026.csv
├── notebooks/
│   └── EDA.ipynb               # Exploratory data analysis
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd EV_Adoption_Classification

# Create and activate virtual environment (optional)
python -m venv venv
# venv\Scripts\activate   (Windows)

# Install dependencies
pip install -r requirements.txt
```

## Local Run

```bash
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`.

## Prediction Workflow

```
User Input (form)  →  utils.predict.predict(dict)  →  validate_input()
                                                        ↓
                                              prepare_dataframe()
                                                        ↓
                                              Pipeline.transform()
                                                        ↓
                                              KNN.predict() / predict_proba()
                                                        ↓
                                              LabelEncoder.inverse_transform()
                                                        ↓
                                              {class, confidence, probabilities}
```

### Using the prediction engine directly (without Streamlit)

```python
from utils.predict import predict, model_information

result = predict({
    "age": 34, "annual_income": 62000, "education_level": "Master",
    "city_type": "Urban", "daily_commute_km": 30.5,
    # ... all 22 features
})
print(result["class"])         # e.g. "High"
print(result["confidence"])    # e.g. 0.83
print(result["probabilities"]) # e.g. {"High": 0.83, "Medium": 0.12, "Low": 0.05}
```

## Training Workflow

```bash
python model/train_model.py
```

The script:
1. Loads and cleans the dataset (clips negative fuel expense to 0)
2. Builds a `ColumnTransformer` (median imputation → StandardScaler for numerical; most-frequent imputation → OrdinalEncoder / OneHotEncoder for categorical)
3. Performs GridSearchCV over k=[1,3,5,7,9,11,13,15], weights=[uniform,distance], metrics=[euclidean,manhattan]
4. Evaluates the best model on the held-out test set
5. Saves all artifacts (`model.pkl`, `encoder.pkl`, etc.) to the `model/` directory

### Best model performance

| Metric | Value |
|--------|-------|
| k (neighbors) | 15 |
| Weighting | distance |
| Metric | euclidean |
| CV F1 (weighted) | 0.8235 |
| Test accuracy | 0.8320 |

## Deployment (Streamlit Cloud)

1. Push the repository to GitHub (ensure all `.pkl` files are committed — `.gitignore` must NOT exclude them).
2. Log in to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select your repository, branch, and set **Main file path** to `app.py`.
4. Click **Deploy**.

No additional configuration is needed — all dependencies are listed in `requirements.txt`.

## Deployment Checklist

- [ ] Repository pushed to GitHub
- [ ] All model `.pkl` files committed and tracked
- [ ] `requirements.txt` includes streamlit, scikit-learn, pandas, numpy
- [ ] `app.py` uses relative paths (no system-specific paths)
- [ ] All imports resolve from project root (`from utils.predict import predict`)
- [ ] Streamlit Cloud app points to `app.py` as entry point
- [ ] No local-only dependencies or absolute paths

## Screenshots

*Screenshots to be added after deployment.*

## Future Improvements

- Add permutation feature importance for model explainability
- Support batch prediction (upload CSV)
- Add more classification algorithms (Random Forest, XGBoost) for comparison
- A/B test different preprocessing strategies
- Add authentication for multi-user scenarios
