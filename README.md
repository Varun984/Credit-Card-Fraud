# Credit Card Fraud Detection

A Streamlit app and XGBoost model for detecting fraudulent transactions from payment logs.

## Features

- **Manual input**: Enter a single transaction and get a fraud prediction with confidence.
- **CSV upload**: Upload a CSV of transactions and get batch predictions with optional download.
- **XGBoost model** with class-weight handling for imbalanced data and ordinal encoding for categorical features.

## Project structure

```
creditcard-fraud-detection/
├── config/           # Path and config (e.g. paths.py)
├── data/             # Raw/input data (e.g. transaction CSV)
├── models/           # Saved model and encoder (.pkl)
├── scripts/          # Training script (train.py)
├── src/              # App entry (Streamlit)
│   ├── input_script.py   # Main Streamlit app
│   └── app.py            # Redirect note (run scripts/train.py)
├── requirements.txt
├── .gitignore
└── README.md
```

## Dependencies

- Python 3.8+
- pandas, scikit-learn, imbalanced-learn, xgboost, streamlit, matplotlib, numpy, joblib

## Setup

From the project root (`creditcard-fraud-detection/`):

```bash
# Optional: create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

## Running

**Train the model** (uses `data/PS_20174392719_1491204439457_log.csv`, saves to `models/`):

```bash
python scripts/train.py
```

**Run the Streamlit app** (loads models from `models/`):

```bash
streamlit run src/input_script.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Data

Place the transaction CSV in `data/`. The training script expects:

- File: `data/PS_20174392719_1491204439457_log.csv`
- Target column: `isFraud`
- Features: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFlaggedFraud

## Contributors

See `contributing.md` and `CODE_OF_CONDUCT.md` for guidelines.
