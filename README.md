# 🛡️ FraudShield AI — Credit Card Fraud Detection

A premium Streamlit dashboard powered by XGBoost for detecting fraudulent credit card transactions with interactive visualizations, batch predictions, and model explainability.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-006600?style=flat-square)

## ✨ Features

- **📊 Interactive Dashboard** — Real-time dataset analytics with Plotly charts (donut, histogram, bar)
- **🔍 Single Transaction Prediction** — Manual input with instant fraud probability & confidence bar
- **📁 Batch CSV Upload** — Bulk prediction with risk-level classification and downloadable results
- **📈 Model Insights** — Feature importance, radar chart, ROC curve, confusion matrix heatmap
- **🎨 Premium Dark UI** — Glass-morphism cards, animated gradients, Inter typography

## 🏗️ Project Structure

```
creditcard-fraud/
├── .streamlit/
│   └── config.toml         # Dark theme configuration
├── config/
│   ├── __init__.py
│   └── paths.py            # Central path configuration
├── data/
│   └── data.csv            # Transaction dataset
├── models/
│   ├── xgb_model.pkl       # Trained XGBoost model
│   └── ordinal_encoder.pkl # Fitted encoder
├── scripts/
│   └── train.py            # Model training script
├── src/
│   ├── app.py              # Legacy training script
│   └── input_script.py     # Main Streamlit dashboard
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (if needed)
python scripts/train.py

# Launch the dashboard
streamlit run src/input_script.py
```

Then open **http://localhost:8501** in your browser.

## 📊 Dataset

Uses the **PaySim** synthetic financial dataset with:

| Feature | Description |
|---------|------------|
| `step` | Hour of simulation (1 step = 1 hour) |
| `type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount |
| `nameOrig` / `nameDest` | Origin / Destination account IDs |
| `oldbalanceOrg` / `newbalanceOrig` | Origin balance before / after |
| `oldbalanceDest` / `newbalanceDest` | Destination balance before / after |
| `isFlaggedFraud` | System flag for amounts > $200K |
| `isFraud` | Target label (binary) |

## 🧠 Model

- **Algorithm:** XGBoost with `scale_pos_weight` for class imbalance
- **Encoding:** OrdinalEncoder for categorical features
- **Evaluation:** Accuracy, Precision, Recall, F1, AUC-ROC

## 🛠️ Tech Stack

`Python` · `Streamlit` · `XGBoost` · `scikit-learn` · `Plotly` · `Pandas` · `NumPy`

## 📜 License

See [LICENSE](LICENSE) for details.

## 🤝 Contributing

See [contributing.md](contributing.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
