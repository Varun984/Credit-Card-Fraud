import pandas as pd 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier, plot_importance
import joblib




# Load data
df = pd.read_csv(r'creditcard-fraud-detection\data\PS_20174392719_1491204439457_log.csv')

# Split features and target
X = df.drop(["isFraud"], axis=1)
Y = df["isFraud"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Ordinal Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(X_train[categorical_cols])

# Transform categorical columns
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
X_train_encoded[categorical_cols] = encoder.transform(X_train[categorical_cols])
X_test_encoded[categorical_cols] = encoder.transform(X_test[categorical_cols])

# Save encoder
joblib.dump(encoder, 'ordinal_encoder.pkl')
# XGBoost: adjust class imbalance
scale_pos_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])
xgb = XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)

# ✅ Train on resampled data
xgb.fit(X_train_encoded, Y_train)

# Predict and evaluate
y_probs =  xgb.predict_proba(X_test_encoded)[:, 1]
y_pred = (y_probs >= 0.9924).astype(int)

# Evaluation metrics
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
print(accuracy_score(Y_test, y_pred))
auc = roc_auc_score(Y_test, y_probs)
print("AUC Score:", auc)

# Save model
joblib.dump(xgb, 'xgb_model.pkl')
