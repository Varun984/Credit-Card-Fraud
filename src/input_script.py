from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Project root so model paths work from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

st.title("Credit Card Fraud Detection!")

model = joblib.load(MODELS_DIR / "xgb_model.pkl")
encoder = joblib.load(MODELS_DIR / "ordinal_encoder.pkl")

categorical_cols = ['type', 'nameOrig', 'nameDest']
def preprocess_input(data):
    # Encode categorical features
    data_encoded = data.copy()
    data_encoded[categorical_cols] = encoder.transform(data[categorical_cols])
    return data_encoded

def predict_fraud(input_data):
    input_data = preprocess_input(input_data)
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[:, 1]
    st.write(f"Prediction Confidence: {confidence[0]:.2f}") 
    return prediction


st.header("Input Data")

# Add tabs for different input methods
tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])

with tab1:
    # Input fields
    type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    if amount>200000:
        isFlaggedFraud = 1
    else:
        isFlaggedFraud = 0
    step = st.number_input("Transaction Step", min_value=0, step=1)
    nameOrig = st.text_input("Origin Account Name")
    oldBalanceOrig = st.number_input("Origin Account Old Balance", min_value=0.0, step=0.01)
    newBalanceOrig = st.number_input("Origin Account New Balance", min_value=0.0, step=0.01)
    nameDest = st.text_input("Destination Account Name")
    oldBalanceDest = st.number_input("Destination Account Old Balance", min_value=0.0, step=0.01)
    newBalanceDest = st.number_input("Destination Account New Balance", min_value=0.0, step=0.01)
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'step': [step],
        'type': [type],
        'amount': [amount],
        'nameOrig': [nameOrig],
        'oldbalanceOrg': [oldBalanceOrig],
        'newbalanceOrig': [newBalanceOrig],
        'nameDest': [nameDest],
        'oldbalanceDest': [oldBalanceDest],
        'newbalanceDest': [newBalanceDest],
        'isFlaggedFraud': [isFlaggedFraud]

    })
    
    if st.button("Predict Fraud"):
        prediction = predict_fraud(input_data)
        if prediction[0] == 1:
            st.error("Fraudulent Transaction Detected!")
        else:
            st.success("Transaction is Legitimate.")

with tab2:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        try:
            csv_data = pd.read_csv(uploaded_file)
            st.write("CSV Preview:")
            st.dataframe(csv_data.head())
            
            # Check if the CSV has the required columns
            required_columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 
                              'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
            
            missing_columns = [col for col in required_columns if col not in csv_data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Process the CSV data
                if st.button("Predict Fraud for All Transactions"):
                    # Create a new column for predictions
                    csv_data_encoded = preprocess_input(csv_data)
                    predictions = model.predict(csv_data_encoded)
                    confidence = model.predict_proba(csv_data_encoded)[:, 1]
                    
                    # Add predictions to the dataframe
                    result_df = csv_data.copy()
                    result_df['fraud_prediction'] = predictions
                    result_df['confidence'] = confidence
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(result_df)
                    
                    # Summary statistics
                    fraud_count = sum(predictions)
                    st.write(f"Total transactions: {len(predictions)}")
                    st.write(f"Fraudulent transactions detected: {fraud_count} ({fraud_count/len(predictions)*100:.2f}%)")
                    
                    # Option to download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")