import streamlit as st
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load the saved MLP model
model_filename = 'C:\\Users\\User\\PycharmProjects\\MLAssignmentWithUI\\venv\\Tumor-Prediction\\bestmlpmodel (1).pkl'
bestmlp = joblib.load(model_filename)

# Load the saved scaler
scaler_filename = 'C:\\Users\\User\\PycharmProjects\\MLAssignmentWithUI\\venv\\Tumor-Prediction\\scaler (1).pkl'
scaler = joblib.load(scaler_filename)

# Define the features
features = ['id', 'diagnosis', 'texture_mean', 'smoothness_mean',
            'compactness_mean', 'symmetry_mean', 'radius_se', 'perimeter_se',
            'area_se', 'concave points_se', 'texture_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'symmetry_worst',
            'fractal_dimension_worst']


def predict_diagnosis(user_input):
    # Normalize the user input using the loaded scaler
    user_input_normalized = scaler.transform(user_input)

    # Make predictions using the MLP model
    prediction = bestmlp.predict(user_input_normalized)

    # Convert the prediction to the corresponding label
    if prediction[0] == 0:
        diagnosis = 'Benign Tumor'
    else:
        diagnosis = 'Malignant Tumor'

    return diagnosis

def main():
    st.title("Breast Cancer Diagnosis Prediction")

    # Create a DataFrame with the user input (excluding 'id' and 'diagnosis')
    user_input = pd.DataFrame(columns=[col for col in features if col not in ['id', 'diagnosis']])

    # Collect user input
    for feature in user_input.columns:
        value = st.number_input(f"Enter {feature}", step=0.00001, format="%.5f")
        user_input[feature] = [value]

    # Show the user input
    st.header("User Input")
    st.write(user_input)

    # Make predictions
    if st.button("Predict"):
        prediction = predict_diagnosis(user_input)
        st.success(f"Predicted diagnosis: {prediction}")

if __name__ == "__main__":
    main()