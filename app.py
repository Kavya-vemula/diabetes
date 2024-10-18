import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Define the prediction function
def predict_diabetes(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

# Streamlit app
st.title('Diabetes Prediction')
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=0)
insulin = st.number_input('Insulin', min_value=0, max_value=846, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, value=0.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.42, value=0.0)
age = st.number_input('Age', min_value=21, max_value=81, value=21)

if st.button('Predict'):
    result = predict_diabetes([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    st.success(f'The person is {result}')
