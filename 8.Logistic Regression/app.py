import streamlit as st
import pandas as pd
from pickle import load

st.title('Diabetes Prediction')
st.write('Predict whether a patient is diabetic or not.')

pregnancies = st.sidebar.number_input('Pregnancies', min_value=0)
glucose = st.sidebar.number_input('Glucose')
bloodpressure = st.sidebar.number_input('Blood Pressure')
skinthickness = st.sidebar.number_input('Skin Thickness')
insulin = st.sidebar.number_input('Insulin')
bmi = st.sidebar.number_input('BMI')
dpf = st.sidebar.number_input('Diabetes Pedigree Function')
age = st.sidebar.slider('Age', min_value=1, max_value=100)

data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': bloodpressure,
    'SkinThickness': skinthickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}

if st.sidebar.button('Submit'):
    features = pd.DataFrame(data, index=[0])
    st.write(features)

    loaded_model = load(open('logistic_model.pkl', 'rb'))
    scaler = load(open('scaler.pkl', 'rb'))

    features = scaler.transform(features)

    res = loaded_model.predict(features)

    if res[0] == 0:
        st.success('Patient is Not Diabetic')
    else:
        st.error('Patient is Diabetic')