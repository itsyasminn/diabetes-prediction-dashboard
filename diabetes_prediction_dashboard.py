import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
best_model = joblib.load("best_diabetes_model.joblib")
importance_df = pd.DataFrame({
     'feature': ['Pregnancies','Glucose','BloodPressure','SkinThickness',
'Insulin','BMI','DiabetesPedigreeFunction','Age'],
    'importance': best_model.named_steps['clf'].feature_importances_ 
})

import sqlite3
from datetime import datetime

conn = sqlite3.connect("diabetes_predictions.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pregnancies INTEGER,
    glucose INTEGER,
    blood_pressure INTEGER,
    skin_thickness INTEGER,
    insulin INTEGER,
    bmi REAL,
    diabetes_pedigree_function REAL,
    age INTEGER,
    prediction INTEGER,
    probability REAL,
    timestamp DATETIME
)
''')
conn.commit()


st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Diabetes Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #34495E;'>Enter patient details to predict diabetes risk</h4>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("Patient Information")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=2)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=120)
bp = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=35)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=100)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=60.0, value=25.5)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=32)

predict_button = st.sidebar.button("Predict")

if predict_button:
    columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age']
    
    data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],columns=columns)

    prediction = best_model.predict(data)[0]
    probability = best_model.predict_proba(data)[:,1][0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Prediction: Diabetes detected")
    else:
        st.success("Prediction: No Diabetes detected")

    st.info(f"Probability of having diabetes: {probability*100:.1f}%")

    st.subheader("Prediction Confidence")
    st.progress(int(probability*100))


    c.execute('''
        INSERT INTO predictions (
            pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
            diabetes_pedigree_function, age, prediction, probability, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, prediction, probability, datetime.now()))
    conn.commit()


    #FEATURE IMPORTANCE CHART
st.subheader("Feature Importance")
plt.figure(figsize=(6,4))
sns.barplot(x=importance_df['importance'], y=importance_df['feature'], palette="Blues_d")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(plt)

st.subheader("Prediction History")
df_history = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
st.dataframe(df_history)