# -*- coding: utf-8 -*-
"""
Created on Fri Dec 9 13:06:59 2022
@author: MITHIL
"""

import numpy as np
import pickle 
import streamlit as st
import os  # Import the os module for working with file paths

# Check and install scikit-learn
try:
    import sklearn
except ModuleNotFoundError:
    os.system('pip install scikit-learn')

# Check the current working directory
print("Current working directory:", os.getcwd())

# Loading the saved model
model_path = './diabetes_model.sav'
if os.path.exists(model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

# Creating the function for prediction
def diabetes_prediction(input_data):
    # Changing input data to numpy array
    input_data_as_array = np.asarray(input_data)
    # Reshaping the data as we are predicting for one instance
    input_data_reshaped = input_data_as_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:  # Assuming it's a binary classification (0 or 1)
        return 'Person is not diabetic'
    else:
        return 'Person has diabetes'

def main():
    # Giving the title
    st.title('Diabetes prediction web app')
    
    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood pressure value')
    SkinThickness = st.text_input('Skin thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value')
    Age = st.text_input('Age of the person')
    
    # Code for prediction
    diagnosis = ''  # Null string
    
    # Creating a button for result
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
