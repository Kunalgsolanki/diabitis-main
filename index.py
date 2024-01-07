import numpy as np
import pickle
import streamlit as st
import os
import matplotlib.pyplot as plt

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

    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value')
    Age = st.text_input('Age of the person')
    
    # Code for prediction
    diagnosis = ''  # Null string
    
    # Creating a button for result
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, DiabetesPedigreeFunction, Age])
        
        # Display result
        st.success(diagnosis)
        
        # Visualize input data as a bar chart
        labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'BMI', 'Diabetes Pedigree Function', 'Age']
        input_values = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
        
        # Plotting bar chart
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        ax_bar.bar(labels, input_values)
        ax_bar.set_title('Input Data Bar Chart')
        ax_bar.set_xlabel('Features')
        ax_bar.set_ylabel('Values')
        
        # Plotting line chart
        fig_line, ax_line = plt.subplots(figsize=(10, 5))
        ax_line.plot(labels, input_values, marker='o', label='Input Data')
        ax_line.set_title('Input Data Line Chart')
        ax_line.set_xlabel('Features')
        ax_line.set_ylabel('Values')
        ax_line.legend()
        
        # Pass the figures to st.pyplot()
        st.pyplot(fig_bar)
        st.pyplot(fig_line)

if __name__ == '__main__':
    main()
