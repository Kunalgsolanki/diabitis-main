# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('./diabetes_model.sav', 'rb'))

input_data = (5,137,108,0,0,48.8,0.227,37)
# changing input data to numoy array
input_data_as_array = np.asarray(input_data)

#reshapeing the data or array as we are predicting for one instance
input_data_reshaped = input_data_as_array.reshape(1, -1)



prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == '0'): # [] is list the predictor gives a list thats why the first values [0] we have to count it or define it
  print('Person is not diabetic')
else:
  print('Person has diabetes')