import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import random

import streamlit as st
from matplotlib import pyplot as plt

import joblib
import time

import warnings
warnings.filterwarnings('ignore')

#st.set_page_config(layout = 'wide')


def read_shift(filepath):
    df = pd.read_csv(filepath)
    df = shuffle(df).reset_index(drop = True)
    return df

df = read_shift('Sonar Data.csv')

def random_num_gen():
    x = random.randint(0,df.shape[0])
    return x

def convert_data():
    n = random_num_gen()
    data_list = df.iloc[n,0:60]
    data_lis = np.asarray(data_list)
    data_lis_reshaped = data_lis.reshape(1,-1)
    return data_lis_reshaped


#### Web App Structure Starts HERE ####

st.title('Live Sonar Simulation')
st.subheader('This is a web app to simulate a Sonar')
st.subheader('This simulation uses AI to find the difference between A Rock and A Mine')
st.markdown("""For more information on the AI techniques used, visit [this link]()""")

st.markdown("""Let's have a look at the dataset""")
st.write(df)
st.markdown("""This dataset contains 198 patterns obtained by bouncing sonar signals 
off a metal cylinder at various angles and under various conditions
and patterns obtained from rocks under similar conditions. The transmitted sonar signal is a frequency-modulated chirp, rising in frequency. 
The data set contains signals obtained from a variety of different aspect angles,
spanning 90 degrees for the cylinder and 180 degrees for the rock.""")

category = st.selectbox('Select a category',('Random Forest Classifier','Support Vector Machine','Artificial Neural Network'))

if(category == 'Random Forest Classifier'):
    model = joblib.load('random_forest_model.joblib')
    if st.button('Detect'):
        time.sleep(10)
    if st.button('Predict'):
        time.sleep(2)
        predicting_data = convert_data()
        prediction = model.predict(predicting_data)
        if prediction==0:
            st.markdown("""Object Idenitifed: Mine""")
        else:
            st.markdown("""Object Identified: Rock""")