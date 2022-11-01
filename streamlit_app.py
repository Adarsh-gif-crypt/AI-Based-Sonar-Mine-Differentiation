import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random

import streamlit as st
from matplotlib import pyplot as plt

import joblib
import time

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout = 'wide',initial_sidebar_state='collapsed')

input_selectbox = st.sidebar.selectbox(
    "Select type of Input",
    ("","Custom Input", "Random Input")
)

def read_shift(filepath):
    df = pd.read_csv(filepath)
    df = shuffle(df).reset_index(drop = True)
    return df

df = read_shift('Sonar Data.csv')

def random_num_gen():
    x = random.randint(0,df.shape[0])
    return x

#debug 
#if st.button('debug'):
#    n = random_num_gen()
#    st.write(n)

s0, h0, s1 = st.columns((2.5,5,2))
h0.title('Sonar vs Mine Predictor Simulation')

st.markdown("""Sonar (sound navigation and ranging) is a technique that uses sound propagation (usually underwater, as in submarine navigation) 
to navigate, measure distances (ranging), communicate with or detect objects on or under the surface of the water, such as other vessels. 
Two types of technology share the name "sonar": passive sonar is essentially listening for the sound made by vessels; active sonar is 
emitting pulses of sounds and listening for echoes. 
This is a web app which simulates an AI based 
predictor which based on the data received, predicts if the object is a Mine or a Rock.
This predictor is useful mainly in Submarines which operate in deep waters where visibility is very low.
This simulation takes data in two forms.
- A random input from the dataset below
- A custom input provided by You (The User) """)


s2, t1, s3 = st.columns((5,6,2))
t1.markdown("""#### Let's have a look at the dataset """)
s4, df0, s5 = st.columns((1,6,1))
df0.write(df)
st.markdown("""This dataset contains 198 patterns obtained by bouncing sonar signals 
off a metal cylinder at various angles and under various conditions
and patterns obtained from rocks under similar conditions. The transmitted sonar signal is a frequency-modulated chirp, rising in frequency. 
The data set contains signals obtained from a variety of different aspect angles,
spanning 90 degrees for the cylinder and 180 degrees for the rock.""")

n = random_num_gen()

def convert_data():
    data_list = df.iloc[n,0:60]
    data_lis = np.asarray(data_list)
    data_lis_reshaped = data_lis.reshape(1,-1)
    return data_lis_reshaped

model_selectbox = st.sidebar.selectbox(
    'Select the model',
    ('','Random Forest Classifier','Support Vector Classifier','Neural Network')
)

if input_selectbox == '':
    s6, t2, s7 = st.columns((3,6,2))
    t2.markdown("""### Go ahead and select a type of input from the sidebar """)
#if input_selectbox == "":
#    h0.markdown("""### Go ahead and select a type of input from the sidebar""")


if input_selectbox == 'Random Input':
    s8, t3, s9 = st.columns((4,6,2))
    if model_selectbox == '':
        t3.markdown("""### Please Select a model from the Sidebar!""")
    if model_selectbox == 'Random Forest Classifier':
        t3.markdown("""### Model Selected: Random Forest Classifier""")
        model = joblib.load('random_forest_model.joblib')
        s10, b0, s11 = st.columns((4,2,3))
        if b0.button('Detect'):
            with st.spinner():
                time.sleep(5)
            b0.success('Anomaly Detected')
        temp3, b1, temp4 = st.columns((3.7,2,3))
        temp5, suct0, temp6 = st.columns((1,5,1))
        if b1.button('Predict Anomaly'):
            with st.spinner():
                time.sleep(2)
            suct0.success('Anomaly Predicted')
            s12, t4, s13 = st.columns((5.2,6,2))
            predicting_data = convert_data()
            prediction = model.predict(predicting_data)
            if prediction==0:
                t4.markdown("""#### Object Idenitifed: Mine""")
            else:
                t4.markdown("""#### Object Identified: Rock""")
            t4.markdown(f"""Selected Row Number: {n}""")
    
    if model_selectbox == 'Support Vector Classifier':
        t3.markdown("""### Model Selected: Support Vector Classifier""")
        model = joblib.load('svc_model.joblib')
        s10, b0, s11 = st.columns((4,2,3))
        if b0.button('Detect'):
            with st.spinner():
                time.sleep(5)
            b0.success('Anomaly Detected')
        temp3, b1, temp4 = st.columns((3.7,2,3))
        temp5, suct0, temp6 = st.columns((1,5,1))
        if b1.button('Predict Anomaly'):
            with st.spinner():
                time.sleep(2)
            suct0.success('Anomaly Predicted')
            s12, t4, s13 = st.columns((5.2,6,2))
            predicting_data = convert_data()
            prediction = model.predict(predicting_data)
            if prediction==0:
                t4.markdown("""#### Object Idenitifed: Mine""")
            else:
                t4.markdown("""#### Object Identified: Rock""")
            t4.markdown(f"""Selected Row Number: {n}""")

    if model_selectbox == 'Neural Network':
        t3.markdown("""### Model Selected: Artificial Neural Network""")
        model = tf.keras.models.load_model('saved_model/my_model')
        s10, b0, s11 = st.columns((4,2,3))
        if b0.button('Detect'):
            with st.spinner():
                time.sleep(5)
            b0.success('Anomaly Detected')
        temp3, b1, temp4 = st.columns((3.7,2,3))
        temp5, suct0, temp6 = st.columns((1,5,1))
        if b1.button('Predict Anomaly'):
            with st.spinner():
                time.sleep(2)
            suct0.success('Anomaly Predicted')
            s12, t4, s13 = st.columns((5.2,6,2))
            predicting_data = convert_data()
            sc = StandardScaler()
            predicting_data = sc.fit_transform(predicting_data)
            prediction = model.predict(predicting_data)
            if prediction==0:
                t4.markdown("""#### Object Idenitifed: Mine""")
            else:
                t4.markdown("""#### Object Identified: Rock""")
            t4.markdown(f"""Selected Row Number: {n}""")

#def convert_raw_data(text):
#    ls = text.split(',')
#    ls = [float[x] for x in ls]
#    return ls

def convert_raw(raw):
    text = raw.split(',')
    text = [float(x) for x in text]
    text = np.asarray(text)
    text = text.reshape(1,-1)
    return text

if input_selectbox == 'Custom Input':
    s8, t3, s9 = st.columns((4,6,2))
    if model_selectbox == '':
        t3.markdown("""### Please Select a model from the Sidebar!""")
    if model_selectbox == 'Random Forest Classifier':
        t3.markdown("""### Model Selected: Random Forest Classifier""")
        model = joblib.load('random_forest_model.joblib')
        text_data = st.text_area('Enter the values seperated by a comma',
        placeholder = '0.0087	,0.0046	,0.0081	,0.0230	,0.0586	,0.0682	,0.0993	,0.0717	,0.0576	,0.0818	,0.1315	,0.1862	,0.2789	,0.2579	,0.2240	,0.2568	,0.2933	,0.2991	,0.3924	,0.4691	,0.5665	,0.6464	,0.6774,0.7577	,0.8856	,0.9419	,1.0000	,0.8564	,0.6790	,0.5587	,0.4147	,0.2946	,0.2025	,0.0688	,0.1171	,0.2157	,0.2216	,0.2776	,0.2309	,0.1444	,0.1513	,0.1745	,0.1756	,0.1424	,0.0908	,0.0138	,0.0469	,0.0480	,0.0159	,0.0045	,0.0015	,0.0052	,0.0038	,0.0079	,0.0114	,0.0050	,0.0030	,0.0064	,0.0058	,0.0030')
        s10, b0, s11 = st.columns((4,2,3))
        if b0.button('Detect',key = '0'):
            with st.spinner():
                time.sleep(5)
            b0.success('Anomaly Detected')
        temp3, b1, temp4 = st.columns((3.7,2,3))
        temp5, suct0, temp6 = st.columns((1,5,1))
        if b1.button('Predict Anomaly',key = '01'):
            prd_data = convert_raw(text_data)
            with st.spinner():
                time.sleep(2)
            suct0.success('Anomaly Predicted')
            s12, t4, s13 = st.columns((5.2,6,2))
            prediction = model.predict(prd_data)
            if prediction==0:
                t4.markdown("""#### Object Idenitifed: Mine""")
            else:
                t4.markdown("""#### Object Identified: Rock""")

    if model_selectbox == 'Support Vector Classifier':
        t3.markdown("""### Model Selected: Support Vector Classifier""")
        model = joblib.load('svc_model.joblib')
        text_dataa = st.text_area('Enter the values seperated by a comma',
        placeholder = '0.0087	,0.0046	,0.0081	,0.0230	,0.0586	,0.0682	,0.0993	,0.0717	,0.0576	,0.0818	,0.1315	,0.1862	,0.2789	,0.2579	,0.2240	,0.2568	,0.2933	,0.2991	,0.3924	,0.4691	,0.5665	,0.6464	,0.6774,0.7577	,0.8856	,0.9419	,1.0000	,0.8564	,0.6790	,0.5587	,0.4147	,0.2946	,0.2025	,0.0688	,0.1171	,0.2157	,0.2216	,0.2776	,0.2309	,0.1444	,0.1513	,0.1745	,0.1756	,0.1424	,0.0908	,0.0138	,0.0469	,0.0480	,0.0159	,0.0045	,0.0015	,0.0052	,0.0038	,0.0079	,0.0114	,0.0050	,0.0030	,0.0064	,0.0058	,0.0030')
        s10, b0, s11 = st.columns((4,2,3))
        if b0.button('Detect', key = '1'):
            with st.spinner():
                time.sleep(5)
            b0.success('Anomaly Detected')
        temp3, b1, temp4 = st.columns((3.7,2,3))
        temp5, suct0, temp6 = st.columns((1,5,1))
        if b1.button('Predict Anomaly',key = '02'):
            prd_data = convert_raw(text_dataa)
            with st.spinner():
                time.sleep(2)
            suct0.success('Anomaly Predicted')
            s12, t4, s13 = st.columns((5.2,6,2))
            prediction = model.predict(prd_data)
            if prediction==0:
                t4.markdown("""#### Object Idenitifed: Mine""")
            else:
                t4.markdown("""#### Object Identified: Rock""")

    if model_selectbox == 'Neural Network':
        t3.markdown("""### Model Selected: Artificial Neural Network""")
        model = tf.keras.models.load_model('saved_model/my_model')
        text_dataaa = st.text_area('Enter the values seperated by a comma',
        placeholder = '0.0087	,0.0046	,0.0081	,0.0230	,0.0586	,0.0682	,0.0993	,0.0717	,0.0576	,0.0818	,0.1315	,0.1862	,0.2789	,0.2579	,0.2240	,0.2568	,0.2933	,0.2991	,0.3924	,0.4691	,0.5665	,0.6464	,0.6774,0.7577	,0.8856	,0.9419	,1.0000	,0.8564	,0.6790	,0.5587	,0.4147	,0.2946	,0.2025	,0.0688	,0.1171	,0.2157	,0.2216	,0.2776	,0.2309	,0.1444	,0.1513	,0.1745	,0.1756	,0.1424	,0.0908	,0.0138	,0.0469	,0.0480	,0.0159	,0.0045	,0.0015	,0.0052	,0.0038	,0.0079	,0.0114	,0.0050	,0.0030	,0.0064	,0.0058	,0.0030',
        key = 'num3')
        s10, b0, s11 = st.columns((4,2,3))
        if b0.button('Detect',key = '2'):
            with st.spinner():
                time.sleep(5)
            b0.success('Anomaly Detected')
        temp3, b1, temp4 = st.columns((3.7,2,3))
        temp5, suct0, temp6 = st.columns((1,5,1))
        if b1.button('Predict Anomaly',key = '03'):
            with st.spinner():
                time.sleep(2)
            suct0.success('Anomaly Predicted')
            s12, t4, s13 = st.columns((5.2,6,2))
            predicting_data = convert_raw(text_dataaa)
            sc = StandardScaler()
            predicting_data = sc.fit_transform(predicting_data)
            prediction = model.predict(predicting_data)
            if prediction==0:
                t4.markdown("""#### Object Idenitifed: Mine""")
            else:
                t4.markdown("""#### Object Identified: Rock""")
