# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 11:52:09 2021

@author: Harish
"""

import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("C:/Users/Harish/Documents/Projects/Air Quality Index/random_forest_regression_model.pkl", 'rb'))

st.write("""Air Quality Index Prediction App""")

st.sidebar.header('User Input Features')

T = st.sidebar.number_input("Average Temperature in °C (T)")
TM = st.sidebar.number_input("Maximum temperature in °C (TM)")
Tm = st.sidebar.number_input("Minimum temperature in °C (Tm)")
SLP = st.sidebar.number_input("Atmospheric pressure at sea level in hPa (SLP)")
H = st.sidebar.number_input("Average relative humidity in % (H)")
VV = st.sidebar.number_input("Average visibility in Km (VV)")
V = st.sidebar.number_input("Average wind speed in Km/h (V)")
VM = st.sidebar.number_input("Maximum sustained wind speed in Km/h (VM)")

test = pd.DataFrame([T, TM, Tm, SLP, H, VV, V, VM]).T
test.columns =['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']

st.table(test)

pred = model.predict(test)

st.write("Predicted PM 2.5 value: \n {}".format(pred[0]))
