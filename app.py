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

T = st.sidebar.number_input("Average annual temperature (T)")
TM = st.sidebar.number_input("Annual average maximum temperature (TM)")
Tm = st.sidebar.number_input("Average annual minimum temperature (Tm)")
SLP = st.sidebar.number_input("Rain or snow precipitation total annual (SLP)")
H = st.sidebar.number_input("Number of days with hail (H)")
VV = st.sidebar.number_input("Number of days with rain (VV)")
V = st.sidebar.number_input("Annual average wind speed (V)")
VM = st.sidebar.number_input("Number of days with storm (VM)")

test = pd.DataFrame([T, TM, Tm, SLP, H, VV, V, VM]).T
test.columns =['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']


st.table(test)

pred = model.predict(test)

st.write("Predicted PM 2.5 value: \n {}".format(pred[0]))
