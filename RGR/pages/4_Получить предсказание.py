import pickle
import random
import numpy as np
import pandas as pd
from sklearn import model_selection
import streamlit as st
import tensorflow as tf


st.set_page_config(layout="wide")

st.title("Предсказания")

st.sidebar.header("Получить предсказание")

st.info(
    """
    Заполните _все_ характеристики, чтобы предсказать возможное возгорание.
    """
)

temperature = st.number_input("Введите температуру",-50.0, 60.0)

humidity = st.number_input("Введите влажность воздуха", 0.0, 100.0)

TVOC = st.number_input("Введите общее количество летучих органических соединений", 0.0, 60000.0)

eCO2 = st.number_input("Введите концентрацию CO2", 0.0, 60000.0)

RawH2 = st.number_input("Введите содержание H2", 10000.0, 13800.0)

RawEthanol = st.number_input("Введите содержание газообразного этанола", 15300, 21400)

Pressure = st.number_input("Введите давление воздуха", 931.0, 940.0)

PM10 = st.number_input("Введите число твердых частиц димаметром менее 1 микрона", 0.0, 14333.69)

PM25 = st.number_input("Введите число твердых частиц димаметром более 1 микрона, но менее 2.5 микрон", 0.0, 45432.26)

NC05 = st.number_input("Введите численную концентрацию твердых частиц в воздухе димаметром менее 0.5 микрона", 0.0, 61482.03)

NC10 = st.number_input("Введите численную концентрацию твердых частиц в воздухе димаметром более 0.5, но менее 1 микрона", 0.0, 51914.68)

NC25 = st.number_input("Введите численную концентрацию твердых частиц в воздухе димаметром более 1, но менее 2.5 микрон", 0.0, 30026.438)


models = st.selectbox(
    "Выберите классификатор", ["KNN", "Bagging", "Neural Network"]
)

if st.button("Предсказать"):
    
    data = [temperature, humidity, TVOC, eCO2, RawH2, RawEthanol, Pressure, PM10, PM25, NC05, NC10, NC25]
    data = np.array(data).reshape((1, -1))
    
    knn_model = pickle.load(open('../models/KNN_model', 'rb'))
    bagging_model =  pickle.load(open('../models/bagging_classification_model', 'rb'))
    neuro_model = tf.keras.models.load_model('C:/Users/verab/Desktop/save/Neuro_classification_model')
    
    if models == "KNN":
        st.title("Предсказание с помощью модели KNN")
        pred = knn_model.predict(data)
        
        if round(float(pred[0])) == 1:
            st.write(":red[_Пожар!!!_] Советуем Вам срочно обратиться в службу пожаротушения.")
        else:
            st.write("Возгорания не обнаружено! Можете спать спокойно.")
                 
    elif models == "Bagging":
        st.title("Предсказание с помощью модели Bagging")
        pred = bagging_model.predict(data)
        
        if round(float(pred[0])) == 1:
            st.write(":red[_Пожар!!!_] Советуем Вам срочно обратиться в службу пожаротушения.")
        elif round(float(pred[0])) == 0:
            st.write("Возгорания не обнаружено! Можете спать спокойно.")
                 
    elif models == "Neural Network":
        st.title("Предсказание с помощью модели TensorFlow")
        pred = neuro_model.predict(data)
        
        if round(float(pred[0])) == 1:
            st.write(":red[_Пожар!!!_] Советуем Вам срочно обратиться в службу пожаротушения.")
        elif round(float(pred[0])) == 0:
            st.write("Возгорания не обнаружено! Можете спать спокойно.")
    
    
    
    
