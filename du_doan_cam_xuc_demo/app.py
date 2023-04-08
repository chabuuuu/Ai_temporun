import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf 
from preprocess import *

st.write("""
# Demo thi AI Temporun

Dự đoán cảm xúc thông qua văn bản


""")

st.sidebar.header('Chỗ để nhập văn bản')




def user_input():
    text = st.sidebar.text_input('Viết gì đó đi ')

    return text
input = user_input()

st.write(input)
encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))


model=tf.keras.models.load_model('my_model2.h5')
input=preprocess(input)

array = cv.transform([input]).toarray()

pred = model.predict(array)

all_res = pred

a=np.argmax(pred, axis=1)
prediction = encoder.inverse_transform(a)[0]


st.subheader('Dự đoán')
if input == '':
    st.write('Bố xin phép dự đoán cảm xúc của m đang là...')
else:

    st.write(prediction)
    st.write(all_res)

