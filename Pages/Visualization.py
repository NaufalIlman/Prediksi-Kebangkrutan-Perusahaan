import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.markdown("<h1 style='color: #22A7EC;'>Visualisasi Data</h1>", unsafe_allow_html=True)
st.write("#### Berikut adalah visualisasi data mengenai company bankruptcy prediction")
st.markdown("______")

#sidebar
st.sidebar.title('Visualisasi apakah perusahaan tersebut bangkrut atau tidak')

data=pd.read_csv('Data_UAS.csv')
#checkbox to show data 
if st.checkbox("Show Data"):
    st.write(data.head(10))

#selectbox + visualisation

# Multiple widgets of the same type may not share the same key.
select=st.sidebar.selectbox('pilih jenis grafik',['Histogram','Pie Chart'],key=0)
bankruptcy=data['Bankrupt?'].value_counts()
bankruptcy=pd.DataFrame({'bankruptcy':bankruptcy.index,'Jumlah':bankruptcy.values})

st.markdown("###  company bankruptcy count")
if select == "Histogram":
        fig = px.bar(bankruptcy, x='bankruptcy', y='Jumlah', color = 'bankruptcy', height= 500)
        st.plotly_chart(fig)
else:
        fig = px.pie(bankruptcy, values='Jumlah', names='bankruptcy')
        st.plotly_chart(fig)