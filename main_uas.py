import streamlit as st
import pandas as pd

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

st.set_page_config(
    page_title = "Home",
    page_icon = "🏠",
)

st.markdown("<h1 style='color: #22A7EC;'>Company Bankruptcy Prediction</h1>", unsafe_allow_html=True)
st.markdown("Aplikasi ini berguna untuk mengklasifikasi kebangkrutan sebuah perusahaan")
st.markdown("______")
st.sidebar.success("pilih halaman")

from PIL import Image
image = Image.open('company.jpg')

st.image(image, caption='~')

st.write(
    """
    # Definisi

    Prediksi kebangkrutan perusahaan adalah proses menggunakan berbagai metode analisis
    untuk mengevaluasi kesehatan keuangan suatu perusahaan dan memprediksi apakah
    perusahaan tersebut berisiko menghadapi kebangkrutan di masa depan.

    Tujuan dari prediksi kebangkrutan perusahaan adalah memberikan informasi kepada
    pemangku kepentingan, seperti pemilik saham, kreditor, dan pemasok, agar mereka dapat
    mengambil langkah-langkah yang tepat untuk mengurangi risiko atau melindungi
    kepentingan mereka.

    Variabel yang akan kami gunakan pada penelitian ini adalah

    x1 = working capital to total assets

    x2 = retained earning to total assets

    x3 = ROA (C) : Return On Total Assets (C) : before interest and depreciation before interest
    
    x4 = net worth / assets

    Machine learning yang akan kami gunakan adalah SVM
    """
)


# About us
st.sidebar.header('About Us')
st.sidebar.markdown('Created by Kelompok 8')