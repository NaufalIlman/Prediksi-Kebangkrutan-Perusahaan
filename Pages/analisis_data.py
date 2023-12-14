import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def model():
    data = pd.read_csv('C:/Users/COMPUTER/Documents/Naufal/Kuliah/Semester 6/Machine Learning/Web app/UAS/UAS.csv')

    # Pisahkan fitur dan target
    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']

    # Pra-pemrosesan data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Bagi data menjadi data pelatihan dan data pengujian
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
    # Langkah 2: Membuat objek Decision Tree Classifier
    clf = DecisionTreeClassifier()

    # Langkah 3: Melatih model menggunakan data training
    clf.fit(X_train, y_train)

    # Langkah 4: Menggunakan model untuk melakukan prediksi pada data testing
    y_pred = clf.predict(X_test)


    # Langkah 5: Evaluasi performa model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    return clf

def main():
    st.markdown("<h1 style='color: #22A7EC;'>Company Bankruptcy Prediction</h1>", unsafe_allow_html=True)
    st.write("#### masukkan data variabel yang diperlukan")

    x1 = st.number_input('masukkan variabel Current Ratio', format="%.9f")
    x2 = st.number_input('masukkan variabel Retained Earnings to Total Assets', format="%.9f")
    x3 = st.number_input('masukkan variabel ROA(C) before interest and depreciation before interest', format="%.9f")
    x4 = st.number_input('masukkan variabel Net worth/Assets', format="%.9f")

    # If button is pressed
    if st.button("Submit"):
        
        # data = pd.read_csv('C:/Users/COMPUTER/Documents/Naufal/Kuliah/Semester 6/Machine Learning/Web app/UAS/UAS.csv')

        # # Pisahkan fitur dan target
        # X = data.drop('Bankrupt?', axis=1)
        # y = data['Bankrupt?']

        # # Pra-pemrosesan data
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)

        # # Bagi data menjadi data pelatihan dan data pengujian
        # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # # Langkah 2: Membuat objek Decision Tree Classifier
        # clf = DecisionTreeClassifier()

        # # Langkah 3: Melatih model menggunakan data training
        # clf.fit(X_train, y_train)
        clf = model()
        # Store inputs into dataframe
        X = pd.DataFrame([[x1, x2, x3, x4]], 
                        columns = ["Current Ratio", "Retained Earnings to Total Assets", "masukkan variabel ROA(C) before interest and depreciation before interest", "Net worth/Assets"])
        # Get prediction
        prediction = clf.predict(X)[0]
        
        if prediction == 0:
            st.write('The company is not bankrupt.')
        else:
            st.write('The company is bankrupt.')
main()
# Train an SVM model
#def user_input_features():
    #form = st.form(key='my_form')
    #x1 = form.number_input('masukkan variabel Current Ratio')
    #x2 = form.number_input('masukkan variabel Retained Earnings to Total Assets')
    #x3 = form.number_input('masukkan variabel ROA(C) before interest and depreciation before interest')
    #x4 = form.number_input('masukkan variabel Net worth/Assets')
    #orm.form_submit_button('prediksi')
    #data = {'Current Ratio': x1,
            #'Retained Earnings to Total Assets': x2,
            #'ROA(C) before interest and depreciation before interest': x3,
            #'Net worth/Assets': x4}
    #features = pd.DataFrame(data, index=[0])
    #return features
#input_df = user_input_features()

#UAS = pd.read_csv('Data_UAS.csv')
#penguins = UAS.drop(columns=['Bankrupt?'])
#df = input_df

# Reads in saved classification model
#load_clf = pickle.load(open('model_svm.pkl', 'rb'))


# Apply model to make predictions
#prediction = load_clf.predict(df)

#st.write("Hasil Prediksi:", prediction)


#st.subheader('Prediction')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
#st.write(penguins_species[prediction])

#st.subheader('Prediction Probability')
#st.write(prediction_proba)

# Display dataframe with prediction column
#st.write('DataFrame with Prediction:')
#st.write(data_with_prediction)

# Display prediction result
#if prediction[0] == 0:
    #st.write('The company is not likely to go bankrupt.')
#else:
    #st.write('The company is likely to go bankrupt.')