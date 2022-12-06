import pickle 
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

#load save model
model = pickle.load(open('model_fraud.sav', 'rb'))

#judul web
st.title('Prediksi SMS')


tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

clean_text = st.text_input('Teks SMS')


#predict
hasil = ''

#tombol
if st.button('Prediksi SMS') :
    hasil = model.predict(loaded_vec.fit_transform([clean_text]))

    if(hasil==0):
        hasil = "SMS Normal"
    elif(hasil==1):
        hasil = "SMS Fraud"
    else:
        hasil = "SMS Promo"
st.success(hasil)

st.write('191351096 - Siti Shintia')