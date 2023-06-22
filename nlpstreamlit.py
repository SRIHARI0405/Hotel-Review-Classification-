# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 00:34:44 2022

@author: rupesh
"""

# -*- coding: utf-8 -*-


import nltk
nltk.download('wordnet')
import pandas as pd
import warnings
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import re
#from rake_nltk import Rake
import pickle
import streamlit as st
import numpy as np
from nltk.stem import PorterStemmer,WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS

pickle_in = open("logistic.pkl","rb")
nlp=pickle.load(pickle_in)

pickle_in = open("tfidf.pkl","rb")
tfidf=pickle.load(pickle_in)

st.header("Predict Ratings for Hotel Reviews")
st.subheader("Enter the review to analyze")

input_text = st.text_area("Type review here", height=50)

#option = st.sidebar.selectbox('Menu bar',['Sentiment Analysis','Keywords'])
#st.set_option('deprecation.showfileUploaderEncoding', False)
#if option == "Sentiment Analysis":
    
    
    
if st.button("Predict "):
      
       wordnet=WordNetLemmatizer()
       text=re.sub('[^A-za-z0-9]',' ',input_text)
       text=text.lower()
       text=text.split(' ')
       text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
       text = ' '.join(text)
       pickle_in = open(r"logistic.pkl", 'rb') 
       nlp = pickle.load(pickle_in)
       pickle_in = open(r"tfidf.pkl", 'rb') 
       tfidf = pickle.load(pickle_in)
       transformed_input = tfidf.transform([text])
       
       if nlp.predict(transformed_input) == 0:
           st.write("Oops! it's a Negative review 😔")
       elif    nlp.predict(transformed_input) == 1:
           st.write("it's Positive review and an happy customer😃")



st.balloons()

