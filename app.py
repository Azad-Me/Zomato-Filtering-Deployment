# -*- coding: utf-8 -*-
"""
Created on Sun May 28 23:05:23 2023

@author: Azad
"""


import streamlit as st
import pandas

import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import wordnet
import contractions
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import pickle
data = pickle.load(open('data.pickle', 'rb'))
similarities = pickle.load(open('similarity.pickle', 'rb'))
hotel = pickle.load(open('hotel.pickle', 'rb'))

tfidf = pickle.load(open('tfidf2.pkl','rb'))
model = pickle.load(open('model_new.pickle','rb'))

def natural_language_processing(text):

    # Fix contractions
    first = contractions.fix(text)

    # Remove URLs
    without_urls = re.sub(r'http\S+|www\S+', '', first)

    # Convert to lowercase
    second = without_urls.lower()

    # Remove punctuation
    punct = set(string.punctuation)
    third = ''.join([i for i in second if i not in punct])

    # Remove white spaces
    fourth = re.sub('\s+', ' ', third).strip()

    # Remove stopwords
    stop = set(stopwords.words('english'))
    fifth = ' '.join([i for i in fourth.split() if i not in stop])

    # Tokenize words
    word_tokens = word_tokenize(fifth)

    # Replace words with synonyms
    rephrased_sentence = []
    for word in word_tokens:
        synonyms = wordnet.synsets(word)
        if len(synonyms) > 0:
            wor = synonyms[0].lemmas()[0].name()
        else:
            wor = word
        rephrased_sentence.append(wor)

    # Join rephrased words
    sixth = ' '.join(rephrased_sentence)

    # Tokenize words again
    seventh = word_tokenize(sixth)

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    eighth = ' '.join([lemmatizer.lemmatize(i) for i in seventh])

    return eighth


def processed_text(text):
    #first = natural_language_processing(text)
    #vectors = tfidf.transform([first]).toarray()
    #result = model.predict(vectors)
    
    url = "https://openai80.p.rapidapi.com/completions"

    payload = {
        	"model": "text-davinci-003",
        	"prompt": f"do sentiment analysis for the text{ text }",
        	"max_tokens": 256,
        	"temperature": 0.7,
        	"top_p": 1,
        	"n": 1,
        	"stream": False,
        	"logprobs": None,
        	"stop": ""
        }
    headers = {
        	"content-type": "application/json",
        	"X-RapidAPI-Key": "8a00eb7337mshf8a2d52e86cf87cp12435fjsnc093d6aa0f8f",
        	"X-RapidAPI-Host": "openai80.p.rapidapi.com"
        }
        
    response = requests.post(url, json=payload, headers=headers)

   if 'positive' in  response.json()['choices'][0]['text']:
      return ('The Review is Good, You can prefer this Hotel.')
   else:
      return ('The Review is Not so Good, there are plenty other options.')



def hotel_recommendation(hotel_name):

  print('Hey!! Gyus, Checkout these Amazing Places')
  print(' ')
  index = hotel[hotel['Name']== hotel_name].index[0]
  similar_content = sorted(list(enumerate(similarities[index])),reverse = True , key = lambda x: x[-1])

  recommend_hotels =[]
  cuisines=[]
  review = []
  rating =[]
  for i,j in similar_content[1:6]:
    recommend_hotels.append(hotel['Name'][i])
    cuisines.append(data[data['Name']== hotel['Name'][i] ]['Cuisines'].iloc[0])
    review.append(data[data['Name']== hotel['Name'][i] ]['Review'].iloc[0])
    rating.append(data[data['Name']== hotel['Name'][i] ]['Rating'].iloc[0])
    
    
  return (recommend_hotels,cuisines,review,rating)



tab1, tab2= st.tabs(["Recommendation Engine", "Sentiment Analysis"])
with tab1:
    st.title('Zomato Hotel Recommendation System')
    option = st.selectbox(
        'The Engine is restricted to 105 Hotels',
        data['Name'].unique())
    
    
    if st.button('Recommend'):
        recommendation , cuisines , rating, review = hotel_recommendation(option)
        for i,j,k,l in zip(recommendation , cuisines , rating, review) :
            st.write(i)
            
            st.caption(j)
            st.caption(k)
            st.caption(l)

with tab2:
        
    st.title('Zomato Hotel Review Sentiment Analysis')
    txt = st.text_area(''' ''')
    if st.button('Should I, Go Here??'):
        
        st.header(processed_text(txt))
        st.caption('Please exercise caution when interpreting the results of this sentiment analysis\
                   model, taking into account its inherent limitations. Keep in mind that factors such \
                       as bias, subjectivity, the possibility of false positives and negatives,\
                           generalization challenges, and the need for ongoing adaptation can impact\
                               the outcomes. Remember, this model serves as a valuable tool for \
                                   understanding sentiment, but it is essential to complement it with\
                                       human judgment and critical analysis for accurate interpretation\
                                           and informed decision-making.')

st.snow()
