#!/usr/bin/env python
#-*- coding: utf-8 -*- 

from flask import Flask, jsonify, render_template, request, redirect, url_for, jsonify


import pandas as pd
import numpy as np
import pickle
import nltk
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#from sklearn.externals import joblib


import flask
app = Flask(__name__)





#please use below code to load glove vectors 

with open('/home/lohit/Desktop/safescity/app/models/glove_vectors', 'rb') as f:
    glove_model = pickle.load(f)
    glove_words =  set(glove_model.keys())









def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)
   

    return phrase

def preprocess(text):
    # convert all the text into lower letters
    # use this function to remove the contractions: https://gist.github.com/anandborad/d410a49a493b56dace4f814ab5325bbd
    # remove all the spacial characters: except space ' '
    text = str(text).lower()
    text = decontractions(text)
    text = re.sub('[^A-Za-z ]+', ' ', text)
    
    return text









nltk.download('stopwords')


stop = stopwords.words('english')





# Use English stemmer.
stemmer = SnowballStemmer("english")



#!pip install autocorrect
from autocorrect import Speller
from nltk.util import ngrams
import collections

spell = Speller(lang='en')




test='was walking along crowded street  holding sums hand  when an elderly man grouped butt  i turned to look at h m and he looked away  and did it again after a while i was   yrs old then' 

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template("index.html")



from nltk.stem import PorterStemmer
ps = PorterStemmer()

def preprocessing_flask(x):
  x = spell(x)
  x = preprocess(x)
  
 
  x = x.split()
  
  x = [item for item in x if item not in stop]
  x = preprocess(x).split()
 
  x = [ps.stem(y) for y in x] 
  
  x = ' '.join(x)
  x.replace('\d+', '')


  # tfidf_w2v
  #load TFIDF
  tfidf_model_test=joblib.load('/home/lohit/Desktop/safescity/app/models/tfidf_model.pkl')
  # we are converting a dictionary with word as a key, and the idf as a value
  dictionary = dict(zip(tfidf_model_test.get_feature_names(), list(tfidf_model_test.idf_)))
  tfidf_words = set(tfidf_model_test.get_feature_names())



  tfidf_w2v_vectors_tr_test = []; # the avg-w2v for each sentence/review is stored in this list
  print(x)

  # for each review/sentence
  vector = np.zeros(300) # as word vectors are of zero length
  tf_idf_weight =0; # num of words with a valid vector in the sentence/review

  for word in x.split(): # for each word in a review/sentence
      if (word in glove_words) and (word in tfidf_words):
        vec = glove_model[word] # getting the vector for each word
          # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
        tf_idf = dictionary[word]*(x.count(word)/len(x.split())) # getting the tfidf value for each word
        vector += (vec * tf_idf) # calculating tfidf weighted w2v
        tf_idf_weight += tf_idf
  if tf_idf_weight != 0:
      vector /= tf_idf_weight
  tfidf_w2v_vectors_tr_test.append(vector)

  print(len(tfidf_w2v_vectors_tr_test))
  print(len(tfidf_w2v_vectors_tr_test[0]))









  # Bigram
  zero_bi_test=[]
  count=0

  
  n = 2
  sentence = x
  unigrams = ngrams(sentence.split(), n)
  tri=[]
  with open("/home/lohit/Desktop/safescity/app/models/final_de_bi.txt", "rb") as fp:   # Unpickling
   final_de = pickle.load(fp)


  di = dict(zip(list(final_de),np.zeros(len(final_de))))

  try:

    for item in unigrams:
    
      tri.append(item[0]+'_'+item[1])

    common= set(tri) & final_de 
  
    for k in common:
      di[k]=1
      
    zero_bi_test.append(di)
  except:
    zero_bi_test.append(di)   



  print(zero_bi_test[0].values())    



  # Tri gram
  with open("/home/lohit/Desktop/safescity/app/models/final_de_tri.txt", "rb") as fp:   # Unpickling
   final_de_tri = pickle.load(fp)

  zero_tri_test=[]
  n = 3
  sentence = x
  unigrams = ngrams(sentence.split(), n)
  tri=[]
  

  
  di=dict(zip(list(final_de_tri),np.zeros(len(final_de_tri))))

  try:
    
    for item in unigrams:
      tri.append(item[0]+'_'+item[1]+'_'+item[2])

    common= set(tri) & final_de_tri 
   

    di=dict(zip(list(final_de_tri),np.zeros(len(final_de_tri))))
    for k in common:
      di[k]=1
    
    zero_tri_test.append(di)
  
  except:

    zero_tri_test.append(di)

  
  # convert bigram into arrays
  z_bi_value_test=[]
  for i in zero_bi_test:
    z_bi_value_test.append(list(i.values()))

  z_bi_value_test=np.array(z_bi_value_test)




  # convert tri gram into arrays
  z_tri_value_test=[]
  for i in zero_tri_test:
    z_tri_value_test.append(list(i.values()))


  z_tri_value_test=np.array(z_tri_value_test)




  # concat all vectors for the given text
  zero_bi_tri_test=pd.DataFrame(np.concatenate((z_bi_value_test,z_tri_value_test),axis=1))

  zero_bi_tri_tf_w2v=np.concatenate((zero_bi_tri_test,tfidf_w2v_vectors_tr_test),axis=1)


  # SGD model

  commenting=joblib.load('/home/lohit/Desktop/safescity/app/models/commenting.pkl')
  staring=joblib.load('/home/lohit/Desktop/safescity/app/models/staring.pkl')
  touching=joblib.load('/home/lohit/Desktop/safescity/app/models/touching.pkl')

  pred_commenting = commenting.predict(zero_bi_tri_tf_w2v)
  pred_staring = staring.predict(zero_bi_tri_tf_w2v)
  pred_touching = touching.predict(zero_bi_tri_tf_w2v)





  if pred_commenting[0]:
        prediction = "Commenting"

  elif pred_staring[0]:
        prediction = "Staring"

  elif pred_touching[0]:
        prediction = "Touching"

  else :
        prediction="Not able to predict"
	
  return prediction



@app.route("/predict", methods=["POST"])
def predict():
	print("Entered")
	to_predict_list = request.form["review_text"]
	print("Test", to_predict_list)
	result = preprocessing_flask(to_predict_list)
	print(result)
	return render_template("index.html", key = "Prediction: ", results = result)


#preprocessing_flask(test)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089)


