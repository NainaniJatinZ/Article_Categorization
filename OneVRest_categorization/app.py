# Importing Libraries
from translate import LibreTranslateAPI
import pandas as pd
import streamlit as st
import joblib
import numpy as np
import time
import re
import preprocessor as p
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Loading Sentence Embedding Model 
import en_core_web_md
nlp1 = en_core_web_md.load()

# Dictionary for mapping SVM Output with category name
category_codes = {                                                                                              
'books and literature':0, 'hobbies & interests':1, 'style & fashion':2,
       'music and audio':3, 'events and attractions':4, 'sports':5, 'pets':6,
       'personal finance':7, 'medical health':8, 'shopping':9, 'travel':10,
       'home & garden':10, 'science':11, 'pop culture':12, 'fine art':13,
       'news and politics':14, 'food & drink':15, 'business and finance':16,
       'video gaming':17, 'technology & computing':18,
       'religion & spirituality':19, 'healthy living':20, 'movies':21, 'careers':22,
       'education':23, 'real estate':24, 'family and relationships':25, 'sensitive topics':26, 'television':27}

p.set_options(p.OPT.URL, p.OPT.MENTION)
def clean(text):
    text = p.clean(text)                    # removes links and mentions

    text = re.sub(r'\W+', ' ', text)        # remove non-alphanumeric characters
    
    text = re.sub(r"\d+", "number", text)   # replace numbers with the word 'number'
    
    if len(text.strip().split()) < 3:       # don't consider sentenced with less than 3 words (i.e. assumed noise)
        return None
    text = text.lower()  # lower case everything
    
    return text.strip() # remove redundant spaces


st.title("One V Rest Categorization v1")

text1 = st.text_area("Article", height=100)         # Input

lt = LibreTranslateAPI("https://translate.argosopentech.com/")      #Translation API 


if st.button("Submit"):

    start_time = time.time()                

    lang = lt.detect(text1)[0]['language']              # Detect Language

    st.write("Detected Language is:", lang)

    if lang !="en": 
        text1 = lt.translate(text1, lang, "en")
        st.text("")
        st.text_area("Translated to English: ", value=text1)        #if Language is not english, show the translated version

    pca_model = joblib.load('data/modelv4/pca_split.pkl')           # load PCA model

    vec = np.expand_dims(nlp1(clean(text1)).vector, axis=0)         # prepare vector for prediction

    x_new = pca_model.transform(vec)

    st.write("")
    st.write("Predicted Categories and Confidence (Out of 100):")
    st.write("")

    for i in range(28):

        svm_model = joblib.load('data/modelv4/svm_'+str(i)+'.pkl')  # loading svm model 

        probability = round(np.amax(svm_model.predict_proba(x_new)[0])*100, 2)  

        prediction = svm_model.predict(x_new)

        del svm_model                                                   # removing model from memory

        threshold = 70                                                  # CHANGE threshold here

        if prediction == 1 and probability>threshold:      

            print(list(category_codes.keys())[list(category_codes.values()).index(i)], probability)
            st.write(list(category_codes.keys())[list(category_codes.values()).index(i)], probability)

    st.write("")
    st.write("Inference time: --- %s seconds ---" % (time.time() - start_time))  

else:
    st.write('Enter text in the box above and hit submit')

st.title("Details")
st.write("Spacy en_core_web_md for embedding followed by 1 vs all SVM for categorization")
st.write("Dataset: from @AdSkate")
st.text_area("Possible Categories ", value="Present categories: books and literature, hobbies & interests, style & fashion, music and audio, events and attractions, sports, pets, personal finance, medical health, shopping, travel, home & garden, science, pop culture, fine art, news and politics, food & drink, business and finance, video gaming, technology & computing, religion & spirituality, healthy living, movies, careers, education, real estate, family and relationships, sensitive topics,television")
st.write("LibreTranslate open source binder for translation")