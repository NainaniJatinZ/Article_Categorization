
from flask import Flask,jsonify,request,make_response
from translate import LibreTranslateAPI
import joblib
import numpy as np
import time
import re
import preprocessor as p
import en_core_web_md
nlp1 = en_core_web_md.load()
app = Flask(__name__)
lt = LibreTranslateAPI("https://translate.argosopentech.com/") 
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

@app.route('/categorize', methods=['GET','POST'])
def categorize():

    if request.method == 'GET':

        return make_response('failure')

    if request.method == 'POST':

        to_translate = request.json['text']
        # dist = {'name':lt.translate(to_translate, "en", "es")}
        lang = lt.detect(to_translate)[0]['language']              # Detect Language

        if lang !="en": 
            to_translate = lt.translate(to_translate, lang, "en")

        pca_model = joblib.load('modelv4/pca_split.pkl')  

        vec = np.expand_dims(nlp1(clean(to_translate)).vector, axis=0)         # prepare vector for prediction

        x_new = pca_model.transform(vec)

        ret_json = {}

        for i in range(28):

            svm_model = joblib.load('modelv4/svm_'+str(i)+'.pkl')  # loading svm model 

            probability = round(np.amax(svm_model.predict_proba(x_new)[0])*100, 2)  

            prediction = svm_model.predict(x_new)

            del svm_model                                                   # removing model from memory

            threshold = 50                                                  # CHANGE threshold here

            if prediction == 1 and probability>threshold:      
                cat = list(category_codes.keys())[list(category_codes.values()).index(i)]
                print(list(category_codes.keys())[list(category_codes.values()).index(i)], probability)
                ret_json[cat] = probability

        return jsonify(ret_json)
        

if __name__ == '__main__':
    app.run()