# Multilingual Article Categorization

## Details:

- Spacy en_core_web_md for embedding followed by 1 vs all SVM for categorization
- Dataset: from @AdSkate
- Present categories: books and literature, hobbies & interests, style & fashion, music and audio, events and attractions, sports, pets, personal finance, medical health, shopping, travel, home & garden, science, pop culture, fine art, news and politics, food & drink, business and finance, video gaming, technology & computing, religion & spirituality, healthy living, movies, careers, education, real estate, family and relationships, sensitive topics,television
- LibreTranslate open source binder for translation


## Links to test the app: 
 
Streamlit Interface: https://article-categorization-v3.herokuapp.com/

Flask API: https://adskate-categorization-api.herokuapp.com/categorize 

## Code Snippet for API: 
```python
import requests
api_url = 'https://adskate-categorization-api.herokuapp.com/categorize'
string_to_category = {'text': 'ARTICLE TEXT HERE'}
r = requests.request("POST", url=api_url, json=string_to_category)
print(r.content.decode('utf-8'))

```



