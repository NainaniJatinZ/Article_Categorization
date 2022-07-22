
import requests
api_url = 'https://adskate-categorization-api.herokuapp.com/categorize'
string_to_category = {'text': 'ARTICLE TEXT HERE'}
r = requests.request("POST", url=api_url, json=string_to_category)
print(r.content.decode('utf-8'))   