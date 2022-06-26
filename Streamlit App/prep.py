from translate import LibreTranslateAPI

lt = LibreTranslateAPI("https://translate.argosopentech.com/")

text = "Hello again, how are you?"
print(lt.detect(text)[0]['language'])