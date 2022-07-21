from translate import LibreTranslateAPI
import pandas as pd
import joblib
import numpy as np
import time
import re
import preprocessor as p
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import en_core_web_md
nlp1 = en_core_web_md.load()

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
    text = p.clean(text)
    text = re.sub(r'\W+', ' ', text)  # remove non-alphanumeric characters
    # replace numbers with the word 'number'
    text = re.sub(r"\d+", "number", text)
    # don't consider sentenced with less than 3 words (i.e. assumed noise)
    if len(text.strip().split()) < 3:
        return None
    text = text.lower()  # lower case everything
    
    return text.strip() # remove redundant spaces


text1 = """Adriana Lastra dimite como vicesecretaria general del PSOE y facilita a Sánchez reordenar el partido
La política, que está embarazada, alega que necesita reposo y cuidados y que ya se encuentra de baja laboral. El presidente aprovechará para hacer cambios ante las elecciones autonómicas Adriana Lastra ha dimitido como vicesecretaria general del PSOE, según ha informado ella misma en la mañana de este lunes a través de una nota de prensa oficial. La política, que tiene 43 años y está embarazada, explica en ese comunicado que su dimisión se debe a los cambios que se han producido en su vida personal y que le exigen “tranquilidad y reposo”, motivo por el cual ha estado de baja laboral en las últimas dos semanas y ni siquiera acudió al Congreso durante el reciente debate sobre el estado de la nación. Lastra le trasladó esa decisión al presidente del Gobierno, Pedro Sánchez, el fin de semana. Su renuncia obliga a Sánchez a reordenar y reforzar la estructura del PSOE pensando sobre todo en las próximas elecciones autonómicas y municipales de mayo de 2023. Esta remodelación coincide con un momento, tras el varapalo sufrido en las recientes elecciones andaluzas, en el que se achaca al Ejecutivo, al PSOE y a sus principales portavocías luchas de poder internas, problemas de coordinación y fallos de comunicación. En el PSOE no se descartan posibles nuevos cambios o relevos, pero ya para después del verano. La política asturiana asegura en ese texto enviado por el PSOE de que informó hace días a Pedro Sánchez de su inminente dimisión, lo que ahora obliga al máximo dirigente socialista a reestructurar la cadena de mando en el partido, aunque sobre ese asunto el comunicado no detalla nada. Sánchez ha tardado muy pocos minutos en reaccionar y ha publicado un tuit en su cuenta oficial en el que califica a Lastra de “socialista ejemplar”, le agradece su compromiso y entrega estos años y acaba: “Seguiremos trabajando juntos”.


Ministras y varios dirigentes socialistas han salido también al paso de ese anuncio para enmarcarlo en claves solo personales, aunque en el partido y el Gobierno hace meses que se había comprobado que la nueva estructura montada entre el Ejecutivo, la sede del PSOE en Ferraz y los portavoces parlamentarios no funcionaba bien y necesitaba importantes ajustes. Lastra, además, quedó muy marcada negativamente en la noche electoral de las elecciones en Andalucía, que entregaron la mayoría absoluta al popular Juan Manuel Moreno. La vicesecretaria general compareció en Ferraz, y tras no felicitar al ganador, arremetió contra el PP de Alberto Núñez Feijóo: “Estas elecciones son la última parada de un camino diseñado por la anterior dirección y ejecutada por el señor Feijóo”. Y luego minusvaloró ese triunfo, al situarlo junto a los de Galicia y Castilla y León como elecciones en territorios favorables al nuevo presidente popular “para consolidar su propia imagen”.

El ministro del Interior, Fernando Grande-Marlaska, ha comentado esta mañana, durante un receso en los cursos de verano de El Escorial: “Mi reconocimiento al trabajo extraordinario de la vicesecretaria general, también como portavoz del grupo parlamentario socialista”. Y ha definido ese trabajo de “magnífico y valiente en momentos no siempre sencillos” para poner en valor “su compromiso con el PSOE y las políticas progresistas”. También han ido en la misma línea las reacciones de la ministra portavoz, Isabel Rodríguez, o de la titular de Hacienda, María Jesús Montero.

Hace justo un año, el líder socialista ya ejecutó una crisis en el Gabinete y también en el partido, tras el fracaso de las elecciones del 4 de mayo en la Comunidad de Madrid, que impulsaron más el fenómeno electoral de la popular Isabel Díaz Ayuso. En aquella remodelación, Sánchez aprovechó para relevar al entonces secretario de Organización del PSOE y ministro de Transportes, José Luis Ábalos, además de a su jefe de gabinete, Iván Redondo, y a la vicepresidenta primera, Carmen Calvo. Fue entonces cuando Sánchez nominó a Lastra como vicesecretaria general del PSOE, aunque le quitó el muy mediático cargo de portavoz en el Congreso, y dejó como secretario de Organización a Santos Cerdán, el número tres del partido, y con el que nunca ha llegado a conectar bien."""

start_time1 = time.time()
lt = LibreTranslateAPI("https://translate.argosopentech.com/")



lang = lt.detect(text1)[0]['language']
print("Detected Lang is: ", lang)
translated = lt.translate(text1, lang, "en")
print("Translated to Eng: ", translated)
print("translation time: --- %s seconds ---" % (time.time() - start_time1))  
start_time = time.time()
pca_model = joblib.load('data/modelv4/pca_split.pkl')
vec = np.expand_dims(nlp1(clean(translated)).vector, axis=0)
print(vec.shape)
x_new = pca_model.transform(vec)
print(x_new.shape)
preds = []

print("Predicted Categories and Confidence (Out of 100):")
for i in range(28):
    svm_model = joblib.load('data/modelv4/svm_'+str(i)+'.pkl')
    probability = round(np.amax(svm_model.predict_proba(x_new)[0])*100, 2)
    prediction = svm_model.predict(x_new)
    del svm_model
    threshold = 70                                                  # CHANGE threshold here

    if prediction == 1 and probability>threshold:      
        print(list(category_codes.keys())[list(category_codes.values()).index(i)], probability)
print("load and predicting: --- %s seconds ---" % (time.time() - start_time))  


