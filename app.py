#Importing the Libraries
import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import os
from sklearn.externals import joblib
import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib
import nltk




#Loading Flask and assigning the model flask runvariable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')





with open('model.pickle', 'rb') as handle:
	model = pickle.load(handle)




@app.route('/')
def home():
    return render_template('main.html')



#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict',methods=['GET','POST'])
def predict():
    res = "Fake"
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    #Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    if pred[0] == 0:
        res = "Fake"
    else:
        res = "Real"
    return render_template('main.html', prediction_text='Analyzed Result: Proned "{}"'.format(res))
    




@app.route('/profile/<name>')
def hello(name=None):
    return render_template('profile.html', name=name)

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/SpuriousTool/')
def about1():
    os.system('PLAY.vbs')
    return render_template('about.html')

if __name__ == '__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
