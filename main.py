# importing useful libraries
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request
app = Flask(__name__)
#Reading the dataset
news_data=pd.read_csv("C:/csv files/train.csv")
news_data.dropna(inplace=True)
#seperating the data and label column
x=news_data.drop(columns='label',axis=1)
y=news_data['label']
#Defining the instance of the porterstemmer class
port_stem=PorterStemmer()
#creating the list of all the stopwords
stopwords=['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'come', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'him', 'himself', 'his', 'how', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'like', 'many', 'me', 'more', 'most', 'my', 'myself', 'never', 'now', 'of', 'on', 'only', 'or', 'other', 'our', 'out', 'over', 'same', 'say', 'see', 'should', 'so', 'some', 'such', 'take', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your', 'yourself']
#defining function for stemming the title column
def stemming(text):
    stemmed_content=re.sub('[^a-zA-Z]',' ',text)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content
#applying the stemming funtion on the title column of the dataset
news_data['title']=news_data['title'].apply(stemming)
#seperating the data and label column
X=news_data['title'].values
Y=news_data['label'].values
# converting text data into feature vectors
Vectorizer=TfidfVectorizer()
Vectorizer.fit(X)
X=Vectorizer.transform(X)
#splitting the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)
def fake_news_det(news):
    news = [stemming(news)]
    news = Vectorizer.transform(news)
    predict = model.predict(news)
    return predict

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)