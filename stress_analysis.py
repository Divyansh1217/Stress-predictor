import streamlit as st
import CLEAN as cl
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
data= pd.read_csv("Stress.csv")
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))
data["text"] = cl.clean(data["text"])
text = " ".join(i for i in data.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt.show())
data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
x=np.array(data['text'])
y=np.array(data['label'])
cv=CountVectorizer()
X=cv.fit_transform(x)
xtrain,xtext,ytrain,ytest=train_test_split(X,y,test_size=0.35,random_state=50)
from sklearn.naive_bayes import BernoulliNB
bn=BernoulliNB()
bn.fit(xtrain,ytrain)
user=st.text_input('Enter a Text:')
data=cv.transform([user]).toarray()
output=bn.predict(data)
st.text(output)



