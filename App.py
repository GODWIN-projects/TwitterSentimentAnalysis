!pip install nltk
import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

working_dir = os.path.dirname(os.path.abspath(__file__))


vect = pickle.load(open(f"{working_dir}/vectorizer.pkl","rb"))
model = pickle.load(open(f"{working_dir}/random_forest.pkl","rb"))

@st.cache()

def prediction(text):
    wnl = WordNetLemmatizer()
    lemmatized_txt = re.sub("[^a-zA-Z]"," ",txt)
    lemmatized_txt = lemmatized_txt.lower()
    lemmatized_txt = lemmatized_txt.split()
    lemmatized_txt = [wnl.lemmatize(word) for word in lemmatized_txt if not word in stopwords.words("english")]
    lemmatized_txt = " ".join(lemmatized_txt)

    vectorized_txt = vect.transform([lemmatized_txt])
    res = model.predict(vectorized_txt)

    if res == 1:
        print("positive text")
    else:
        print("negative text")


def main():

    html_temp = """
            <div style = "background-color:yellow;padding:13px">
            <h1 style="color:black;text-align:center;">
            Tweet Sentiment Analysis
            </h1></div>"""
    text = st.text_area(placeholder="Enter the text from the tweet", height=250)
    result = ""

    if st.button("Predict"):
        result = prediction(text)
        st.success(f"Your tweet is mostly {result}")


if __name__ == "__main__":
    main()
