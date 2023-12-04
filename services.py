import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer

def readData(url):
    f = open(url)
    data = json.load(f)
   
    contents = []
    #only get contents
    for item in data:
        contents.append(item.get('content'))

    return contents

def getFeatures(data, num_features = 128, noise = 10, log = False):
    tfidf_vectorizer = TfidfVectorizer(max_features=num_features, min_df=noise ,stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    if log:
        features = tfidf_vectorizer.get_feature_names_out()
        tf = pd.DataFrame(tfidf_matrix.toarray(), columns = features)
        print(tf)
    return tfidf_vectorizer,tfidf_matrix

#return number of words
def numberOfWords(data):
    words = set()
    for item in data:
        for word in item.split():
            words.add(word)
    return len(words)

#return a dictionary of each word with its occurence in the corpus
def wordCount(data):
    words = {}
    for item in data:
        for word in item.split():
            words[word] = words.get('word',0) + 1
    return words

def getWordEmbedding(embedding):
    #print(embedding)
    model = Word2Vec.load(embedding)
    return model