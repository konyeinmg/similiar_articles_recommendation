import numpy as np

import services

def getCosineSimilarity(candidate, reference):
    dot_product = np.dot(candidate, reference)
    magnitue_candidate = np.linalg.norm(candidate)
    magnitue_reference = np.linalg.norm(reference)
    consine_similarity = dot_product / (magnitue_candidate * magnitue_reference)
    return consine_similarity

def getSimilarArticles(candidate, documents, corpus, num_similar_articles = 5):
    scores = []
    for docuemnt in documents:
        score = getCosineSimilarity(candidate, docuemnt)   
        scores += [score]
    print(scores)
    #sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1] #reverse
    #print(sorted_indices)
    similar_articles_indices = sorted_indices[:num_similar_articles]
    similar_articles = []
    for index in similar_articles_indices:
        similar_articles += [corpus[index]]
    return similar_articles
    