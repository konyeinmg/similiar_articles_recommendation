import numpy as np

import services
import Similarlity

data = services.readData('data.json')
services.logger('Data Read')
#print(len(data))
#print(data[:5])

train_percent = 0.9
train, test = services.train_test_split(data, train_percent)
#print('Train data len : ', len(train))
#print('Test data len : ', len(test))
services.logger('Train Test Split')

num_features = 1024
noise = 10
tfidf_vectorizer, tfidf_matrix = services.getFeatures(train, num_features, noise)
services.logger('Feature Extracted')


word_embeddings = services.getWordEmbedding('GoogleNews-vectors-negative300.bin')
services.logger('Word Embeddings Loaded')
#print(word_embeddings['apple'])

document_embeddings = []
for i in range(len(train)):
    document_embeddings += [services.getDocumentEmbeddings(train[i], i, word_embeddings, tfidf_matrix, tfidf_vectorizer)]
document_embeddings = np.array(document_embeddings)
services.logger('Document Embeddings Calculated')

candidate = test[10]
services.logger(candidate)
test_tfidf = tfidf_vectorizer.transform([candidate])
test_document_embedding = services.getDocumentEmbeddings(candidate, 0, word_embeddings, test_tfidf, tfidf_vectorizer)

services.logger('Similarity Scores Calculating....')
services.logger('Similar Articles.........')
similar_articles = Similarlity.getSimilarArticles(test_document_embedding, document_embeddings, train)
for article in similar_articles:
    services.logger(article)