import numpy as np

import services
import Similarlity
from LSH import LSH

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
ind2doc = {}
for i in range(len(train)):
    document_embedding = services.getDocumentEmbeddings(train[i], i, word_embeddings, tfidf_matrix, tfidf_vectorizer)
    ind2doc[i] = document_embedding
    document_embeddings.append(document_embedding)

document_embeddings = np.vstack(document_embeddings)
#print('Document embeddings shape : ',document_embeddings.shape)
#print('Document embeddings dict len : ',len(ind2doc))
services.logger('Document Embeddings Calculated')


candidate = test[201] #change id of document u want to calculate
services.logger(candidate)
test_tfidf = tfidf_vectorizer.transform([candidate])
test_document_embedding = services.getDocumentEmbeddings(candidate, 0, word_embeddings, test_tfidf, tfidf_vectorizer)

services.logger('Similarity Scores Calculating....')
services.logger('Similar Articles.........')
similar_articles, indices = Similarlity.getSimilarArticles(test_document_embedding, document_embeddings, train)
print('Similar articles indices by pure cosine similarity : ', indices) #you can print similar articles

#LSH
lsh = LSH()
lsh.make_hash_table(document_embeddings)
lsh_similar_articles, lsh_indices = lsh.similar_articles(test_document_embedding, train) #you can print similar articles
print('Similar articles indices by LSH : ', lsh_indices)
