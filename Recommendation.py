import services

'''
data = services.readData('data.json')
#print(len(data))
#print(data[:5])

num_features = 1024
noise = 10
tfidf_vectorizer, tfidf_matrix = services.getFeatures(data, num_features, noise)
'''
word_embeddings = services.getWordEmbedding('word2vec-google-news-300.model')
print(word_embeddings.shape)