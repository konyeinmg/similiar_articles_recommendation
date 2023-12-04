import services

'''
data = services.readData('data.json')
#print(len(data))
#print(data[:5])

num_features = 1024
noise = 10
tfidf_vectorizer, tfidf_matrix = services.getFeatures(data, num_features, noise)
'''
word_embeddings = services.getWordEmbedding('GoogleNews-vectors-negative300.bin')
#print(word_embeddings.shape())