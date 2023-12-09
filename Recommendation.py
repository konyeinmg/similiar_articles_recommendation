import services

data = services.readData('data.json')
services.logger('Data Read')
#print(len(data))
#print(data[:5])

num_features = 1024
noise = 10
tfidf_vectorizer, tfidf_matrix = services.getFeatures(data, num_features, noise)
services.logger('Feature Extracted')

word_embeddings = services.getWordEmbedding('GoogleNews-vectors-negative300.bin')
services.logger('Word Embeddings Loaded')
#print(word_embeddings['apple'])