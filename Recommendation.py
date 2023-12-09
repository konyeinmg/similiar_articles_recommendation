import services

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
