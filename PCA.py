from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

import services

word_embeddings_dict = services.getWordEmbedding('GoogleNews-vectors-negative300.bin')
services.logger('Word Embeddings Loaded')

plt.figure(figsize=(10,8))

opposite_pairs = [('good', 'bad')]

for pair in opposite_pairs:
    word1 = word_embeddings_dict[pair[0]]
    word2 = word_embeddings_dict[pair[1]]
    word_embeddings = np.array([word1,word2])
    #print(word_embeddings)
    pca = PCA(n_components=2)
    word_embeddings_reduced = pca.fit_transform(word_embeddings)
    #emb1 = word_embeddings_reduced[list(word_embeddings_dict.keys()).index(pair[0])]
    #emb2 = word_embeddings_reduced[list(word_embeddings_dict.keys()).index(pair[1])]
    plt.scatter(word_embeddings_reduced[0][0], word_embeddings_reduced[0][1], marker='x', color='red', s=100, label = pair[0])
    plt.scatter(word_embeddings_reduced[1][0], word_embeddings_reduced[1][1], marker='x', color='green', s=100, label = pair[1])

plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Opposite words in plot')
plt.grid(True)
plt.show()