from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

import services

word_embeddings_dict = services.getWordEmbedding('GoogleNews-vectors-negative300.bin')
services.logger('Word Embeddings Loaded')

plt.figure(figsize=(5,4))

opposite_pairs = [('good', 'bad'), ('king', 'queen'), ('hot', 'cold')]

for pair in opposite_pairs:
    word1 = word_embeddings_dict[pair[0]]
    word2 = word_embeddings_dict[pair[1]]
    word_embeddings = np.array([word1,word2])
    #print(word_embeddings)
    pca = PCA(n_components=2)
    word_embeddings_reduced = pca.fit_transform(word_embeddings)
    emb1 = word_embeddings_reduced[0]
    emb2 = word_embeddings_reduced[1]
    plt.scatter(emb1[0],emb1[1], marker='x', color='red', s=100)
    plt.scatter(emb2[0],emb2[1], marker='x', color='green', s=100)
    plt.text(emb1[0],emb1[1], pair[0], fontsize=8, ha='right', va='bottom', color='blue')
    plt.text(emb2[0],emb2[1], pair[1], fontsize=8, ha='right', va='bottom', color='blue')

plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Opposite words in plot')
plt.grid(True)
plt.show()