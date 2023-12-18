from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

import services

word_embeddings_dict = services.getWordEmbedding('GoogleNews-vectors-negative300.bin')
services.logger('Word Embeddings Loaded')

embeddings = list(word_embeddings_dict.values())

word_embeddings = np.array(embeddings)

pca = PCA(n_components=2)
word_embeddings_reduced = pca.fit_transform(word_embeddings)