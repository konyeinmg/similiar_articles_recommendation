import services

word_embeddings = services.getWordEmbedding('GoogleNews-vectors-negative300.bin')
services.logger('Word Embeddings Loaded')