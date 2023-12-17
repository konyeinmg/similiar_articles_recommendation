import numpy as np

import Similarlity

class LSH:

    def __init__(self, N_PLANES = 8, N_DIMS = 300):
        self.N_PLANES = N_PLANES #for 213 buckets, 16 vecors in each bucket
        self.N_DIMS = N_DIMS #word embedding dimension
        self.planes = np.random.normal(size = (self.N_DIMS, self.N_PLANES))
    
    def hash_value_of_vector(v):
        dot_product = np.dot(v,self.planes)

        sign_of_dot_product = np.sign(dot_product)

        h = sign_of_dot_product >= 0
        h = np.squeeze(h)

        hash_value = 0
        for i in range(self.N_PLANES):
            hash_value += np.power(2,i) * h[i]
        hash_value = int(hash_value)
        return hash_value
    
    def make_hash_table(vecs):
        num_buckets = 2 ** self.N_PLANES

        self.hash_table = {i:[] for i in range(num_buckets)}
        self.id_table = {i:[] for i in range(num_buckets)}

        for i,v in enumerate(vecs):
            h = hash_value_of_vector(v)
            
            self.hash_table[h].append(v)
            self.id_table[h].append(i)
        
    def similar_articles(vec,corpus, num_of_articles = 5):
        h = hash_value_of_vector(vec)
        documents_to_consider = self.hash_table[h]
        docuemnt_ids_to_consider = self.id_table[h]

        scores = {}
        for doc_id,document in zip(docuemnt_ids_to_consider, documents_to_consider):
            score = Similarlity.getCosineSimilarity(vec, document)
            scores[doc_id] = score