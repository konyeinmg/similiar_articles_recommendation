import numpy as np

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