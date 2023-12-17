class LSH:

    def __init__(self, N_PLANES = 8, N_DIMS = 300):
        self.N_PLANES = N_PLANES #for 213 buckets, 16 vecors in each bucket
        self.N_DIMS = N_DIMS #word embedding dimension
        self.planes = np.random.normal(size = (self.N_DIMS, self.N_PLANES))