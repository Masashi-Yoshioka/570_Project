import numpy as np

# Covariance matrix of X
def fn_generate_cov(dim, corr):
    
    acc  = []
    for i in range(dim):
        row = np.ones((1, dim)) * corr
        row[0][i] = 1
        acc.append(row)
    
    return np.concatenate(acc, axis = 0)

