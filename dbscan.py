from numpy.linalg import norm

def sorted_k_dist(X, k):

    n, d = X.shape[0], X.shape[1]
    k_dist = []
    
    for i in range(n):
        k_dist.append(np.sort([norm(X[i, :] - X[j, :]) for j in range(n)])[k])
    
    return np.sort(k_dist)[::-1]
