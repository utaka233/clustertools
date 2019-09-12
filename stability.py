import numpy as np
from clustertools.scorer import count_matrix_calculator, maximum_jaccard_calculator


def bootstrap_stability_index(model, X, n_bootstrap):
    # step 1. make base clustering model
    base = model
    base.fit(X)
    cluster_base = base.predict(X)
    
    # step 2. count a number of clusters
    n_clusters = np.unique(cluster_base).shape[0]
    
    # step 3. bootstraping
    size = X.shape[0]
    base_index = np.arange(size)
    
    maximum_jaccard = np.zeros((n_bootstrap, n_clusters)) 
    
    for i in range(n_bootstrap):
        bootstrap_index = np.random.choice(base_index, size = size)
        X_bootstrap = X[bootstrap_index]
        
        new = model
        new.fit(X_bootstrap)
        
        comparison_index = np.unique(bootstrap_index)
        cluster_base_comparison = cluster_base[comparison_index]
        cluster_new_comparison = new.predict(X[comparison_index, :])
        count_matrix = count_matrix_calculator(cluster_base_comparison, cluster_new_comparison)
        
        maximum_jaccard[i] = maximum_jaccard_calculator(count_matrix)
    
    mean_maximum_jaccard = maximum_jaccard.mean(axis = 0)
    dissolved_counter = np.where(maximum_jaccard <= 0.5, 1, 0).sum(axis = 0) / n_bootstrap

    return mean_maximum_jaccard, dissolved_counter