import numpy as np


def count_matrix_calculator(y_true, y_pred):
    # step 1. check a number of labels
    n_labels = np.unique(y_true).shape[0]
    n_clusters = np.unique(y_pred).shape[0]
    size = np.maximum(n_labels, n_clusters)

    # step 2. count combinations of (y_true, y_pred)
    results = np.array([y_true, y_pred]).T    # make [y_true, y_pred] for each datapoints
    indexes, counts = np.unique(results, axis = 0, return_counts = True)

    # step 3. make a cost matrix
    count_matrix = np.zeros((size, size))
    for i in np.arange(counts.size):
        index = tuple(indexes[i])
        count = counts[i]
        count_matrix[index] = count
    
    # step 4. output
    return count_matrix


def accuracy_matrix_calculator(y_true, y_pred):
    # step 1. make a count matrix
    count_matrix = count_matrix_calculator(y_true, y_pred)
    
    # step 2. calculate accuracies
    accuracy_matrix = count_matrix / count_matrix.sum()
    
    # step 3. output
    return accuracy_matrix


def macro_f1_matrix_calculator(y_true, y_pred):
    # step 1. make a count matrix
    count_matrix = accuracy_matrix_calculator(y_true, y_pred)
    
    # step 2. calculate precision and recall
    epsilon = np.finfo(float).eps
    precision = count_matrix / (count_matrix.sum(axis = 0) + epsilon)
    recall = count_matrix / (count_matrix.sum(axis = 1) + epsilon)
    macro_f1_matrix = (2 * precision * recall) / (precision + recall + epsilon)
    
    # step 3. output
    return macro_f1_matrix


SCORERS = dict(accuracy = accuracy_matrix_calculator,
               macro_f1 = macro_f1_matrix_calculator)


def get_cost_matrix_calculator(scoring):
    try:
        cost_matrix_calculator = SCORERS[scoring]
    except KeyError:
        raise ValueError("%r is not a valid scoring value. "
                         "Use clustertools.scorer.SCORERS.keys() "
                         "and check your scoring including the dictionary keys." % (scoring))
        
    return cost_matrix_calculator


def maximum_jaccard_calculator(count_matrix):
    n_rows = count_matrix.shape[0]
    n_cols = count_matrix.shape[1]
    maximum_jaccard = np.zeros(n_rows)
    
    record_jaccard = np.zeros(n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
             record_jaccard[j] = count_matrix[i, j] / \
                (count_matrix.sum(axis = 1)[i] + count_matrix.sum(axis = 0)[j] - count_matrix[i, j])
        maximum_jaccard[i] = np.max(record_jaccard)
        record_jaccard = np.zeros(n_cols)
    
    return maximum_jaccard