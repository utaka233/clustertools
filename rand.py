# 各データポイントが同じクラスターに所属しているかを表現する。
import numpy as np
from sklearn.metrics import confusion_matrix

def encode_y_to_cluster(y):
    n_sample = y.shape[0]
    diff = np.triu(y.reshape(n_sample, 1) - y.reshape(1, n_sample))
    diff_1darray = diff[np.triu_indices(n = n_sample, k = 1)]
    same_or_not = np.where(diff_1darray == 0, 0, 1)
    return same_or_not

def rand_index(y_true, y_pred, adjusted = True):
    cluster_true = encode_y_to_cluster(y_true)
    cluster_pred = encode_y_to_cluster(y_pred)
    rand_matrix = confusion_matrix(y_true, y_pred)
    rand_index = np.diag(rand_matrix).sum() / rand_matrix.sum()
    if adjusted == False:
        return (rand_matrix, rand_index)
    else:
        n_cluster = np.unique(y_true).shape[0]
        n_combinations = np.sum(rand_matrix)
        marginal_row = np.sum(rand_matrix, axis = 1).reshape(n_cluster, 1)
        marginal_column = np.sum(rand_matrix, axis = 0).reshape(1, n_cluster)
        expected_values = marginal_row @ marginal_column / n_combinations
        adjusted_rand_score = (np.diag(rand_matrix).sum() - np.diag(expected_values).sum()) / (rand_matrix.sum() - np.diag(expected_values).sum())
        return (rand_matrix, rand_index, adjusted_rand_score)
