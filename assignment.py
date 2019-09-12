import numpy as np
from scipy.optimize import linear_sum_assignment
from .scorer import get_cost_matrix_calculator

'''
Class for assign label indexes to cluster indexes with Hungarian algorithm.
* We assign label indexes to maximize the supervised accuracy in labeled data.

'''
class Assigner():
    def __init__(self, scoring):
        self.scoring = scoring
    
    def fit(self, y_true, y_pred):
        # step 1. calculate a cost matrix for scoring
        cost_matrix_calculator = get_cost_matrix_calculator(self.scoring)
        cost_matrix = cost_matrix_calculator(y_true, y_pred)

        # step 2. solve linear sum assignment problem with Hungarian algorithm
        label_id, cluster_id = linear_sum_assignment(-1 * cost_matrix)    # Remark : the method sloves min-problem.

        # step 3. make cluster_id -> label_id
        self.cluster_to_label = label_id[np.argsort(cluster_id)]    # sort by label_id with accending cluster_id
        
    def transform(self, y_pred):
        # compute label 
        return self.cluster_to_label[y_pred]