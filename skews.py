import numpy as np
import random
from sklearn.cluster import KMeans
from collections import Counter
from scipy.special import softmax
from copy import deepcopy
from sklearn.metrics import accuracy_score

import load_data


# applies no skew
# takes train data (features and labels) and some meta data
# returns array of batched indices
def apply_homogenous_data_distribution(unassigned_data_features, unassigned_data_labels, num_clients, num_classes):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels) 
    net_dataidx_map = {}
    
    N = len(unassigned_data_features)
    assert N == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    idxs = np.random.permutation(N)
    batch_idxs = np.array_split(idxs, num_clients)
    net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}
    
    return net_dataidx_map


# applies label distribution skew, i.e., manipulates the distribution over all classes at each client
# takes alpha, train data (features and labels), and some meta data
# returns array of batched indices
def apply_label_distribution_skew(alpha, unassigned_data_features, unassigned_data_labels, num_clients, num_classes, is_regression=False, min_samples_per_class=0):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels) 
    net_dataidx_map = {}
    
    N = len(unassigned_data_features)
    assert N == len(unassigned_data_labels), "features and labels must have same length at dimension 0"

    if is_regression:
        unassigned_data_labels_quantiles = np.array(load_data.assign_quantiles(unassigned_data_labels, unassigned_data_labels, num_classes))

    idx_batch = [[] for _ in range(num_clients)]
    
    #iterate all classes
    for idx_class in range(num_classes):
        
        if is_regression: 
            idx_k = np.where(unassigned_data_labels_quantiles == idx_class)[0]
        else:
            idx_k = np.where(unassigned_data_labels == idx_class)[0]

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < (N / num_clients)) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        if min_samples_per_class > 0:
            for i in range(len(proportions)):
              if i == 0 and proportions[0] == 0:
                  proportions[0] = min_samples_per_class
                  proportions[1] = proportions[1] - min_samples_per_class
              elif i == len(proportions)-1 and proportions[i] < proportions[i-1] + min_samples_per_class:
                  missing = min_samples_per_class - (proportions[i] - proportions[i-1])
                  proportions[i] = min(proportions[i] + missing, len(idx_k)-2)
              elif i > 0 and proportions[i] < proportions[i-1] + min_samples_per_class:
                  missing = min_samples_per_class - (proportions[i] - proportions[i-1])
                  proportions[i+1] = proportions[i+1] - missing
                  proportions[i] = proportions[i] + missing

        
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    
    largest_client_idx = np.argmax([len(x) for x in idx_batch])
    for j in range(num_clients):
        if len(idx_batch[j]) < 5:
            transfer = 5 - len(idx_batch[j])
            idx_batch[j].extend(idx_batch[largest_client_idx][-transfer:])
            idx_batch[largest_client_idx] = idx_batch[largest_client_idx][:-transfer]
    
    for j in range(num_clients):     
        net_dataidx_map[j] = idx_batch[j]
    
    return net_dataidx_map

# splits features + samples in accordance to prior sample-to-client assignment
# takes train data (features and labels) and prior sample-to-client assignment
# returns dict of feature vectors and labels for each client
def apply_no_attribute_skew(unassigned_data_features, unassigned_data_labels, samples_to_clients):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels)
    
    assert len(unassigned_data_features) == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    feature_dict, label_dict = {}, {}
    
    for key in samples_to_clients.keys():
        indices = samples_to_clients[key]
        feature_dict[key] = list(unassigned_data_features[indices])
        label_dict[key] = list(unassigned_data_labels[indices])
        
    return feature_dict, label_dicts
