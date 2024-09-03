import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
from sklearn.cluster import KMeans
import sys
import statistics
import torch
import torchvision
import torchvision.transforms as transforms
import random
import h5py
import skews
from keras.preprocessing.image import img_to_array, array_to_img
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# loads dataset from files and sets config parameters accordingly
def load_raw_data(config):
    
    if config["dataset_name"].lower() == "covtype":
        
        config["num_features"] = 54
        config["num_classes"] = 2

        if config["model_name"] == "auto":
            config["model_name"] = "ann"
        
        X,y = [],[]
        with open("./data/CovType/covtype.libsvm.binary.scale") as file:
            for line in file:
                s = line.rstrip()
                s = s.split(" ")
                y.append(int(s[0])-1) 
                xi = [0.] * config["num_features"]
                for e in s[1:]:
                    e = e.split(":")
                    i = int(e[0])
                    f = float(e[1])
                    xi[i-1] = f
                X.append(xi)
        print(sum(y)/len(y))

    elif config["dataset_name"].lower() == "mnist":

        config["num_features"] = (1, 320)
        config["num_classes"] = 10

        if config["model_name"] == "auto":
            config["model_name"] = "toycnn"

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        X = trainset.data[:,None,:,:].tolist()    
        y = trainset.targets.tolist() 
        
    else:
        raise ValueError("dataset " + config["dataset_name"] + " has not been configured yet")
        
    config["X_raw"] = X
    config["y_raw"] = y

    return config



# apply skews and distribution on row data
def distribute_skewed_data(config):
    
    # train/shared/test splitting
    shared_frac = config["shared_set_fraction"]
    test_frac = config["test_set_fraction"]

    is_regression = not config["num_classes"] > 1
    num_classes = config["num_classes"] if not is_regression else config["num_quantiles"]
    
    X_train, X_shared_and_test, y_train, y_shared_and_test = train_test_split(config["X_raw"], 
                                                                              config["y_raw"], 
                                                                              test_size = test_frac + shared_frac
                                                                             )
    X_test, X_shared, y_test, y_shared = train_test_split(X_shared_and_test, 
                                                          y_shared_and_test, 
                                                          test_size = shared_frac / (shared_frac + test_frac)
                                                         )
    del config["X_raw"]
    del config["y_raw"]

    if config["global_skew"] is not None: 

            proportions = np.random.dirichlet(np.repeat(config["global_skew"], config["num_classes"]))
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            remove_data_fraction = config["remove_data_fraction"] if "remove_data_fraction" in config else .5
            overall_num_to_remove = int(remove_data_fraction * len(y_train))

            for i, class_proportion in enumerate(proportions):

                indices_with_label_i = np.where(y_train == i)[0]

                candidate_removal = int(overall_num_to_remove * class_proportion)
                num_to_remove = min(len(indices_with_label_i), candidate_removal)


                indices_to_remove = np.random.choice(indices_with_label_i, num_to_remove, replace=False)

                mask = np.ones(len(y_train), dtype=bool)
                mask[indices_to_remove] = False

                X_train = X_train[mask]
                y_train = y_train[mask]

    # apply either label or quantity skew
    if config["label_skew"] is None or config["label_skew"] == "homogeneous":
        sample_to_client_assignment = skews.apply_homogenous_data_distribution(X_train, y_train, config["num_clients"], num_classes)
    elif config["label_skew"] == "label_distribution" and config["label_alpha"] is not None:
        samples_per_class = config["min_samples_per_class"] if "min_samples_per_class" in config else 0
        sample_to_client_assignment = skews.apply_label_distribution_skew(float(config["label_alpha"]), X_train, y_train, config["num_clients"], num_classes, is_regression, min_samples_per_class=samples_per_class)
    else: 
        raise ValueError("label/quantity skew " + config["label_skew"] + " is not defined yet")
    
    # apply attribute skew
    if config["attribute_skew"] is None:
        clients_feature_dict, clients_label_dict = skews.apply_no_attribute_skew(X_train, y_train, sample_to_client_assignment)
    else: 
        raise ValueError("attribute skew " + config["attribute_skew"] + " is not defined yet")
    
    config["X_train"] = X_train
    config["y_train"] = y_train
    config["X_test"] = X_test
    config["y_test"] = y_test
    config["X_shared"] = X_shared
    config["y_shared"] = y_shared
    config["clients_feature_dict"] = clients_feature_dict
    config["clients_label_dict"] = clients_label_dict
    config["sample_to_client_assignment"] = sample_to_client_assignment
    
    return config



# measures all three types of imbalances for data distribution
# takes the config and filename to write to
# returns global label imbalance, global quantity imbalance, dict of local label imbalances, dict of cosine similarities, global feature imbalance, and dict of local feature imbalances
def measure_imbalance(config, filename=None, log=True):
    
    writer = open(filename, 'w') if filename is not None else sys.stdout
    
    local_label_imbalances, global_label_imbalance, local_label_distribution_imbalances = measure_label_imbalance(config)
    global_quantity_imbalance, local_quantity_imbalances = measure_quantity_imbalance(config)
    global_feature_imbalance, local_feature_imbalances = measure_feature_imbalance(config)

    global_cs_median = statistics.median(list(local_label_distribution_imbalances.values()))
    global_cs_stdev = statistics.stdev(list(local_label_distribution_imbalances.values()))
    local_feature_median = statistics.median(list(local_feature_imbalances.values()))
    local_feature_stdev = statistics.stdev(list(local_feature_imbalances.values()))
    
    if log:
        writer.write("\nGlobal Label imbalance "+str(global_label_imbalance))
        writer.write("\nGlobal Label Distribution imbalance Median:"+str(global_cs_median)+", Std.Dev.:"+str(global_cs_stdev))
        writer.write("\nGlobal Quantity imbalance "+str(global_quantity_imbalance))
        writer.write("\nGlobal Feature imbalance "+str(global_feature_imbalance))
        writer.write("\n   Local Feature imbalance Mean:"+str(local_feature_median) + ", Std.Dev.:" + str(local_feature_stdev))
        
    for i in range(config["num_clients"]):
        if log:
            writer.write("\n\nClient "+str(i))
            writer.write("\n  Local Label Imbalance "+str(local_label_imbalances[i]))
            writer.write("\n  Local Label Distribution Imbalance "+str(local_label_distribution_imbalances[i]))
            writer.write("\n  Local Quantity Imbalance "+str(local_quantity_imbalances[i]))
            writer.write("\n  Local Feature Imbalance "+str(local_feature_imbalances[i]))
        
    if filename is not None:
        writer.close()
        
    return global_label_imbalance, local_label_imbalances, global_quantity_imbalance, local_quantity_imbalances, (global_cs_median, global_cs_stdev), local_label_distribution_imbalances, global_feature_imbalance, local_feature_imbalances


# measures local and gloabl imbalance as well as mismatch between class distributions for each client
# takes only the config
# returns dict of local imbalances, the global imbalance and a dict of cosine similarities between class distributions
def measure_label_imbalance(config):
    
    num_quantiles = config["num_quantiles"] if "num_quantiles" in config else 4
    num_classes_analysis = config["num_classes"] if config["num_classes"] > 1 else num_quantiles

    if config["num_classes"] > 1:
        global_counter = Counter(config["y_train"])
    #regression is handled using quartiles
    else:
        quartiles = assign_quantiles(config["y_train"], config["y_train"], num_quantiles)
        global_counter = Counter(quartiles)
    
    global_imbalance = float(max(list(global_counter.values())) / min(list(global_counter.values()))) if len(list(global_counter.values())) > 1 else float("inf")
    global_distribution = [dict(global_counter)[x] if x in dict(global_counter) else 0 for x in range(num_classes_analysis)]
    
    local_imbalances, mismatches = {}, {}
    
    for key, value in config["clients_label_dict"].items():
        
        if len(value) > 0:
          if config["num_classes"] > 1:
              local_counter = Counter(value)
          #regression is handled using quartiles
          else:
              quartiles = assign_quantiles(config["y_train"], value, num_quantiles)
              local_counter = Counter(quartiles)
          
          local_imbalance = float(max(list(local_counter.values()), default=1) / min(list(local_counter.values()), default=1)) if len(list(local_counter.values())) > 1 else float(max(list(local_counter.values())))
          local_imbalances[key] = local_imbalance
          
          local_distribution = [dict(local_counter)[x] if x in dict(local_counter) else 0 for x in range(num_classes_analysis)]
          cos_sim = dot(local_distribution, global_distribution)/(norm(local_distribution)*norm(global_distribution))
          mismatches[key] = 1 - cos_sim

    return local_imbalances, global_imbalance, mismatches


# measures quantity imbalance
# takes only the config
# returns quantity imbalance
def measure_quantity_imbalance(config):
    
    clients_data_count = [len(x) for x in list(config["clients_label_dict"].values())]
    global_quantity_imbalance = float(max(clients_data_count) / min(clients_data_count)) if min(clients_data_count) > 0 else float("inf")
    N = sum(clients_data_count) / len(clients_data_count)
    local_quantity_imbalances = {key: len(value)/N for key, value in config["clients_label_dict"].items()}
    return global_quantity_imbalance, local_quantity_imbalances
                 
                 
# measures the feature imbalance using the purity metric
# takes only the config
# returns the global feature imbalance and a dict of local feature imbalances
def measure_feature_imbalance(config):
    
    N = sum([len(x) for x in list(config["clients_label_dict"].values())]) 
    previous_assignment = np.zeros(N)
    for client_idx, val in config["sample_to_client_assignment"].items():
        for feature_idx in val:
            previous_assignment[feature_idx] = client_idx

    x_k_means = np.array(config["X_train"]).reshape((len(config["X_train"]), -1))

    # compute initial centroids
    centroids = []
    for i in range(config["num_clients"]):

        indices = np.where(previous_assignment == i)
        vals = np.array(x_k_means)[indices]
        centroid = np.mean(vals, axis=0)
        centroids.append(centroid)

    # apply k-means
    kmeans = KMeans(n_clusters=config["num_clients"], init=np.array(centroids), max_iter=100).fit(x_k_means)
    assignments = np.array(kmeans.labels_)
    assert N == len(assignments), "mismatching lengths of X_train and all clients"

    num_true_assignments = 0
    per_cluster_purity = {}
  
    
    # compute purity
    for i in range(config["num_clients"]):
        indices = np.where(assignments == i)
        vals = previous_assignment[indices]
        cluster_label = max(set(list(vals)), key=list(vals).count)  
        true_assignments = sum(1 for j in vals if j == cluster_label)
        num_true_assignments += true_assignments
        per_cluster_purity[i] = true_assignments / len(vals)
    
    purity = num_true_assignments / N
                 
    return purity, per_cluster_purity


# takes a population which is to be devided in quantiles and a list of values
# assignes quantiles to values in accordance with data distribution in population
# returns a list of quantile indices
def assign_quantiles(population, values, q = 4):
    quantiles = []
    for i in range(q+1):
        quantile = np.quantile(population, i/q)
        quantiles.append(quantile)
    quantiles[0] = min(population)-1
    quantiles[-1] = max(population)+1
    ret = []
    for value in values:
        for i, quantile in enumerate(quantiles[:-1]):
            quantile_plus = quantiles[i+1]
            if float(value) >= float(quantile) and float(value) < float(quantile_plus):
                ret.append(i)
                break
    return ret