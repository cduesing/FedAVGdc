import models
import model_utils
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error
import numpy as np
import sys
import random
from scipy import spatial
import itertools


# used for inherence of different methods
class FedBlueprint:
    
    # init the learning strategy
    def __init__(self, config):
        self.config = config
        self.central_model = models.get_model_by_name(config)
        self.is_bert = config["model_name"].lower() == "bert"
    
    # runs training and evaluation procedure on provided data and config
    def run(self, config, filename=None, log_per_round=False, return_f1s=False):
        
        self.config = config
        best_central_model = None
        
        writer = open(filename, 'w') if filename is not None else sys.stdout
        accuracies, precisions, recalls, f1s, f12s, all_distributions = [],[],[],[],[], []
        
        for i in tqdm(range(config["rounds"])):
            self.train_round()
            torch.cuda.empty_cache()

            if "2nd_evaluation_averaging" in self.config:
              acc, pre, rec, f1, f12, all_predicitions = self.evaluate()
            else:
              f12 = 0
              acc, pre, rec, f1, all_predicitions = self.evaluate()

            if best_central_model is None:
                best_central_model = deepcopy(self.central_model.cpu())
            elif f1 > f1s[np.argmax(f1s)] and config["num_classes"] > 1:
                best_central_model = deepcopy(self.central_model.cpu())
            elif f1 < f1s[np.argmin(f1s)] and config["num_classes"] == 1:
                best_central_model = deepcopy(self.central_model.cpu())

            accuracies.append(acc)
            precisions.append(pre)
            recalls.append(rec)
            f1s.append(f1)
            f12s.append(f12)
            all_distributions.append(np.unique(all_predicitions, return_counts=True))
            
            if log_per_round:
                model_utils.write_metrices(writer, "Round "+str(i), acc, pre, rec, f1, np.unique(all_predicitions, return_counts=True), is_classification=self.config["num_classes"]>1)
        
        idx = np.argmax(f1s) if config["num_classes"] > 1 else np.argmin(f1s)
        if "2nd_evaluation_averaging" not in self.config:
          model_utils.write_metrices(writer, "Best performance at round: "+str(idx), accuracies[idx], precisions[idx], recalls[idx], f1s[idx], all_distributions[idx], is_classification=self.config["num_classes"]>1)
        
        if filename is not None:
            writer.close()
        
        if return_f1s:
            if "2nd_evaluation_averaging" in self.config:
              return best_central_model, f1s, f12s
            return best_central_model, f1s

        return best_central_model

    # runs training and evaluation procedure on provided data and config
    def run_till_threshold(self, config, filename=None, log_per_round=False, return_f1s=False,f1_threshold=0.7):
        
        best_central_model = None
        
        writer = open(filename, 'w') if filename is not None else sys.stdout
        accuracies, precisions, recalls, f1s, all_distributions = [],[],[],[],[]
        
        num_till_f1 = None
        for i in tqdm(range(config["rounds"])):
            self.train_round()
            acc, pre, rec, f1, all_predicitions = self.evaluate()

            if best_central_model is None:
                best_central_model = deepcopy(self.central_model.cpu())
            elif f1 > f1s[np.argmax(f1s)] and config["num_classes"] > 1:
                best_central_model = deepcopy(self.central_model.cpu())
            elif f1 < f1s[np.argmin(f1s)] and config["num_classes"] == 1:
                best_central_model = deepcopy(self.central_model.cpu())

            accuracies.append(acc)
            precisions.append(pre)
            recalls.append(rec)
            f1s.append(f1)
            all_distributions.append(np.unique(all_predicitions, return_counts=True))
            
            if log_per_round:
                model_utils.write_metrices(writer, "Round "+str(i), acc, pre, rec, f1, np.unique(all_predicitions, return_counts=True), is_classification=self.config["num_classes"]>1)

            if f1 >= f1_threshold:
                num_till_f1 = i+1
                break
        print("Rounds till convergence:", num_till_f1)

        idx = np.argmax(f1s) if config["num_classes"] > 1 else np.argmin(f1s)
        model_utils.write_metrices(writer, "Best performance at round: "+str(idx), accuracies[idx], precisions[idx], recalls[idx], f1s[idx], all_distributions[idx], is_classification=self.config["num_classes"]>1)
        
        if filename is not None:
            writer.close()
        
        if return_f1s:
            return best_central_model, f1s

        return best_central_model
    
    # computes accuracy, precision, recall and f1
    # takes test samples and labels
    # returns metrices
    def evaluate(self):

        x_test = torch.tensor(self.config["X_test"]).float()
        y_test = torch.tensor(self.config["y_test"])
        self.central_model.eval()
        if self.config["num_classes"] > 1:    
            all_predicitions, _ = model_utils.perform_inference(self.central_model, x_test, self.config["batch_size"], self.config["device"], is_bert=self.is_bert)
            acc = np.sum(np.array(all_predicitions) == y_test.numpy()) / len(y_test)
            pre, rec, f1, _ = precision_recall_fscore_support(y_test, all_predicitions, labels=list(range(self.config["num_classes"])), average=self.config["evaluation_averaging"])
            
            if "2nd_evaluation_averaging" in self.config:
              _, _, f12, _ = precision_recall_fscore_support(y_test, all_predicitions, labels=list(range(self.config["num_classes"])), average=self.config["2nd_evaluation_averaging"])
              return acc, pre, rec, f1, f12, all_predicitions

            return acc, pre, rec, f1, all_predicitions
        else: 
            _, all_predictions = model_utils.perform_inference(self.central_model, x_test, self.config["batch_size"], self.config["device"], is_bert=self.is_bert)
            mae = mean_absolute_error(y_test, all_predictions)
            mse = mean_squared_error(y_test, all_predictions, squared=True)
            rmse = mean_squared_error(y_test, all_predictions, squared=False)
            return _, mae, mse, rmse, all_predictions
        
        
# average model weights by number of samples for training
# with 'weighted=False', the model weights are averaged without being weighted by the client sample counter
class FedAvg(FedBlueprint):
    
    # aggregates the central model using weighted average from local models
    # performs a single learning round
    def train_round(self):

        self.central_model.train()
        local_models = []
        x_train_clients = self.config["clients_feature_dict"]
        y_train_clients = self.config["clients_label_dict"]
        
        # select active clients
        if "active_clients" in self.config and isinstance(self.config["active_clients"], list):
            active_clients = self.config["active_clients"]
        elif "active_clients" in self.config:
            num_active_clients = int(self.config["num_clients"] * self.config["active_clients"])
            print("number of active clients", num_active_clients)
            active_clients = random.sample(range(self.config["num_clients"]), num_active_clients)
        else: 
            active_clients =  list(range(self.config["num_clients"]))
        ys_for_weighting = {}

        for i, index in enumerate(active_clients):
            local_model = deepcopy(self.central_model)
            x_local_train = torch.tensor(x_train_clients[index]).float()
            y_local_train = torch.tensor(y_train_clients[index])
            ys_for_weighting[i] = y_local_train
            
            if len(x_local_train) > 0:
                learning_rate = self.config["learning_rate"] if "learning_rate" in self.config else 0.001
                local_model.train()
                local_model = model_utils.perform_training(local_model, x_local_train, y_local_train, self.config["batch_size"], self.config["local_epochs"], self.config["device"], is_bert=self.is_bert, learning_rate=learning_rate)

            local_models.append(local_model)
            del local_model
        
        self.central_model = model_utils.aggregate_models(self.central_model, local_models, ys_for_weighting, weighted=self.config["weighted"])
        del local_models


# distribution-corrected FedAvg with heuristics
class DCFedAvg(FedBlueprint):
    
    # performs a single learning round
    def train_round(self):
        local_models = []
        x_train_clients = self.config["clients_feature_dict"]
        y_train_clients = self.config["clients_label_dict"]
        
        # select active clients
        if "active_clients" in self.config and isinstance(self.config["active_clients"], list):
            active_clients = self.config["active_clients"]
        elif "active_clients" in self.config:
            num_active_clients = int(self.config["num_clients"] * self.config["active_clients"])
            active_clients = random.sample(range(self.config["num_clients"]), num_active_clients)

            # calc each clients label distribution
            client_label_counts = []
            for client in range(self.config["num_clients"]):
                labels = self.config["clients_label_dict"][client]
                label_counts = [0] * self.config["num_classes"]
                for label in labels:
                    label_counts[label] += 1
                client_label_counts.append(label_counts)
            client_label_counts = np.array(client_label_counts)

            # calc label distribution of active clients
            global_label_distribution = np.sum(client_label_counts, axis=0)
            active_label_distribution = np.sum(client_label_counts[active_clients], axis=0)
  
            if "dc_target" in self.config and self.config["dc_target"] == "balanced":
                target_label_distribution = [1] * self.config["num_classes"]
            else: 
                target_label_distribution = global_label_distribution

            current_cosine_distance = spatial.distance.cosine(target_label_distribution, active_label_distribution)
            
            num_combinations = self.config["dc_combinations"] if "dc_combinations" in self.config else 1
            
            for iteration in range(num_combinations):

                inactive_clients = np.setdiff1d(list(range(self.config["num_clients"])), active_clients)
                cosine_distances = []

                for client in inactive_clients:
                    candidate_label_distribution = np.sum([active_label_distribution, client_label_counts[client]], axis=0)
                    cosine_distance = spatial.distance.cosine(target_label_distribution, candidate_label_distribution)
                    cosine_distances.append(cosine_distance)
                
                best_distance_index = np.argmin(cosine_distances)
                if cosine_distances[best_distance_index] <= current_cosine_distance:
                    current_cosine_distance = cosine_distances[best_distance_index]
                    added_client = inactive_clients[best_distance_index]
                    active_clients.append(added_client)
                    active_label_distribution = np.sum([active_label_distribution, client_label_counts[added_client]], axis=0)

                else:
                    break

        else: 
            active_clients =  list(range(self.config["num_clients"]))
        ys_for_weighting = {}

        for i, index in enumerate(active_clients):
            local_model = deepcopy(self.central_model)
            x_local_train = torch.tensor(x_train_clients[index]).float()
            y_local_train = torch.tensor(y_train_clients[index])
            ys_for_weighting[i] = y_local_train
            
            if len(x_local_train) > 0:
                local_model = model_utils.perform_training(local_model, x_local_train, y_local_train, self.config["batch_size"], self.config["local_epochs"], self.config["device"], is_bert=self.is_bert)

            local_models.append(local_model)
            del local_model
        
        self.central_model = model_utils.aggregate_models(self.central_model, local_models, ys_for_weighting, weighted=self.config["weighted"])
        del local_models



# takes a string as input
# returns a learning w.r.t. the provided name
# checks if all required arguments are provided
def get_strategy_by_name(config):
    if config["strategy_name"].lower() == "fedavg":
        print("FedAvg ignores these parameters: 'stepsize', 'reset_per_round'")
        return FedAvg(config)
    elif config["strategy_name"].lower() == "dcfedavg":
        return DCFedAvg(config)
    else:
        raise ValueError("strategy " + config["strategy_name"] + " has not been configured yet")