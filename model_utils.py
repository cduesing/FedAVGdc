import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import statistics
from scipy import linalg
import matplotlib.pyplot as plt
from collections import Counter

# model update w.r.t. provided features and labels
# takes model, x tensor, y tensor, batchsize, number of epochs and device
# returns updated model
def perform_training(model, x_train, y_train, batch_size=8, epochs=1, device="cpu", is_bert=False, learning_rate=0.001):
    
    if hasattr(model, 'config') and hasattr(model.config, "problem_type"):
        model.config.problem_type = None
    model.to(device)
    optimizer = Adam(model.parameters(), lr = learning_rate)
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)

    losses = 0
    for _ in range(epochs):
        model.train()
        model.zero_grad()

        for batch in train_dataloader:
            if len(batch[1]) > 1:
              x = batch[0].float().to(device)
              y = batch[1].to(device)
              loss, _ = model(x,labels=y)
              losses += loss.item()

              loss.backward()
              optimizer.step()
            
    return model


# use model to perform inference
# takes model, x tensor, batch_size and device
# returns 1D list containing predicted class indices and 2D list of logits
def perform_inference(model, x_test, batch_size=8, device="cpu", output_logits=False, is_bert=False, y_test=None):
    
    losses = []
    all_predicitions, all_logits = [], []
    model.to(device)
    test_dataset = TensorDataset(x_test) if y_test == None else TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, sampler = None, batch_size = batch_size)

    model.eval()
    with torch.no_grad():    
        for batch in test_dataloader:
            loss = 0
            y = batch[1].to(device) if y_test is not None else None
            if is_bert is False:
                x = batch[0].float().to(device)
                (loss, logits) = model(x,labels=y)
            else:
                x = batch[0].long().to(device)
                logits = model(x,labels=y)[0]
        
            predictions = torch.argmax(logits, dim=1).tolist()
            all_predicitions.extend(predictions)
            all_logits.extend(logits.tolist())
            losses.append(loss.item() if y_test != None else 0)
    
    if y_test is not None:
        return all_predicitions, all_logits, statistics.mean(losses)

    return all_predicitions, all_logits


# computes weighted (if weighted==True) or unweighted (if weighted==False) average of a number of local models to create an updated central model
# takes the old central model, the list of local models, the sample distribution (for weighting), boolean if to weighten the average
# returns new central model
def aggregate_models(central_model,local_model_list, y_train_clients, weighted=True, attentive=False, stepsize=1.2, dp=0.00):
    
    new_central_model_dict = deepcopy(central_model.state_dict())
    keys = list(new_central_model_dict.keys())

    for k in keys:
            
        # weighted average
        if weighted: 
            counts = [len(y_train_clients[key]) for key in y_train_clients]
            if local_model_list[0].state_dict()[k].dtype == torch.float32:
                l = [torch.mul(model.state_dict()[k], counts[i]) for i,model in enumerate(local_model_list)]
                aggregated = torch.div(torch.sum(torch.stack(l), dim=0), sum(counts))
            else:
                aggregated = local_model_list[0].state_dict()[k]
        
        # unweighted average
        else: 
            if local_model_list[0].state_dict()[k].dtype == torch.float32:
                l = [model.state_dict()[k] for model in local_model_list]
                aggregated = torch.mean(torch.stack(l), dim=0)
            else:
                aggregated = local_model_list[0].state_dict()[k]

        new_central_model_dict[k] = aggregated
        
    central_model.load_state_dict(new_central_model_dict)
    
    return central_model


# writes evaluation metrices to writer
def write_metrices(writer, title, acc, pre, rec, f1, dist, is_classification=True):
    writer.write(str(title))
    if is_classification: 
        writer.write("\n Accuracy: "+str(acc))
        writer.write("\n Precision: "+str(pre))
        writer.write("\n Recall: "+str(rec))
        writer.write("\n F1-Score: "+str(f1))
        writer.write("\n Label Distribution: "+str(dist))
    else: 
        writer.write("\n MAE: "+str(pre))
        writer.write("\n MSE: "+str(rec))
        writer.write("\n RMSE: "+str(f1))
    writer.write("\n\n")