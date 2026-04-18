import torch
import models
import torch.nn as nn
from utils import train_model, evaluate_model, save_model
from IPython.display import clear_output
import os
import shutil

def compare_architecture(train_dataloader, 
                         validation_dataloader,
                         test_dataloader,
                         device=None,
                         reload=False):
    '''
    Loads all *.py model classifiers found in /models, either train or load their corresponding models and produce a metric table of the model classifiers
    '''    
    # Retrieve all models from models dir
    model_names = [m for m in models.__all__]
    model_list = [getattr(models, m)().to(device) for m in model_names]
    n = len(model_list)
    criterion = nn.CrossEntropyLoss()
    
    train_metrics = []
    validation_metrics = []
    test_metrics = []

    model_dir = "./outputs/models/compare"
    eval_dir = "./outputs/compare/out"

    if (not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    else:
        if (reload):
            shutil.rmtree(f"{model_dir}/")
            os.mkdir(model_dir)
            
    for i in range(n):
        curr_name = model_names[i]

        # Load already trained default comparison models
        load_model = False 
        for file in os.listdir(model_dir):
            if model_names[i] in file:
                load_model = True
                model_list[i] = torch.load(f"{model_dir}/{file}", weights_only=False)

        # Train model if unable to load anything
        if (not load_model):
            train_save_model(model_list[i], model_dir, criterion, train_dataloader, validation_dataloader, device, verbose=False)
        else:
            print(f"Successfully retrieved {curr_name} model...")
        
        # Prepare metrics for evaluation
        train, val, test = save_metrics(model_list[i], train_dataloader, validation_dataloader, test_dataloader, device, eval_dir)
        train_metrics.append(train)
        validation_metrics.append(val)
        test_metrics.append(test)

        # Clearing space for other operations
        model_list[i] = None
        torch.cuda.empty_cache()

    clear_output() # Clear Print to make way for Evaluation results
    
    for i in range(n):
        print_metrics(model_names[i], train_metrics[i], validation_metrics[i], test_metrics[i])

# Assuming that the models here are all already trained
def compare_models(models, train_dataloader, validation_dataloader, test_dataloader, device=None):
    
    model_list = []
    train_metrics = []
    validation_metrics = []
    test_metrics = []

    # If models is a string of the dir containing the models to be compared
    if isinstance(models, str):
        if (os.path.exists(models)):
            for model in os.listdir(models):
                try:
                    model_list.append(torch.load(model, weights_only=False))
                except Exception as e:
                    print("Unable to load model.", e)
                    return
        else:
            print("Error: Model directory does not exist.")
            return
    else:
        model_list = models
    
    for model in model_list:
        train, val, test = save_metrics(model, train_dataloader, validation_dataloader, test_dataloader, device)
        train_metrics.append(train)
        validation_metrics.append(val)
        test_metrics.append(test)
    
    for i in range(len(model_list)):
        print_metrics(model_list[i], train_metrics[i], validation_metrics[i], test_metrics[i])
    

def train_save_model(model, model_dir, criterion, train_dataloader, validation_dataloader, device, verbose=False):
    model_name = model._name
    print(f"Training {model._name} model...")
    train_model(model, criterion, train_dataloader, validation_dataloader, device, verbose=verbose)
    print("Training finished, saving model...")
    save_model(model, model_name, 20, model_dir)
        
def save_metrics(model, train_dataloader, validation_dataloader, test_dataloader, device=None, eval_dir="./outputs/compare/out"):
    # Prepare metrics for evaluation comparison
    model_name = model._name
    print(f"Evaluating {model_name} model...")
    train_metrics = (evaluate_model(model, train_dataloader, "{}/{}/train".format(eval_dir, model_name), device))
    validation_metrics = (evaluate_model(model, validation_dataloader, "{}/{}/val".format(eval_dir, model_name), device))
    test_metrics = (evaluate_model(model, test_dataloader, "{}/{}/test".format(eval_dir, model_name), device))
    print("Evaluation finished...")

    return train_metrics, validation_metrics, test_metrics
      
def print_metrics(model, train_metrics, validation_metrics, test_metrics):
    if isinstance(model, str):
        name = model
    else:
        name = model._name
    print("{}".format("-"*len(name))       
            + f"\n{name}"        
            + "\n{}".format("-"*len(name))        
            + f"\nTrain | Accuracy: {train_metrics[0]:.2f}% | F1_Score: {train_metrics[1]:.2f} | F2_Score: {train_metrics[2]:.2f} | Anomaly Detection: {train_metrics[3]:.2f}%"
            + f"\n  Val | Accuracy: {validation_metrics[0]:.2f}% | F1_Score: {validation_metrics[1]:.2f} | F2_Score: {validation_metrics[2]:.2f} | Anomaly Detection: {validation_metrics[3]:.2f}%"
            + f"\n Test | Accuracy: {test_metrics[0]:.2f}% | F1_Score: {test_metrics[1]:.2f} | F2_Score: {test_metrics[2]:.2f} | Anomaly Detection: {test_metrics[3]:.2f}%")