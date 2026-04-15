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
    eval_dir = "./outputs/training/compare/out"

    if (not os.path.exists(model_dir)) or reload:
        if reload and os.path.exists(model_dir):
            shutil.rmtree(f"{model_dir}/")
        os.makedirs(model_dir)

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
            print(f"Training {curr_name} model...")
            train_model(model_list[i], criterion, train_dataloader, validation_dataloader, device, verbose=False)
            print("Training finished, saving model...")
            save_model(model_list[i], curr_name, 20, model_dir)
        else:
            print(f"Successfully retrieved {curr_name} model...")
        
        # Prepare metrics for evaluation
        print(f"Evaluating {curr_name} model...")
        train_metrics.append(evaluate_model(model_list[i], train_dataloader, "{}/{}/train".format(eval_dir, model_names[i]), device))
        validation_metrics.append(evaluate_model(model_list[i], validation_dataloader, "{}/{}/val".format(eval_dir, model_names[i]), device))
        test_metrics.append(evaluate_model(model_list[i], test_dataloader, "{}/{}/test".format(eval_dir, model_names[i]), device))
        print("Evaluation finished...")
        
        # Clearing space for other operations
        model_list[i] = None
        torch.cuda.empty_cache()

    clear_output() # Clear Print to make way for Evaluation results
    
    for i in range(n):
        name = model_names[i]
        print("{}".format("-"*len(name))       
              + f"\n{name}"        
              + "\n{}".format("-"*len(name))        
              + f"\nTrain | Accuracy: {train_metrics[i][0]:.2f}% | F1_Score: {train_metrics[i][1]:.2f} | F2_Score: {train_metrics[i][2]:.2f} | Anomaly Detection: {train_metrics[i][3]:.2f}%"
              + f"\n  Val | Accuracy: {validation_metrics[i][0]:.2f}% | F1_Score: {validation_metrics[i][1]:.2f} | F2_Score: {validation_metrics[i][2]:.2f} | Anomaly Detection: {validation_metrics[i][3]:.2f}%"
              + f"\n Test | Accuracy: {test_metrics[i][0]:.2f}% | F1_Score: {test_metrics[i][1]:.2f} | F2_Score: {test_metrics[i][2]:.2f} | Anomaly Detection: {test_metrics[i][3]:.2f}%")

def compare_models():
    pass