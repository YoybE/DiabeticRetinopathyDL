import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
import shutil

from .io import save_image
from .visualization import plot_anomaly_distribution

def train_model(model, 
                criterion, 
                train_dataloader: DataLoader,
                validation_dataloader: DataLoader,
                device=None, 
                lr=0.001, 
                num_epochs=20, 
                verbose=True):
    '''
    Trains model and plots Training/Validation Loss/Accuracy graphs

    Inputs: 
        - model: Pytorch model to train
        - criterion: Loss function to minimize
        - train_dataloader: Dataloader for training dataset
        - train_dataloader: Dataloader for validation dataset
        - device: device default to cpu
        - lr: float representing Learning Rate
        - num_epochs: integer representing total number of epochs for training 
        - verbose: boolean to toggle print statements of loss & accuracy
    '''
    
    if (device is None):
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    train = []
    validation = []

    # Initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0
        val_running_loss = 0
        correct = 0
        val_correct = 0
        total = 0
        val_total = 0

        # Training
        model.train()
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = [torch.argmax(p) for p in outputs]
            correct += sum(p==t for p,t in zip(pred, labels))
            train.append(sum(t==1 for t in labels))
            total += len(labels)
            
        epoch_acc = correct/total

        # Validation
        model.eval()
        with torch.no_grad():
            for images, labels in validation_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                pred = [torch.argmax(p) for p in outputs]
                val_correct += sum(p==t for p,t in zip(pred, labels))
                validation.append(sum(t==1 for t in labels))
                val_total += len(labels)
        
        val_epoch_acc = val_correct/val_total
        
        epoch_loss = running_loss / len(train_dataloader)
        val_epoch_loss = val_running_loss / len(validation_dataloader)

        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)
        val_loss_list.append(val_epoch_loss)
        val_acc_list.append(val_epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        if (verbose):
            print(f" Training Loss: {epoch_loss:.6f}   | Training Accuracy: {epoch_acc:.4f}\n Validation Loss: {val_epoch_loss:.6f} | Validation Accuracy: {val_epoch_acc:.4f}")
    
    plot_anomaly_distribution(train, validation, train_dataloader.batch_size)
    plot_train_val(train_loss_list, train_acc_list, val_loss_list, val_acc_list)

def plot_train_val(train_loss_list, train_acc_list, val_loss_list, val_acc_list):
    '''
    Plots the training and validation loss/accuracy obtained during training
    Training graphs are used to visualize the loss and accuracy during training
    Validation graphs can help to determine signs of underfitting/overfitting
    '''

    train_acc_list = [v.cpu() for v in train_acc_list]
    val_acc_list = [v.cpu() for v in val_acc_list]

    train_fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(train_loss_list, "orange")
    ax2.plot(train_acc_list)
    _ = ax1.set_title("Training Loss")
    _ = ax2.set_title("Training Accuracy")
    _ = ax1.set_xlabel("Epoch")
    _ = ax1.set_ylabel("Loss")
    _ = ax2.set_xlabel("Epoch")
    _ = ax2.set_ylabel("Accuracy")
    plt.subplots_adjust(right=2)

    val_fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(val_loss_list, "orange")
    ax2.plot(val_acc_list)
    _ = ax1.set_title("Validation Loss")
    _ = ax2.set_title("Validation Accuracy")
    _ = ax1.set_xlabel("Epoch")
    _ = ax1.set_ylabel("Loss")
    _ = ax2.set_xlabel("Epoch")
    _ = ax2.set_ylabel("Accuracy")
    plt.subplots_adjust(right=2)

def evaluate_model(model, dataloader, output_dir="./outputs", device=None):
    '''
    Calculates the accuracy, F1 and F2 scores and 
    Saves their output as images if the dataloader has shuffle=False

    Input:
        - model: Pytorch model to evaluate
        - dataloader: Dataloader
        - output_dir: string of folder/dir to save images at
        - device: device default to cpu

    Output:
        - accuracy: Acc. of Model based on Dataloader (in %)
        - f1_score: F1 Score of Model
        - f2_score: F2 Score of Model
        - recall: Recall of Model (% of anomalies detected correctly)
    '''
    if (device is None):
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    model.eval()

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    save = (type(dataloader.sampler) != torch.utils.data.sampler.RandomSampler)
    
    # Ensures that old evaluation images are not intertwined with new ones
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    with torch.no_grad(): # Ensures that weights do not accidentally get updated
        batch = 0
        for images, labels in dataloader: # Iterates through the dataloader, and accumulatively calculates tp,tn,fp,fn
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            if (save):
                save_image(model.unet_output, predicted, output_dir, "out", dataloader.batch_size, batch)
                batch += 1
            
            tp += sum((p == t == 1) for p,t in zip(predicted, labels))
            tn += sum((p == t == 0) for p,t in zip(predicted, labels))
            fp += sum(((p == 1) & (t == 0)) for p,t in zip(predicted, labels))
            fn += sum(((p == 0) & (t == 1)) for p,t in zip(predicted, labels))
        
        # Reset model's unet_output to free memory
        model.unet_output = None

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    acc = (tp+tn)/len(dataloader.dataset)
    f1_score = 2*((precision*recall)/(precision+recall))
    if (torch.isnan(f1_score)):
        f1_score = 0

    f2_score = 5*((precision*recall)/(4*precision+recall))
    if (torch.isnan(f2_score)):
        f2_score = 0
    
    return acc*100, f1_score, f2_score, recall*100