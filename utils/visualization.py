import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import os

def visualize_dataset():
    '''
    Plots 5 images for each class within the DiabeticRetinopathy dataset for visualization
    '''

    fig, axes = plt.subplots(2, 5, figsize=(8,4))

    for idx in range(5):
        # Set all axes to hug left (west)
        axes[0][idx].set_anchor('W')
        axes[1][idx].set_anchor('W')

        # 'Healthy' Images
        axes[0][idx].set_xticks([])
        axes[0][idx].set_yticks([])
        healthy_fp = "Healthy_{}.png".format(idx+2)
        _ = axes[0][idx].set_title(healthy_fp, fontsize=10)
        img = plt.imread("./dataset/Healthy/{}".format(healthy_fp))
        axes[0][idx].imshow(img)

        # 'Severe DR' Images
        axes[1][idx].set_xticks([])
        axes[1][idx].set_yticks([])
        severe_fp = "Severe DR_{}.png".format(idx+2)
        _ = axes[1][idx].set_title(severe_fp, fontsize=10)
        img = plt.imread("./dataset/Severe DR/{}".format(severe_fp))
        axes[1][idx].imshow(img)

    plt.tight_layout()
    plt.show()

def plot_train_val(train_loss_list, train_acc_list, val_loss_list, val_acc_list, save=False, name=None, save_dir="./outputs/compare/plots"):
    '''
    Plots the training and validation loss/accuracy obtained during training
    Training graphs are used to visualize the loss and accuracy during training
    Validation graphs can help to determine signs of underfitting/overfitting
    '''
    train_fig, train_axes = plt.subplots(1,2, layout="constrained")
    plot_train(train_axes, train_loss_list, train_acc_list)
    save_plot(save, save_dir, "train", name)

    val_fig, val_axes = plt.subplots(1,2, layout="constrained")
    plot_val(val_axes, val_loss_list, val_acc_list)
    save_plot(save, save_dir, "val", name)

def plot_train(axes, train_loss_list, train_acc_list, label=None):
    try:
        train_acc_list = [v.cpu() for v in train_acc_list]
    except:
        pass

    axes[0].plot(train_loss_list, label=label)
    axes[1].plot(train_acc_list, label=label)
    _ = axes[0].set_title("Training Loss")
    _ = axes[1].set_title("Training Accuracy")
    _ = axes[0].set_xlabel("Epoch")
    _ = axes[0].set_ylabel("Loss")
    _ = axes[1].set_xlabel("Epoch")
    _ = axes[1].set_ylabel("Accuracy")
    _ = axes[0].legend()
    _ = axes[1].legend()

def plot_val(axes, val_loss_list, val_acc_list, label=None):
    try:
        val_acc_list = [v.cpu() for v in val_acc_list]
    except:
        pass

    axes[0].plot(val_loss_list, label=label)
    axes[1].plot(val_acc_list, label=label)
    _ = axes[0].set_title("Validation Loss")
    _ = axes[1].set_title("Validation Accuracy")
    _ = axes[0].set_xlabel("Epoch")
    _ = axes[0].set_ylabel("Loss")
    _ = axes[1].set_xlabel("Epoch")
    _ = axes[1].set_ylabel("Accuracy")
    _ = axes[0].legend()
    _ = axes[1].legend()

def multiplot_train_val(metrics, save, name="", save_dir="./outputs/compare/plots"):
    if (not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    train_fig, train_axes = plt.subplots(1,2, layout="constrained")
    val_fig, val_axes = plt.subplots(1,2, layout="constrained")

    for k,v in metrics.items():
        train_loss_list = v["train"]["loss"]
        train_acc_list = v["train"]["acc"]
        val_loss_list = v["val"]["loss"]
        val_acc_list = v["val"]["acc"]
        plot_train(train_axes, train_loss_list, train_acc_list, k)
        plot_val(val_axes, val_loss_list, val_acc_list, k)

    save_plot(save, save_dir, "multiplot", "val"+name)
    save_plot(save, save_dir, "multiplot", "train"+name)

def plot_anomalies(validation_dataset, 
                   test_dataset, 
                   val_dir="./outputs/val", 
                   test_dir="./outputs/test"):
    '''
    Plots images for anomaly against their original inputs 
    for Validation & Test Dataloaders for visualization
    '''

    # Based on our dataset, we know that for val/test, the anomaly indices stay from 100 to 118
    indices = list(range(100,119))
    n_indices = len(indices)
    fig, axes = plt.subplots(n_indices,4, figsize=(10,50))
    
    if (not (os.path.exists(val_dir) and os.path.exists(test_dir))):
        print("Unable to plot validation & test anomalies as directories do not exist!")
        return

    # Removes '/' ending for standardization  
    if (val_dir[-1] == '/'):
        val_dir = val_dir[:-1]
    if (test_dir[-1] == '/'):
        test_dir = test_dir[:-1]

    for idx in range(n_indices):
        # Set all axes to hug left (west)
        axes[idx][0].set_anchor('W')
        axes[idx][1].set_anchor('W')
        axes[idx][2].set_anchor('W')
        axes[idx][3].set_anchor('W')

        # Validation Input
        axes[idx][0].set_xticks([])
        axes[idx][0].set_yticks([])
        _ = axes[idx][0].set_title("Original Input (Val)")
        axes[idx][0].imshow(validation_dataset[indices[idx]][0].permute(1,2,0), aspect="equal")

        # Validation Output
        axes[idx][1].set_xticks([])
        axes[idx][1].set_yticks([])
        try:
            img = plt.imread("{}/out{}_0.png".format(val_dir, indices[idx]))
            _ = axes[idx][1].set_xlabel("Wrongly Labelled as 'Normal'", fontsize=10, color='r')
        except:
            img = plt.imread("{}/out{}_1.png".format(val_dir, indices[idx]))
            _ = axes[idx][1].set_xlabel("Correctly Labelled as 'Abnormal'", fontsize=10, color='g')

        axes[idx][1].imshow(img, aspect="equal")
        _ = axes[idx][1].set_title("UNet Output (Val)") 

        # Test Input
        axes[idx][2].set_xticks([])
        axes[idx][2].set_yticks([])
        _ = axes[idx][2].set_title("Original Input (Test)")
        axes[idx][2].imshow(test_dataset[indices[idx]][0].permute(1,2,0), aspect="equal")

        # Test Output
        axes[idx][3].set_xticks([])
        axes[idx][3].set_yticks([])
        try:
            img = plt.imread("{}/out{}_0.png".format(test_dir, indices[idx]))
            _ = axes[idx][3].set_xlabel("Wrongly Labelled as 'Normal'", fontsize=10, color='r')
        except:
            img = plt.imread("{}/out{}_1.png".format(test_dir, indices[idx]))
            _ = axes[idx][3].set_xlabel("Correctly Labelled as 'Abnormal'", fontsize=10, color='g')

        axes[idx][3].imshow(img, aspect="equal")
        _ = axes[idx][3].set_title("UNet Output (Test)")    

    plt.tight_layout()
    plt.show()

def plot_anomaly_distribution(train, validation, batch_size, save=False, name=None, save_dir="./outputs/compare/plots"):
    train = [v.cpu() for v in train]
    train_normal = [batch_size-v for v in train]
    validation = [v.cpu() for v in validation]
    validation_normal = [batch_size-v for v in validation]
    n_train = len(train)
    n_validation = len(validation)

    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    # Set all axes to hug left (west)
    axes[0].set_anchor('W')
    axes[1].set_anchor('W')

    # Train Distribution
    _ = axes[0].set_title("Anomaly Distribution (Train)")
    bs_list = list(range(n_train))
    norm = mcolors.Normalize(min(train), max(train))
    axes[0].bar(bs_list, train, color='red')
    norm = mcolors.Normalize(min(train_normal), max(train_normal))
    axes[0].bar(bs_list, train_normal, bottom=train, color='green')

    # Validation Distribution
    _ = axes[1].set_title("Anomaly Distribution (Val)")
    bs_list = list(range(n_validation))
    norm = mcolors.Normalize(min(validation), max(validation))
    axes[1].bar(bs_list, validation, color='red')
    norm = mcolors.Normalize(min(validation_normal), max(validation_normal))
    axes[1].bar(bs_list, validation_normal, bottom=validation, color='green')

    plt.tight_layout()
    save_plot(save, save_dir, "dist", name) 

def save_plot(save, save_dir, plot_type, name):
    if (save):
        plt.savefig("{}/{}_{}.jpeg".format(save_dir, plot_type, name))
        plt.close()
    else:
        plt.show()