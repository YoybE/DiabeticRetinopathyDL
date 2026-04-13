import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import os

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

def plot_anomaly_distribution(train_dataloader, validation_dataloader, test_dataloader):

    torch.manual_seed(42)
    train = [sum(l == 1) for i,l in train_dataloader]
    n_train = len(train)
    validation = [sum(l == 1) for i,l in validation_dataloader]
    n_validation = len(validation)
    test = [sum(l == 1) for i,l in test_dataloader]
    n_test = len(test)

    cmap = plt.get_cmap('OrRd')
    fig, axes = plt.subplots(1, 3, figsize=(9,3))

    # Set all axes to hug left (west)
    axes[0].set_anchor('W')
    axes[1].set_anchor('W')
    axes[2].set_anchor('W')

    # Train Distribution
    _ = axes[0].set_title("Anomaly Distribution (Train)")
    norm = mcolors.Normalize(min(train), max(train))
    axes[0].bar(list(range(n_train)), train, color=cmap(norm(train)))

    # Validation Distribution
    _ = axes[1].set_title("Anomaly Distribution (Val)")
    norm = mcolors.Normalize(min(validation), max(validation))
    axes[1].bar(list(range(n_validation)), validation, color=cmap(norm(validation)))

    # Test Distribution
    _ = axes[2].set_title("Anomaly Distribution (Test)")
    norm = mcolors.Normalize(min(test), max(test))
    axes[2].bar(list(range(n_test)), test, color=cmap(norm(test)))

    plt.tight_layout()
    plt.show()