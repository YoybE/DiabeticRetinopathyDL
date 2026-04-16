import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def import_dataset():
    import kagglehub
    import shutil
    import os

    # For copying to local folder
    destination = "./dataset"

    if not os.path.exists(destination): # Skips downloading dataset if dataset folder already exists
        # Download dataset
        path = kagglehub.dataset_download("sachinkumar413/diabetic-retinopathy-dataset")
        print("Downloaded to:", path)

        shutil.copytree(path, destination, dirs_exist_ok=True)
        print("Copied to:", destination)

        # Check structure
        for root, dirs, files in os.walk(destination):
            print("Current path:", root)
            print("Folders:", dirs)
            print("------")
            break

        # Remove unwanted folders
        folders_to_remove = ["Mild DR", "Moderate DR", "Proliferate DR"]

        for folder in folders_to_remove:
            folder_path = os.path.join(destination, folder)

            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"Removed: {folder_path}")
            else:
                print(f"Folder not found: {folder_path}")
    else:
        print("Dataset folder already exists, ignoring...")
    
    complete_path = "./dataset/.complete"

    if os.path.exists(complete_path):
        shutil.rmtree(complete_path)
        print("Removed .complete folder")

def initialize_dataset(t=None):
    # For loading the Diabetic Retinopathy Dataset
    if t is None:
        transforms_base = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])]) # Normalization values from Pytorch's recommended ImageNet values
    else:
        transforms_base = t

    # full_dataset[i] is a a tuple of (image tensor, class index)
    full_dataset = torchvision.datasets.DatasetFolder(root="./dataset", loader=torchvision.datasets.folder.default_loader, transform=transforms_base, extensions=[".png"])

    return full_dataset

def split_dataset(full_dataset):
    '''
    Training-Test-Validation Split
    Train dataset: 80% of 'Healthy' + 80% of 'Severe'
    Test dataset: 10% of 'Healthy' + 10% of 'Severe'
    Validation dataset: 10% of 'Healthy' + 10% of 'Severe'
    '''
    
    print("Classes:", full_dataset.class_to_idx)
    print("Total dataset size:", len(full_dataset))

    healthy = 0
    severe = 0
    for data in full_dataset:
        if data[1] == 0:
            healthy += 1
        elif data[1] == 1:
            severe += 1
    print("Total 'Healthy' samples:", healthy)
    print("Total 'Severe' samples:", severe)
    
    targets = full_dataset.targets
    train_indices = []
    test_indices = []
    validation_indices = []

    for c in range(len(full_dataset.classes)):
        class_indices = [i for i, t in enumerate(targets) if t==c]
        torch.manual_seed(42)
        perm = torch.randperm(len(class_indices)).tolist()
        if c == 0:
            r = healthy
        else:
            r = severe
        a = int(r*0.8)
        b = int(r*0.9)
        train_indices.extend([class_indices[p] for p in perm[:a]])
        test_indices.extend([class_indices[p] for p in perm[a:b]])
        validation_indices.extend([class_indices[p] for p in perm[b:r]])

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    validation_dataset = torch.utils.data.Subset(full_dataset, validation_indices)

    return train_dataset, test_dataset, validation_dataset

def initialize_dataloaders(train_dataset, test_dataset, validation_dataset, batch_size=32):
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=False)
    return train_dataloader, test_dataloader, validation_dataloader