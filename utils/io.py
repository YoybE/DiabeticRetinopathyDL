import torch
import torchvision
import os

def save_image(img_out, label_out, dir, type, batch_size, batch=0):
    '''
    Saves images from tensor outputs to help showcase the image segmentation/attention...etc
    
    Inputs: 
        - img_out: Tensor of Images (N,C,H,W)
        - label_out: Tensor of Predicted Labels (N,)
        - dir: string of folder/dir to save images at 
        - type: string of the preferred name of the output
        - batch_size: integer representing batch size of dataloader
        - batch: integer representing current batch

    Output:
        No returned output 
        but saves images in format "{dir}/{type}{img_no}_{normal:0 | anomaly:1}.png"
    '''
    n = img_out.shape[0]

    # Move to CPU to ensure that backpropagation is unaffected & save GPU memory
    out = img_out.detach().cpu()
    pred = label_out.detach().cpu()

    # Creates directory folder if doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in range(n):
        img_no = i+batch_size*batch
        flattened_img = torch.mean(out[i], dim=0, keepdim=True)
        image = torchvision.transforms.functional.to_pil_image(flattened_img, mode="L")
        filepath = "{}/{}{}_{}.png".format(dir, type, str(img_no), pred[i]) # E.g. "./outputs/out20_0.png"
        image.save(filepath)

        # Helps to resolve memory issues by clearing these temporary outputs
        del flattened_img, image
        torch.cuda.empty_cache()

    del out, pred

def save_model(model, name, epoch, dir="./models", ckpt=False, **kwargs):
    '''
    Saves model to disk in the format {name}_{epoch}_{kwargs}-{date}
    
    E.g. save_model(model, "test", 20, lr=0.001) is saved as "test_20_lr0.001-20260413.pt"
    '''
    from datetime import datetime

    meta = ""
    date = datetime.now().strftime("%Y%m%d")

    if (kwargs):
        for k, v in kwargs.items():
            meta += "_{}{}".format(k,v)
    
    # Removes '/' ending for standardization  
    if (dir[-1] == '/'):
        dir = dir[:-1]

    # If checkpointing, then save as .ckpt file
    if (ckpt):
        ext = ".ckpt"
    else:
        ext = ".pt"

    save_name = "{}_{}{}-{}{}".format(name, epoch, meta, date, ext)
    save_dir = "{}/{}".format(dir, save_name)

    try:
        torch.save(model, save_dir)
        print(f"Saved {save_name} to {save_dir}")
    except:
        print("Failed to save model...")