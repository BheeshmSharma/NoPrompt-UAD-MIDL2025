# Import standard libraries
import os
import shutil
import json

# Import third-party libraries
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# Import custom modules
from metrics.metrics import eval_metrics

## CREATING FOLDERS IN CREATE_MAPS DPT.py
def create_directory(path):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def clear_and_create_directory(path):
    """
    Clears the directory if it exists, then creates it.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def setup_folders(base_path):
    """
    Sets up the required folder structure for inference.
    """
    folder_paths = {
        'images': os.path.join(base_path, 'img'),
        'prediction': os.path.join(base_path, 'pred'),
        'masks': os.path.join(base_path, 'mask'),
    }

    # Create or reset necessary directories
    for folder in folder_paths.values():
        clear_and_create_directory(folder)

    return folder_paths


def write_results_to_file(save_path, total_images, aggregated_metrics,total_tumor_images=0,zero_images=0):
    """
    Write results to a file.
    """
    with open(save_path, "w") as file:

        for metric_type, thresholds in aggregated_metrics.items():
            for threshold, total in thresholds.items():
                file.write(f"AVG {metric_type.upper()} - {threshold}: {total / total_images:.4f}\n")

def get_file_names(folder):
    file_names=[]
    lister=os.listdir(f'inference/{folder}/mask/')
    for i in lister:
        file_names.append(i.split('/')[0])
    return file_names


#function to return mask
def get_image_heat_map_new(img, attentions, head_num=-1, token=0, model="ZeroshotCLIP"):

    patch_size = 32 # default

    w_featmap = img.shape[2] // patch_size
    h_featmap = img.shape[1] // patch_size


    if(head_num < 0):
        attentions = attentions.reshape(1, w_featmap, h_featmap).mean(dim=0)
    else:
        attentions = attentions.reshape(1, w_featmap, h_featmap)[head_num]

    attention = np.asarray(Image.fromarray((attentions*255).detach().numpy().astype(np.uint8)).resize((h_featmap * patch_size, w_featmap * patch_size))).copy()
   
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(img)

    mask = cv2.resize(attention / attention.max(), pil_image.size)[..., np.newaxis]

    result = (mask * pil_image).astype("uint8")

    return result,mask

def process_image(img, impath, folder_paths,dataset,input_size,mis_pred):
        """
        Processes the image and saves results in the specified folders.

        Parameters:
            img (torch.Tensor): Input image tensor.
            impath (str): Path to the image.
            folder_paths (dict): Dictionary containing folder paths.
            label (int): Ground truth label.
        """
        file_path = impath.split('/')[-1].split('.')[0]

        # Convert tensor to image and save
        transform_to_pil = transforms.ToPILImage()
        image = transform_to_pil(img)
        image.save(f"{folder_paths['images']}/{file_path}.png")

        # Load attentions and generate heatmap
        attentions = torch.load('Attn_map.pt')[0, 0, 1:50]
        _, mask_pred = get_image_heat_map_new(img, attentions)

        # Save attention heatmap
        mask_pred = torch.tensor(mask_pred).permute(2, 0, 1)
        mask_pred = transform_to_pil(mask_pred)
        mask_pred.save(f"{folder_paths['prediction']}/{file_path}.png")

        # Load and save the mask

        if(mis_pred==False):
            mask_file = f"MASK/{dataset}/{file_path}.png" 
            mask = Image.open(mask_file)
            mask.save(f"{folder_paths['masks']}/{file_path}.png")
        else:
            mask=torch.zeros(input_size)
            mask=transform_to_pil(mask)
            mask.save(f"{folder_paths['masks']}/{file_path}.png")





