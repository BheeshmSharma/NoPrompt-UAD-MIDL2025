import os
import shutil
import json
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from metrics.metrics import eval_metrics
from PIL import Image
import numpy as np
import cv2

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


## BETTER METRIC CALCULATION IN generate_metric DPT.py

import json
from tqdm import tqdm

def calculate_batch_metrics(batch, folder, threshold, median, classify):
    """
    Calculate metrics for a given batch.

    Parameters:
        batch (dict): Dictionary containing lists of data for the batch.
            Keys include ['label', 'domain', 'impath', 'index', 'img'].
        folder (str): Folder where the evaluation data is stored.
        threshold (float): Threshold for metrics calculation.
        median (bool): Whether to apply median filtering.

    Returns:
        tuple: Aggregated metrics, batch history, zero images count, tumor images count.
    """
    batch_metrics = {
        'dice': {'0.4': 0, '0.5': 0, '0.6': 0},
        'iou': {'0.4': 0, '0.5': 0, '0.6': 0},
        'auprc': {'0.4': 0, '0.5': 0, '0.6': 0},
        'f1': {'0.4': 0, '0.5': 0, '0.6': 0},
        'recall': {'0.4': 0, '0.5': 0, '0.6': 0},
        'precision': {'0.4': 0, '0.5': 0, '0.6': 0},
    }
    batch_history = {}
    zero_images = 0
    tumor_images = 0

    # Iterate through the batch size (assume all lists have the same length)
    batch_size = len(batch['label'])  # Number of elements in the batch
    for i in range(batch_size):
        label = batch['label'][i]
        impath = batch['impath'][i]

        # Process only if the label indicates a tumor image
        if label != 1:
            continue

        # Evaluate metrics for the image
        metrics = eval_metrics(impath, folder, threshold, median)

        if not metrics:
            zero_images += 1
            continue

        # Store metrics for the current image
        batch_history[impath] = metrics
        tumor_images += 1

        # Aggregate metrics
        for metric_type in batch_metrics:
            for threshold_key in batch_metrics[metric_type]:
                batch_metrics[metric_type][threshold_key] += metrics[f"{metric_type}-{threshold_key}"]

    return batch_metrics, batch_history, zero_images, tumor_images



def write_results_to_file(save_path, total_images, aggregated_metrics,total_tumor_images=0,zero_images=0):
    """
    Write results to a file.
    """
    with open(save_path, "w") as file:
        #file.write(f"TOTAL TUMOR IMAGES: {tumor_images}\n")
        #file.write(f"IGNORED IMAGES: {zero_images}\n\n")

        for metric_type, thresholds in aggregated_metrics.items():
            for threshold, total in thresholds.items():
                file.write(f"AVG {metric_type.upper()} - {threshold}: {total / total_images:.4f}\n")

def get_file_names(folder):
    file_names=[]
    lister=os.listdir(f'inference/{folder}/mask/')
    for i in lister:
        file_names.append(i.split('/')[0])
    return file_names

def get_image_heat_map_new(img, attentions, head_num=-1, token=0, model="ZeroshotCLIP"):

    patch_size = 32 # default

    n_heads=1

    w_featmap = img.shape[2] // patch_size
    h_featmap = img.shape[1] // patch_size


    if(head_num < 0):
        attentions = attentions.reshape(1, w_featmap, h_featmap).mean(dim=0)
    else:
        attentions = attentions.reshape(1, w_featmap, h_featmap)[head_num]

    attention = np.asarray(Image.fromarray((attentions*255).detach().numpy().astype(np.uint8)).resize((h_featmap * patch_size, w_featmap * patch_size))).copy()
   
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(img)

    #print('pil image',pil_image.size)
    
    #print('attention', attention.shape)

    mask = cv2.resize(attention / attention.max(), pil_image.size)[..., np.newaxis]
    #print('mask shape',mask.shape)

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

        '''
        if(mis_pred==False):
            mask_file = f"MASK/{dataset}/{file_path}.png" 
            mask = Image.open(mask_file)
            mask.save(f"{folder_paths['masks']}/{file_path}.png")
        else:
            mask=torch.zeros(input_size)
            mask=transform_to_pil(mask)
            mask.save(f"{folder_paths['masks']}/{file_path}.png")
        '''

def save_mask(img, impath, path):

    file_path = impath.split('/')[-1].split('.')[0]
    transform_to_pil = transforms.ToPILImage()

    attentions = torch.load('Attn_map.pt')[0, 0, 1:50]
    _, mask_pred = get_image_heat_map_new(img, attentions)


        # Save attention heatmap
    mask_pred = torch.tensor(mask_pred).permute(2, 0, 1)
    mask_pred = transform_to_pil(mask_pred)
    mask_pred.save(os.path.join(path,f'{file_path}.png'))




