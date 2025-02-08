import os
import cv2 
import torch
import random
import config
import requests
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import SamModel, SamProcessor

import warnings
warnings.filterwarnings("ignore")

# Notebook Workflow
device = args.device
checkpoint_path = args.checkpoints


device = "cuda" if torch.cuda.is_available() else "cpu"

MedSAM = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(device)
processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
MedSAM.eval()


# Function to generate random points inside a bounding box and round them
def generate_random_points(bbox, num_points=10, decimal_places=0):
    x_min, y_min, x_max, y_max = bbox
    points = []
    labels = []  # Corresponding labels for the points
    for _ in range(num_points):
        # Generate random x and y coordinates inside the bounding box
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        
        # Round the coordinates to the specified number of decimal places
        points.append([round(x, decimal_places), round(y, decimal_places)])
        
        # Assign label '1' for all points (you can modify this if needed)
        labels.append(1)
    
    return points, labels

# Function to get bounding box
def get_bounding_box(ground_truth_map):
    ground_truth_map = np.array(ground_truth_map)
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:  # If there are no non-zero elements
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # Add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    
    return bbox

# Function to get bounding boxes for a batch
def get_bounding_boxes_for_batch(batch_masks):
    bounding_boxes = []
    for mask in batch_masks:
        bbox = get_bounding_box(mask)
        if bbox:
            bounding_boxes.append(bbox)
    return bounding_boxes

Dataset_name = "BraTS20" ## Keep Dataset folder name here
Prompt_type = "PointsPrompt" ## Keep which type of prompt inference to be performed (default: Box, Point)

Filenames = OS.listdir(f'./DATA/{Dataset_name}/images/Unhealthy/')
directory = f'./DATA/{Dataset_name}/MedSAM_Mask_with_DDPT_Prompt_{Prompt_type}/'
image_folder_path = f'./DATA/{Dataset_name}/images/Unhealthy/'
DDPT_mask_folder_path = f'./DATA/{Dataset_name}/DDPT_Mask/'

os.makedirs(directory, exist_ok=True)
count = 0.0
for filename in tqdm(Filenames):
    image_path = os.path.join(image_folder_path, filename)
    image = Image.open(image_path).convert("RGB")

    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    corrected_array = np.power(image_array, 2.0)  # Apply gamma correction
    image = [Image.fromarray((corrected_array*255).astype(np.uint8))]  # Rescale to [0, 255]
    
    # Load and normalize the DPT mask image
    DDPT_mask_image = cv2.imread(os.path.join(DDPT_mask_folder_path, filename), cv2.IMREAD_GRAYSCALE)
    DDPT_mask_image = (DDPT_mask_image - DDPT_mask_image.min()) / (DDPT_mask_image.max() - DDPT_mask_image.min())
    DDPT_mask_image = np.where(DDPT_mask_image > 0.5, 1.0, 0.0)

    # Get bounding boxes for the batch
    input_boxes = get_bounding_box(DDPT_mask_image)
    # Transform the list of bounding boxes
    transformed_boxes = [[box] for box in input_boxes]
    
    if Prompt_type == "Box":
        # Prepare inputs for the model
        inputs = processor(image, input_boxes=[transformed_boxes], return_tensors="pt").to(device)

    elif Prompt_type == "Box":
        points, labels = generate_random_points(input_boxes)  # Generate 10 points and labels for each bounding box
        # Prepare inputs for the model
        inputs = processor(image, input_points=[points], input_labels = [labels], return_tensors="pt").to(device)

    # Predict
    with torch.no_grad():
        outputs = MedSAM(**inputs, multimask_output=False)

    preds = torch.sigmoid(outputs.pred_masks.squeeze(1)).squeeze(1)
    preds = (preds - preds.min()) / (preds.max() - preds.min())
    preds = torch.where(preds > 0.5, 1.0, 0.0).squeeze(0)

    numpy_array = preds.cpu().numpy()
    image = Image.fromarray(numpy_array.astype(np.uint8)*255)
    image.save(os.path.join(directory, filename))