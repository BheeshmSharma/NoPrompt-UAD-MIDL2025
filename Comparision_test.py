import os
import numpy as np
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn import CosineSimilarity
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import logging
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
import random
import torch.nn as nn
import argparse
import config
import sys
from Modules import *
from loss import *
import time
import torch.nn.functional as F
import math
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def load_and_convert_to_tensor(image_folder_path, mask_folder_path, DPT_mask_folder_path, MedSAM_mask_folder_path, file_list, image_mode='L', mask_mode='L', target_size=(128, 128)):
    # Helper function to convert an image to a tensor
    def to_tensor(image):
        return torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)  # Add channel dimension

    def apply_gamma_correction(image):
        """Apply gamma correction to a PIL image."""
        image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        corrected_array = np.power(image_array, 2.0)  # Apply gamma correction
        corrected_image = Image.fromarray((corrected_array*255).astype(np.uint8))  # Rescale to [0, 255]
        return corrected_image

    # Helper function to crop and resize the image based on non-zero rows/columns
    def crop_and_resize(image, mask, MedSAM_mask, DPT_mask, target_size=(128, 128), threshold=64):
        image_array = np.array(image)
        mask_array = np.array(mask)
        MedSAM_mask_array = np.array(MedSAM_mask)
        DPT_mask_array = np.array(DPT_mask)
        
        # Find non-zero rows and columns
        non_zero_rows = np.any(image_array, axis=1)
        non_zero_cols = np.any(image_array, axis=0)
        
        row_indices = np.where(non_zero_rows)[0]
        col_indices = np.where(non_zero_cols)[0]

        # Check if there are sufficient non-zero rows/columns
        if len(row_indices) >= threshold and len(col_indices) >= threshold:
            cropped_image = image_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
            cropped_mask = mask_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
            cropped_DPT_mask = DPT_mask_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
            cropped_MedSAM_mask = MedSAM_mask_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
        else:
            # Crop the central 64x64 area
            top = 32-((max(row_indices) - min(row_indices)) // 2)
            left = 32-((max(col_indices) - min(col_indices)) // 2)
            cropped_image = image_array[min(row_indices)-top:max(row_indices)+top, min(col_indices)-left:max(col_indices)+left]
            cropped_mask = mask_array[min(row_indices)-top:max(row_indices)+top, min(col_indices)-left:max(col_indices)+left]
            cropped_DPT_mask = DPT_mask_array[min(row_indices)-top:max(row_indices)+top, min(col_indices)-left:max(col_indices)+left]
            cropped_MedSAM_mask = MedSAM_mask_array[min(row_indices)-top:max(row_indices)+top, min(col_indices)-left:max(col_indices)+left]

        # Convert to PIL image and resize to 128x128
        cropped_image = Image.fromarray(cropped_image).resize(target_size, Image.BILINEAR)
        cropped_mask = Image.fromarray(cropped_mask).resize(target_size, Image.BILINEAR)
        cropped_DPT_mask = Image.fromarray(cropped_DPT_mask).resize((256,256), Image.BILINEAR)
        cropped_MedSAM_mask = Image.fromarray(cropped_MedSAM_mask).resize((256,256), Image.BILINEAR)
        
        return cropped_image, cropped_mask, cropped_MedSAM_mask, cropped_DPT_mask
    
    # Initialize lists for valid images and masks
    valid_images = []
    valid_masks = []
    valid_DPT_masks = []
    valid_MedSAM_masks = []
    
    # Load, resize, and convert masks
    for file in file_list:
        # Open and process image
        image = Image.open(os.path.join(image_folder_path, file)).convert(image_mode)      
        image = apply_gamma_correction(image)  

        mask = Image.open(os.path.join(mask_folder_path, file)).convert(mask_mode).resize(image.size, Image.BILINEAR)
        MedSAM_mask = Image.open(os.path.join(MedSAM_mask_folder_path, file)).convert(mask_mode).resize((256, 256), Image.BILINEAR)
        DPT_mask = Image.open(os.path.join(DPT_mask_folder_path, file)).convert(mask_mode).resize((256, 256), Image.BILINEAR) 

        cropped_resized_image, cropped_resized_mask, cropped_MedSAM_mask, cropped_DPT_mask = crop_and_resize(image, mask, MedSAM_mask, DPT_mask, target_size)
        image_tensor = to_tensor(cropped_resized_image)
        mask_tensor = to_tensor(cropped_resized_mask)
        DPT_mask_tensor = to_tensor(cropped_DPT_mask)        
        MedSAM_mask_tensor = to_tensor(cropped_MedSAM_mask)

        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension if needed

        # Add valid image and mask to the lists
        valid_images.append(image_tensor)  # Add the processed image tensor
        valid_masks.append(mask_tensor)  # Add the mask tensor
        valid_DPT_masks.append(DPT_mask_tensor)  # Add the mask tensor        
        valid_MedSAM_masks.append(MedSAM_mask_tensor)  # Add the mask tensor        

    # Convert the list of valid images and masks into batches
    batch_images = torch.stack(valid_images) if valid_images else torch.empty(0)
    batch_masks = torch.stack(valid_masks) if valid_masks else torch.empty(0)
    batch_DPT_masks = torch.stack(valid_DPT_masks) if valid_DPT_masks else torch.empty(0)
    batch_MedSAM_masks = torch.stack(valid_MedSAM_masks) if valid_MedSAM_masks else torch.empty(0)

    batch_masks = torch.where(batch_masks > 127.5, 1.0, 0.0)
    batch_DPT_masks = torch.where(batch_DPT_masks > 127.5, 1.0, 0.0)
    batch_MedSAM_masks = torch.where(batch_MedSAM_masks > 127.5, 1.0, 0.0)
    
    return batch_images, batch_masks, batch_DPT_masks, batch_MedSAM_masks

def compute_metrics_per_sample(preds, targets, epsilon=1e-7):
    """
    Compute AUROC, AUPRC, and additional metrics for a single sample.

    Parameters:
    - preds: Tensor of shape (H, W), predicted probabilities.
    - targets: Tensor of shape (H, W), binary ground truth labels (0 or 1).
    - epsilon: Small value to avoid division by zero.

    Returns:
    - metrics: Dictionary containing AUROC, AUPRC, precision, recall, sensitivity,
               specificity, and Jaccard index (IoU)
    """
    # preds = torch.where(preds > 0.5, 1.0, 0.0)
    preds_flat = preds.flatten().detach().cpu().numpy()
    targets_flat = targets.flatten().detach().cpu().numpy()

    # Ensure targets are binary
    if not np.all(np.isin(targets_flat, [0, 1])):
        raise ValueError("Targets must be binary (0 or 1).")

    # Handle cases where no positive or negative samples exist in the target
    if np.sum(targets_flat) < epsilon or np.sum(1 - targets_flat) < epsilon:
        return {
            "auroc": 0.0,
            "auprc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "jaccard": 0.0,
            "FP_rate": 0.0,
        }

    # Threshold predictions at 0.5
    preds_binary = (preds_flat >= 0.5).astype(int)

    # Compute AUROC and AUPRC
    auroc = roc_auc_score(targets_flat, preds_flat)
    auprc = average_precision_score(targets_flat, preds_flat)

    # Compute confusion matrix
    tp = np.sum((preds_binary == 1) & (targets_flat == 1))
    tn = np.sum((preds_binary == 0) & (targets_flat == 0))
    fp = np.sum((preds_binary == 1) & (targets_flat == 0))
    fn = np.sum((preds_binary == 0) & (targets_flat == 1))

    # Compute precision, recall, sensitivity, specificity, and Jaccard index
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)  # Same as sensitivity
    specificity = tn / (tn + fp + epsilon)
    jaccard = tp / (tp + fp + fn + epsilon)
    FP_rate = fp / (fp + tn + epsilon)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "jaccard": jaccard,
        "FP_rate": FP_rate,
    }

def compute_average_metrics(preds, targets):
    """
    Compute the average AUROC, AUPRC, and additional metrics for a batch of predictions.

    Parameters:
    - preds: Tensor of shape (B, H, W), predicted probabilities for the batch.
    - targets: Tensor of shape (B, H, W), binary ground truth labels for the batch.

    Returns:
    - avg_metrics: Dictionary containing average values for all metrics across the batch.
    """
    batch_size = preds.size(0)
    metrics_sum = {
        "auroc": 0.0,
        "auprc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "jaccard": 0.0,
        "FP_rate": 0.0,
    }

    for i in range(batch_size):
        metrics = compute_metrics_per_sample(preds[i], targets[i])
        for key in metrics:
            metrics_sum[key] += metrics[key]

    avg_metrics = {key: metrics_sum[key] / batch_size for key in metrics_sum}
    return avg_metrics



'''Weight Initialization'''
'''::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::'''
import torch.nn.init as init
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)



'''Main class'''
''':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::'''
if __name__ == "__main__":
    basic_parser = argparse.ArgumentParser(description="arser for config file path")
    basic_parser.add_argument('config_file', type=str, help='Path to the configuration file')
    basic_args, remaining_argv = basic_parser.parse_known_args()
    args = config.parse_args(basic_args.config_file, remaining_argv)

    # Define the condition
    use_seed = args.seed  # Change this to False if you don't want to set a seed
    
    # if use_seed:
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)  # For NumPy operations
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
        # torch.cuda.manual_seed_all(seed_value)  # If using multiple GPUs

    '''Data Loading and path setup'''
    ''':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::'''
    # Paths to the folders containing images
    if args.test_dataset != "":
        dataset = args.test_dataset
        print(f"Evaluating with cross domain test data: {args.test_dataset}")
    else:
        dataset = args.dataset_name
        print(f"Evaluating with same domain training data: {args.dataset_name}")

    '''Data Loading and path setup'''
    ''':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::'''
    Save_path = "./Experiments/" + str(args.dataset_name) + '_' + str(args.folder_name) + "/"
    
    tumor_image_folder_path =  './DATA/' + dataset +'/Frames/'
    dpt_mask_folder_path = './DATA/DPT_inference/' + dataset + '/'
    MedSAM_mask_folder_path = './DATA/MedSAM_inference/' + dataset + '/'
    mask_folder_path = './DATA/' + dataset + '/GT_Mask/'
    
    tumor_image_files_org_test = read_file_to_list('./DATA/' + args.dataset_name +'/test_file_names.txt')

    # Setup device
    device = args.device
    
    Encoder_Model = Encoder().to(device)
    Decoder_Model = Mask_Decoder().to(device)
    
    checkpoint_path = "./Experiments/" + str(args.dataset_name) + '_' + str(args.folder_name) + "/Fold_0/"
    print( f"Inferecing with model {checkpoint_path}")
    
    Encoder_Model.load_state_dict(torch.load(checkpoint_path + "Encoder_Model.pth", map_location=device), strict = False)
    Decoder_Model.load_state_dict(torch.load(checkpoint_path + "Decoder_Model.pth", map_location=device), strict = False)
    
    np.random.shuffle(tumor_image_files_org_test)
    tumor_val_files = tumor_image_files_org_test
    
    point_embedding_loss = Point_Embedding_Loss(loss_type='mse')
    BCM = Boundary_coordinate_and_Mask(grid_size=32)
    
    Candidate_Prompt_Embedding = torch.load('./Fixed_Candidate_Embeddings/Candidate_Prompt_Embedding' + str(i) + '.pt').to(device)
                  
    # Validation
    Encoder_Model.eval()
    Decoder_Model.eval()
    
    loss_Seg_Mask = 0.0
    loss_Point_Act = 0.0
    Attention_Dice_Score = 0.0
    
    WithTrueMask_Decoder_Dice_Score = 0.0
    WithTrueMask_AUROC_Score = 0.0
    WithTrueMask_AUPRC_Score = 0.0
    WithTrueMask_Precision_Score = 0.0
    WithTrueMask_Rcall_Score = 0.0
    WithTrueMask_Specificity_Score = 0.0
    WithTrueMask_Jaccard_Score = 0.0
    WithTrueMask_FP_rate_Score = 0.0
    
    WithDPTMask_Decoder_Dice_Score = 0.0
    WithDPTMask_AUROC_Score = 0.0
    WithDPTMask_AUPRC_Score = 0.0
    WithDPTMask_Precision_Score = 0.0
    WithDPTMask_Rcall_Score = 0.0
    WithDPTMask_Specificity_Score = 0.0
    WithDPTMask_Jaccard_Score = 0.0
    WithDPTMask_FP_rate_Score = 0.0
    
    WithMedSAMMask_Decoder_Dice_Score = 0.0
    WithMedSAMMask_AUROC_Score = 0.0
    WithMedSAMMask_AUPRC_Score = 0.0
    WithMedSAMMask_Precision_Score = 0.0
    WithMedSAMMask_Rcall_Score = 0.0
    WithMedSAMMask_Specificity_Score = 0.0
    WithMedSAMMask_Jaccard_Score = 0.0
    WithMedSAMMask_FP_rate_Score = 0.0

    num_sample_val_epoch = 0.0
    
    with torch.no_grad():
        for tumor_batch in tqdm(create_batches(tumor_val_files, args.batch_size), total=len(tumor_val_files) // args.batch_size):
            # Batch processing
            tumor_images, target_masks, DPT_Mask, MedSAM_Prediction_Mask = load_and_convert_to_tensor(tumor_image_folder_path, mask_folder_path, dpt_mask_folder_path, MedSAM_mask_folder_path, tumor_batch)
            tumor_images, target_masks, DPT_Mask, MedSAM_Prediction_Mask = tumor_images.to(device), target_masks.to(device), DPT_Mask.to(device), MedSAM_Prediction_Mask.to(device)
            bs = tumor_images.shape[0]
            Candidate_Prompt_Embedding_batch = Candidate_Prompt_Embedding.repeat(bs, 1, 1)

            if target_masks.dim() == 1:
                continue
            
            # Create mask for the points which are inside the bounding box using DPT Mask
            box_point_mask = BCM.create_masks(target_masks).to(device).float()
            box_point_mask = box_point_mask.view(bs, -1)
                        
            # Point based prompt embedding from MedSAM Prompt Encoder
            Candidate_Prompt_Embedding_batch = Candidate_Prompt_Embedding_batch.detach()
                        
            # Pass images and point embedding of grid point from encoder
            candidate_spatial_embedding_activations, candidate_spatial_embedding, img_emd = Encoder_Model(tumor_images, Candidate_Prompt_Embedding_batch)
                        
            ## Mask Decoder 
            predicted_mask = Decoder_Model(img_emd, candidate_spatial_embedding)
            predicted_mask = torch.sigmoid(predicted_mask).squeeze(1)
            point_sparse_embedding_activations = candidate_spatial_embedding_activations.squeeze(2)
            
            # Loss Computation
            Segmentation_loss = combined_weighted_compute_average_ELT_loss(predicted_mask, MedSAM_Prediction_Mask, DPT_Mask)
            Point_Activation_loss = compute_average_ELdice_loss(candidate_spatial_embedding_activations, box_point_mask)
                
            Attention_Dice_Score_sum = compute_average_dice(candidate_spatial_embedding_activations, box_point_mask) * bs
            resized_target_mask_256 = F.interpolate(target_masks, size=(256, 256), mode='bilinear', align_corners=False)
            resized_DPT_256 = F.interpolate(DPT_Mask, size=(256, 256), mode='bilinear', align_corners=False)
            resized_target_mask_256 = torch.where(resized_target_mask_256 > 0.0, 1.0, 0.0)
            resized_DPT_256 = torch.where(resized_DPT_256 > 0.0, 1.0, 0.0)
            MedSAM_Prediction_Mask = torch.where(MedSAM_Prediction_Mask > 0.0, 1.0, 0.0)
            
            
            WithMedSAMMask_Decoder_Dice_sum = compute_average_dice(MedSAM_Prediction_Mask, resized_target_mask_256) * bs
            WithDPTMask_Decoder_Dice_sum = compute_average_dice(resized_DPT_256, resized_target_mask_256) * bs
            WithTrueMask_Decoder_Dice_sum = compute_average_dice(predicted_mask, resized_target_mask_256) * bs
            
            True_avg_metrics = compute_average_metrics(predicted_mask, resized_target_mask_256)
            MedSAM_avg_metrics = compute_average_metrics(MedSAM_Prediction_Mask, resized_target_mask_256)
            DPT_avg_metrics = compute_average_metrics(resized_DPT_256, resized_target_mask_256)

            True_SUM_auroc = True_avg_metrics["auroc"] * bs
            True_SUM_auprc = True_avg_metrics["auprc"] * bs
            True_SUM_precision= True_avg_metrics["precision"] * bs
            True_SUM_recall = True_avg_metrics["recall"] * bs
            True_SUM_specificity = True_avg_metrics["specificity"] * bs
            True_SUM_jaccard = True_avg_metrics["jaccard"] * bs
            True_SUM_FP_rate = True_avg_metrics["FP_rate"] * bs

            MedSAM_SUM_auroc = MedSAM_avg_metrics["auroc"] * bs
            MedSAM_SUM_auprc = MedSAM_avg_metrics["auprc"] * bs
            MedSAM_SUM_precision= MedSAM_avg_metrics["precision"] * bs
            MedSAM_SUM_recall = MedSAM_avg_metrics["recall"] * bs
            MedSAM_SUM_specificity = MedSAM_avg_metrics["specificity"] * bs
            MedSAM_SUM_jaccard = MedSAM_avg_metrics["jaccard"] * bs
            MedSAM_SUM_FP_rate = MedSAM_avg_metrics["FP_rate"] * bs

            DPT_SUM_auroc = DPT_avg_metrics["auroc"] * bs
            DPT_SUM_auprc = DPT_avg_metrics["auprc"] * bs
            DPT_SUM_precision= DPT_avg_metrics["precision"] * bs
            DPT_SUM_recall = DPT_avg_metrics["recall"] * bs
            DPT_SUM_specificity = DPT_avg_metrics["specificity"] * bs
            DPT_SUM_jaccard = DPT_avg_metrics["jaccard"] * bs
            DPT_SUM_FP_rate = DPT_avg_metrics["FP_rate"] * bs
            
            num_sample_val_epoch += bs

            loss_Seg_Mask += Segmentation_loss.item()
            loss_Point_Act += Point_Activation_loss.item()
            Attention_Dice_Score += Attention_Dice_Score_sum.item()

            WithMedSAMMask_Decoder_Dice_Score += WithMedSAMMask_Decoder_Dice_sum.item()
            WithDPTMask_Decoder_Dice_Score += WithDPTMask_Decoder_Dice_sum.item()
            WithTrueMask_Decoder_Dice_Score += WithTrueMask_Decoder_Dice_sum.item()
            
            WithTrueMask_AUROC_Score += True_SUM_auroc.item()
            WithTrueMask_AUPRC_Score += True_SUM_auprc.item()
            WithTrueMask_Precision_Score += True_SUM_precision.item()
            WithTrueMask_Rcall_Score += True_SUM_recall.item()
            WithTrueMask_Specificity_Score += True_SUM_specificity.item()
            WithTrueMask_Jaccard_Score += True_SUM_jaccard.item()
            WithTrueMask_FP_rate_Score += True_SUM_FP_rate.item()

            WithDPTMask_AUROC_Score += DPT_SUM_auroc.item()
            WithDPTMask_AUPRC_Score += DPT_SUM_auprc.item()
            WithDPTMask_Precision_Score += DPT_SUM_precision.item()
            WithDPTMask_Rcall_Score += DPT_SUM_recall.item()
            WithDPTMask_Specificity_Score += DPT_SUM_specificity.item()
            WithDPTMask_Jaccard_Score += DPT_SUM_jaccard.item()
            WithDPTMask_FP_rate_Score += DPT_SUM_FP_rate.item()

            WithMedSAMMask_AUROC_Score += MedSAM_SUM_auroc.item()
            WithMedSAMMask_AUPRC_Score += MedSAM_SUM_auprc.item()
            WithMedSAMMask_Precision_Score += MedSAM_SUM_precision.item()
            WithMedSAMMask_Rcall_Score += MedSAM_SUM_recall.item()
            WithMedSAMMask_Specificity_Score += MedSAM_SUM_specificity.item()
            WithMedSAMMask_Jaccard_Score += MedSAM_SUM_jaccard.item()
            WithMedSAMMask_FP_rate_Score += MedSAM_SUM_FP_rate.item()

mean_loss_Seg_Mask = loss_Seg_Mask / num_sample_val_epoch
mean_loss_Point_Act = loss_Point_Act / num_sample_val_epoch
mean_Attention_Dice_Score = Attention_Dice_Score / num_sample_val_epoch

mean_WithTrueMask_Decoder_Dice_Score = WithTrueMask_Decoder_Dice_Score / num_sample_val_epoch
mean_WithTrueMask_AUROC_Score = WithTrueMask_AUROC_Score / num_sample_val_epoch
mean_WithTrueMask_AUPRC_Score = WithTrueMask_AUPRC_Score / num_sample_val_epoch
mean_WithTrueMask_Precision_Score = WithTrueMask_Precision_Score / num_sample_val_epoch
mean_WithTrueMask_Recall_Score = WithTrueMask_Rcall_Score / num_sample_val_epoch
mean_WithTrueMask_Specificity_Score = WithTrueMask_Specificity_Score / num_sample_val_epoch
mean_WithTrueMask_Jaccard_Score = WithTrueMask_Jaccard_Score / num_sample_val_epoch
mean_WithTrueMask_FP_rate_Score = WithTrueMask_FP_rate_Score / num_sample_val_epoch

mean_WithDPTMask_Decoder_Dice_Score = WithDPTMask_Decoder_Dice_Score / num_sample_val_epoch
mean_WithDPTMask_AUROC_Score = WithDPTMask_AUROC_Score / num_sample_val_epoch
mean_WithDPTMask_AUPRC_Score = WithDPTMask_AUPRC_Score / num_sample_val_epoch
mean_WithDPTMask_Precision_Score = WithDPTMask_Precision_Score / num_sample_val_epoch
mean_WithDPTMask_Recall_Score = WithDPTMask_Rcall_Score / num_sample_val_epoch
mean_WithDPTMask_Specificity_Score = WithDPTMask_Specificity_Score / num_sample_val_epoch
mean_WithDPTMask_Jaccard_Score = WithDPTMask_Jaccard_Score / num_sample_val_epoch
mean_WithDPTMask_FP_rate_Score = WithDPTMask_FP_rate_Score / num_sample_val_epoch

mean_WithMedSAMMask_Decoder_Dice_Score = WithMedSAMMask_Decoder_Dice_Score / num_sample_val_epoch
mean_WithMedSAMMask_AUROC_Score = WithMedSAMMask_AUROC_Score / num_sample_val_epoch
mean_WithMedSAMMask_AUPRC_Score = WithMedSAMMask_AUPRC_Score / num_sample_val_epoch
mean_WithMedSAMMask_Precision_Score = WithMedSAMMask_Precision_Score / num_sample_val_epoch
mean_WithMedSAMMask_Recall_Score = WithMedSAMMask_Rcall_Score / num_sample_val_epoch
mean_WithMedSAMMask_Specificity_Score = WithMedSAMMask_Specificity_Score / num_sample_val_epoch
mean_WithMedSAMMask_Jaccard_Score = WithMedSAMMask_Jaccard_Score / num_sample_val_epoch
mean_WithMedSAMMask_FP_rate_Score = WithMedSAMMask_FP_rate_Score / num_sample_val_epoch

# Printing in a table format

print(f":::::::::::::::::::::::::Folder Name {args.folder_name}:::::::::::::::::::::::::::::::")
print(f"{'Metric':<40} {'True Mask':<20} {'MedSAM Mask':<20} {'DPT Mask':<20}")
print(f"{'Decoder Dice Score':<40} {mean_WithTrueMask_Decoder_Dice_Score:<20} {mean_WithMedSAMMask_Decoder_Dice_Score:<20} {mean_WithDPTMask_Decoder_Dice_Score:<20}")
print(f"{'AUROC Score':<40} {mean_WithTrueMask_AUROC_Score:<20} {mean_WithMedSAMMask_AUROC_Score:<20} {mean_WithDPTMask_AUROC_Score:<20}")
print(f"{'AUPRC Score':<40} {mean_WithTrueMask_AUPRC_Score:<20} {mean_WithMedSAMMask_AUPRC_Score:<20} {mean_WithDPTMask_AUPRC_Score:<20}")
print(f"{'Precision Score':<40} {mean_WithTrueMask_Precision_Score:<20} {mean_WithMedSAMMask_Precision_Score:<20} {mean_WithDPTMask_Precision_Score:<20}")
print(f"{'Recall Score':<40} {mean_WithTrueMask_Recall_Score:<20} {mean_WithMedSAMMask_Recall_Score:<20} {mean_WithDPTMask_Recall_Score:<20}")
print(f"{'Specificity Score':<40} {mean_WithTrueMask_Specificity_Score:<20} {mean_WithMedSAMMask_Specificity_Score:<20} {mean_WithDPTMask_Specificity_Score:<20}")
print(f"{'Jaccard Score':<40} {mean_WithTrueMask_Jaccard_Score:<20} {mean_WithMedSAMMask_Jaccard_Score:<20} {mean_WithDPTMask_Jaccard_Score:<20}")
print(f"{'FP Rate Score':<40} {mean_WithTrueMask_FP_rate_Score:<20} {mean_WithMedSAMMask_FP_rate_Score:<20} {mean_WithDPTMask_FP_rate_Score:<20}")