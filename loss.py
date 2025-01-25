from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import os
import numpy as np
import torch
from PIL import Image
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
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import math
import time

import warnings
warnings.filterwarnings("ignore")



def calculate_tversky_index(inputs, targets, delta, lambda_reg=0.01):
    # Flatten predictions and target masks
    preds_flat = inputs.view(-1)
    target_masks_flat = targets.view(-1)
    
    # Calculate true positives (TP), false negatives (FN), and false positives (FP)
    TP = torch.sum(preds_flat * target_masks_flat)
    FN = torch.sum((1 - preds_flat) * target_masks_flat)
    FP = torch.sum(preds_flat * (1 - target_masks_flat))
    
    # Tversky index
    tversky_index = TP / (TP + delta * FN + (1 - delta) * FP + 1e-6)
    
    return tversky_index



def compute_average_ELTversky_loss(preds, target_masks_resized):
    batch_size = preds.size(0)
    total_dice = 0.0
    delta = 0.4
    
    for i in range(batch_size):
        dice = calculate_tversky_index(preds[i], target_masks_resized[i], delta)
        dice = torch.clamp(torch.pow(-torch.log(dice), 0.3), 0, 2)
        total_dice += dice
        
    avg_dice = total_dice / batch_size
    return avg_dice



def dice_coefficient(preds, target_masks_resized, epsilon=1e-7):
    # Flatten predictions and target masks
    preds_flat = preds.view(-1)
    target_masks_flat = target_masks_resized.view(-1)
    # preds_flat = torch.where(preds_flat > 0.5, 1.0, 0.0)
    
    # Intersection and union
    intersection = torch.sum(preds_flat * target_masks_flat)
    union = torch.sum(preds_flat) + torch.sum(target_masks_flat)
    
    # Calculate Dice coefficient
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice_score

def dice_coefficient1(preds, target_masks_resized, epsilon=1e-7):
    # Flatten predictions and target masks
    preds_flat = preds.view(-1)
    target_masks_flat = target_masks_resized.view(-1)
    preds_flat = torch.where(preds_flat > 0.5, 1.0, 0.0)
    
    # Intersection and union
    intersection = torch.sum(preds_flat * target_masks_flat)
    union = torch.sum(preds_flat) + torch.sum(target_masks_flat)
    
    # Calculate Dice coefficient
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice_score
    
def compute_average_dice(preds, target_masks_resized):
    batch_size = preds.size(0)
    total_dice = 0.0
    
    for i in range(batch_size):
        dice = dice_coefficient1(preds[i], target_masks_resized[i])
        total_dice += dice
        
    avg_dice = total_dice / batch_size
    return avg_dice

def compute_average_ELdice_loss(preds, target_masks_resized):
    batch_size = preds.size(0)
    total_dice = 0.0
    
    for i in range(batch_size):
        dice = dice_coefficient(preds[i], target_masks_resized[i])
        dice = torch.clamp(torch.pow(-torch.log(dice), 0.3), 0, 2)
        total_dice += dice
        
    avg_dice = total_dice / batch_size
    return avg_dice


def compute_weight_map(mask, sigma=10):
    mask_np = mask.cpu().numpy()
    weight_map = []
    for m in mask_np:
        smoothed_mask = gaussian_filter(m[0], sigma=sigma)
        normalized_weights = smoothed_mask / smoothed_mask.max()
        weight_map.append(normalized_weights)
    weight_map = torch.tensor(weight_map, dtype=torch.float32).to(mask.device)
    return weight_map.unsqueeze(1)


# Compute Weighted EL-Dice Loss
def combined_weighted_compute_average_ELdice_loss(preds, target_masks_MedSAM, target_masks_DPT, delta=0.5):
    batch_size = preds.size(0)
    total_loss = 0.0
    sigma=10
    
    bce_loss_fn = nn.BCELoss()  # Binary cross-entropy loss

    for i in range(batch_size):
        # Compute weight map for the current target mask
        weight_map_MedSAM = compute_weight_map(target_masks_SAM[i:i+1], sigma=sigma).squeeze()
        weight_map_MedSAM = weight_map_MedSAM.detach().unsqueeze(0)

        weight_map_DPT = compute_weight_map(target_masks_DPT[i:i+1], sigma=sigma).squeeze()
        weight_map_DPT = weight_map_DPT.detach().unsqueeze(0)
        
        weight_map_DPT_FP = compute_weight_map(1 - target_masks_DPT[i:i+1], sigma=15).squeeze()
        weight_map_DPT_FP = weight_map_DPT_FP.detach().unsqueeze(0)

        # Compute Dice coefficient
        dice_MedSAM = dice_coefficient(preds[i], target_masks_MedSAM[i]*(1.0 - weight_map_MedSAM))
        el_dice_MedSAM = torch.clamp(torch.pow(-torch.log(dice_MedSAM), 0.3), 0, 2)

        dice_DPT = dice_coefficient(preds[i], target_masks_DPT[i]*weight_map_DPT)
        el_dice_DPT = torch.clamp(torch.pow(-torch.log(dice_DPT), 0.3), 0, 2)

        dice_DPT_FP = dice_coefficient((1 - preds[i]) * (1 - target_masks_DPT[i]), (1 - target_masks_DPT[i]) * weight_map_DPT_FP)
        el_dice_DPT_FP = torch.clamp(torch.pow(-torch.log(dice_DPT_FP), 0.3), 0, 2)

        D_factor = dice_coefficient(target_masks_MedSAM[i], target_masks_DPT[i])
        
        # Compute false positives
        fp_DPT = preds[i] * (1 - target_masks_DPT[i])
        fp_penalty_DPT = fp_DPT.sum() / (target_masks_DPT[i].numel())

        if math.isnan(el_dice_MedSAM):
            total_loss = total_loss + el_dice_DPT + 0.6 * el_dice_DPT_FP + 0.2 * fp_penalty_DPT
        else:
            total_loss = total_loss + (D_factor * el_dice_MedSAM) + el_dice_DPT + 0.6 * el_dice_DPT_FP  + 0.6 * fp_penalty_DPT 
    
    avg_loss = total_loss / batch_size
    return avg_loss

# Compute Weighted EL-Tversky Loss
def combined_weighted_compute_average_ELT_loss(preds, target_masks_MedSAM, target_masks_DPT, delta=0.5):
    batch_size = preds.size(0)
    total_loss = 0.0
    sigma=10
    delta = 0.4

    for i in range(batch_size):
        # Compute weight map for the current target mask
        weight_map_MedSAM = compute_weight_map(target_masks_MedSAM[i:i+1], sigma=sigma).squeeze()
        weight_map_MedSAM = weight_map_MedSAM.detach().unsqueeze(0)

        weight_map_DPT = compute_weight_map(target_masks_DPT[i:i+1], sigma=sigma).squeeze()
        weight_map_DPT = weight_map_DPT.detach().unsqueeze(0)

        weight_map_DPT_FP = compute_weight_map(1 - target_masks_DPT[i:i+1], sigma=15).squeeze()
        weight_map_DPT_FP = weight_map_DPT_FP.detach().unsqueeze(0)

        # Compute Dice coefficient
        dice_MedSAM = calculate_tversky_index(preds[i], target_masks_MedSAM[i] * (1.0 - weight_map_MedSAM), 0.3)
        el_dice_MedSAM = torch.clamp(torch.pow(-torch.log(dice_MedSAM), 0.3), 0, 2)

        dice_DPT = calculate_tversky_index(preds[i], target_masks_DPT[i] * weight_map_DPT, 0.3)
        el_dice_DPT = torch.clamp(torch.pow(-torch.log(dice_DPT), 0.3), 0, 2)

        dice_DPT_FP = dice_coefficient((1 - preds[i]) * (1 - target_masks_DPT[i]), (1 - target_masks_DPT[i]) * weight_map_DPT_FP)
        el_dice_DPT_FP = torch.clamp(torch.pow(-torch.log(dice_DPT_FP), 0.3), 0, 2)

        D_factor = dice_coefficient(target_masks_MedSAM[i], target_masks_DPT[i])

        # Compute false positives
        fp_DPT = preds[i] * (1 - target_masks_DPT[i])  # Pixels predicted as positive but are negative
        fp_penalty_DPT = fp_DPT.sum() / (target_masks_DPT[i].numel())  # Normalize by total pixels
        
        if math.isnan(el_dice_MedSAM):
            total_loss = total_loss + el_dice_DPT + 0.6 * el_dice_DPT_FP + 0.6 * fp_penalty_DPT
        else:
            total_loss = total_loss + el_dice_DPT + 0.6 * el_dice_DPT_FP + 0.6 * fp_penalty_DPT + (D_factor * el_dice_MedSAM)
    
    avg_loss = total_loss / batch_size

    return avg_loss

class ActivePointEmbeddingLoss_MSE(nn.Module):
    def __init__(self, reduction: str = 'mean', alpha: float = 1.0, beta: float = 1.0):
        super(ActivePointEmbeddingLoss_MSE, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'"
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta

    def forward(self, point_sparse_embedding: torch.Tensor, target_point_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, embedding_dim = point_sparse_embedding.shape
        assert target_point_mask.shape == (batch_size, num_points), "Mask shape mismatch."
        
        active_label = torch.ones(embedding_dim, device=point_sparse_embedding.device)  # [+1] for active points
        inactive_label = -torch.ones(embedding_dim, device=point_sparse_embedding.device)  # [-1] for inactive points
        
        losses = []
        
        for b in range(batch_size):
            embeddings = point_sparse_embedding[b]  # Shape: (num_points, embedding_dim)
            mask = target_point_mask[b]  # Shape: (num_points)
            
            # Extract active and inactive embeddings
            active_embeddings = embeddings[mask == 1]  # Shape: (num_active_points, embedding_dim)
            inactive_embeddings = embeddings[mask == 0]  # Shape: (num_inactive_points, embedding_dim)

            active_loss = torch.tensor(0.0, device=point_sparse_embedding.device)
            inactive_loss = torch.tensor(0.0, device=point_sparse_embedding.device)

            # Active point loss (force embeddings to be close to +1)
            if active_embeddings.shape[0] > 0:
                active_loss = F.mse_loss(active_embeddings, active_label.unsqueeze(0).repeat(active_embeddings.shape[0], 1))

            # Inactive point loss (force embeddings to be close to -1)
            if inactive_embeddings.shape[0] > 0:
                inactive_loss = F.mse_loss(inactive_embeddings, inactive_label.unsqueeze(0).repeat(inactive_embeddings.shape[0], 1))

            # Combine the losses with weights alpha and beta
            total_loss = self.alpha * active_loss + self.beta * inactive_loss
            losses.append(total_loss)

        # Stack losses and apply reduction
        losses = torch.stack(losses)  # Shape: (batch_size,)
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses  # No reduction

    
# Function to read file and store lines in a list
def read_file_to_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines

# Helper function to create batches
def create_batches(files, batch_size):
    for i in range(0, len(files), batch_size):
        yield files[i:i + batch_size]

# Normalize cosine similarity to be between 0 and 1
def normalize_sim(sim):
    return (sim - sim.min()) / (sim.max() - sim.min())