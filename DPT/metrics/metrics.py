# Standard library imports
import scipy

# Third-party library imports
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F



def dice(predicted_mask, target_mask,threshold):
  
  thresholded_mask = torch.where(predicted_mask > threshold, torch.tensor(1),torch.tensor(0))
  
  intersection = torch.sum(thresholded_mask * target_mask)
  union = torch.sum(thresholded_mask) + torch.sum(target_mask)
  dice = (2.0 * intersection) / (union + 1e-8)  # Adding epsilon to avoid division by zero
  return dice

def iou(predicted_mask, target_mask,threshold):

  thresholded_mask = torch.where(predicted_mask > threshold, torch.tensor(1),torch.tensor(0))
  intersection = torch.sum(thresholded_mask * target_mask)
  union = torch.sum(predicted_mask) + torch.sum(target_mask) - intersection
  iou = (intersection) / (union + 1e-8)  # Adding epsilon to avoid division by zero
  return iou


def auprc(predicted_mask, target_mask,threshold):

  thresholded_mask = torch.where(predicted_mask > threshold, torch.tensor(1),torch.tensor(0))
  threshold_mask_flat = thresholded_mask.flatten()
  target_mask_flat = target_mask.flatten()
  
  # Calculate precision-recall curve
  precision, recall, _ = precision_recall_curve(target_mask_flat, threshold_mask_flat)

  # Calculate area under the precision-recall curve
  auprc = auc(recall, precision)
  return auprc
  
def f1(predicted_mask, target_mask,threshold):
  thresholded_mask = torch.where(predicted_mask > threshold, torch.tensor(1),torch.tensor(0))
  f1 = f1_score(thresholded_mask.flatten(), target_mask.flatten())
  precision = precision_score(thresholded_mask.flatten(), target_mask.flatten())
  recall = recall_score(thresholded_mask.flatten(), target_mask.flatten())
  
  return f1,precision,recall

def apply_2d_median_filter(pred, kernelsize=5): 
    pred = scipy.ndimage.median_filter(pred, size=kernelsize)
    return pred
  


#THIS FUNCTION EXTRACTS THE ATTENTION AND THE MASK TENSORS, AND THEN PASSES IT THROUGH THE DICE FUNCTION TO GET THE FINAL VALUE FOR THE 2 TENSORS
def eval_metrics(file_name,folder,ground_threshold,median=None):

  #file_name=impath.split('/')[-1].split('.')[0]
  
  transform = transforms.ToTensor()
  
  #mask_file=f'inference/{folder}/mask/{file_name}.png'
  mask_file=f'inference/{folder}/mask/{file_name}'
   
  mask = Image.open(mask_file)
  mask = mask.convert("L")
  mask = transform(mask)    
  mask = F.interpolate(mask.unsqueeze(0), size=(224), mode='bilinear', align_corners=False)[0]
  mask = torch.where(mask > 0.0, torch.tensor(1),torch.tensor(0))  
  #DO THIS ONLY FOR THE BRATS21 WHERE THE IMG IS NOT OF 1 CHANNEL
  

  
  #pred_file=f'inference/{folder}/pred/{file_name}.png'
  pred_file=f'inference/{folder}/pred/{file_name}'
  mask_pred = Image.open(pred_file)
  if(median):
    mask_pred=apply_2d_median_filter(mask_pred)
  
  mask_pred = transform(mask_pred)
  
  
    
  
  #APPLYING THE THRESHOLD ON THE GROUND TRUTH MASKS
  
  if (ground_threshold==0):                   #ZERO THRESHOLD CASE
    if(torch.sum(mask)==ground_threshold):
      #print('PRINTING PREDICTING MASK SCORE', torch.sum(mask_pred))
      return False
  else:                                       #200, 400 THRESHOLD CASE
    if(torch.sum(mask)<ground_threshold):
      #print('PRINTING PREDICTING MASK SCORE', torch.sum(mask_pred))
      return False
  

  
  
  thresholds=[0.4,0.5,0.6]
  metrics = {
            'dice': {'0.4': 0, '0.5': 0, '0.6': 0},
            'iou': {'0.4': 0, '0.5': 0, '0.6': 0},
            'auprc': {'0.4': 0, '0.5': 0, '0.6': 0},
            'f1': {'0.4': 0, '0.5': 0, '0.6': 0},
            'recall': {'0.4': 0, '0.5': 0, '0.6': 0},
            'precision': {'0.4': 0, '0.5': 0, '0.6': 0},
        }
  
  
  for i in thresholds:
  
    
    dice_score=dice(mask_pred,mask,i)
    iou_score=iou(mask_pred,mask,i)
    auprc_score=auprc(mask_pred,mask,i)
    
    f1_score,precision_score,recall_score=f1(mask_pred,mask,i)
  
  
    metrics['dice'][f'{i}']=dice_score.item()
    metrics['iou'][f'{i}']=iou_score.item()
    metrics['auprc'][f'{i}']=auprc_score
    metrics['f1'][f'{i}']=f1_score
    metrics['precision'][f'{i}']=precision_score
    metrics['recall'][f'{i}']=recall_score
  
  return metrics
  