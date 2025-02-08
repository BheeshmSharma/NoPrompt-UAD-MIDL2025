import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
import cv2

# Open an image file

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

def dice(predicted_mask, target_mask,threshold):
  
  thresholded_mask = torch.where(predicted_mask > threshold, torch.tensor(1),torch.tensor(0))
  
  
  #print('THRESHOLD',thresholded_mask.shape,np.sum(thresholded_mask))
  #print('TARGET',target_mask.shape,np.sum(target_mask))
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
  #thresholded_mask=predicted_mask
  threshold_mask_flat = thresholded_mask.flatten()
  target_mask_flat = target_mask.flatten()
  
  #print('thresh',threshold_mask_flat.shape)
  #print('target',target_mask_flat.shape)
  
  #basically a vector of size [50176]
  
  

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
  


#THIS FUNCTION EXTRACTS THE ATTENTION AND THE MASK TENSORS, AND THEN PASSES IT THROUGH THE DICE FUNCTION TO GET THE FINAL VALUE FOR THE 2 TENSORS
def eval_metrics(impath,folder,ground_threshold):

  #file_name=impath.split('/')[-1].split('.')[0]
  file_name=impath.split('/')[-1]
  
  transform = transforms.ToTensor()
  
  #mask_file=f'inference/{folder}/mask/{file_name}.png'
  mask_file=f'inference/{folder}/mask/{file_name}'
  mask = Image.open(mask_file)
  
  #DO THIS ONLY FOR THE BRATS21 WHERE THE IMG IS NOT OF 1 CHANNEL
  #mask = mask.convert("L")
  
  
  mask = transform(mask)
  
  
  
  mask = F.interpolate(mask.unsqueeze(0), size=(224), mode='bilinear', align_corners=False)[0]
  mask = torch.where(mask > 0.0, torch.tensor(1),torch.tensor(0))

  
  #pred_file=f'inference/{folder}/pred/{file_name}.png'
  pred_file=f'inference/{folder}/pred/{file_name}'
  mask_pred = Image.open(pred_file)
  mask_pred = transform(mask_pred)

  
  
  
  if (ground_threshold==0):                   #ZERO THRESHOLD CASE
    if(torch.sum(mask)==ground_threshold):
      return False
  else:                                       #200, 400 THRESHOLD CASE
    if(torch.sum(mask)<ground_threshold):
      return False
  

  
  
  thresholds=[0.4,0.5,0.6]
  metrics={}
  
  
  for i in thresholds:
  
    
    dice_score=dice(mask_pred,mask,i)
    iou_score=iou(mask_pred,mask,i)
    auprc_score=auprc(mask_pred,mask,i)
    
    f1_score,precision_score,recall_score=f1(mask_pred,mask,i)
  
  
    metrics[f'dice-{i}']=dice_score.item()
    metrics[f'iou-{i}']=iou_score.item()
    metrics[f'auprc-{i}']=auprc_score
    metrics[f'f1-{i}']=f1_score
    metrics[f'precision-{i}']=precision_score
    metrics[f'recall-{i}']=recall_score
  
  
  return metrics
  