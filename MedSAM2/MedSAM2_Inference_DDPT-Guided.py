import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# Custom imports
import cfg
from conf import settings
import func_2d.function as function
from func_2d.dataset import *
from func_2d.utils import *

# Configure warnings
warnings.filterwarnings("ignore")

# Set CUDA blocking mode (for debugging)
CUDA_LAUNCH_BLOCKING = 1


dataset_name = f"BraTS20" ## Keep Dataset folder name here

file_list = os.listdir(f'../DATA/{dataset_name}/images/Unhealthy/')
directory = f'./DATA/{Dataset_name}/MedSAM2_Mask_with_DDPT_Prompt_Point/'
image_dir = f'../DATA/{dataset_name}/images/Unhealthy/'
mask_dir = f'./DATA/{Dataset_name}/DDPT_Mask/'

os.makedirs(directory, exist_ok=True)



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
    # print(y_indices, x_indices)
    if len(y_indices) == 0 or len(x_indices) == 0:  # If there are no non-zero elements
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # Add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 5))
    x_max = min(W, x_max + np.random.randint(0, 5))
    y_min = max(0, y_min - np.random.randint(0, 5))
    y_max = min(H, y_max + np.random.randint(0, 5))
    bbox = [x_min, y_min, x_max, y_max]
    
    return bbox

# Function to get bounding boxes for a batch
def get_bounding_boxes_for_batch(batch_masks):
    bounding_boxes = []
    for i in range(batch_masks.shape[0]):
        bbox = get_bounding_box(batch_masks[i])
        if bbox:
            bounding_boxes.append(bbox)
    return bounding_boxes

# use bfloat16 for the entire work
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)


args.pretrain = './MedSAM2_pretrain.pth'
args.net = 'sam2'
args.exp_name = 'REFUGE_MedSAM2'
args.vis = 1
args.sam_ckpt = './checkpoints/sam2_hiera_tiny.pt'
args.sam_config = 'sam2_hiera_t'
args.image_size = 1024
args.out_size = 1024
args.val_freq = 1

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = "none")

def load_checkpoint(model, checkpoint_path, device='cuda:0'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # print("Checkpoint Keys:", checkpoint.keys())
    
    # Load the state dict from the checkpoint
    checkpoint_dict = checkpoint['model']
    
    # Get the model's state dict
    model_dict = model.state_dict()
    
    # Count the total number of layers in the checkpoint and the model
    total_checkpoint_layers = len(checkpoint_dict)
    total_model_layers = len(model_dict)
    
    # Filter out incompatible layers from the checkpoint and only keep the matching ones
    compatible_checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    # Count the number of matching layers
    matching_layers = len(compatible_checkpoint_dict)
        
    # Update model's state dict with the compatible layers from the checkpoint
    model_dict.update(compatible_checkpoint_dict)
    
    # Load the updated state dict into the model
    model.load_state_dict(model_dict, strict=False)
    
    # print("Checkpoint loaded successfully!")
    return model

# Example usage:
net = load_checkpoint(net, './MedSAM2_pretrain.pth', device='cuda:0')

def preprocess_sample(image_path, mask_path, image_mode='L', mask_mode='L', target_size=(1024, 1024)):
    
    # Crop and resize
    def crop_and_resize(image, mask, target_size):
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        non_zero_rows = np.any(image_array, axis=1)
        non_zero_cols = np.any(image_array, axis=0)
        
        row_indices = np.where(non_zero_rows)[0]
        col_indices = np.where(non_zero_cols)[0]
        
        # print(mask_array.shape)
        if len(row_indices) > 0 and len(col_indices) > 0:
            cropped_image = image_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
            cropped_mask = mask_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
        else:
            cropped_image = image_array
            cropped_mask = mask_array
            
        # print(cropped_image)
        cropped_image = Image.fromarray(cropped_image).resize(target_size, Image.BILINEAR)
        cropped_mask = Image.fromarray(cropped_mask).resize(target_size, Image.BILINEAR)
        return cropped_image, cropped_mask

    image = Image.open(image_path).convert(image_mode)
    mask = Image.open(mask_path).convert(mask_mode).resize(image.size, Image.BILINEAR)

    cropped_image, cropped_mask = crop_and_resize(image, mask, target_size)
    
    image_tensor = torch.tensor(np.array(cropped_image), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dim
    mask_tensor = torch.tensor(np.array(cropped_mask), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return image_tensor, mask_tensor

class MedicalDataset(Dataset):
    def __init__(self, file_list, image_dir, mask_dir, transform=None):
        self.file_list = file_list
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Get filename from the list
        filename = self.file_list[idx]
        
        # Construct full paths for the image and depth mask
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        image_tensor, mask_tensor = preprocess_sample(image_path, mask_path)

        mask_tensor = torch.where(mask_tensor > 0.0, 1.0, 0.0)  # Threshold at 0.5

        return filename, image_tensor, mask_tensor

# Function to read file and store lines in a list
def read_file_to_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines


# Create the dataset instance
dataset = MedicalDataset(file_list=file_list, image_dir=image_dir, mask_dir=mask_dir)
# Create the DataLoader
train_loader = DataLoader(dataset, batch_size = 1, shuffle=True)
filenames_list = []
# Example: Iterate over the DataLoader and print the batch shape
for batch_idx, (filename, images, masks) in tqdm(enumerate(train_loader)):
    filename = filename
    image_tensor = images
    mask_tensor = masks

    image_tensor = image_tensor.to(torch.bfloat16)  # Convert input to BFloat16
    image_tensor = image_tensor.view(image_tensor.shape[0],1,1024,1024).cuda()  # Convert input to BFloat16
    image_tensor = image_tensor.repeat(1, 3, 1, 1)
    mask_tensor = mask_tensor.view(image_tensor.shape[0],1024,1024)#cuda()  # Convert input to BFloat16

    input_boxes = get_bounding_boxes_for_batch(mask_tensor)
    # Generate random points for each bounding box and store them in the desired format
    batch_random_points = []
    batch_random_points_labels = []
    
    # For each bounding box, generate points and append to the batch
    for bbox in input_boxes:
        points, labels = generate_random_points(bbox)  # Generate 10 points and labels for each bounding box
        batch_random_points.append([points])  # [batchsize, num_points, point_location, labels]
        batch_random_points_labels.append([labels])  # [batchsize, num_points, point_location, labels]
    
    coords_torch = torch.as_tensor(batch_random_points, dtype=torch.float, device=GPUdevice).squeeze(1)
    labels_torch = torch.as_tensor(batch_random_points_labels, dtype=torch.int, device=GPUdevice).squeeze(1)

    memory_bank_list = []
    feat_sizes = [(256, 256), (128,128), (64,64)]
    with torch.no_grad():
    
        """ image encoder """
        backbone_out = net.forward_image(image_tensor)
        _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
        B = vision_feats[-1].size(1) 
    
        """ memory condition """
        if len(memory_bank_list) == 0:
            vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
    
        else:
            for element in memory_bank_list:
                maskmem_features = element[0]
                maskmem_pos_enc = element[1]
                to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                to_cat_image_embed.append((element[3]).cuda(non_blocking=True)) # image_embed
                
            memory_stack_ori = torch.stack(to_cat_memory, dim=0)
            memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
            image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)
    
            vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, 64, 64) 
            vision_feats_temp = vision_feats_temp.reshape(B, -1)
    
            image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
            vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
            similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
    
            similarity_scores = F.softmax(similarity_scores, dim=1) 
            sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  # Shape [batch_size, 16]
    
            memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
            memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
    
            memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
            memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
    
    
    
            vision_feats[-1] = net.memory_attention(
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos_embeds[-1]],
                memory=memory,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=0
                )
        
        feats = [feat.permute(1, 2, 0).reshape(B, -1, *feat_size) 
                 for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
        
        image_embed = feats[-1]
        high_res_feats = feats[:-1]
    
        """ prompt encoder """
        points = (coords_torch, labels_torch)
    
        
        se, de = net.sam_prompt_encoder(
            points=points, 
            boxes=None,
            masks=None,
            batch_size=B,
        )    
              
        low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=net.sam_prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False, 
            repeat_image=False,  
            high_res_features = high_res_feats
        )
    
        # prediction
        pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                        mode="bilinear", align_corners=False)
        
    for j in range(image_tensor.shape[0]):
        image_sample = image_tensor[j, 0].to(torch.float32).cpu().numpy()  # (1024, 1024)
        mask_sample = mask_tensor[j].to(torch.float32).cpu().numpy()    # (1024, 1024)
    
        pred_sample_sig = torch.sigmoid(high_res_multimasks[j, 0])
        pred_sample_norm = pred_sample_sig / pred_sample_sig.max()
        pred_sample_binary = torch.where(pred_sample_norm > 0.0, 1.0, 0.0).squeeze(0)
        pred_sample_binary = 1 - pred_sample_binary

        numpy_array = pred_sample_binary.to(torch.float32).cpu().numpy()
        image = Image.fromarray(numpy_array.astype(np.uint8)*255)
        image.save(os.path.join(directory, filename[j]))
