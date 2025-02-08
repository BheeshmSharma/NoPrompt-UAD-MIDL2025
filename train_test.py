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
import config
import sys
from Modules import *
from loss import *
import time
import torch.nn.functional as F
import math


import warnings
warnings.filterwarnings("ignore")

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

    print("Using the following configuration:")
    print(args)

    # Define the condition
    use_seed = args.seed  # Change this to False if you don't want to set a seed
    
    if use_seed:
        seed_value = 42
        random.seed(seed_value)
        np.random.seed(seed_value)  # For NumPy operations
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        # torch.cuda.manual_seed_all(seed_value)  # If using multiple GPUs

    '''Data Loading and path setup'''
    ''':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::'''
    Save_path = "./Experiments/" + str(args.dataset_name) + '_' + str(args.folder_name) + "/"
    
    tumor_image_folder_path =  './DATA/' + args.dataset_name +'/Frames/'
    ddpt_mask_folder_path = './DATA/DDPT_inference/' + args.dataset_name + '/'
    MedSAM_mask_folder_path = './DATA/MedSAM_inference/' + args.dataset_name + '/'
    mask_folder_path = './DATA/' + args.dataset_name + '/GT_Mask/'
    
    tumor_image_files_org_train = read_file_to_list('./DATA/' + args.dataset_name +'/train_file_names.txt')
    tumor_image_files_org_test = read_file_to_list('./DATA/' + args.dataset_name +'/val_file_names.txt')
    
    # Setup device
    device = args.device
    '''For Each Fold Model Initialization'''
    ''':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::'''
    # Ensure the folds how many need to be trained
    for fold in range(args.fold[0], args.fold[1]):
        Encoder_Model = Image_Encoder().to(device)
        Decoder_Model = Mask_Decoder().to(device)

        if args.pre_training == False:
            Encoder_Model.apply(initialize_weights)
            Decoder_Model.apply(initialize_weights)
        else:
            checkpoint_path = "./Experiments/" + str(args.dataset_name) + '_' + str(args.folder_name) + "/Fold_0/"
            Encoder_Model.load_state_dict(torch.load(checkpoint_path + "Encoder_Model.pth", map_location=device), strict=False)
            Decoder_Model.load_state_dict(torch.load(checkpoint_path + "Decoder_Model.pth", map_location=device), strict=False)
        
        # Directory to save model
        save_path_f =  Save_path + "Fold_" + str(fold) + '/'
        save_path_f_latest =  Save_path + "Fold_" + str(fold) + '/latest/'
        
        # Create the directory if it does not exist
        if not os.path.exists(save_path_f):
            os.makedirs(save_path_f)
            
        if not os.path.exists(save_path_f_latest):
            os.makedirs(save_path_f_latest)

        
        '''Setup logging'''
        logging.basicConfig(filename= save_path_f + '/training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
        
        # Get the root logger and clear existing handlers
        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Add a new handler
        handler = logging.FileHandler(filename= save_path_f + '/training_log.txt')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Log the parameters
        logging.info(f"Training Setup===:")
        logging.info(f"Device: {args.device}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}, Number of epochs: {args.num_epochs}, Loss function: {args.loss_function}, Dataset name: {args.dataset_name}")
        
        # Split into training and validation sets
        tumor_train_files, tumor_val_files = train_test_split(tumor_image_files_org_train, test_size=0.1, random_state=42)
        tumor_test_files = tumor_image_files_org_test

        # Optimizer
        params = list(Encoder_Model.parameters()) + list(Decoder_Model.parameters())
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params), lr=args.learning_rate, momentum=0.9)

        warmup_scheduler = WarmUpLR(optimizer, warmup_steps=1, base_lr=args.learning_rate, max_lr=1.0)

        BCM = Boundary_coordinate_and_Mask(grid_size=32)
        
        Candidate_Prompt_Embedding = torch.load('./Fixed_Candidate_Embeddings/Candidate_Prompt_Embedding' + str(i) + '.pt').to(device)
        Point_Embedding_Loss = ActivePointEmbeddingLoss_MSE()
        
        # Training loop
        best = 0.0
        gamma = 0
        for epoch in range(args.num_epochs):
            Encoder_Model.train()
            Decoder_Model.train()
            
            running_loss_Seg_Mask = 0.0
            running_loss_Point_Act = 0.0
            running_loss_Emd = 0.0
            running_Attention_Dice_Score = 0.0
            running_Decoder_Dice_Score = 0.0
            running_WithTrueMask_Decoder_Dice_Score = 0.0
            count = 0
            num_sample_epoch = 0

            # Shuffle data at the start of each epoch
            np.random.shuffle(tumor_train_files)

            # Process in batches
            for tumor_batch in tqdm(create_batches(tumor_train_files, args.batch_size), total=len(tumor_train_files) // args.batch_size):
                tumor_images, target_masks, MedSAM_Prediction_Mask, target_masks_256, MedSAM_Prediction_Mask_256 = load_and_convert_to_tensor(tumor_image_folder_path, ddpt_mask_folder_path, MedSAM_mask_folder_path, tumor_batch)
                tumor_images, target_masks, MedSAM_Prediction_Mask, target_masks_256, MedSAM_Prediction_Mask_256 = tumor_images.to(device), target_masks.to(device), MedSAM_Prediction_Mask.to(device), target_masks_256.to(device), MedSAM_Prediction_Mask_256.to(device)
                bs = tumor_images.shape[0]
                Candidate_Prompt_Embedding_batch = Candidate_Prompt_Embedding.repeat(bs, 1, 1)

                if target_masks.dim() == 1:
                    continue
                    
                # First Forward Pas
                optimizer.zero_grad()
                start_time = time.time()

                # Create mask for the points which are inside the bounding box using DDPT Mask
                box_point_mask = BCM.create_masks(target_masks).to(device).float()
                box_point_mask = box_point_mask.view(bs, -1)
                
                # Point based prompt embedding from MedSAM Prompt Encoder
                Candidate_Prompt_Embedding_batch = Candidate_Prompt_Embedding_batch.detach()

                # Pass images and point embedding of grid point from encoder
                candidate_spatial_embedding_activations, candidate_spatial_embedding, img_emd = Encoder_Model(tumor_images, Candidate_Prompt_Embedding_batch)

                ## Mask Decoder 
                predicted_mask = Decoder_Model(img_emd, candidate_spatial_embedding)
                predicted_mask = torch.sigmoid(predicted_mask)
                candidate_spatial_embedding_activations = candidate_spatial_embedding_activations.squeeze(2)

                # Loss Computation
                Point_Activation_loss = compute_average_ELdice_loss(candidate_spatial_embedding_activations, box_point_mask)
                Emd_loss = Point_Embedding_Loss(candidate_spatial_embedding, box_point_mask)
                Segmentation_loss = combined_weighted_compute_average_ELT_loss(predicted_mask, MedSAM_Prediction_Mask_256, target_masks_256)
                    
                Attention_Dice_sum = compute_average_dice(candidate_spatial_embedding_activations, box_point_mask) * bs
                Decoder_Dice_sum = compute_average_dice(predicted_mask, MedSAM_Prediction_Mask_256) * bs
                WithTrueMask_Decoder_Dice_sum = compute_average_dice(predicted_mask, target_masks_256) * bs

                num_sample_epoch += bs       

                # Total Loss
                total_loss =  Segmentation_loss + Emd_loss + Point_Activation_loss
                total_loss.backward()
                optimizer.step()
                running_loss_Seg_Mask += Segmentation_loss.item()
                running_loss_Point_Act += Point_Activation_loss.item()
                running_loss_Emd += Emd_loss.item()
                running_Attention_Dice_Score += Attention_Dice_sum.item()
                running_Decoder_Dice_Score += Decoder_Dice_sum.item()
                running_WithTrueMask_Decoder_Dice_Score += WithTrueMask_Decoder_Dice_sum.item()
                count = count + 1
                
            # Validation
            Encoder_Model.eval()
            Decoder_Model.eval()
            
            val_running_loss_Seg_Mask = 0.0
            val_running_loss_Point_Act = 0.0
            val_running_loss_Emd = 0.0
            val_running_Attention_Dice_Score = 0.0
            val_running_Decoder_Dice_Score = 0.0 
            val_running_WithTrueMask_Decoder_Dice_Score = 0.0
            num_sample_val_epoch = 0.0

            
            with torch.no_grad():
                for tumor_batch in tqdm(create_batches(tumor_val_files, args.batch_size), total=len(tumor_val_files) // args.batch_size):
                    # Batch processing
                    tumor_images, target_masks, MedSAM_Prediction_Mask, target_masks_256, MedSAM_Prediction_Mask_256 = load_and_convert_to_tensor(tumor_image_folder_path, mask_folder_path, MedSAM_mask_folder_path, tumor_batch)
                    tumor_images, target_masks, MedSAM_Prediction_Mask, target_masks_256, MedSAM_Prediction_Mask_256 = tumor_images.to(device), target_masks.to(device), MedSAM_Prediction_Mask.to(device), target_masks_256.to(device), MedSAM_Prediction_Mask_256.to(device)
                    bs = tumor_images.shape[0]
                    Candidate_Prompt_Embedding_batch = Candidate_Prompt_Embedding.repeat(bs, 1, 1)
                    
                    if target_masks.dim() == 1:
                        continue
                    
                    # Create mask for the points which are inside the bounding box using DDPT Mask
                    box_point_mask = BCM.create_masks(target_masks).to(device).float()
                    box_point_mask = box_point_mask.view(bs, -1)
    
                    # Point based prompt embedding from MedSAM Prompt Encoder
                    Candidate_Prompt_Embedding_batch = Candidate_Prompt_Embedding_batch.detach()
                    
                    # Pass images and point embedding of grid point from encoder
                    candidate_spatial_embedding_activations, candidate_spatial_embedding, img_emd = Encoder_Model(tumor_images, Candidate_Prompt_Embedding_batch)
    
                    ## Mask Decoder 
                    predicted_mask = Decoder_Model(img_emd, candidate_spatial_embedding)
                    predicted_mask = torch.sigmoid(predicted_mask)
                    candidate_spatial_embedding_activations = candidate_spatial_embedding_activations.squeeze(2)

                    # Loss Computation
                    Point_Activation_loss = compute_average_ELdice_loss(candidate_spatial_embedding_activations, box_point_mask)
                    Emd_loss = Point_Embedding_Loss(candidate_spatial_embedding, box_point_mask)
                    Segmentation_loss = combined_weighted_compute_average_ELT_loss(predicted_mask, MedSAM_Prediction_Mask_256, target_masks_256)
                            
                    Attention_Dice_sum = compute_average_dice(candidate_spatial_embedding_activations, box_point_mask) * bs
                    Decoder_Dice_sum = compute_average_dice(predicted_mask, MedSAM_Prediction_Mask_256) * bs
                    WithTrueMask_Decoder_Dice_sum = compute_average_dice(predicted_mask, target_masks_256) * bs

                    num_sample_val_epoch += bs

                    val_running_loss_Seg_Mask += Segmentation_loss.item()
                    val_running_loss_Point_Act += Point_Activation_loss.item()
                    val_running_loss_Emd += Emd_loss.item()
                    val_running_Attention_Dice_Score += Attention_Dice_sum.item()
                    val_running_Decoder_Dice_Score += Decoder_Dice_sum.item()
                    val_running_WithTrueMask_Decoder_Dice_Score += WithTrueMask_Decoder_Dice_sum.item()

            # Validation unseen Data
            test_running_loss_Seg_Mask = 0.0
            test_running_loss_Point_Act = 0.0
            test_running_loss_Emd = 0.0
            test_running_Attention_Dice_Score = 0.0
            test_running_Decoder_Dice_Score = 0.0 
            test_running_WithTrueMask_Decoder_Dice_Score = 0.0
            num_sample_test_epoch = 0.0

            
            with torch.no_grad():
                for tumor_batch in tqdm(create_batches(tumor_test_files, args.batch_size), total=len(tumor_test_files) // args.batch_size):
                    # Batch processing
                    tumor_images, target_masks, MedSAM_Prediction_Mask, target_masks_256, MedSAM_Prediction_Mask_256 = load_and_convert_to_tensor(tumor_image_folder_path, mask_folder_path, MedSAM_mask_folder_path, tumor_batch)
                    tumor_images, target_masks, MedSAM_Prediction_Mask, target_masks_256, MedSAM_Prediction_Mask_256 = tumor_images.to(device), target_masks.to(device), MedSAM_Prediction_Mask.to(device), target_masks_256.to(device), MedSAM_Prediction_Mask_256.to(device)
                    bs = tumor_images.shape[0]
                    Candidate_Prompt_Embedding_batch = Candidate_Prompt_Embedding.repeat(bs, 1, 1)
                    
                    if target_masks.dim() == 1:
                        continue
                    
                    # Create mask for the points which are inside the bounding box using DDPT Mask
                    box_point_mask = BCM.create_masks(target_masks).to(device).float()
                    box_point_mask = box_point_mask.view(bs, -1)
                        
                    # Point based prompt embedding from MedSAM Prompt Encoder
                    Candidate_Prompt_Embedding_batch = Candidate_Prompt_Embedding_batchCandidate_Prompt_Embedding_batch.detach()
                    
                    # Pass images and point embedding of grid point from encoder
                    candidate_spatial_embedding_activations, candidate_spatial_embedding, img_emd = Encoder_Model(tumor_images, Candidate_Prompt_Embedding_batch)
    
                    ## Mask Decoder 
                    predicted_mask = Decoder_Model(img_emd, candidate_spatial_embedding)
                    predicted_mask = torch.sigmoid(predicted_mask)
                    candidate_spatial_embedding_activations = candidate_spatial_embedding_activations.squeeze(2)

                    # Loss Computation
                    Point_Activation_loss = compute_average_ELdice_loss(candidate_spatial_embedding_activations, box_point_mask)
                    Emd_loss = Point_Embedding_Loss(candidate_spatial_embedding, box_point_mask)
                    Segmentation_loss = combined_weighted_compute_average_ELT_loss(predicted_mask, MedSAM_Prediction_Mask_256, target_masks_256)
                            
                    Attention_Dice_sum = compute_average_dice(candidate_spatial_embedding_activations, box_point_mask) * bs
                    Decoder_Dice_sum = compute_average_dice(predicted_mask, MedSAM_Prediction_Mask_256) * bs
                    WithTrueMask_Decoder_Dice_sum = compute_average_dice(predicted_mask, target_masks_256) * bs

                    num_sample_test_epoch += bs

                    test_running_loss_Seg_Mask += Segmentation_loss.item()
                    test_running_loss_Point_Act += Point_Activation_loss.item()
                    test_running_loss_Emd += Emd_loss.item()
                    test_running_Attention_Dice_Score += Attention_Dice_sum.item()
                    test_running_Decoder_Dice_Score += Decoder_Dice_sum.item()
                    test_running_WithTrueMask_Decoder_Dice_Score += WithTrueMask_Decoder_Dice_sum.item()

            mean_val_running_loss_Seg_Mask = val_running_loss_Seg_Mask / num_sample_val_epoch
            mean_val_running_loss_Point_Act = val_running_loss_Point_Act / num_sample_val_epoch
            mean_val_running_loss_Emd = val_running_loss_Emd / num_sample_val_epoch
            mean_val_running_Attention_Dice_Score = val_running_Attention_Dice_Score / num_sample_val_epoch
            mean_val_running_Decoder_Dice_Score = val_running_Decoder_Dice_Score / num_sample_val_epoch
            mean_val_running_WithTrueMask_Decoder_Dice_Score = val_running_WithTrueMask_Decoder_Dice_Score / num_sample_val_epoch

            mean_test_running_loss_Seg_Mask = test_running_loss_Seg_Mask / num_sample_test_epoch
            mean_test_running_loss_Point_Act = test_running_loss_Point_Act / num_sample_test_epoch
            mean_test_running_loss_Emd = test_running_loss_Emd / num_sample_test_epoch
            mean_test_running_Attention_Dice_Score = test_running_Attention_Dice_Score / num_sample_test_epoch
            mean_test_running_Decoder_Dice_Score = test_running_Decoder_Dice_Score / num_sample_test_epoch
            mean_test_running_WithTrueMask_Decoder_Dice_Score = test_running_WithTrueMask_Decoder_Dice_Score / num_sample_test_epoch

            mean_train_running_loss_Seg_Mask = running_loss_Seg_Mask / num_sample_epoch
            mean_train_running_loss_Point_Act = running_loss_Point_Act / num_sample_epoch
            mean_train_running_loss_Emd = running_loss_Emd / num_sample_epoch
            mean_train_running_Attention_Dice_Score = running_Attention_Dice_Score / num_sample_epoch
            mean_train_running_Decoder_Dice_Score = running_Decoder_Dice_Score / num_sample_epoch
            mean_train_running_WithTrueMask_Decoder_Dice_Score = running_WithTrueMask_Decoder_Dice_Score / num_sample_epoch

            # Step the scheduler
            warmup_scheduler.step()
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Learning Rate: {warmup_scheduler.get_lr()[0]:.10f}")
            
            print(f"""
            Epoch [{epoch + 1}/{args.num_epochs}] |||| Learning Rate {warmup_scheduler.get_lr()[0]:.10f} |||| Folder Name {args.folder_name}|||| Datset {args.dataset_name}
            ----------------------------------------------------------------------------------------------------------
            {"Metric":<40}{"Training":<25}{"Validation":<25}{"Test":<25}
            ----------------------------------------------------------------------------------------------------------
            {"Number of Samples:":<40}{num_sample_epoch:<25}{num_sample_val_epoch:<25}{num_sample_test_epoch:<25}
            {"Segmentation Loss:":<40}{mean_train_running_loss_Seg_Mask:<25}{mean_val_running_loss_Seg_Mask:<25}{mean_test_running_loss_Seg_Mask:<25}
            {"Point Activation Loss:":<40}{mean_train_running_loss_Point_Act:<25}{mean_val_running_loss_Point_Act:<25}{mean_test_running_loss_Point_Act:<25}
            {"Point Sparse Embedding Loss:":<40}{mean_train_running_loss_Emd:<25}{mean_val_running_loss_Emd:<25}{mean_test_running_loss_Emd:<25}
            {"Attention Dice Score:":<40}{mean_train_running_Attention_Dice_Score:<25}{mean_val_running_Attention_Dice_Score:<25}{mean_test_running_Attention_Dice_Score:<25}
            {"Decoder Dice Score with MedSAM:":<40}{mean_train_running_Decoder_Dice_Score:<25}{mean_val_running_Decoder_Dice_Score:<25}{mean_test_running_Decoder_Dice_Score:<25}
            {"Decoder Dice Score with True Mask:":<40}{mean_train_running_WithTrueMask_Decoder_Dice_Score:<25}{mean_val_running_WithTrueMask_Decoder_Dice_Score:<25}{mean_test_running_WithTrueMask_Decoder_Dice_Score:<25}
            ----------------------------------------------------------------------------------------------------------
            """)


            logging.info(f"""
            Epoch {epoch + 1}/{args.num_epochs} |||| Learning Rate {warmup_scheduler.get_lr()[0]:.10f} |||| Folder Name {args.folder_name}
            --------------------------------------------------------------
            {{"Metric":<50}}{"Training":<25}{"Validation":<25}{"Test":<25}
            ----------------------------------------------------------------------------------------------------------
            {{"Number of Samples:"}}{num_sample_epoch:<25}{num_sample_val_epoch:<25}{num_sample_test_epoch:<25}
            {{"Segmentation Loss:"}}{mean_train_running_loss_Seg_Mask:<25}{mean_val_running_loss_Seg_Mask:<25}{mean_test_running_loss_Seg_Mask:<25}
            {{"Point Activation Loss:"}}{mean_train_running_loss_Point_Act:<25}{mean_val_running_loss_Point_Act:<25}{mean_test_running_loss_Point_Act:<25}
            {{"Point Sparse Embedding Loss:"}}{mean_train_running_loss_Emd:<25}{mean_val_running_loss_Emd:<25}{mean_test_running_loss_Emd:<25}
            {{"Attention Dice Score:"}}{mean_train_running_Attention_Dice_Score:<25}{mean_val_running_Attention_Dice_Score:<25}{mean_test_running_Attention_Dice_Score:<25}
            {{"Decoder Dice Score with MedSAM:"}}{mean_train_running_Decoder_Dice_Score:<25}{mean_val_running_Decoder_Dice_Score:<25}{mean_test_running_Decoder_Dice_Score:<25}
            {{"Decoder Dice Score with True Mask:"}}{mean_train_running_WithTrueMask_Decoder_Dice_Score:<25}{mean_val_running_WithTrueMask_Decoder_Dice_Score:<25}{mean_test_running_WithTrueMask_Decoder_Dice_Score:<25}
            ----------------------------------------------------------------------------------------------------------
            """)
            
            torch.save(Encoder_Model.state_dict(), save_path_f_latest + 'Encoder_Model.pth')
            torch.save(Decoder_Model.state_dict(), save_path_f_latest + 'Decoder_Model.pth')
                
            if best < mean_val_running_WithTrueMask_Decoder_Dice_Score:
                torch.save(Encoder_Model.state_dict(), save_path_f + 'Encoder_Model.pth')
                torch.save(Decoder_Model.state_dict(), save_path_f + 'Decoder_Model.pth')
                best = mean_val_running_WithTrueMask_Decoder_Dice_Score
                print(f"Saving best model at Epoch {epoch + 1}/{args.num_epochs}")
                gamma = 0
            else:
                gamma = gamma + 1
                
            if args.early_stop == True:
                if gamma == 15:
                    print(f"Early Stop:::>>> Saving model")
                    torch.cuda.empty_cache()
                    break