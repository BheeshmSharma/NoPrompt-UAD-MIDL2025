import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageEnhance
import random
import cv2  # For elastic deformations


def load_and_convert_to_tensor(image_folder_path, mask_folder_path, MedSAM_mask_folder_path, file_list, image_mode='L', mask_mode='L', target_size=(128, 128)):
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
    def crop_and_resize(image, mask, MedSAM_mask, target_size=(128, 128), threshold=64):
        image_array = np.array(image)
        mask_array = np.array(mask)
        MedSAM_mask_array = np.array(MedSAM_mask)
        
        # Find non-zero rows and columns
        non_zero_rows = np.any(image_array, axis=1)
        non_zero_cols = np.any(image_array, axis=0)
        
        row_indices = np.where(non_zero_rows)[0]
        col_indices = np.where(non_zero_cols)[0]

        # Check if there are sufficient non-zero rows/columns
        if len(row_indices) >= threshold and len(col_indices) >= threshold:
            cropped_image = image_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
            cropped_mask = mask_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
            cropped_MedSAM_mask = MedSAM_mask_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
        else:
            # Crop the central 64x64 area
            top = 32-((max(row_indices) - min(row_indices)) // 2)
            left = 32-((max(col_indices) - min(col_indices)) // 2)
            cropped_image = image_array[min(row_indices)-top:max(row_indices)+top, min(col_indices)-left:max(col_indices)+left]
            cropped_mask = mask_array[min(row_indices)-top:max(row_indices)+top, min(col_indices)-left:max(col_indices)+left]
            cropped_MedSAM_mask = MedSAM_mask_array[min(row_indices)-top:max(row_indices)+top, min(col_indices)-left:max(col_indices)+left]
        
        # Convert to PIL image and resize to 128x128
        cropped_image = Image.fromarray(cropped_image).resize(target_size, Image.BILINEAR)
        cropped_mask_128 = Image.fromarray(cropped_mask).resize(target_size, Image.BILINEAR)
        cropped_MedSAM_mask_128 = Image.fromarray(cropped_MedSAM_mask).resize(target_size, Image.BILINEAR)
        
        cropped_mask_256 = Image.fromarray(cropped_mask).resize((256,256), Image.BILINEAR)
        cropped_MedSAM_mask_256 = Image.fromarray(cropped_MedSAM_mask).resize((256,256), Image.BILINEAR)
        
        return cropped_image, cropped_mask_128, cropped_MedSAM_mask_128, cropped_mask_256, cropped_MedSAM_mask_256
    
    def apply_augmentations(image, mask, MedSAM_mask, cropped_mask_256, cropped_MedSAM_mask_256):
        # Random rotation
        if random.random() < 0.7:
            angle = random.randint(-90, 90)  # Rotate between -30 and 30 degrees
            image = image.rotate(angle, resample=Image.BILINEAR)
            
            mask = mask.rotate(angle, resample=Image.NEAREST)
            MedSAM_mask = MedSAM_mask.rotate(angle, resample=Image.NEAREST)
            
            cropped_mask_256 = cropped_mask_256.rotate(angle, resample=Image.NEAREST)
            cropped_MedSAM_mask_256 = cropped_MedSAM_mask_256.rotate(angle, resample=Image.NEAREST)

        # Random horizontal flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            MedSAM_mask = MedSAM_mask.transpose(Image.FLIP_LEFT_RIGHT)
    
            cropped_mask_256 = cropped_mask_256.transpose(Image.FLIP_LEFT_RIGHT)
            cropped_MedSAM_mask_256 = cropped_MedSAM_mask_256.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            MedSAM_mask = MedSAM_mask.transpose(Image.FLIP_TOP_BOTTOM)
    
            cropped_mask_256 = cropped_mask_256.transpose(Image.FLIP_TOP_BOTTOM)
            cropped_MedSAM_mask_256 = cropped_MedSAM_mask_256.transpose(Image.FLIP_TOP_BOTTOM)

        # Random brightness and contrast adjustments
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))  # Brightness range [0.8, 1.2]

        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))  # Contrast range [0.8, 1.2]

        # Add Gaussian noise
        if random.random() < 0.7:
            image_array = np.array(image).astype(np.float32)
            noise = np.random.normal(0, 10, image_array.shape).astype(np.float32)  # Mean 0, std 10
            noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_image)

        return image, mask, MedSAM_mask, cropped_mask_256, cropped_MedSAM_mask_256
    
    # Initialize lists for valid images and masks
    valid_images = []
    valid_masks = []
    valid_MedSAM_masks = []
    valid_masks_256 = []
    valid_MedSAM_masks_256 = []
    
    # Load, resize, and convert masks
    for file in file_list:
        # Open and process image
        image = Image.open(os.path.join(image_folder_path, file)).convert(image_mode)      
        image = apply_gamma_correction(image)  

        mask = Image.open(os.path.join(mask_folder_path, file)).convert(mask_mode).resize(image.size, Image.BILINEAR)
        MedSAM_mask = Image.open(os.path.join(MedSAM_mask_folder_path, file)).convert(mask_mode).resize((256, 256), Image.BILINEAR) if os.path.exists(os.path.join(MedSAM_mask_folder_path, file)) else Image.fromarray(np.zeros((256, 256), dtype=np.uint8), mode=mask_mode)

        mask_array = np.array(mask)
        mask_array1 = np.where(mask_array > 127.5, 1.0, 0.0)

        non_zero_pixels = np.count_nonzero(mask_array1)

        # Check if mask has fewer than 200 non-zero pixels
        if non_zero_pixels < 200:
            continue  # Skip this image-mask pair
        
        cropped_resized_image, cropped_resized_mask, cropped_MedSAM_mask, cropped_mask_256, cropped_MedSAM_mask_256 = crop_and_resize(image, mask, MedSAM_mask, target_size)
        cropped_resized_image, cropped_resized_mask, cropped_MedSAM_mask, cropped_mask_256, cropped_MedSAM_mask_256 = apply_augmentations(cropped_resized_image, cropped_resized_mask, cropped_MedSAM_mask, cropped_mask_256, cropped_MedSAM_mask_256)

        image_tensor = to_tensor(cropped_resized_image)

        mask_tensor = to_tensor(cropped_resized_mask)
        MedSAM_mask_tensor = to_tensor(cropped_MedSAM_mask)
        mask_tensor_256 = to_tensor(cropped_mask_256)
        MedSAM_mask_tensor_256 = to_tensor(cropped_MedSAM_mask_256)
        
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension if needed

        # Add valid image and mask to the lists
        valid_images.append(image_tensor)  # Add the processed image tensor
        valid_masks.append(mask_tensor)  # Add the mask tensor
        valid_MedSAM_masks.append(MedSAM_mask_tensor)  # Add the mask tensor        
        valid_masks_256.append(mask_tensor_256)  # Add the mask tensor
        valid_MedSAM_masks_256.append(MedSAM_mask_tensor_256)  # Add the mask tensor        

    # Convert the list of valid images and masks into batches
    batch_images = torch.stack(valid_images) if valid_images else torch.empty(0)
    batch_masks = torch.stack(valid_masks) if valid_masks else torch.empty(0)
    batch_MedSAM_masks = torch.stack(valid_MedSAM_masks) if valid_MedSAM_masks else torch.empty(0)
    batch_masks_256 = torch.stack(valid_masks_256) if valid_masks_256 else torch.empty(0)
    batch_MedSAM_masks_256 = torch.stack(valid_MedSAM_masks_256) if valid_MedSAM_masks_256 else torch.empty(0)
    
    batch_masks = torch.where(batch_masks > 127.5, 1.0, 0.0)
    batch_masks_256 = torch.where(batch_masks_256 > 127.5, 1.0, 0.0)
    
    batch_MedSAM_masks = torch.where(batch_MedSAM_masks > 127.5, 1.0, 0.0)
    batch_MedSAM_masks_256 = torch.where(batch_MedSAM_masks_256 > 127.5, 1.0, 0.0)
    
    return batch_images, batch_masks, batch_MedSAM_masks, batch_masks_256, batch_MedSAM_masks_256