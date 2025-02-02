import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

def save_frame(volume_frame, mask_frame, frame_index, volume_name, frame_folder, mask_folder, non_zero_file, zero_file):
    # Convert the volume and mask frames to images
    volume_img = Image.fromarray(volume_frame.astype(np.uint8))
    mask_img = Image.fromarray(mask_frame.astype(np.uint8))

    # Construct filenames
    base_filename = f"{volume_name.split('.nii')[0]}_{frame_index}.png"
    volume_filename = os.path.join(frame_folder, base_filename)
    mask_filename = os.path.join(mask_folder, base_filename)

    # Save the frame and mask
    volume_img.save(volume_filename)
    mask_img.save(mask_filename)

    # Save filenames in respective text files
    if np.any(mask_frame != 0):  # Check if mask has any non-zero value
        non_zero_file.write(base_filename + "\n")
    else:
        zero_file.write(base_filename + "\n")

def process_nii_files(volume_folder, gt_mask_folder, frame_folder, mask_folder, non_zero_txt, zero_txt):
    # Ensure output directories exist
    os.makedirs(frame_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # Open text files for writing
    with open(non_zero_txt, "w") as non_zero_file, open(zero_txt, "w") as zero_file:
        # Get list of all volume files in the folder
        volume_files = [f for f in os.listdir(volume_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

        for volume_file in tqdm(volume_files):
            # Load the volume (e.g., MRI or CT scan)
            volume_path = os.path.join(volume_folder, volume_file)
            volume_nii = nib.load(volume_path)
            volume_data = volume_nii.get_fdata()

            # Get corresponding GT mask (assuming it's named similarly)
            gt_mask_path = os.path.join(gt_mask_folder, volume_file)
            gt_mask_nii = nib.load(gt_mask_path)
            gt_mask_data = gt_mask_nii.get_fdata()

            # Extract volume name without extension
            volume_name = os.path.splitext(volume_file)[0]

            # Iterate through the frames (3rd dimension of the volume)
            for frame_index in range(volume_data.shape[2]):
                volume_frame = volume_data[:, :, frame_index]
                gt_mask_frame = gt_mask_data[:, :, frame_index]

                # Save frame and categorize based on mask values
                save_frame(volume_frame, gt_mask_frame, frame_index, volume_name, frame_folder, mask_folder, non_zero_file, zero_file)

if __name__ == "__main__":
    # Input paths
    volume_folder = './Dataste/Volumes/'
    gt_mask_folder = './Dataste/Volumes/'
    frame_folder = './Dataste/Frames/'
    mask_folder = './Dataste/Masks/'
    
    # Output text files
    Healthy = "./Dataste/Healthy_frame_names.txt"
    Unhealthy = "./Dataste/Unhealthy_frame_names.txt"

    # Process the nii files
    process_nii_files(volume_folder, gt_mask_folder, frame_folder, mask_folder, Unhealthy, Healthy)
