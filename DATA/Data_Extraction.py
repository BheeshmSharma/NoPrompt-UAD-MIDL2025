import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import random


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


def read_patient_filenames(txt_file):
    """ Reads a TXT file and groups filenames by patient ID. """
    patient_dict = {}

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        filename = line.strip()  # Remove newline characters
        patient_id = filename.split('_')[0]  # Extract patient ID (assuming first part before '_')

        if patient_id not in patient_dict:
            patient_dict[patient_id] = []
        patient_dict[patient_id].append(filename)

    return patient_dict

def split_patients(patient_ids, train_ratio=0.80, val_ratio=0.05):
    """ Splits patient IDs into train, val, and test sets while maintaining consistency. """
    random.shuffle(patient_ids)  # Shuffle for randomness

    total = len(patient_ids)
    train_idx = int(train_ratio * total)
    val_idx = train_idx + int(val_ratio * total)

    train_patients = set(patient_ids[:train_idx])
    val_patients = set(patient_ids[train_idx:val_idx])
    test_patients = set(patient_ids[val_idx:])

    return train_patients, val_patients, test_patients

def save_split_frames(patient_dict, patient_set, filename):
    """ Saves frames of selected patients into a given file. """
    with open(filename, "w") as f:
        for patient in patient_set:
            for frame in patient_dict.get(patient, []):
                f.write(frame + "\n")

if __name__ == "__main__":
    # Input paths
    volume_folder = './Dataset/Volumes/'
    gt_mask_folder = './Dataset/Volumes/'
    frame_folder = './Dataset/Frames/'
    mask_folder = './Dataset/Masks/'
    
    # Output text files
    Healthy = "./Dataset/Healthy_frame_names.txt"
    Unhealthy = "./Dataset/Unhealthy_frame_names.txt"

    # Process the nii files
    process_nii_files(volume_folder, gt_mask_folder, frame_folder, mask_folder, Unhealthy, Healthy)

    # Output folder
    output_folder = "./Dataset/"
    os.makedirs(output_folder, exist_ok=True)

    # Read filenames grouped by patient
    Unhealthy_patients = read_patient_filenames(Unhealthy)
    Healthy_patients = read_patient_filenames(Healthy)

    # Get all unique patient IDs
    all_patients = set(Unhealthy_patients.keys()).union(set(Healthy_patients.keys()))

    # Split patients consistently across train, val, and test
    train_patients, val_patients, test_patients = split_patients(list(all_patients))

    # Save frames while keeping patient-wise consistency
    save_split_frames(Healthy_patients, train_patients, os.path.join(output_folder, "Healthy_train.txt"))
    save_split_frames(Healthy_patients, val_patients, os.path.join(output_folder, "Healthy_val.txt"))
    save_split_frames(Healthy_patients, test_patients, os.path.join(output_folder, "Healthy_test.txt"))

    save_split_frames(Unhealthy_patients, train_patients, os.path.join(output_folder, "Unhealthy_train.txt"))
    save_split_frames(Unhealthy_patients, val_patients, os.path.join(output_folder, "Unhealthy_val.txt"))
    save_split_frames(Unhealthy_patients, test_patients, os.path.join(output_folder, "Unhealthy_test.txt"))
