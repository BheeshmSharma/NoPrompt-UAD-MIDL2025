import torch
import numpy as np

def get_bounding_box(ground_truth_map):
    # print(ground_truth_map.shape)
    ground_truth_map = ground_truth_map.detach().cpu().numpy()
    # print(np.unique(ground_truth_map))
    y_indices, x_indices = np.where(ground_truth_map > 0)
    # print(y_indices, x_indices)

    if len(y_indices) == 0 or len(x_indices) == 0:  # If there are no non-zero elements
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # Add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    # print(H, W)
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    
    return bbox

def get_bounding_boxes_for_batch(batch_masks):
    batch_size = batch_masks.shape[0]
    bounding_boxes = []
    for i in range(batch_size):
        mask = batch_masks[i, 0, :, :]
        bbox = get_bounding_box(mask)
        if bbox:
            bounding_boxes.append(bbox)
        else:
            bounding_boxes.append([0, 0, 0, 0])  # If no bbox found, return a default bbox
    return bounding_boxes

def masks_to_bounding_boxes_tensor(batch_masks):
    # Extract bounding boxes
    bboxes = get_bounding_boxes_for_batch(batch_masks)
    
    # Transform to tensor
    bbox_tensor = torch.tensor(bboxes, dtype=torch.float32)
    return bbox_tensor