import numpy as np
import torch
import matplotlib.pyplot as plt

class Boundary_coordinate_and_Mask:
    def __init__(self, grid_size=16, image_size=128):
        """
        Initialize the BoundaryMaskCreator.

        Parameters:
        - grid_size: The size of the grid for boundary point extraction
        - image_size: The dimensions of the image (assumed square)
        """
        self.grid_size = grid_size
        self.image_size = image_size
        self.gap = image_size // grid_size
        self.boundary_points = self._generate_boundary_points()
        self.boundary_points_tensor = torch.tensor(self.boundary_points, dtype=torch.float32)

    def _generate_boundary_points(self):
        """
         
        
        Returns:
        - boundary_points: Array of boundary points
        """
        boundary_points = []
        for i in range(self.grid_size):
            # Horizontal lines
            y = i * self.gap
            for x in range(0, self.image_size, self.gap):
                boundary_points.append((x, y))
            
            # Vertical lines
            x = i * self.gap
            for y in range(0, self.image_size, self.gap):
                boundary_points.append((x, y))
        
        boundary_points = np.array(boundary_points)
        boundary_points = np.unique(boundary_points, axis=0)
        return boundary_points


    def create_masks(self, mask):
        boundary_points = torch.tensor(self.boundary_points, device=mask.device)
        boundary_rows, boundary_cols = boundary_points[:, 0], boundary_points[:, 1]
        sample = mask[:, 0, boundary_rows, boundary_cols]
        sample = sample.view(mask.shape[0], 32, 32)
        sample = torch.where(sample > 0.0, 1.0, 0.0)
        return sample

        

    def plot_masks(self, bounding_boxes, masks):
        """
        Plot the boundary points and mask points for each bounding box.

        Parameters:
        - bounding_boxes: PyTorch tensor of shape (batch_size, 4) with bounding boxes
        - masks: PyTorch tensor of shape (batch_size, num_points) with binary masks
        """
        bounding_boxes_np = bounding_boxes.numpy()
        masks_np = masks.numpy()

        plt.figure(figsize=(12, 12))

        # Plot all boundary points in red
        plt.scatter(self.boundary_points[:, 0], self.boundary_points[:, 1], c='red', s=10, label='Boundary Points')

        # Plot mask points in green for each bounding box
        for i, mask in enumerate(masks_np):
            mask_points = self.boundary_points[mask == 1]
            plt.scatter(mask_points[:, 0], mask_points[:, 1], s=10, label=f'Mask Points for Bounding Box {i + 1}')
        
        plt.title('Boundary Points and Mask Points')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
        plt.legend()
        plt.grid(True)
        plt.show()