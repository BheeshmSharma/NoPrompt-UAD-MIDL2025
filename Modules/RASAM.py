import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Region_Aware_Spatial_Attention(nn.Module):
    def __init__(self, emb_dim=256, out_emb_dim=256, num_heads=4, grid_size=(256, 256), num_channels=32):
        super(Region_Aware_Spatial_Attention, self).__init__()
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_channels = num_channels

        self.query_projections = nn.ModuleList([nn.Linear(emb_dim, out_emb_dim) for _ in range(num_heads)])
        self.key_projections = nn.ModuleList([nn.Linear(self.num_channels, out_emb_dim) for _ in range(num_heads)])
        self.value_projections = nn.ModuleList([nn.Linear(self.num_channels, out_emb_dim) for _ in range(num_heads)])

        self.output_proj = nn.Linear(emb_dim * 4, emb_dim)

        self.positional_noise_encoding = nn.Parameter(torch.randn(4096, 256))
        self.FF = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),    
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),      
            nn.Sigmoid()
        )
        
    def forward(self, img_emb, point_emb):
        batch_size = img_emb.size(0)
        
        # Flatten Image Embeddings
        img_emb = img_emb.permute(0, 2, 3, 1).view(batch_size, -1, img_emb.size(1))
        
        # Add positional encoding
        pos_noise = self.positional_noise_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)
        pos_noise = pos_noise.squeeze(1)
        img_with_pos_emb = img_emb + pos_noise

        attended_heads = []
        for i in range(self.num_heads):
            query = self.query_projections[i](point_emb)
            key = self.key_projections[i](img_emb)
            value = self.value_projections[i](img_with_pos_emb)
            
            attention_scores = torch.matmul(query, key.transpose(-2, -1))
            attention_scores = attention_scores / (self.emb_dim ** 0.5)
            attention_scores = F.softmax(attention_scores, dim=-1)
            attended_values = torch.matmul(attention_scores, value)
            
            attended_heads.append(attended_values)

        attended_values_concat = torch.cat(attended_heads, dim=-1)
        point_spatial_embedding = self.output_proj(attended_values_concat)
        
        point_spatial_embedding = point_spatial_embedding + point_emb
        point_spatial_embedding_reshaped, _ = self._reshape_tensor(point_spatial_embedding)

        point_spatial_embedding_activations = self.FF(point_spatial_embedding_reshaped)
        point_spatial_embedding_activations = point_spatial_embedding_activations.view(batch_size, -1, 1)
                                                     
        return point_spatial_embedding_activations, point_spatial_embedding

    def _reshape_tensor(self, tensor):
        batch_size, num_point, embd = tensor.shape
        grid_dim = int(num_point ** 0.5)
        tensor_reshaped = tensor.view(batch_size, grid_dim, grid_dim, embd)
        return tensor_reshaped, grid_dim

    # def euclidean_distance(self, sampled_points_emd):
    #     mean_sampled_points_emd = torch.mean(sampled_points_emd, dim=1)
    #     mean_sampled_points_emd = mean_sampled_points_emd.unsqueeze(1)
        
    #     error =  torch.sqrt(torch.sum((mean_sampled_points_emd - sampled_points_emd) ** 2, dim=-1))
    #     summed_error = torch.sum(error, dim=1)
    #     batch_avg_error = torch.mean(summed_error)
        
    #     return batch_avg_error