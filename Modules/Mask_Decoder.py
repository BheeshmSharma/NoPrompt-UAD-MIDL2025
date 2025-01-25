import torch
import torch.nn as nn


class Mask_Decoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_points=1024, output_size=256):
        super(Mask_Decoder, self).__init__()

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Positional encoding (learnable positional embedding)
        self.positional_encoding = nn.Parameter(torch.randn(1, embed_dim, 64, 64))  # Assuming 64x64 feature map size
        self.regularization_strength = 1e-5

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(num_points, 128, kernel_size=4, stride=2, padding=1),  # 1024 -> 128 channels
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64 channels
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)  # 64 -> 1 channel
        )
        

        # Adding LayerNorm layers after cross-attention and feed-forward layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(num_points)
        
        self.cross_attention_dropout = nn.Dropout(p=0.1)

    def forward(self, image_emb, sparse_emb):
        bs, num_points, embed_dim = sparse_emb.size()
        
        # Add positional encoding to image embedding
        image_emb_pos = image_emb + self.positional_encoding  # Broadcasting positional encoding across batch size
        residual = sparse_emb

        # Flatten the image embedding
        image_emb_flat = image_emb.view(bs, embed_dim, -1).permute(0, 2, 1)  # [bs, 4096, embed_dim]
        image_emb_pos_flat = image_emb_pos.view(bs, embed_dim, -1).permute(0, 2, 1)  # [bs, 4096, embed_dim]

        # Cross-attention
        cross_attn_output, _ = self.cross_attention(query=sparse_emb, key=image_emb_flat, value=image_emb_pos_flat)  # [bs, num_points, embed_dim]

        # Apply dropout after cross-attention
        cross_attn_output = self.cross_attention_dropout(cross_attn_output)

        cross_attn_output = cross_attn_output + residual  # Residual connection

        # Apply LayerNorm after cross-attention
        cross_attn_output = self.norm1(cross_attn_output)

        # Reshape cross_attn_output for matrix multiplication
        cross_attn_output = cross_attn_output.permute(0, 2, 1).contiguous().view(bs, embed_dim, num_points)

        # Matrix multiplication === Image Embedding X Cross Attention (transpose)
        output = torch.bmm(image_emb_flat, cross_attn_output)  # [bs, 4096, num_points]
        
        # Apply LayerNorm after matrix multiplication
        output = self.norm2(output)

        # Reshape output for upsampling
        h = w = int(output.shape[1] ** 0.5)  # Calculate height and width based on num_points
        output = output.permute(0, 2, 1).contiguous().view(bs, num_points, h, w)  # Shape: [bs, embed_dim, h, w]

        # Upsample to [bs, 1, 256, 256]
        final_output = self.upsample(output)
        
        return final_output