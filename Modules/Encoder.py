import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, ReLU
from torch import Tensor
from .RASAM import Region_Aware_Sparse_Attention
import time

class Refinement_Block(nn.Module):
    def __init__(self):
        super(Refinement_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.skip_conv = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=1)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1) 
    def forward(self, x):
        conv_out = self.conv(x)
        upconv1_out = self.upconv1(conv_out)
        upconv2_out = self.upconv2(upconv1_out)
        skip_out = self.skip_conv(x)
        out = upconv2_out + skip_out
        out = self.global_avg_pool(out)
        return out


# SamLayerNorm class inspired by SAM model
class SamLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# Convolutional block with two convolutions and ReLU
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

# Encoder block with a convolution block and max pooling
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, dropout_prob=0.2):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.norm = SamLayerNorm(out_c, data_format="channels_first")
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x) 
        x = self.dropout(x)
        p = self.pool(x)
        return x, p


# Main UNet Encoder class
class Image_Encoder(nn.Module):
    def __init__(self, training=True, batch_size=16, dropout_prob=0.1):
        super(Image_Encoder, self).__init__()
        self.training = training
        self.batch_size = batch_size
        
        """ Encoder """
        self.e1 = encoder_block(1, 32, dropout_prob=dropout_prob)
        self.e2 = encoder_block(32, 64, dropout_prob=dropout_prob)
        self.e3 = encoder_block(64, 128, dropout_prob=dropout_prob)
        self.e4 = encoder_block(128, 256, dropout_prob=dropout_prob)

        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.adapter1 = Residual_Refinement_Block()
        self.adapter2 = Region_Aware_Sparse_Attention(emb_dim=256, num_heads=4, grid_size=(32, 32), num_channels=256)

    def forward(self, inputs, point_emb):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        
        refined_features = self.adapter1(s1)
        
        s2, p2 = self.e2(s1)
        
        s3, p3 = self.e3(s2)
        
        s4, p4 = self.e4(p3)

        point_sparse_embedding_activations, point_sparse_embedding = self.adapter2(refined_features, point_emb)

        return point_sparse_embedding_activations, point_sparse_embedding, s4
