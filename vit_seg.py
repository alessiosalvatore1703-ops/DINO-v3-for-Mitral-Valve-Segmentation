import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder."""
    
    def __init__(
        self, 
        img_size=128, 
        patch_size=16, 
        in_channels=1, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        dropout=0.0
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x


class SegmentationHead(nn.Module):
    """Segmentation decoder head with progressive upsampling."""
    
    def __init__(self, embed_dim, img_size, patch_size, n_classes=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_per_side = img_size // patch_size
        
        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(32, n_classes, kernel_size=1),
        )
        
    def forward(self, x):
        # x: (B, n_patches, embed_dim)
        B, N, C = x.shape
        
        # Reshape to spatial grid
        x = x.transpose(1, 2).reshape(B, C, self.patches_per_side, self.patches_per_side)
        
        # Decode
        x = self.decoder(x)
        return x


class ViTSeg(nn.Module):
    """Vision Transformer for Segmentation.
    
    A simple ViT-based segmentation model with:
    - ViT encoder to extract features from patches
    - Progressive upsampling decoder for dense prediction
    """
    
    def __init__(
        self, 
        img_size=128, 
        patch_size=8,       # Smaller patches for better spatial resolution
        in_channels=1, 
        n_classes=1,
        embed_dim=384,      # Smaller embedding for efficiency
        depth=6,            # Fewer layers
        num_heads=6, 
        mlp_ratio=4.0, 
        dropout=0.1
    ):
        super().__init__()
        
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.decoder = SegmentationHead(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            n_classes=n_classes
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # x: (B, C, H, W)
        features = self.encoder(x)  # (B, n_patches, embed_dim)
        out = self.decoder(features)  # (B, n_classes, H, W)
        return out
    
    def predict(self, x):
        return torch.sigmoid(self.forward(x))


# Predefined configurations
def vit_tiny_seg(img_size=128, in_channels=1, n_classes=1):
    """Tiny ViT for segmentation - fast training."""
    return ViTSeg(
        img_size=img_size,
        patch_size=8,
        in_channels=in_channels,
        n_classes=n_classes,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1
    )


def vit_small_seg(img_size=128, in_channels=1, n_classes=1):
    """Small ViT for segmentation - balanced."""
    return ViTSeg(
        img_size=img_size,
        patch_size=8,
        in_channels=in_channels,
        n_classes=n_classes,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    )


def vit_base_seg(img_size=128, in_channels=1, n_classes=1):
    """Base ViT for segmentation - higher capacity."""
    return ViTSeg(
        img_size=img_size,
        patch_size=16,
        in_channels=in_channels,
        n_classes=n_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    )

