import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, dim, max_len=21, depth=2, heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):  # (B,L,C)
        B, L, C = x.shape
        x = x + self.pos[:, :L, :]
        x = self.enc(x)
        return x[:, L // 2, :]  # (B,C) center context


class TemporalConditionedUNet(nn.Module):
    """
    Wraps an existing UNet. Concats a learned context map (from temporal transformer)
    to the input image channels, then calls the UNet.
    """
    def __init__(self, unet, emb_dim=1024, window=7, ctx_ch=8, depth=2, heads=8):
        super().__init__()
        self.unet = unet
        self.window = window

        self.temporal = TemporalTransformer(dim=emb_dim, max_len=window, depth=depth, heads=heads)
        self.to_ctx = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, ctx_ch)
        )
        self.ctx_ch = ctx_ch

    def forward(self, img, emb_win):
        # img: (B,1,H,W), emb_win: (B,L,C)
        ctx = self.temporal(emb_win)              # (B,emb_dim)
        ctx = self.to_ctx(ctx)                    # (B,ctx_ch)
        ctx_map = ctx[:, :, None, None].expand(-1, -1, img.shape[-2], img.shape[-1])  # (B,ctx_ch,H,W)
        x = torch.cat([img, ctx_map], dim=1)      # (B,1+ctx_ch,H,W)
        return self.unet(x)

    def predict(self, img, emb_win, threshold=0.5):
        logits = self.forward(img, emb_win)
        return (torch.sigmoid(logits) > threshold).float()
