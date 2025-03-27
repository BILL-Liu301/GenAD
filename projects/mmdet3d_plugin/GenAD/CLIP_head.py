import torch
import torch.nn as nn
import torch.nn.functional as F

from CLIP import clip

class CLIPHead(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dtype = torch.float16
        self.feature_dim = 256

        self.model, _ = clip.load('ViT-B/32', device)
        self.model = self.model.to(self.dtype)
        self.model = self.model.train()

        # 多视角的特征融合
        self.fuse_feat = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=128, kernel_size=5, stride=2, padding=2, dtype=self.dtype),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, dtype=self.dtype),
            nn.ReLU(inplace=True)
        )

        self.text = {}
        self.query = nn.ModuleDict()

        # 红绿的特征
        self.text.update({
            'traffic_light': [
                'a red traffic light',
                'a green traffic light',
                'a yellow traffic light'
            ]
        })
        self.query.update({
            'traffic_light': nn.Parameter(torch.randn(1, 1, size=self.feature_dim, dtype=self.dtype, device=device))
        })
        

    def forward(self, imgs):
        B, N = imgs.shape[:2]
        imgs = imgs.reshape(B*N, *imgs.shape[2:])
        imgs_reshape = F.upsample(imgs, size=(244, 244), mode='bilinear', align_corners=True)
        img_feats = self.model.encode_image(imgs_reshape)
        img_feats = img_feats.reshape(B, N, *img_feats.shape[1:])
        img_feats = self.fuse_feat(img_feats)
        return img_feats