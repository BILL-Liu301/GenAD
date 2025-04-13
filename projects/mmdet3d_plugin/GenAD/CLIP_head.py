import torch
import torch.nn as nn
import torch.nn.functional as F

from CLIP import clip

class CLIPHead(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dtype = torch.float32
        self.feature_dim = 512

        self.model, _ = clip.load('ViT-B/32', device)
        self.model = self.model.train()
        self.model = self.model.to(self.dtype)

    def forward(self, description, device):
        assert isinstance(description, str)
        tokenize = clip.tokenize(description.split('. '))
        text_feat = self.model.encode_text(tokenize.to(device))
        text_feat = text_feat.unsqueeze(0)
        return text_feat
