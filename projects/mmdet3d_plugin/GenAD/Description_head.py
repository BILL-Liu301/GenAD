import torch
import torch.nn as nn
import torch.nn.functional as F

import clip

class DescriptionHead(nn.Module):
    def __init__(self):
        super(DescriptionHead, self).__init__()
        self.dtype = torch.float32

        model, _ = clip.load('ViT-B/32')
        model = model.train()
        model = model.to(self.dtype)

        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        # self.text_projection = model.text_projection
        self.mlp = nn.Sequential(
            nn.Linear(512, 512, dtype=self.dtype),
            nn.ReLU(),
            nn.LayerNorm(512, dtype=self.dtype),
            nn.Linear(512, 256, dtype=self.dtype),
            nn.ReLU(),
            nn.LayerNorm(256, dtype=self.dtype)
        )

    def forward(self, description, device):
        assert isinstance(description, str)
        tokenize = clip.tokenize(description, truncate=True)
        text_feat = self.encode_text(tokenize.to(device))
        return text_feat

    def encode_text(self, text):
        # 基本照搬CLIP的encode_text
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        x = self.mlp(x)
        return x
