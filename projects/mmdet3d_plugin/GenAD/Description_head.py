import clip.model
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip

class DescriptionHead(nn.Module):
    def __init__(self):
        super(DescriptionHead, self).__init__()
        self.dtype = torch.float32
        self.context_length = 128
        self.transformer_width = 512
        self.transformer_heads = 8
        self.transformer_layers = 12
        self.embed_dim = 256

        # 参考CLIP的实现
        self.token_embedding = nn.Embedding(49408, self.transformer_width)
        self.positional_embedding = nn.Parameter(torch.randn(self.context_length, self.transformer_width, requires_grad=True))
        self.transformer = clip.model.Transformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.ln_final = clip.model.LayerNorm(self.transformer_width)

        self.mlp = nn.Sequential(
            nn.Linear(self.transformer_width, self.transformer_width),
            nn.ReLU(),
            nn.LayerNorm(self.transformer_width),
            nn.Linear(self.transformer_width, self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim)
        )

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, description, device):
        assert isinstance(description, str)
        tokenize = clip.tokenize(description, self.context_length, truncate=False)
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
