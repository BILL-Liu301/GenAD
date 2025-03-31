import torch
import torch.nn as nn
import torch.nn.functional as F

from CLIP import clip

class CLIPHead(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dtype = torch.float16
        self.feature_dim = 512

        self.model, _ = clip.load('ViT-B/32', device)
        self.model = self.model.train()

        self.multi_img_feats_fuse = nn.Conv1d(in_channels=6, out_channels=1, kernel_size=1, bias=False, dtype=self.dtype)

        self.text = {}
        self.query = {}
        self.fuse = nn.ModuleDict()
        self.keys = ['traffic_light', 'traffic_condition']

        # 红绿的特征，traffic_light
        key = self.keys[0]
        self.text.update({
            key: [
                'a red traffic light',
                'a green traffic light',
                'a yellow traffic light',
                'No traffic light'
            ]
        })
        self.query.update({
            key: nn.Parameter(torch.randn(1, 1, self.feature_dim, dtype=self.dtype, device=device), requires_grad=True)
        })
        self.fuse.update({
            key: nn.Linear(self.feature_dim * 2, self.feature_dim, bias=False, dtype=self.dtype)
        })

        # 交通情况，traffic_condition
        key = self.keys[1]
        self.text.update({
            key: [
                'a free traffic',
                'a smooth traffic',
                'a normal traffic',
                'a heavy traffic'
            ]
        })
        self.query.update({
            key: nn.Parameter(torch.randn(1, 1, self.feature_dim, dtype=self.dtype, device=device), requires_grad=True)
        })
        self.fuse.update({
            key: nn.Linear(self.feature_dim * 2, self.feature_dim, bias=False, dtype=self.dtype)
        })


    def forward(self, imgs):
        B, N = imgs.shape[:2]
        imgs = imgs.reshape(B*N, *imgs.shape[2:])
        imgs_reshape = F.upsample(imgs, size=(244, 244), mode='bilinear', align_corners=True)
        img_feats = self.model.encode_image(imgs_reshape)
        img_feats = img_feats.reshape(B, N, *img_feats.shape[1:])

        img_feats_oup = []
        descriptions_oup = {}
        for key in self.keys:
            img_feats_ = self.fuse_img_feats_query(img_feats, key)
            text_feats_ = self.get_text_feats(key, img_feats_.dtype, img_feats_.device)

            # normalize
            img_feats_ /= img_feats_.norm(dim=-1, keepdim=True)
            text_feats_ /= text_feats_.norm(dim=-1, keepdim=True)

            # 特征之间交互
            descriptions_ = img_feats_ @ text_feats_.T

            img_feats_oup.append(img_feats_)
            descriptions_oup.update({key: descriptions_})

        img_feats_oup = torch.cat(img_feats_oup, dim=1)

        return img_feats_oup, descriptions_oup

    # 往当前feats中添加query，并融合多视角的图像特征
    def fuse_img_feats_query(self, img_feats, key):
        B, N = img_feats.shape[:2]

        # 提取当前的query
        query = self.query[key].repeat(B, N, 1)

        # 将query融入到feats中
        feats = torch.cat([img_feats, query], dim=-1)
        feats_fuse = self.fuse[key](feats)

        # 将多视角的图像特征融合
        feats_fuse = self.multi_img_feats_fuse(feats_fuse)

        return feats_fuse
    
    def get_text_feats(self, key, dtype, device):
        text_input = torch.cat([clip.tokenize(text) for text in self.text[key]], dim=0).to(device=device).long()
        text_feats = self.model.encode_text(text_input)
        return text_feats
