import copy
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.GenAD.utils.traj_lr_warmup import get_traj_warmup_loss_weight
from projects.mmdet3d_plugin.GenAD.utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)

from projects.mmdet3d_plugin.GenAD.generator import DistributionModule, PredictModel
from projects.mmdet3d_plugin.GenAD.generator import FuturePrediction


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit * 2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


@HEADS.register_module()
class GenADHead(DETRHead):
    """Head of VAD model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                 loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                     type='ChamferDistance', loss_src_weight=1.0, loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=1.0),
                 tot_epoch=None,
                 use_traj_lr_warmup=False,
                 motion_decoder=None,
                 motion_map_decoder=None,
                 use_description=False,
                 description_ca_bev=False,
                 description_ca_map=False,
                 description_ca_motion=False,
                 description_bev_ca=None,
                 description_map_ca=None,
                 description_motion_ca=None,
                 use_pe=False,
                 motion_det_score=None,
                 map_thresh=0.5,
                 dis_thresh=0.2,
                 pe_normalization=True,
                 ego_his_encoder=None,
                 ego_fut_mode=5,  # 3 -> 5，5种类型的指令
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanAgentDisLoss', loss_weight=0.1),
                 loss_plan_dir=dict(type='PlanMapThetaLoss', loss_weight=0.1),
                 ego_agent_decoder=None,
                 ego_map_decoder=None,
                 query_thresh=None,
                 query_use_fix_pad=None,
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 agent_dim=300,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.tot_epoch = tot_epoch
        self.use_traj_lr_warmup = use_traj_lr_warmup
        self.motion_decoder = motion_decoder
        self.motion_map_decoder = motion_map_decoder
        self.use_description = use_description
        self.description_ca_bev = description_ca_bev
        self.description_ca_map = description_ca_map
        self.description_ca_motion = description_ca_motion
        self.description_bev_ca = description_bev_ca
        self.description_map_ca = description_map_ca
        self.description_motion_ca = description_motion_ca
        self.use_pe = use_pe
        self.motion_det_score = motion_det_score
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.pe_normalization = pe_normalization
        self.ego_his_encoder = ego_his_encoder
        self.ego_fut_mode = ego_fut_mode
        self.query_thresh = query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.agent_dim = agent_dim
        self.with_cur = True

        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
            self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_query = map_num_query
        self.map_num_classes = map_num_classes
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_dir_interval = map_dir_interval

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        if map_class_weight is not None and (self.__class__ is GenADHead):
            assert isinstance(map_class_weight, float), 'Expected ' \
                                                        'class_weight to have type float. Found ' \
                                                        f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                                                         'bg_cls_weight to have type float. Found ' \
                                                         f'{type(map_bg_cls_weight)}.'
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            self.map_bg_cls_weight = map_bg_cls_weight

        self.traj_bg_cls_weight = 0

        super(GenADHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(self.map_code_weights, requires_grad=False), requires_grad=False)

        if self.use_description:
            if self.description_ca_bev:
                self.bev_query_pos = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
                self.description_bev_ca = build_transformer_layer_sequence(self.description_bev_ca)
            if self.description_ca_map:
                self.map_query_pos = nn.Embedding(self.map_num_query, self.embed_dims)
                self.description_map_ca = build_transformer_layer_sequence(self.description_map_ca)
            if self.description_ca_motion:
                self.description_motion_ca = build_transformer_layer_sequence(self.description_motion_ca)

        # cmd部分
        # self.cmd_query = nn.Embedding(self.ego_fut_mode, self.embed_dims)
        # self.cmd_pos = nn.Embedding(self.ego_fut_mode, self.embed_dims)
        # self.cmd_query_mlp = nn.Linear(self.embed_dims * 2, self.embed_dims)
        # self.cmd_pos_mlp = nn.Linear(self.embed_dims * 2, self.embed_dims)

        if kwargs['train_cfg'] is not None:
            assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided ' \
                                                          'when train_cfg is set.'
            map_assigner = kwargs['train_cfg']['map_assigner']
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                           'should be exactly the same.'
            assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'

            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.map_sampler = build_sampler(sampler_cfg, context=self)

        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)
        self.loss_map_bbox = build_loss(loss_map_bbox)
        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_iou = build_loss(loss_map_iou)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_plan_dir = build_loss(loss_plan_dir)
        self.loss_vae_gen = build_loss(loss_vae_gen)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        if self.with_cur:
            traj_in_dim = self.embed_dims * 4
        else:
            traj_in_dim = self.embed_dims * 2
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(traj_in_dim, traj_in_dim))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(traj_in_dim, 2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        # for _ in range(self.num_reg_fcs):
        traj_cls_branch.append(Linear(self.embed_dims * 14, self.embed_dims * 2))
        traj_cls_branch.append(nn.LayerNorm(self.embed_dims * 2))
        traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims * 2, self.embed_dims * 2))
        traj_cls_branch.append(nn.LayerNorm(self.embed_dims * 2))
        traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims * 2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                        self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)

        if self.motion_decoder is not None:
            self.motion_decoder = build_transformer_layer_sequence(self.motion_decoder)
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError('Not implement yet')

        if self.motion_map_decoder is not None:
            self.lane_encoder = LaneNet(256, 128, 3)
            self.motion_map_decoder = build_transformer_layer_sequence(self.motion_map_decoder)
            if self.use_pe:
                self.pos_mlp = nn.Linear(2, self.embed_dims)

        if self.ego_his_encoder is not None:
            self.ego_his_encoder = LaneNet(2, self.embed_dims // 2, 3)
        else:
            self.ego_query = nn.Embedding(1, self.embed_dims)

        self.ego_agent_pos_mlp = nn.Linear(2, self.embed_dims)

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims * 2 + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims * 2
        if self.with_cur:
            ego_fut_dec_in_dim = int(ego_fut_dec_in_dim * 2)
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, self.ego_fut_mode * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        self.ego_coord_mlp = nn.Linear(2, 2)

        # generator motion & planning
        self.layer_dim = 4
        self.present_distribution_in_channels = 512
        self.future_distribution_in_channels = 524
        self.now_pred_in_channels = 64
        self.probabilistic = True
        self.latent_dim = 32
        self.min_log_sigma = -5.0
        self.max_log_sigma = 5.0

        self.present_distribution = DistributionModule(
            self.present_distribution_in_channels,
            self.latent_dim,
            min_log_sigma=self.min_log_sigma,
            max_log_sigma=self.max_log_sigma,
        )

        self.future_distribution = DistributionModule(
            self.future_distribution_in_channels,
            self.latent_dim,
            min_log_sigma=self.min_log_sigma,
            max_log_sigma=self.max_log_sigma,
        )

        # Future prediction
        self.predict_model = PredictModel(
            in_channels=self.latent_dim,
            out_channels=self.embed_dims * 2,
            hidden_channels=self.latent_dim * 4,
            num_layers=self.layer_dim
        )






    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_map_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_traj_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.traj_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.map_reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.map_reg_branches[0][-1].bias.data[2:], 0.)
        if self.motion_decoder is not None:
            for p in self.motion_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.motion_mode_query.weight)
            if self.use_pe:
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
        if self.motion_map_decoder is not None:
            for p in self.motion_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.lane_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            if self.use_pe:
                xavier_init(self.pos_mlp, distribution='uniform', bias=0.)
        if self.ego_his_encoder is not None:
            for p in self.ego_his_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # @auto_fp16(apply_to=('mlvl_feats'))

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    # @force_fp32(apply_to=('mlvl_feats', 'prev_bev', 'img_feats_from_vlm'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                gt_labels_3d=None,
                gt_attr_labels=None,
                ego_fut_trajs=None,
                ego_fut_cmd=None,
                description_feats=None
                ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        assert bs==1, 'only support bs=1'

        # 从query_embedding、map_pts_embedding和map_instance_embedding中提取各自的weight，作为embed和query
        # 不理解，按理说上述三个都是nn.Embedding类型的对象，直接用就好了，为什么要特意提取各自的weight？
        object_query_embeds = self.query_embedding.weight.to(dtype)
        if self.map_query_embed_type == 'all_pts':
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == 'instance_pts':
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)  # [1, 20, 512]
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)  # [100, 1, 512]
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)  # [2000, 512]
        bev_queries = self.bev_embedding.weight.to(dtype)  # [10000, 512]

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)  # [1, 100, 100]
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  # [1, 256, 100, 100]

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                map_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                description_feat=description_feats['bev'] if self.use_description and self.description_ca_bev else None,
                bev_query_pos=self.bev_query_pos.weight if self.use_description and self.description_ca_bev else None,
                description_bev_ca=self.description_bev_ca if self.use_description and self.description_ca_bev else None
            )

        # bev_embed: bev features
        # hs: agent_query
        # init_reference: reference points init
        # inter_references: reference points processing
        # map_hs: map_query
        # map_init_reference: reference points init
        # map_inter_references: reference points processing

        bev_embed, hs, init_reference, inter_references, map_hs, map_init_reference, map_inter_references = outputs

        if self.use_description and self.description_ca_map:
            # 将map_hs和vlm提取的特征进行特征融合
            map_hs = map_hs.permute(1, 0, 2, 3).flatten(1, 2)  # [3, 2000, 1, 256] -> [2000, 3, 1, 256] -> [2000, 3, 256]
            map_query_pos = self.map_query_pos.weight.unsqueeze(1).repeat(1, map_hs.shape[1], 1).to(dtype)  # [2000, 256] -> [2000, 3, 256]
            description_feat = description_feats['map']
            map_hs = self.description_map_ca(  # [2000, 3, 256]
                query=map_hs,
                key=description_feat.repeat(1, map_hs.shape[1], 1),
                value=description_feat.repeat(1, map_hs.shape[1], 1),
                query_pos=map_query_pos,
            )
            map_hs = map_hs.permute(1, 0, 2).unsqueeze(2)  # [2000, 3, 256] -> [3, 2000, 256] -> [3, 2000, 1, 256]

        hs = hs.permute(0, 2, 1, 3)  # [3, 300, 1, 256] -> [3, 1, 300, 256]
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_trajs = []
        outputs_trajs_classes = []

        map_hs = map_hs.permute(0, 2, 1, 3)  # [3, 2000, 1, 256] -> [3, 1, 2000, 256]
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        for lvl in range(hs.shape[0]):  # hs.shape = [3, 1, 300, 256]
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])  # [1, 300, 256] -> [1, 300, 10]
            tmp = self.reg_branches[lvl](hs[lvl])  # [1, 300, 256] -> [1, 300, 10]

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](  # [1, 2000, 256] -> [1, 100, 20, 256] -> [1, 100, 256] -> [1, 100, 3]
                map_hs[lvl].view(bs, self.map_num_vec, self.map_num_pts_per_vec, -1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])  # [1, 2000, 256] -> [1, 2000, 2]
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()  # cx,cy,w,h
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)

        # motion prediction
        # 重点关注对象

        # motion query
        if self.motion_decoder is not None:
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]  # [1, 300, 2]，历史数据对应的最后一帧的bev特征坐标系
            # motion_query
            motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]  [3, 1, 300, 256] -> [1, 300, 256] -> [300, 1, 256]
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]  [6, 256]
            # [M, B, D], M=A*fut_mode
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)  # [300, 6, 1, 256] -> [1800, 1, 256]

            # A: 300, B: 1, D: 256, fut_mode: 6, M: 1800

            if self.use_pe:
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]  [1, 300, 2]
                motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]  [1, 300, 2] -> [1, 300, 256]
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)  # [1, 300, 256] -> [1, 300, 1, 256] -> [1, 300, 6, 256] -> [1, 1800, 256]
                motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]  [1800, 1, 256]
            else:
                assert False
                motion_pos = None

            # 尚未传入self.motion_det_score
            if self.motion_det_score is not None:
                assert False
                motion_score = outputs_classes[-1]
                max_motion_score = motion_score.max(dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
            else:
                invalid_motion_idx = None

            # ego query
            # batch = batch_size
            if self.ego_his_encoder is not None:
                ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
            else:
                ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [1, 256] -> [1, 1, 256] -> [1, 1, 256]
            ego_query = ego_his_feats.permute(1, 0, 2)  # [1, 1, 256]
            ego_pos = torch.zeros((batch_size, 1, 2), device=ego_query.device).permute(1, 0, 2)  # [1, 1, 2]
            ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)  # [1, 1, 2] -> [1, 1, 256]

            # ego <-> agent Interaction
            motion_query = torch.cat([motion_query, ego_query], dim=0)  # [1801, 1, 256]
            motion_pos = torch.cat([motion_pos, ego_pos_emb], dim=0)  # [1801, 1, 256]

            if self.use_description and self.description_ca_motion:
                # 将motion_query和vlm提取的特征进行特征融合
                description_feat = description_feats['motion']
                motion_query = self.description_motion_ca(
                    query=motion_query,
                    key=description_feat,
                    value=description_feat,
                    query_pos=motion_pos,  # [1801, 1, 256]
                )

            # 此处的decoder就是TransformerDecoder
            # 公式5
            motion_hs = self.motion_decoder(  # [1801, 1, 256]
                query=motion_query,  # [1801, 1, 256]
                key=motion_query,  # [1801, 1, 256]
                value=motion_query,  # [1801, 1, 256]
                query_pos=motion_pos,  # [1801, 1, 256]
                key_pos=motion_pos,  # [1801, 1, 256]
                key_padding_mask=invalid_motion_idx
            )

            if self.motion_map_decoder is not None:
                # map preprocess
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]  [1, 300, 2]
                motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)  # [1, 300, 2] -> [1, 300, 1, 2] -> [1, 300, 6, 2] -> [1, 1800, 2]

                # ego_coords = torch.Tensor(1, 1, 2).cuda(1)
                ego_coords = torch.zeros([batch_size, 1, 2], device=motion_hs.device)  # [1, 1, 2]
                ego_coords_embd = self.ego_coord_mlp(ego_coords)  # [1, 1, 2] -> [1, 1, 2]
                # ego_coords_embd = torch.zeros([batch_size, 1, 2], device=motion_hs.device)
                motion_coords = torch.cat([motion_coords, ego_coords_embd], dim=1)  # [1, 1801, 2]

                map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)  # [1, 2000, 256] -> [1, 100, 20, 256]
                map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]  [1, 100, 20, 256] -> [1, 100, 256]q
                map_score = map_outputs_classes[-1]  # [1, 100, 3]
                map_pos = map_outputs_coords_bev[-1]  # [1, 100, 20, 2]
                map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(  # [1801, 1, 256], [1801, 1, 2], [1801, 1]
                    motion_coords, map_query, map_score, map_pos,
                    map_thresh=self.map_thresh, dis_thresh=self.dis_thresh,
                    pe_normalization=self.pe_normalization, use_fix_pad=True
                )
                map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]  [1, 1801, 256]
                sa_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)  # [1801, 1, 256] -> [1, 1801, 256] -> [1801, 256] -> [1, 1801, 256]

                # position encoding
                if self.use_pe:
                    (num_query, batch) = sa_motion_query.shape[:2]  # [1, 1801, 256]
                    motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)  # [1, 1801, 2]
                    motion_pos = self.pos_mlp(motion_pos)  # [1, 1801, 2] -> [1, 1801, 256]
                    map_pos = map_pos.permute(1, 0, 2)  # [1801, 1, 2] -> [1, 1801, 2]
                    map_pos = self.pos_mlp(map_pos)  # [1, 1801, 2] -> [1, 1801, 256]
                else:
                    motion_pos, map_pos = None, None

                # 此处的decoder就是TransformerDecoder
                # 公式6
                ca_motion_query = self.motion_map_decoder(  # [1, 1801, 256]
                    query=sa_motion_query,  # [1, 1801, 256]
                    key=map_query,  # [1, 1801, 256]
                    value=map_query,  # [1, 1801, 256]
                    query_pos=motion_pos,  # [1, 1801, 256]
                    key_pos=map_pos,  # [1, 1801, 256]
                    key_padding_mask=key_padding_mask)  # [1801, 1]
            else:
                assert False
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            ########################################
            # generator for planning & motion
            current_states = torch.cat([  # [1, 1801, 512]
                motion_hs.permute(1, 0, 2),  # [1801, 1, 256] -> [1, 1801, 256]
                # ca_motion_query.reshape(batch_size, -1, self.embed_dims)  # [1, 1801, 256] -> [1, 1801, 256]
                ca_motion_query  # [1, 1801, 256]
            ], dim=2)
            distribution_comp = {}
            # states = torch.randn((2, 1, 64, 200, 200), device=motion_hs.device)
            # future_distribution_inputs = torch.randn((2, 5, 6, 200, 200), device=motion_hs.device)
            noise = None
            if self.training:
                future_distribution_inputs = self.get_future_labels(
                    gt_labels_3d, gt_attr_labels,
                    ego_fut_trajs, motion_hs.device
                )
            else:
                future_distribution_inputs = None

            # 1. model CVA distribution for state
            # if self.fut_ts > 0:
            #     # present_state = states[:, :1].contiguous()
            #     if self.probabilistic:
            #         # Do probabilistic computation
            #         sample, output_distribution = self.distribution_forward(  # [1, 32, 1801], dict
            #             current_states, future_distribution_inputs, noise
            #         )
            #         # distribution_comp = {**distribution_comp, **output_distribution}
            #         distribution_comp = output_distribution
            # else:
            #     assert False
            assert self.fut_ts > 0, "future time steps should be greater than 0. It is usually set to 6."
            assert self.probabilistic, "probabilistic computation should be enabled."
            # Do probabilistic computation
            sample, distribution_comp = self.distribution_forward(  # [1, 32, 1801], dict
                current_states, future_distribution_inputs, noise
            )
            states_hs, future_states_hs = self.future_states_predict(  # [6, 1, 1801, 1024], [6, 1, 1801, 512]
                batch_size=batch_size,  # 1
                sample=sample,  # [1, 32, 1801]
                hidden_states=current_states,  # [1, 1801, 512]
                current_states=current_states  # [1, 1801, 512]
            )

            # 2. predict future state from distribution
            # 因为上面将ego的相关数据cat到了最后一个channel，此处其实就是进行分离
            # ego_query_hs = states_hs[:, :, self.agent_dim * self.fut_mode, :]
            ego_query_hs = states_hs[:, :, -1, :]  # [6, 1, 1024]
            ego_query_hs = ego_query_hs.unsqueeze(1).permute(0, 2, 1, 3)  # [6, 1, 1024] -> [6, 1, 1, 1024] -> [6, 1, 1, 1024]
            # motion_query_hs = states_hs[:, :, 0:self.agent_dim * self.fut_mode, :]
            motion_query_hs = states_hs[:, :, :-1, :]  # [6, 1, 1800, 1024]
            motion_query_hs = motion_query_hs.reshape(self.fut_ts, batch_size, -1, self.fut_ts, motion_query_hs.shape[-1])  # [6, 1, 300, 6, 1024]
            ego_fut_trajs_list = []
            motion_fut_trajs_list = []
            # 遍历未来每一帧
            for i in range(self.fut_ts):
                outputs_ego_trajs = self.ego_fut_decoder(  # [1, 3, 2]
                    ego_query_hs[i]  # [1, 1, 1024]
                ).reshape(batch_size, self.ego_fut_mode, 2)
                ego_fut_trajs_list.append(outputs_ego_trajs)
                outputs_agent_trajs = self.traj_branches[0](motion_query_hs[i])  # [1, 300, 6, 2]
                motion_fut_trajs_list.append(outputs_agent_trajs)

            ego_trajs = torch.stack(ego_fut_trajs_list, dim=2)  # [1, 3, 6, 2]，个人认为是直接预测了3个路径
            agent_trajs = torch.stack(motion_fut_trajs_list, dim=3)  # [1, 300, 6, 6, 2]
            agent_trajs = agent_trajs.reshape(batch_size, 1, self.agent_dim, self.fut_mode, -1)  # [1, 1, 300, 6, 6, 2]

        # future_hs = future_states_hs[:, :, 0:self.agent_dim * self.fut_mode, :].reshape(
        #     batch_size, self.agent_dim, self.fut_mode, -1)
        # current_hs = current_states[:, 0:self.agent_dim * self.fut_mode, :].reshape(
        #     batch_size, self.agent_dim, self.fut_mode, -1)
        #
        # motion_cls_hs = torch.cat((future_hs, current_hs), dim=-1)
        motion_cls_hs = torch.cat((  # [1, 300, 6, 3584]
            future_states_hs[:, :, 0:-1, :].reshape(batch_size, self.agent_dim, self.fut_mode, -1),  # [6, 1, 1800, 512] -> [1, 300, 6, 3072]
            current_states[:, 0:self.agent_dim * self.fut_mode, :].reshape(batch_size, self.agent_dim, self.fut_mode, -1)  # [1, 1800, 512] -> [1, 300, 6, 512]
        ), dim=-1)

        outputs_traj_class = self.traj_cls_branches[0](motion_cls_hs).squeeze(-1)  # [1, 300, 6]
        outputs_trajs_classes.append(outputs_traj_class)

        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = agent_trajs.permute(1, 0, 2, 3, 4)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)
        # outputs_trajs = outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1)
        # outputs_trajs_classes = outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1)

        outs = {
            'bev_embed': bev_embed,  # [10000, 1, 256]
            'all_cls_scores': outputs_classes,  # [3, 1, 300, 10]
            'all_bbox_preds': outputs_coords,  # [3, 1, 300, 10]
            'all_traj_preds': outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),  # [3, 1, 300, 6, 12]
            'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),  # [3, 1, 300, 6]
            'map_all_cls_scores': map_outputs_classes,  # [3, 1, 100, 3]
            'map_all_bbox_preds': map_outputs_coords,  # [3, 1, 100, 4]
            'map_all_pts_preds': map_outputs_pts_coords,  # [3, 1, 100, 20, 2]
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'map_enc_cls_scores': None,
            'map_enc_bbox_preds': None,
            'map_enc_pts_preds': None,
            'ego_fut_preds': ego_trajs,  # [1, 3, 6, 2]
            'loss_vae_gen': distribution_comp,  # dict
        }

        return outs

    def map_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                               self.map_num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts * 2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts * 2:self.fut_ts * 3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                  dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    def _map_get_target_single(self,
                               cls_score,
                               bbox_pred,
                               pts_pred,
                               gt_labels,
                               gt_bboxes,
                               gt_shifts_pts,
                               gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                                              gt_bboxes, gt_labels, gt_shifts_pts,
                                                              gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                                  gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                                          pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, traj_targets_list, traj_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    def map_get_targets(self,
                        cls_scores_list,
                        bbox_preds_list,
                        pts_preds_list,
                        gt_bboxes_list,
                        gt_labels_list,
                        gt_shifts_pts_list,
                        gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_planning(self,
                      ego_fut_preds,
                      ego_fut_gt,
                      ego_fut_masks,
                      ego_fut_cmd,
                      lane_preds,
                      lane_score_preds,
                      agent_preds,
                      agent_fut_preds,
                      agent_score_preds,
                      agent_fut_cls_preds):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        loss_plan_bound = self.loss_plan_bound(
            ego_fut_preds[ego_fut_cmd == 1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        loss_plan_col = self.loss_plan_col(
            ego_fut_preds[ego_fut_cmd == 1],
            agent_preds,
            agent_fut_preds,
            agent_score_preds,
            agent_fut_cls_preds,
            weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
        )

        loss_plan_dir = self.loss_plan_dir(
            ego_fut_preds[ego_fut_cmd == 1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_col = torch.nan_to_num(loss_plan_col)
            loss_plan_dir = torch.nan_to_num(loss_plan_dir)

        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = loss_plan_l1
        loss_plan_dict['loss_plan_bound'] = loss_plan_bound
        loss_plan_dict['loss_plan_col'] = loss_plan_col
        loss_plan_dict['loss_plan_dir'] = loss_plan_dir

        return loss_plan_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traj_preds,
                    traj_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # traj regression loss
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

        neg_inds = (bbox_weights[:, 0] == 0)
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks, neg_inds)

        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan],
            traj_targets[isnotnan],
            traj_weights[isnotnan],
            avg_factor=num_total_pos)

        if self.use_traj_lr_warmup:
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
                              num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    def get_best_fut_preds(self,
                           traj_preds,
                           traj_targets,
                           gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts * 2)

        return best_traj_preds

    def get_traj_cls_target(self,
                            traj_preds,
                            traj_targets,
                            gt_fut_masks,
                            neg_inds):
        """"Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    def map_loss_single(self,
                        cls_scores,
                        bbox_preds,
                        pts_preds,
                        gt_bboxes_list,
                        gt_labels_list,
                        gt_shifts_pts_list,
                        gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list, pts_preds_list,
                                               gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
                                               gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                      align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.map_dir_interval:, :] - \
                                 denormed_pts_preds[:, :-self.map_dir_interval, :]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:, :] - pts_targets[:, :-self.map_dir_interval, :]

        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    def distribution_loss(self, output):
        kl_loss = self.loss_vae_gen(output)
        return kl_loss

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None
            ):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        ego_fut_preds = preds_dicts['ego_fut_preds']
        distribution_pred = preds_dicts['loss_vae_gen']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list)

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            raise NotImplementedError
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        map_losses_cls, map_losses_bbox, map_losses_iou, \
        map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_traj'] = loss_traj[-1]
        loss_dict['loss_traj_cls'] = loss_traj_cls[-1]
        # loss from the last decoder layer
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        batch, num_agent = all_traj_preds[-1].shape[:2]
        agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
        agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        loss_plan_input = [ego_fut_preds, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                           map_all_pts_preds[-1], map_all_cls_scores[-1].sigmoid(),
                           all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                           all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid()]

        loss_planning_dict = self.loss_planning(*loss_plan_input)
        loss_dict['loss_plan_reg'] = loss_planning_dict['loss_plan_reg']
        loss_dict['loss_plan_bound'] = loss_planning_dict['loss_plan_bound']
        loss_dict['loss_plan_col'] = loss_planning_dict['loss_plan_col']
        loss_dict['loss_plan_dir'] = loss_planning_dict['loss_plan_dir']

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
                map_losses_cls[:-1],
                map_losses_bbox[:-1],
                map_losses_iou[:-1],
                map_losses_pts[:-1],
                map_losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
            map_enc_loss_pts, map_enc_loss_dir = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir

        loss_dict['loss_vae_gen'] = self.loss_vae_gen(distribution_pred)

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
            'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list

    def select_and_pad_pred_map(
            self,
            motion_pos,
            map_query,
            map_score,
            map_pos,
            map_thresh=0.5,
            dis_thresh=None,
            pe_normalization=True,
            use_fix_pad=False
    ):
        """select_and_pad_pred_map.
        Args:
            motion_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2]
            selected_padding_mask: [B*A, P1(+1)]
        """

        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0] ** 2 + map_pos[..., 1] ** 2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)],
                                            dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0] ** 2 + selected_map_dist[..., 1] ** 2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask

    def select_and_pad_query(
            self,
            query,
            query_pos,
            query_score,
            score_thresh=0.5,
            use_fix_pad=True
    ):
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)],
                                            dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask

    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
        """distribution_forward.
        Args:
            present_features:: output features of transformer model.
            future_distribution_inputs: the agent and ego gt trajectory in the future.
            noise: gaussian noise.
        Returns:
            sample: sample taken from present/future distribution
            present_distribution_mu: mean value of present gaussian distribution with shape (B, S, D)
            present_distribution_log_sigma: variance of present gaussian distribution with shape (B, S, D)
            future_distribution_mu: mean value of future gaussian distribution with shape (B, S, D)
            future_distribution_log_sigma: variance of future gaussian distribution with shape (B, S, D)
        """

        b = present_features.shape[0]
        c = present_features.shape[1]

        # 根据经过一系列计算得到的特征，计算当前的分布
        present_mu, present_log_sigma = self.present_distribution(  # [1, 1, 32], [1, 1, 32]
            present_features  # [1, 1801, 512]
        )

        # # 计算未来的分布
        # future_mu, future_log_sigma = None, None
        # if future_distribution_inputs is not None:
        #     # Concatenate future labels to z_t
        #     # future_features = future_distribution_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
        #     future_features = torch.cat([  # [1, 1801, 524]
        #         present_features,  # [1, 1801, 512]
        #         future_distribution_inputs  # [1, 1801, 12]
        #     ], dim=2)
        #     future_mu, future_log_sigma = self.future_distribution(future_features)

        # noise = torch.randn_like(present_mu) if noise is None else noise
        # if self.training:
        #     mu = future_mu
        #     sigma = torch.exp(future_log_sigma)
        # else:
        #     mu = present_mu
        #     sigma = torch.exp(present_log_sigma)
        # sample = mu + sigma * noise  # [1, 1, 32]

        # 计算未来的分布
        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # 训练阶段
            # Concatenate future labels to z_t
            # future_features = future_distribution_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
            future_features = torch.cat([  # [1, 1801, 524]
                present_features,  # [1, 1801, 512]
                future_distribution_inputs  # [1, 1801, 12]
            ], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)
            mu = future_mu  # [1, 1, 32]
            sigma = torch.exp(future_log_sigma)  # [1, 1, 32]
        else:
            # 测试阶段
            mu = present_mu  # [1, 1, 32]
            sigma = torch.exp(present_log_sigma)  # [1, 1, 32]
        noise = torch.randn_like(present_mu) if noise is None else noise
        sample = mu + sigma * noise  # [1, 1, 32]

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)  # [1, 1, 32] -> [1, 32, 1] -> [1, 32, 1801]

        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution

    def get_future_labels(self, gt_labels_3d, gt_attr_labels, ego_fut_trajs, device):

        """get_future_label.
        Args:
            gt_labels_3d: agent future 3d labels
            gt_attr_labels: agent future 3d labels
            ego_fut_trajs: ego future trajectory.
            device: gpu device id
        Returns:
            gt_trajs: [B, A, T, 2]
        """

        agent_dim = 300
        veh_list = [0, 1, 3, 4]
        mapped_class_names = [
            'car',
            'truck', 
            'construction_vehicle', # 忽略
            'bus',
            'trailer', 
            'barrier', # 忽略
            'motorcycle', # 忽略
            'bicycle', # 忽略
            'pedestrian', 
            'traffic_cone' # 忽略
        ]
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        batch_size = len(gt_labels_3d)

        # gt_label = gt_labels_3d[0]
        # gt_attr_label = gt_attr_labels[0]

        gt_fut_trajs_bz_list = []

        # 遍历所有batch
        for bz in range(batch_size):
            gt_fut_trajs_list = []

            # 从每个batch中提取label数据
            gt_label = gt_labels_3d[bz]
            gt_attr_label = gt_attr_labels[bz]

            # 遍历所有标签
            for i in range(gt_label.shape[0]):
                # 当这个标签在[0, 1, 3, 4]中，直接修改为0，那么从mapped_class_names中提取出的就是car
                gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
                box_name = mapped_class_names[gt_label[i]]
                # 忽略一些类别
                if box_name in ignore_list:
                    continue

                gt_fut_masks = gt_attr_label[i][self.fut_ts * 2:self.fut_ts * 3]
                num_valid_ts = sum(gt_fut_masks == 1)
                gt_fut_traj = gt_attr_label[i][:self.fut_ts * 2].reshape(-1, 2)
                gt_fut_traj = gt_fut_traj[:num_valid_ts]
                if gt_fut_traj.shape[0] == 0:
                    gt_fut_traj = torch.zeros([self.fut_ts - gt_fut_traj.shape[0], 2], device=device)
                if gt_fut_traj.shape[0] < self.fut_ts:
                    gt_fut_traj = torch.cat(
                        (gt_fut_traj, torch.zeros([self.fut_ts - gt_fut_traj.shape[0], 2], device=device)), 0)
                gt_fut_trajs_list.append(gt_fut_traj)

            if len(gt_fut_trajs_list) != 0 & len(gt_fut_trajs_list) < agent_dim:
                gt_fut_trajs = torch.cat(
                    (torch.stack(gt_fut_trajs_list),
                     torch.zeros([agent_dim - len(gt_fut_trajs_list), self.fut_ts, 2], device=device)), 0)
            else:
                gt_fut_trajs = torch.zeros([agent_dim, self.fut_ts, 2], device=device)

            gt_fut_trajs_bz_list.append(gt_fut_trajs)

        if len(gt_fut_trajs_bz_list) != 0:
            gt_trajs = torch.cat((torch.stack(gt_fut_trajs_bz_list).repeat(1, 6, 1, 1), ego_fut_trajs), dim=1)
        else:
            gt_trajs = ego_fut_trajs
        # future_states =  gt_trajs.reshape(batch_size, gt_trajs.shape[1], -1)

        # [bz, a, t, 2]
        return gt_trajs.reshape(batch_size, gt_trajs.shape[1], -1)

    def future_states_predict(self, batch_size, sample, hidden_states, current_states):
        """get_future_label.
              Args:
                  batch_size: batch size
                  sample: sample taken from present/future distribution
                  hidden_states: hidden states input of autoregressive model.
                  current_states: current states input of autoregressive model.
              Returns:
                  states_hs: the final features combined with the generative features and current features
                  future_states_hs: the generative features predicted by generate model(VAE)
              """

        # 为未来每一帧都生成一个样本
        future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)  # [1, 32, 1801] -> [1, 1, 32, 1801] -> [6, 1, 32, 1801]
        # 这里直接reshape了，感觉有点问题
        # 在future_prediction_input中，future_prediction_input[0] == future_prediction_input[1] == ...   == future_prediction_input[-1]
        # 这里分析future_prediction_input_0 = future_prediction_input[0, 0]
        # 在future_prediction_input_0中，future_prediction_input_0[:, 0] == future_prediction_input_0[:, 1] == ... == future_prediction_input_0[:, -1]
        # 在直接reshape之后，特征全被打乱
        # 所以我认为，正确的代码应该是进行砖置，然后让1801这个channel和batch_size合并
        # future_prediction_input = future_prediction_input.permute(0, 3, 1, 2).reshape(self.fut_ts, -1, self.latent_dim)  # [6, 1, 32, 1801] -> [6, 1801, 1, 32] -> [6, 1801, 32]
        future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim)  # [6, 1, 32, 1801] -> [6, 1801, 32]

        # 在输入的时候，hidden_states = current_states
        # 同上，我感觉此处的reshape很奇怪
        hidden_state = hidden_states.reshape(self.layer_dim, -1, int(self.embed_dims / 2))  # [1, 1801, 512] -> [4, 1801, 128]
        
        # 此处这个self.predict_model包含了GRU
        future_states = self.predict_model(  # [6, 1801, 512]
            future_prediction_input,  # [6, 1801, 32]
            hidden_state  # [4, 1801, 128]
        )

        current_states_hs = current_states.unsqueeze(0).repeat(self.fut_ts, 1, 1, 1)  # [1, 1801, 512] -> [1, 1, 1801, 512] -> [6, 1, 1801, 512]
        future_states_hs = future_states.reshape(self.fut_ts, batch_size, -1, future_states.shape[2])  # [6, 1801, 512] -> [6, 1, 1801, 512]

        if self.with_cur:
            states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)  # [6, 1, 1801, 1024]
        else:
            states_hs = future_states_hs

        return states_hs, future_states_hs
