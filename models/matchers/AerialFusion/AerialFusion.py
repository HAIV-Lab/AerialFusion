from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from numpy import ndarray, dtype
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, List, Callable, Any
import math
import matplotlib.pyplot as plt
from numba import njit
import cv2
from torchsummary import summary
import kornia
import os

from .error import compute_reprojection_error

try:
    from flash_attn.modules.mha import FlashCrossAttention

    FOUND_OFFICIAL_FLASH = True
except ModuleNotFoundError:
    FOUND_OFFICIAL_FLASH = False

from sklearn.neighbors import NearestNeighbors

from .Dynamic_Process import Dynamic_Matcher                        # .
from .Dynamic_Process.Dynamic_Matcher import SkewedGaussianGPU      # .
from .Dynamic_Process import Dynamic_Fusion                         # .

torch.backends.cudnn.deterministic = True

device = torch.device(os.environ.get("AERIALFUSION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
        kpts: torch.Tensor,
        size: Optional[List[int]] = None,
        shape: Optional[List[int]] = None) -> torch.Tensor:
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one * w, one * h])[None]
    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) / 2
    kpts = (kpts - shift[:, None]) / scale[:, None, None]

    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def apply_cached_rotary_emb(
        freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    """可学习-傅里叶-位置-编码器"""
    def __init__(self, M: int, dim: int, F_dim: int = None,
                 gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        encode position vector
        编码位置向量
        """
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return repeat(emb, '... n -> ... (n r)', r=2)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super(TokenConfidence, self).__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ get confidence tokens """
        return (
            self.token(desc0.detach().float()).squeeze(-1),
            self.token(desc1.detach().float()).squeeze(-1))
    
    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices)
        correct1 = (la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices)
        return (self.loss_fn(logit0, correct0.float()).mean(-1)
                + self.loss_fn(logit1, correct1.float()).mean(-1)) / 2.0


class FastAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.s = dim ** -0.5

    def forward(self, q, k, v) -> torch.Tensor:
        if hasattr(F, 'scaled_dot_product_attention'):
            q, k, v = [x.contiguous() for x in [q, k, v]]
            return F.scaled_dot_product_attention(q, k, v)
        else:
            s = self.s
            attn = F.softmax(torch.einsum('...id,...jd->...ij', q, k) * s, -1)
            return torch.einsum('...ij,...jd->...id', attn, v)


class FlashAttention(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        if FOUND_OFFICIAL_FLASH:
            self.flash = FlashCrossAttention()

    def forward(self, q, k, v) -> torch.Tensor:
        if FOUND_OFFICIAL_FLASH:
            q, k, v = [x.transpose(-2, -3) for x in [q, k, v]]
            m = self.flash(q.half(), torch.stack([k, v], 2).half())
            return m.transpose(-2, -3).to(q.dtype)
        else:
            args = [x.half().contiguous() for x in [q, k, v]]
            return F.scaled_dot_product_attention(*args).to(q.dtype)


class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        attn = FlashAttention if flash else FastAttention
        self.inner_attn = attn(self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )

    def _forward(self, x: torch.Tensor,
                 encoding: Optional[torch.Tensor] = None):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b n (h d three) -> b h n d three',
                        three=3, h=self.num_heads)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if encoding is not None:
            q = apply_cached_rotary_emb(encoding, q)
            k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v)                                                                              # 内部 注意
        message = self.out_proj(rearrange(context, 'b h n d -> b n (h d)'))
        return x + self.ffn(torch.cat([x, message], -1))

    def forward(self, x0, x1, encoding0=None, encoding1=None):
        return self._forward(x0, encoding0), self._forward(x1, encoding1)


class CrossTransformer(nn.Module):
    """交叉transformer"""
    def __init__(self, embed_dim: int, num_heads: int,
                 flash: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )

        if flash:
            self.flash = FastAttention(dim_head)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            (qk0, qk1, v0, v1))
        if self.flash is not None:
            m0 = self.flash(qk0, qk1, v1)
            m1 = self.flash(qk1, qk0, v0)
        else:
            qk0, qk1 = qk0 * self.scale ** 0.5, qk1 * self.scale ** 0.5
            sim = torch.einsum('b h i d, b h j d -> b h i j', qk0, qk1)
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum('bhij, bhjd -> bhid', attn01, v1)
            m1 = torch.einsum('bhji, bhjd -> bhid', attn10.transpose(-2, -1), v0)
        m0, m1 = self.map_(lambda t: rearrange(t, 'b h n d -> b n (h d)'),
                           m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


def sigmoid_log_double_softmax(sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """
    create the log assignment matrix from logits and similarity;
    根据对数和相似度创建日志分配矩阵;
    """
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = (scores0 + scores1 + certainties)
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    """匹配任务"""
    def __init__(self, dim: float) -> None:
        super(MatchAssignment, self).__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """
        build assignment matrix from descriptors
        描述符 构建 匹配矩阵
        """
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d ** .25, mdesc1 / d ** .25
        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def scores(self, desc0: torch.Tensor, desc1: torch.Tensor):
        m0 = torch.sigmoid(self.matchability(desc0)).squeeze(-1)
        m1 = torch.sigmoid(self.matchability(desc1)).squeeze(-1)
        return m0, m1


def filter_matches(scores: torch.Tensor, th: float):
    """
    obtain matches from a log assignment matrix [B x M+1 x N+1]
    匹配矩阵 获取 匹配关系
    """
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    mutual0 = torch.arange(m0.shape[1]).to(m0)[None] == m1.gather(1, m0)
    mutual1 = torch.arange(m1.shape[1]).to(m1)[None] == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    if th is not None:
        valid0 = mutual0 & (mscores0 > th)
    else:
        valid0 = mutual0
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, m0.new_tensor(-1))
    m1 = torch.where(valid1, m1, m1.new_tensor(-1))
    return m0, m1, mscores0, mscores1

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron 多层感知机"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def Point_Proc(kpt_t_track_0, kpt_t_track_1, desc_t_track_0, desc_t_track_1, scores_t_track_0, scores_t_track_1, maxl, ratio):
    """处理特征点跟踪序列"""
    kpt_t_0, kpt_t_1 = [], []
    desc_t_0, desc_t_1 = [], []
    scores_t_0, scores_t_1 = [], []
    for i in range(0, len(kpt_t_track_0)):
        if len(kpt_t_track_0[i]) >= int(round(maxl * ratio)):
            assert len(kpt_t_track_0[i]) == len(desc_t_track_0[i])
            kpt_t_0.append(kpt_t_track_0[i])
            desc_t_0.append(desc_t_track_0[i])
            scores_t_0.append(scores_t_track_0[i])

    for j in range(0, len(kpt_t_track_1)):
        if len(kpt_t_track_1[j]) >= int(round(maxl * ratio)):
            assert len(kpt_t_track_1[j]) == len(desc_t_track_1[j])
            kpt_t_1.append(kpt_t_track_1[j])
            desc_t_1.append(desc_t_track_1[j])
            scores_t_1.append(scores_t_track_1[j])

    return kpt_t_0, kpt_t_1, desc_t_0, desc_t_1, scores_t_0, scores_t_1

def predict_future_positions(kpt_sequence, predict_steps = 1):
    """
    预测所有特征点未来N帧的位置
    :param kpt_sequence: (T, N, 2)的历史序列
    :param predict_steps: 预测帧数
    :return: (N, 2)的预测位置数组
    """
    T, N, _ = kpt_sequence.shape
    predictors = [Dynamic_Matcher.FeaturePointPredictor(kpt_sequence[0, i]) for i in range(N)]

    for t in range(1, T):
        for i in range(N):
            predictors[i].update(kpt_sequence[t, i])

    future_pos = np.array([predictors[i].predict_next(predict_steps) for i in range(N)])

    return future_pos

def compute_motion_vectors(feature_coords):
    diffs = np.diff(feature_coords, axis=0)                        # 结果形状 (0.5 * mal - 1, n, 2)

    weights = np.arange(1, diffs.shape[0] + 1)                     # 每帧位移系数, 线性权重

    avg_motion = np.average(diffs, weights = weights, axis=0)       # 结果形状 (n, 2)

    return avg_motion

class AerialFusion(nn.Module):              # nn.Module
    default_conf = {
        'name': 'lightglue',                # just for interfacing
        'input_dim': 256,                   # input descriptor dimension (autoselected from weights)
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'n_layers': 9,
        'num_heads': 4,
        'flash': False,                     # enable FlashAttention
        'mp': False,                        # enable mixed precision
        'filter_threshold': 0.1,            # match threshold
        'depth_confidence': -1,             # -1 is no early stopping, recommend: 0.95
        'width_confidence': -1,             # -1 is no point pruning, recommend: 0.99
        'weights': './weights/superpoint_lightglue.pth',
        'weights_test': './weights/checkpoint_best.tar',                                         # ./weights/checkpoint_best.tar
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,}}

    pretrained = {'superpoint': ('superpoint_lightglue', 256)}

    def __init__(self, pretrained = 'superpoint', **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        if pretrained is not None:
            assert (pretrained in list(self.pretrained.keys()))
            self.conf['weights'], self.conf['input_dim'] = self.pretrained[pretrained]

        self.conf = conf = SimpleNamespace(**self.conf)
        
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(
                conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.self_attn = nn.ModuleList(                                                             # 自注意力
            [Transformer(d, h, conf.flash) for _ in range(n)])
        self.cross_attn = nn.ModuleList(                                                            # 交叉注意力
            [CrossTransformer(d, h, conf.flash) for _ in range(n)])

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])

        """Feature Fusion"""
        self.fusion = Dynamic_Fusion.Dynamic_Fusion()

        path = Path(__file__).parent
        path_test = path / self.conf.weights_test
        state_dict = torch.load(str(path_test), map_location = 'cpu')                                       # map_location='cpu'
        matcher_params = {
            k.replace('matcher.', ''): v
            for k, v in state_dict["model"].items()
            if k.startswith('matcher.')}
        self.load_state_dict(matcher_params, strict = False)                                                 # 加载原有 keys

        print('Loaded AerialFusion model !!!')

    def forward(self, data: dict = None,
                 kpt_t_track_0 = None,
                 kpt_t_track_1 = None,
                 desc_t_track_0 = None,
                 desc_t_track_1 = None,
                 scores_t_track_0 = None,
                 scores_t_track_1 = None,
                 scales0 = None,
                 scales1 = None,
                 img_width = None,
                 img_height = None) -> dict:
        """
        Match keypoints and descriptors between two images

        Input (dict):
            keypoints0: [B x M x 2], descriptors0: [B x M x D]
            keypoints1: [B x N x 2], descriptors1: [B x N x D]

        Output (dict):
            matches0: [B x M], matching_scores0: [B x M]
            matches1: [B x N], matching_scores1: [B x N]
            log_assignment: [B x M+1 x N+1]
        """

        global i
        kpts0_or, kpts1_or = data['keypoints0'], data['keypoints1']

        """记录预测相关属性"""
        scores_pre = torch.tensor([])
        m0_pre, m1_pre = torch.tensor([]), torch.tensor([])
        all_desc0_pre, all_desc1_pre = [], []
        mscores0_pre, mscores1_pre = torch.tensor([]), torch.tensor([])
        prune0_pre, prune1_pre = torch.tensor([]), torch.tensor([])
        pre_kpts0_torch, pre_kpts1_torch = torch.tensor([]), torch.tensor([])
        pre_m_kpts0_torch, pre_m_kpts1_torch = torch.tensor([]), torch.tensor([])
        Skewed_Gaussian_0, Skewed_Gaussian_1 = [], []
        Trans_H_torch, Fusion_torch = torch.tensor([]), torch.tensor([])

        """SGauss匹配运行,进行梯度计算"""
        if (kpt_t_track_0 != None and kpt_t_track_1 != None
                and desc_t_track_0 != None and desc_t_track_1 != None
                and scores_t_track_0 != None and scores_t_track_1 != None):

            if (len(kpt_t_track_0) > 0 and len(kpt_t_track_1) > 0
                    and len(desc_t_track_0) > 0 and len(desc_t_track_1) > 0
                    and len(scores_t_track_0) > 0 and len(scores_t_track_1) > 0):

                kpt_t_0, kpt_t_1, desc_t_0, desc_t_1, scores_t_0, scores_t_1 = Point_Proc(kpt_t_track_0,
                                                                                          kpt_t_track_1,
                                                                                          desc_t_track_0,
                                                                                          desc_t_track_1,
                                                                                          scores_t_track_0,
                                                                                          scores_t_track_1,
                                                                                          maxl = 16,
                                                                                          ratio = 0.5)

                """获取 跟踪序列 最短长度值"""
                if len(kpt_t_0) > 0 and len(kpt_t_1) > 0:
                    print("building SG:", 666666)
                    min_len_0 = len(kpt_t_0[0])
                    for i in range(0, len(kpt_t_0)):
                        if len(kpt_t_0[i]) < min_len_0:
                            min_len_0 = len(kpt_t_0[i])

                    min_len_1 = len(kpt_t_1[0])
                    for j in range(0, len(kpt_t_1)):
                        if len(kpt_t_1[j]) < min_len_1:
                            min_len_1 = len(kpt_t_1[j])

                    feature_kpts0, feature_kpts1 = [], []
                    feature_desc0, feature_desc1 = [], []
                    feature_scores_0, feature_scores_1 = [], []
                    for n in range(0, min_len_0):
                        kpts0_ = [item_kpt_0[-1 - n] for item_kpt_0 in kpt_t_0]
                        desc0_ = [item_desc_0[-1 - n] for item_desc_0 in desc_t_0]
                        scores0_ = [item_score_0[-1 - n] for item_score_0 in scores_t_0]

                        feature_kpts0.append(kpts0_)
                        feature_desc0.append(desc0_)
                        feature_scores_0.append(scores0_)

                    for m in range(0, min_len_1):
                        kpts1_ = [item_kpt_1[-1 - m] for item_kpt_1 in kpt_t_1]
                        desc1_ = [item_desc_1[-1 - m] for item_desc_1 in desc_t_1]
                        scores1_ = [item_score_1[-1 - m] for item_score_1 in scores_t_1]

                        feature_kpts1.append(kpts1_)
                        feature_desc1.append(desc1_)
                        feature_scores_1.append(scores1_)

                    """
                    feature_kpts0, feature_kpts1: 包含相同 跟踪长度 特征点-跟踪 序列
                    feature_desc0, feature_desc1: 包含相同 跟踪长度 描述符-跟踪 序列
                    """

                    predicted_pos_0 = predict_future_positions(np.array(feature_kpts0)[:-1],
                                                               predict_steps=1)  # 卡尔曼滤波 预测 当前帧特征点
                    predicted_pos_1 = predict_future_positions(np.array(feature_kpts1)[:-1], predict_steps=1)

                    predicted_pos_0 = np.transpose(predicted_pos_0, (2, 0, 1))  # 预测 当前帧特征点位置 作为均值向量
                    predicted_pos_1 = np.transpose(predicted_pos_1, (2, 0, 1))

                    direction_0 = compute_motion_vectors(np.array(feature_kpts0)[:-1])  # 获取 前帧点 位移方向向量 及其 权重均值
                    direction_1 = compute_motion_vectors(np.array(feature_kpts1)[:-1])

                    mu_scores_0 = np.array(feature_scores_0)[-2]  # 获得 前帧特征点置信度 作为均值处概率（即最大概率）
                    mu_scores_1 = np.array(feature_scores_1)[-2]

                    Skewed_Gaussian_GPU = Dynamic_Matcher.SkewedGaussianGPU(width=img_width, height=img_height, grid_stride=1)
                    gaussian_lut = Skewed_Gaussian_GPU.build_gaussian_lut(size=256, sigma=3.0)                  # 生成高斯分布查询表

                    predicted_pos_0 = torch.from_numpy(predicted_pos_0)
                    predicted_pos_1 = torch.from_numpy(predicted_pos_1)

                    pre_kpts0_torch = predicted_pos_0.squeeze(0)
                    pre_kpts1_torch = predicted_pos_1.squeeze(0)

                    """特征点位置处理"""

                    Skewed_Gaussian_0 = Skewed_Gaussian_GPU.compute(pre_kpts0_torch, -1 * direction_0, mu_scores_0, gaussian_lut)
                    Skewed_Gaussian_1 = Skewed_Gaussian_GPU.compute(pre_kpts1_torch, -1 * direction_1, mu_scores_1, gaussian_lut)

                    b, m, _ = predicted_pos_0.shape
                    b, n, _ = predicted_pos_1.shape

                    kpts0 = normalize_keypoints(predicted_pos_0, size=None, shape=[1, 1, img_height, img_width])
                    kpts1 = normalize_keypoints(predicted_pos_1, size=None, shape=[1, 1, img_height, img_width])

                    assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
                    assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)

                    desc0 = torch.from_numpy(np.array(feature_desc0[-2])).detach()
                    desc1 = torch.from_numpy(np.array(feature_desc1[-2])).detach()

                    assert (desc0.shape[-1] == self.conf.input_dim)
                    assert (desc1.shape[-1] == self.conf.input_dim)

                    if torch.is_autocast_enabled():
                        desc0 = desc0.half()
                        desc1 = desc1.half()

                    desc0 = self.input_proj(desc0)
                    desc1 = self.input_proj(desc1)

                    kpts0, kpts1 = kpts0.to(device).to(torch.float32), kpts1.to(device).to(torch.float32)
                    desc0, desc1 = desc0.to(torch.float32).to(device), desc1.to(torch.float32).to(device)

                    """ Keypoint_Motion Description. 描述子 + 编码(特征点位置 + 特征点运动) """
                    desc_SG_0 = Dynamic_Matcher.extract_circular_patches_gpu(pre_kpts0_torch, Skewed_Gaussian_0)
                    desc_SG_1 = Dynamic_Matcher.extract_circular_patches_gpu(pre_kpts1_torch, Skewed_Gaussian_1)
                    desc0 = desc0.unsqueeze(0) + desc_SG_0.unsqueeze(0).to(device)
                    desc1 = desc1.unsqueeze(0) + desc_SG_1.unsqueeze(0).to(device)

                    encoding0 = self.posenc(kpts0)
                    encoding1 = self.posenc(kpts1)

                    ind0 = torch.arange(0, m).to(device=kpts0.device)[None]
                    ind1 = torch.arange(0, n).to(device=kpts0.device)[None]
                    prune0_pre = torch.ones_like(ind0)                        # store layer where pruning is detected 检测修剪 存储层
                    prune1_pre = torch.ones_like(ind1)
                    dec, wic = self.conf.depth_confidence and not self.training, self.conf.width_confidence and not self.training
                    token0, token1 = None, None

                    """self + cross attention"""
                    for i in range(self.conf.n_layers):
                        desc0, desc1 = self.self_attn[i](desc0, desc1, encoding0, encoding1)
                        desc0, desc1 = self.cross_attn[i](desc0, desc1)

                        if self.training or i == self.conf.n_layers - 1:
                            all_desc0_pre.append(desc0)
                            all_desc1_pre.append(desc1)
                            continue  # no early stopping or adaptive width at last layer 没有提前停止或自适应宽度的最后一层

                        if dec > 0:  # early stopping 提前停止
                            token0, token1 = self.token_confidence[i](desc0, desc1)
                            if self.stop(token0, token1, self.conf_th(i), dec, m + n):
                                break
                        if wic > 0:  # point pruning 特征点 修剪
                            match0, match1 = self.log_assignment[i].scores(desc0, desc1)
                            mask0 = self.get_mask(token0, match0, self.conf_th(i), 1 - wic)
                            mask1 = self.get_mask(token1, match1, self.conf_th(i), 1 - wic)
                            ind0, ind1 = ind0[mask0][None], ind1[mask1][None]
                            desc0, desc1 = desc0[mask0][None], desc1[mask1][None]
                            if desc0.shape[-2] == 0 or desc1.shape[-2] == 0:
                                break
                            encoding0 = encoding0[:, :, mask0][:, None]
                            encoding1 = encoding1[:, :, mask1][:, None]

                        prune0_pre[:, ind0] += 1
                        prune1_pre[:, ind1] += 1

                    if wic > 0:      # scatter with indices after pruning 修剪 指数分数
                        scores__pre, _ = self.log_assignment[i](desc0, desc1)
                        dt_pre, dev_pre = scores__pre.dtype, scores__pre.device
                        scores_pre = torch.zeros(b, m + 1, n + 1, dtype=dt_pre, device=dev_pre)
                        scores_pre[:, :-1, :-1] = -torch.tensor(float('inf'))
                        scores_pre[:, ind0[0], -1] = scores__pre[:, :-1, -1]
                        scores_pre[:, -1, ind1[0]] = scores__pre[:, -1, :-1]
                        x, y = torch.meshgrid(ind0[0], ind1[0])
                        scores_pre[:, x, y] = scores__pre[:, :-1, :-1]
                    else:
                        scores_pre, _ = self.log_assignment[i](desc0, desc1)

                    """当前帧 预测匹配"""
                    m0_pre, m1_pre, mscores0_pre, mscores1_pre = filter_matches(scores_pre, self.conf.filter_threshold)         # 过滤 匹配
                    matches_pre = m0_pre[0].to('cpu', non_blocking=True)                                                        # .cpu().numpy()
                    valid_pre = matches_pre > -1                                                                                # bool

                    pre_m_kpts0_torch, pre_m_kpts1_torch = pre_kpts0_torch[valid_pre], pre_kpts1_torch[matches_pre[valid_pre]]  # 最佳匹配的特征点

        """原始匹配运行"""
        b, m, _ = kpts0_or.shape
        b, n, _ = kpts1_or.shape

        kpts0 = normalize_keypoints(kpts0_or, size=data.get('image_size0'), shape=data['img0'].shape)
        kpts1 = normalize_keypoints(kpts1_or, size=data.get('image_size1'), shape=data['img1'].shape)

        assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
        assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)

        desc0 = data['descriptors0'].detach()
        desc1 = data['descriptors1'].detach()

        assert (desc0.shape[-1] == self.conf.input_dim)
        assert (desc1.shape[-1] == self.conf.input_dim)

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        kpts0, kpts1 = kpts0.to(device).to(torch.float32), kpts1.to(device).to(torch.float32)
        desc0, desc1 = desc0.to(device).to(torch.float32), desc1.to(device).to(torch.float32)

        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        ind0 = torch.arange(0, m).to(device=kpts0.device)[None]
        ind1 = torch.arange(0, n).to(device=kpts0.device)[None]
        prune0 = torch.ones_like(ind0)                                    # store layer where pruning is detected 检测修剪 存储层
        prune1 = torch.ones_like(ind1)
        dec, wic = self.conf.depth_confidence, self.conf.width_confidence
        token0, token1 = None, None

        """记录原始desc0, desc1"""
        all_desc0, all_desc1 = [], []

        for i in range(self.conf.n_layers):
            desc0, desc1 = self.self_attn[i](desc0, desc1, encoding0, encoding1)  # 自注意力
            desc0, desc1 = self.cross_attn[i](desc0, desc1)  # 交叉注意力

            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer 没有提前停止或自适应宽度的最后一层

            if dec > 0:  # early stopping 提前停止
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.stop(token0, token1, self.conf_th(i), dec, m + n):
                    break

            if wic > 0:  # point pruning 特征点 修剪
                match0, match1 = self.log_assignment[i].scores(desc0, desc1)
                mask0 = self.get_mask(token0, match0, self.conf_th(i), 1 - wic)
                mask1 = self.get_mask(token1, match1, self.conf_th(i), 1 - wic)
                ind0, ind1 = ind0[mask0][None], ind1[mask1][None]
                desc0, desc1 = desc0[mask0][None], desc1[mask1][None]
                if desc0.shape[-2] == 0 or desc1.shape[-2] == 0:
                    break
                encoding0 = encoding0[:, :, mask0][:, None]
                encoding1 = encoding1[:, :, mask1][:, None]

            prune0[:, ind0] += 1
            prune1[:, ind1] += 1

        if wic > 0:  # scatter with indices after pruning 修剪 指数分数
            scores_, _ = self.log_assignment[i](desc0, desc1)
            dt, dev = scores_.dtype, scores_.device
            scores = torch.zeros(b, m + 1, n + 1, dtype=dt, device=dev)
            scores[:, :-1, :-1] = -torch.tensor(float('inf'))
            scores[:, ind0[0], -1] = scores_[:, :-1, -1]
            scores[:, -1, ind1[0]] = scores_[:, -1, :-1]
            x, y = torch.meshgrid(ind0[0], ind1[0])
            scores[:, x, y] = scores_[:, :-1, :-1]
        else:
            scores, _ = self.log_assignment[i](desc0, desc1)


        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)              # 过滤 匹配
        matches = m0[0].to('cpu', non_blocking=True)
        valid = matches > -1  # bool

        kpts0_torch = kpts0_or.to('cpu', non_blocking=True).squeeze(0)
        kpts1_torch = kpts1_or.to('cpu', non_blocking=True).squeeze(0)

        """特征点位置处理"""

        """当前帧原始匹配"""
        m_kpts0_torch, m_kpts1_torch = kpts0_torch[valid], kpts1_torch[matches[valid]]                   # 最佳匹配的特征点

        if len(pre_m_kpts0_torch) > 0 and len(pre_m_kpts1_torch) > 0:
            pre_kpts0_torch_ = torch.vstack((pre_kpts0_torch, kpts0_torch))
            pre_kpts1_torch_ = torch.vstack((pre_kpts1_torch, kpts1_torch))
            pre_m_kpts0_torch_ = torch.vstack((pre_m_kpts0_torch, m_kpts0_torch))
            pre_m_kpts1_torch_ = torch.vstack((pre_m_kpts1_torch, m_kpts1_torch))
        else:
            pre_kpts0_torch_ = kpts0_torch
            pre_kpts1_torch_ = kpts1_torch
            pre_m_kpts0_torch_ = m_kpts0_torch
            pre_m_kpts1_torch_ = m_kpts1_torch



        frame_0_torch = data["img0"].to('cpu', non_blocking = True) * 255.0
        frame_1_torch = data["img1"].to('cpu', non_blocking = True) * 255.0
        frame_0 = np.array(data["img0"].squeeze(0).permute(1, 2, 0).cpu()) * 255.0
        frame_1 = np.array(data["img1"].squeeze(0).permute(1, 2, 0).cpu()) * 255.0

        if pre_m_kpts0_torch_.shape[0] >= 5 and pre_m_kpts1_torch_.shape[0] >= 5 and data["modal"] == 'multi':
            """实现TransFormation"""
            Trans_H_torch = kornia.geometry.homography.find_homography_dlt(
                        pre_m_kpts1_torch_.unsqueeze(0), pre_m_kpts0_torch_.unsqueeze(0),                               # 添加batch维度
                        weights = None).squeeze(0)

            t_frame_1_torch = kornia.geometry.transform.warp_perspective(frame_1_torch, Trans_H_torch.unsqueeze(0), dsize = (img_height, img_width))
            t_frame_1 = np.array(t_frame_1_torch.squeeze(0).permute(1, 2, 0).cpu())
            frame_0_gray = cv2.cvtColor(frame_0, cv2.COLOR_RGB2GRAY)
            t_frame_1_gray = cv2.cvtColor(t_frame_1, cv2.COLOR_RGB2GRAY)

            print(frame_0_gray.shape, np.array(t_frame_1_gray).shape,frame_0.shape)

            """运行Dynamic_Fusion"""
            if Skewed_Gaussian_0 is not None and Skewed_Gaussian_1 is not None:
                if len(Skewed_Gaussian_0) > 0 and len(Skewed_Gaussian_1) > 0:
                    Fusion_torch = self.fusion.fusion_color(frame_0_gray, t_frame_1_gray, frame_0,                      # vi_gray, ir, vi_color
                                                      SG_vi = Skewed_Gaussian_0, SG_ir = Skewed_Gaussian_1)
                else:
                    Fusion_torch = self.fusion.fusion_color(frame_0_gray, t_frame_1_gray, frame_0, SG_vi = None, SG_ir = None)

        if len(all_desc0_pre) > 0 and len(all_desc1_pre) > 0:
            return {
                "return_or":
                    {'matches0': m0,
                    'matches1': m1,
                    'matching_scores0': mscores0,
                    'matching_scores1': mscores1,
                    'log_assignment': scores,
                    'prune0': prune0,
                    'prune1': prune1,
                    'kpts0': kpts0_torch,
                    'kpts1': kpts1_torch,
                    'm_kpts0': m_kpts0_torch,
                    'm_kpts1': m_kpts1_torch},

                "return_pre":
                    {'matches0': m0_pre,
                    'matches1': m1_pre,
                    'matching_scores0': mscores0_pre,
                    'matching_scores1': mscores1_pre,
                    'log_assignment': scores_pre,
                    'prune0': prune0_pre,
                    'prune1': prune1_pre,
                    'kpts0': pre_kpts0_torch,
                    'kpts1': pre_kpts1_torch,
                    'm_kpts0': pre_m_kpts0_torch,
                    'm_kpts1': pre_m_kpts1_torch,

                'SG_vi': Skewed_Gaussian_0,
                'SG_ir': Skewed_Gaussian_1,
                "Fusion": Fusion_torch,
                "Trans_H": Trans_H_torch},

            "pre_kpts0": pre_kpts0_torch_,
            "pre_kpts1": pre_kpts1_torch_,
            "pre_m_kpts0": pre_m_kpts0_torch_,
            "pre_m_kpts1": pre_m_kpts1_torch_
            }
        else:
            return {
                "return_or":
                    {'matches0': m0,
                     'matches1': m1,
                     'matching_scores0': mscores0,
                     'matching_scores1': mscores1,
                     'log_assignment': scores,
                     'prune0': prune0,
                     'prune1': prune1,
                     'kpts0': kpts0_torch,
                     'kpts1': kpts1_torch,
                     'm_kpts0': m_kpts0_torch,
                     'm_kpts1': m_kpts1_torch},

                "return_pre":
                    {'matches0': m0_pre,
                     'matches1': m1_pre,
                     'matching_scores0': mscores0_pre,
                     'matching_scores1': mscores1_pre,
                     'log_assignment': scores_pre,
                     'prune0': prune0_pre,
                     'prune1': prune1_pre,
                     'kpts0': pre_kpts0_torch,
                     'kpts1': pre_kpts1_torch,
                     'm_kpts0': pre_m_kpts0_torch,
                     'm_kpts1': pre_m_kpts1_torch,

                     'SG_vi': Skewed_Gaussian_0,
                     'SG_ir': Skewed_Gaussian_1,
                     "Fusion": Fusion_torch,
                     "Trans_H": Trans_H_torch},

                "pre_kpts0": pre_kpts0_torch_,
                "pre_kpts1": pre_kpts1_torch_,
                "pre_m_kpts0": pre_m_kpts0_torch_,
                "pre_m_kpts1": pre_m_kpts1_torch_
            }


    def conf_th(self, i: int) -> float:
        """
         scaled confidence threshold
         缩放 置信阈值
        """
        return np.clip(0.8 + 0.1 * np.exp(-4.0 * i / self.conf.n_layers), 0, 1)

    def get_mask(self, confidence: torch.Tensor, match: torch.Tensor,
                 conf_th: float, match_th: float) -> torch.Tensor:
        """ mask points which should be removed """
        if conf_th and confidence is not None:
            mask = torch.where(confidence > conf_th, match,
                               match.new_tensor(1.0)) > match_th
        else:
            mask = match > match_th
        return mask

    def stop(self, token0: torch.Tensor, token1: torch.Tensor,
             conf_th: float, inl_th: float, seql: int) -> torch.Tensor:
        """ evaluate stopping condition"""
        tokens = torch.cat([token0, token1], -1)
        if conf_th:
            pos = 1.0 - (tokens < conf_th).float().sum() / seql
            return pos > inl_th
        else:
            return tokens.mean() > inl_th
