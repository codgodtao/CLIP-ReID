import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)  # B, B
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosine_dist(x, y):
    """
    Cosine distance (不是余弦相似度).

    cos(x, y) ∈ [-1, 1], cos_dist = (1 - cos) / 2 ∈ [0, 1]
    cos_dist 越小表示越相似，与 euclidean_dist 一致，方便统一使用。

    注意: 此函数假设 x, y 已经 L2 归一化。
    """
    m, n = x.size(0), y.size(0)
    xy_intersection = torch.mm(x, y.t())  # cos similarity (假设已归一化)
    dist = (1. - xy_intersection) / 2  # 映射到 [0, 1]
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices.
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss with HARDER example mining.

    支持两种距离度量:
    1. 欧氏距离 (Euclidean): 适用于未归一化的原始特征
    2. 余弦距离 (Cosine):   适用于 L2 归一化后的特征

    重要修复: 当 normalize_feature=True 时, 必须使用余弦距离!
    原代码使用欧氏距离计算归一化特征的 triplet loss,
    导致 dist_ap ≈ √2 (难正样本), dist_an ≈ 0 (难负样本),
    dist_an - dist_ap ≈ -√2 → loss 永远 ≈ 1, 不可优化。
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            # 关键修复: 归一化特征必须用余弦距离!
            # 欧氏距离在归一化后范围是 [0, √2], 而余弦距离始终 [0, 1],
            # 且方向与 metric learning 的目标一致 (同身份近, 不同身份远)。
            global_feat = normalize(global_feat, axis=-1)
            dist_mat = cosine_dist(global_feat, global_feat)
        else:
            dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
