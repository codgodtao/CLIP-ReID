"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised contrastive loss for CLIP-ReID stage1 (image-text alignment).

    改进点:
    1. temperature 默认 0.07 (原 1.0 过高, 导致对比信号过弱, 不同身份的
       image/text 特征难以拉开)。CLIP 系列及 SupCon 论文均使用 0.01~0.1。
    2. 对 image_features / text_features 做 L2 归一化后再计算 logits,
       使相似度仅取决于方向(余弦), 不受特征幅度影响。否则 stage1 只优化
       prompt_learner, 文本特征幅度会随训练漂移, 损失被幅度主导。
    """
    def __init__(self, device, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature

    def forward(self, text_features, image_features, t_label, i_targets):
        # L2 归一化, 使内积 = 余弦相似度, 范围 [-1, 1]
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)

        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size, batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # 防止 mask.sum(1)=0 (某个 batch 内没有正样本) 出现 nan
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = - mean_log_prob_pos.mean()

        return loss
