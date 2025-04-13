import os
os.environ["HF_ENDPOINT"]  = "https://hf-mirror.com"
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from transformers import AutoProcessor, CLIPConfig, CLIPModel, ViTModel, ViTConfig
import loralib as lora
import copy

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='clip2')
class CLIP2Detector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(self.config['num_heads']*768, 2)
        # self.head = nn.Linear(768, 2)
        self.loss_func = self.build_loss(config)
        self.mutli_head = MultiHeadRegionAttention(num_heads=self.config['num_heads'])
        
    def build_backbone(self, config):
        # prepare the backbone
        model_name = config['model_weight']
        if config['pretrained'] != '':
            clip_config = CLIPConfig.from_pretrained(model_name)
            model = CLIPModel(clip_config)
            model.load_state_dict(torch.load(config['pretrained']))
            backbone = model.vision_model
            print('加载微调模型')
        else:
            _, backbone = get_clip_visual(model_name=model_name)
        return backbone
        
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        self.partition_loss = PartitionLoss()
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['last_hidden_state']  # b 50 768
        feat, attn_weights = self.mutli_head(feat) 
        self.attn_weights = attn_weights
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss1 = self.loss_func(pred, label)
        loss = loss1
        # _, intra_loss, inter_loss = attention_loss(self.attn_weights)
        # print('loss1: {} intra_loss: {}, inter_loss: {}'.format(loss1, intra_loss, inter_loss))
        # loss = loss1 + intra_loss + inter_loss
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict


def get_clip_visual(model_name = "openai/clip-vit-base-patch16"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model


class MultiHeadRegionAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4):
        super(MultiHeadRegionAttention, self).__init__()
        self.num_heads = num_heads
        # 为每个head设置一个查询向量参数 (embed_dim)
        self.query = nn.Parameter(torch.randn(num_heads, embed_dim)) # num_head embed_dim
        # 可选：为每个头设置一个独立的线性变换用于输出特征
        self.fc = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads)])
    
    def forward(self, x):
        # x shape: (B, 50, 768)
        B, N, D = x.shape
        # 输出每个头的全局特征列表
        head_features = []
        attn_weights_all = []  # 用于存储每个head的注意力权重
        for i in range(self.num_heads):
            # 计算 head i 对每个patch特征的注意力权重: attn_weights shape (B, N)
            # 使用 query_i 与 x 的内积作为注意力分数
            query_i = self.query[i]  # shape (768,)
            attn_scores = torch.matmul(x, query_i)  # 内积：(B, N, D) · (D,) -> (B, N)
            attn_weights = F.softmax(attn_scores, dim=1)  # 对每个图像的N个patch归一化
            attn_weights_all.append(attn_weights)
            # 使用注意力权重对x加权求和，得到该头的全局特征 (B, D)
            head_feat = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, 1, N)*(B, N, D)->(B,1,D)->(B,D)
            # 通过该头的线性层（可选）
            head_feat = self.fc[i](head_feat)
            head_features.append(head_feat)
        # 拼接所有头的特征 (B, num_heads*D)
        multi_head_feat = torch.cat(head_features, dim=1)  # (B, num_heads*D)
        # 将所有 head 的注意力权重堆叠成 (B, num_heads, N)
        attn_weights_all = torch.stack(attn_weights_all, dim=1)
        return multi_head_feat, attn_weights_all

# class MultiHeadRegionAttention(nn.Module):
#     def __init__(self, embed_dim=768, num_heads=4):
#         super(MultiHeadRegionAttention, self).__init__()
#         self.num_heads = num_heads
#         # 为每个head设置一个查询向量参数 (embed_dim)
#         self.query = nn.Parameter(torch.randn(num_heads, embed_dim)) # num_head embed_dim
#         # 可选：为每个头设置一个独立的线性变换用于输出特征
#         self.fc = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads)])
    
#     def forward(self, x):
#         # x shape: (B, 50, 768)
#         B, N, D = x.shape
#         # 输出每个头的全局特征列表
#         head_features = []
#         attn_weights_all = []  # 用于存储每个head的注意力权重
#         for i in range(self.num_heads):
#             # 计算 head i 对每个patch特征的注意力权重: attn_weights shape (B, N)
#             # 使用 query_i 与 x 的内积作为注意力分数
#             query_i = self.query[i]  # shape (768,)
#             attn_scores = torch.matmul(x, query_i)  # 内积：(B, N, D) · (D,) -> (B, N)
#             attn_weights = F.softmax(attn_scores, dim=1)  # 对每个图像的N个patch归一化
#             attn_weights_all.append(attn_weights)
#             # 使用注意力权重对x加权求和，得到该头的全局特征 (B, D)
#             head_feat = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, 1, N)*(B, N, D)->(B,1,D)->(B,D)
#             # 通过该头的线性层（可选）
#             head_feat = self.fc[i](head_feat)
#             head_features.append(head_feat)
#         # 对所有头的输出求和
#         multi_head_feat = torch.sum(torch.stack(head_features, dim=1), dim=1)  # (B, D)
#         # 将所有 head 的注意力权重堆叠成 (B, num_heads, N)
#         attn_weights_all = torch.stack(attn_weights_all, dim=1)
#         return multi_head_feat, attn_weights_all

def attention_loss(attn_weights, lambda_intra=0.1, lambda_inter=1.0):
    """
    attn_weights: tensor of shape (B, num_heads, N)
    lambda_intra: 权重因子，用于调节内部集中损失的影响
    lambda_inter: 权重因子，用于调节跨head分散损失的影响
    """
    B, H, N = attn_weights.shape
    eps = 1e-6  # 避免 log(0)
    
    # Intra-head Loss: 计算每个 head 的熵
    # 每个 head 的熵: -sum(p * log(p))，然后平均
    entropy = - (attn_weights + eps) * torch.log(attn_weights + eps)
    intra_loss = entropy.sum(dim=2).mean()  # (B, H) -> scalar
    
    # Inter-head Loss: 对每个样本计算所有 head 之间的相似度
    inter_loss = 0.0
    for b in range(B):
        # A: (H, N)
        A = attn_weights[b]
        # Gram 矩阵 G: (H, H)
        G = torch.matmul(A, A.t())
        # 去除对角线（对角线上值为1，因为每个分布内积自身为1）
        off_diag = G - torch.diag(torch.diag(G))
        inter_loss += (off_diag ** 2).sum()
    inter_loss = inter_loss / B
    
    total_loss = lambda_intra * intra_loss + lambda_inter * inter_loss
    return total_loss, intra_loss, inter_loss
