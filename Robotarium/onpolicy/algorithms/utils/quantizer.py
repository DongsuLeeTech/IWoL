import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.gat_comm import GAT_Comm
from onpolicy.algorithms.r_mappo.algorithm.transformer_comm import Transformer_Comm
from onpolicy.algorithms.r_mappo.algorithm.scheduler import Scheduler
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import scipy.linalg as linalg

class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(NearestEmbed, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x, weight_sg=False):
        # Flatten the input except for the last dimension
        flat_x = x.view(-1, self.embedding_dim)

        # Compute distances to embedding vectors
        distances = (flat_x.pow(2).sum(dim=1, keepdim=True)
                     + self.embeddings.weight.pow(2).sum(dim=1)
                     - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))

        # Find closest embedding
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view_as(x)

        # Stop gradients to embedding vectors if weight_sg is True
        if weight_sg:
            quantized = x + (quantized - x).detach()

        return quantized, encoding_indices

class VQVAEQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.1):
        super(VQVAEQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize codebook (embedding table)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # Flatten input except for the last dimension
        flat_x = x.view(-1, self.embedding_dim)

        # Compute distances between input and embedding vectors
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        # Get the nearest embedding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).view_as(x)

        # Compute commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss, encoding_indices


class RVQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        flat_x = x.view(-1, self.embedding_dim)

        # (입력-코드북) 거리 계산
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        # 최솟값 인덱스
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # 코드북에서 벡터 가져오기
        quantized = self.embedding(encoding_indices).view_as(x)

        # Loss 계산 (commitment + straight-through)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized_st = x + (quantized - x).detach()
        return quantized_st, loss, encoding_indices


class MultiStageRVQ(nn.Module):
    def __init__(self, num_stages, num_embeddings, embedding_dim, commitment_cost=0.1):
        super().__init__()
        # 스테이지 수만큼 SingleVQ를 만들고, ModuleList로 묶는다.
        self.num_stages = num_stages
        self.quantizers = nn.ModuleList([
            RVQ(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_stages)
        ])

    def forward(self, x):
        residual = x
        total_loss = 0
        all_indices = []

        for stage in range(self.num_stages):
            quantized, stage_loss, idx = self.quantizers[stage](residual)
            residual = residual - quantized

            total_loss += stage_loss
            all_indices.append(idx)

        return residual, total_loss, all_indices
