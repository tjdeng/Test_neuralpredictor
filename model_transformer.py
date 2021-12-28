# Most of this code is from https://github.com/ultmaster/neuralpredictor.pytorch
# which was authored by Yuge Zhang, 2020

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normalize_adj(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def get_laplacian_matrix(adj):
    rowsum = adj.sum(2)
    degree_matrix = torch.zeros(adj.shape, device=adj.device)
    for i in range(len(adj)):
        degree_matrix[i] = torch.diag(rowsum[i])
    return torch.sub(degree_matrix, adj)


class TransformerPredictorV1(nn.Module):
    def __init__(self, operation_dim, position_dim, trans_hidden=80, linear_hidden=96):
        super().__init__()
        # self.operation_encoding = nn.Linear(operation_dim, trans_hidden, bias=False)
        self.operation_encoding = nn.Embedding(operation_dim, trans_hidden, padding_idx=None)
        self.position_encoding = nn.Linear(position_dim, trans_hidden, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=trans_hidden, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear((operation_dim+2)*trans_hidden, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)

        self.pos_map = nn.Linear(position_dim, operation_dim+2)

    def forward(self, inputs):
        numv, adj, out = inputs["num_vertices"], inputs["adjacency"], inputs["features"]

        # print("adj:", adj[:2])
        # print("out:", out[:2])
        # exit()

        out = out.long()
        adj = get_laplacian_matrix(adj)
        opt_out = self.operation_encoding(out)
        # map dim
        pos_out = self.pos_map(adj).transpose(1, 2)
        pos_out = self.position_encoding(pos_out)

        trans_input = opt_out + pos_out
        out = self.transformer_encoder(trans_input)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out
