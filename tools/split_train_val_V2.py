import numpy as np
import h5py
import dgl
import h5py
import hashlib
import numpy as np
from copy import deepcopy
import random
from scipy import sparse as sp

import torch
from torch.utils.data import Dataset


def generate_lapla_matrix(adjacency, note):
    normalized_lapla_matrix = []
    unnormalized_lapla_matrix = []

    for index in range(len(adjacency)):
        # normalized lapla_matrix
        adj_matrix = adjacency[index]
        degree = torch.tensor(np.sum(adj_matrix, axis=1))
        degree = sp.diags(dgl.backend.asnumpy(degree).clip(1) ** -0.5, dtype=float)
        normalized_lapla = np.array(sp.eye(adj_matrix.shape[0]) - degree * adj_matrix * degree)
        normalized_lapla_matrix.append(normalized_lapla)

        # un-normalized lapla_matrix
        adj_matrix = adjacency[index]
        degree = np.diag(np.sum(adj_matrix, axis=1))
        unnormalized_lapla = degree - adj_matrix
        unnormalized_lapla_matrix.append(unnormalized_lapla)

        if index % 100 == 0:
            print(note, index)

    np.save('%s_lapla_matrix.npy' % note, np.array(unnormalized_lapla_matrix))
    np.save('%s_lapla_nor_matrix.npy' % note, np.array(normalized_lapla_matrix))


def denoise_nasbench(metrics, threshold=0.8):
    val_metrics = metrics[:, -1, :, -1, 2]
    index = np.where(val_metrics[:, 0] > threshold)
    return index[0]


with h5py.File("data/nasbench.hdf5", mode="r") as f:
    total_count = len(f["hash"][()])
    metrics = f["metrics"][()]
random_state = np.random.RandomState(0)
result = dict()
for n_samples in [172, 334, 860]:
    split = random_state.permutation(total_count)[:n_samples]
    result[str(n_samples)] = split

# >91
valid91 = denoise_nasbench(metrics, threshold=0.91)
for n_samples in [172, 334, 860]:
    result["91-" + str(n_samples)] = np.intersect1d(result[str(n_samples)], valid91)
result["denoise-91"] = valid91

result["denoise-80"] = denoise_nasbench(metrics)
np.savez("data/train.npz", **result)
