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


class Nb101DatasetV1(Dataset):
    """
        New and effective data split.
    """
    def __init__(self, split=None, debug=False, candidate_ops=5, pos_enc_dim=2, data_seed='s0', data_type='train'):
        self.hash2id = dict()
        with h5py.File("data/nasbench101/nasbench.hdf5", mode="r") as f:
            for i, h in enumerate(f["hash"][()]):
                self.hash2id[h.decode()] = i
            self.num_vertices = f["num_vertices"][()]
            self.trainable_parameters = f["trainable_parameters"][()]
            self.adjacency = f["adjacency"][()]
            self.operations = f["operations"][()]
            self.metrics = f["metrics"][()]
        self.random_state = np.random.RandomState(0)
        # if split is not None and split != "all":
        #     self.sample_range = np.load("data/nasbench101/train_%s.npz" % data_seed)[str(split)]
        if split is not None and split != "all":
            self.sample_range = np.load("data/nasbench101/train.npz")[str(split)]
        else:
            self.sample_range = list(range(len(self.hash2id)))
        self.debug = debug
        self.seed = 0
        self.candidate_ops = candidate_ops
        # self.lapla_eigen_pos_enc = laplacian_positional_encoding(self.adjacency, pos_enc_dim=pos_enc_dim)
        # self.lapla_eigen_pos_enc = np.load('data/nasbench101/bench101_pos_enc.npy')
        # generate_lapla_matrix(self.adjacency)

        # self.lapla = np.load('data/nasbench101/lapla_matrix.npy')
        # self.lapla_nor = np.load('data/nasbench101/lapla_nor_matrix.npy')

        # self.random_pos = np.load('./data/nasbench101/random_pos.npy')[2]

        self.data_type = data_type
        # self.val_mean_mean, self.val_mean_std = 0.90243375, 0.05864741

        # # all
        self.val_mean_mean, self.val_mean_std = 0.908192, 0.023961  # all val mean/std (from SemiNAS)
        self.test_mean_mean, self.test_mean_std = 0.8967984, 0.05799569  # all test mean/std

        # val_acc_list = []
        # test_acc_list = []
        # for index in range(len(self.metrics)):
        #     val_acc = self.metrics[index, -1, :, -1, 2]
        #     val_acc = np.mean(val_acc[0])
        #     test_acc = self.metrics[index, -1, :, -1, 3]
        #     # test_acc = self._check_validity(test_acc)
        #     test_acc = np.mean(test_acc)
        #     val_acc_list.append(val_acc)
        #     test_acc_list.append(test_acc)
        #     if index % 10000 == 0:
        #         print(index)
        # print(np.mean(val_acc_list), np.std(val_acc_list))
        # print(np.mean(test_acc_list), np.std(test_acc_list))
        # print('')

    def __len__(self):
        return len(self.sample_range)

    def _check(self, item):
        n = item["num_vertices"]
        ops = item["operations"]
        adjacency = item["adjacency"]
        mask = item["mask"]
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0

    def mean_acc(self):
        return np.mean(self.metrics[:, -1, self.seed, -1, 2])

    def std_acc(self):
        return np.std(self.metrics[:, -1, self.seed, -1, 2])

    def normalize(self, num):
        if self.data_type == 'train':
            return (num - self.val_mean_mean) / self.val_mean_std
        elif self.data_type == 'test':
            return (num - self.test_mean_mean) / self.test_mean_std
        else:
            raise ValueError('Wrong data_type!')

    def denormalize(self, num):
        if self.data_type == 'train':
            return num * self.val_mean_std + self.val_mean_mean
        elif self.data_type == 'test':
            return num * self.test_mean_std + self.test_mean_mean
        else:
            raise ValueError('Wrong data_type!')

    def resample_acc(self, index, split="val"):
        # when val_acc or test_acc are out of range
        assert split in ["val", "test"]
        split = 2 if split == "val" else 3
        for seed in range(3):
            acc = self.metrics[index, -1, seed, -1, split]
            if not self._is_acc_blow(acc):
                return acc
        if self.debug:
            print(index, self.metrics[index, -1, :, -1])
            raise ValueError
        return np.array(self.val_mean_mean)

    def _is_acc_blow(self, acc):
        return acc < 0.2

    def _rand_flip(self, batch_pos):
        batch_lap_pos_enc = torch.from_numpy(batch_pos)
        sign_flip = torch.rand(batch_lap_pos_enc.size(1))
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        return batch_lap_pos_enc

    def _check_validity(self, data_list, threshold=0.3):
        _data_list = deepcopy(data_list).tolist()
        del_idx = []
        for _idx, _data in enumerate(_data_list):
            if _data < threshold:
                del_idx.append(_idx)
        for _idx in del_idx[::-1]:
            _data_list.pop(_idx)
        if len(_data_list) == 0:
            return data_list
        else:
            return np.array(_data_list)

    def __getitem__(self, index):
        # paper method
        index = self.sample_range[index]
        # val_acc, test_acc = self.metrics[index, -1, self.seed, -1, 2:]
        val_acc = self.metrics[index, -1, :, -1, 2]
        test_acc = self.metrics[index, -1, :, -1, 3]
        # val_acc = np.mean(val_acc)
        val_acc = val_acc[0]  # 0/1/2 yeilds similar rankings

        test_acc = self._check_validity(test_acc)
        test_acc = np.mean(test_acc)
        if self._is_acc_blow(val_acc):
            val_acc = self.resample_acc(index, "val")

        n = self.num_vertices[index]
        ops_onehot = np.array([[i == k + 2 for i in range(self.candidate_ops)]
                               for k in self.operations[index]], dtype=np.float32)
        if n < 7:
            ops_onehot[n:] = 0.
        features = np.expand_dims(np.array([i for i in range(self.candidate_ops)]), axis=0)
        features = np.tile(features, (len(ops_onehot), 1))
        features = ops_onehot * features
        features = np.sum(features, axis=-1)

        # random_pos = self.random_pos[index].astype(np.float32)

        adjacency_matrix = self.adjacency[index]

        result = {
            "num_vertices": n,
            "adjacency": adjacency_matrix,
            # "lapla": self.lapla[index],
            # "lapla_nor": self.lapla_nor[index].astype(np.float32),
            "operations": ops_onehot,
            "features": torch.from_numpy(features).long(),
            "mask": np.array([i < n for i in range(7)], dtype=np.float32),
            "val_acc": torch.tensor(self.normalize(val_acc), dtype=torch.float32),
            "test_acc": torch.tensor(float(self.normalize(test_acc)), dtype=torch.float32)
        }
        if self.debug:
            self._check(result)
        return result
