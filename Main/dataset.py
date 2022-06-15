# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 18:59
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import json
import torch
import numpy as np
from itertools import repeat
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data, InMemoryDataset, Batch
from Main.utils import clean_comment


class WeiboFTDataset(InMemoryDataset):
    def __init__(self, root, word2vec, clean=True, transform=None, pre_transform=None, pre_filter=None):
        self.word2vec = word2vec
        self.clean = clean
        self.set_aug_mode('none')
        self.set_aug_ratio(0.2)
        self.set_aug_prob(np.ones(25) / 25)
        self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, lambda x, y: x]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        if self.clean:
            limit_num = 600
            pass_comment = ['', '转发微博', '转发微博。', '轉發微博', '轉發微博。']
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                if 'label' in post['source'].keys():
                    y.append(post['source']['label'])
                pass_num = 0
                id_to_index = {}
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    id_to_index[comment['comment id']] = i
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        pass_num += 1
                        continue
                    post['comment'][i]['comment id'] -= pass_num
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        continue
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    if comment['parent'] == -1:
                        row.append(0)
                    else:
                        row.append(post['comment'][id_to_index[comment['parent']]]['comment id'] + 1)
                    col.append(comment['comment id'] + 1)
                edge_index = [row, col]
                edge_attr = torch.ones(len(row), 1)
                y = torch.LongTensor(y)
                edge_index = torch.LongTensor(edge_index)
                one_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr) if 'label' in post[
                    'source'].keys() \
                    else Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(one_data)
        else:
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                if 'label' in post['source'].keys():
                    y.append(post['source']['label'])
                for i, comment in enumerate(post['comment']):
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    row.append(comment['parent'] + 1)
                    col.append(comment['comment id'] + 1)
                edge_index = [row, col]
                edge_attr = torch.ones(len(row), 1)
                y = torch.LongTensor(y)
                edge_index = torch.LongTensor(edge_index)
                one_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr) if 'label' in post[
                    'source'].keys() \
                    else Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])

    def set_aug_mode(self, aug_mode='none'):
        self.aug_mode = aug_mode

    def set_aug_ratio(self, aug_ratio=0.2):
        self.aug_ratio = aug_ratio

    def set_aug_prob(self, prob):
        if prob.ndim == 2:
            prob = prob.reshape(-1)
        self.aug_prob = prob


class WeiboDataset(InMemoryDataset):
    def __init__(self, root, word2vec, clean=True, transform=None, pre_transform=None, pre_filter=None):
        self.word2vec = word2vec
        self.clean = clean
        self.set_aug_mode('none')
        self.set_aug_ratio(0.2)
        self.set_aug_prob(np.ones(25) / 25)
        self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, lambda x, y: x]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        if self.clean:
            limit_num = 600
            pass_comment = ['', '转发微博', '转发微博。', '轉發微博', '轉發微博。']
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                if 'label' in post['source'].keys():
                    y.append(post['source']['label'])
                pass_num = 0
                id_to_index = {}
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    id_to_index[comment['comment id']] = i
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        pass_num += 1
                        continue
                    post['comment'][i]['comment id'] -= pass_num
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        continue
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    if comment['parent'] == -1:
                        row.append(0)
                    else:
                        row.append(post['comment'][id_to_index[comment['parent']]]['comment id'] + 1)
                    col.append(comment['comment id'] + 1)
                edge_index = [row, col]
                edge_attr = torch.ones(len(row), 1)
                y = torch.LongTensor(y)
                edge_index = torch.LongTensor(edge_index)
                one_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr) if 'label' in post[
                    'source'].keys() \
                    else Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(one_data)
        else:
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                if 'label' in post['source'].keys():
                    y.append(post['source']['label'])
                for i, comment in enumerate(post['comment']):
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    row.append(comment['parent'] + 1)
                    col.append(comment['comment id'] + 1)
                edge_index = [row, col]
                edge_attr = torch.ones(len(row), 1)
                y = torch.LongTensor(y)
                edge_index = torch.LongTensor(edge_index)
                one_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr) if 'label' in post[
                    'source'].keys() \
                    else Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])

    def set_aug_mode(self, aug_mode='none'):
        self.aug_mode = aug_mode

    def set_aug_ratio(self, aug_ratio=0.2):
        self.aug_ratio = aug_ratio

    def set_aug_prob(self, prob):
        if prob.ndim == 2:
            prob = prob.reshape(-1)
        self.aug_prob = prob

    def get(self, idx):
        data, data1, data2 = self.data.__class__(), self.data.__class__(), self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes, data1.num_nodes, data2.num_nodes = self.data.__num_nodes__[idx], self.data.__num_nodes__[
                idx], self.data.__num_nodes__[idx]
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key], data1[key], data2[key] = item[s], item[s], item[s]

        # pre-defined augmentations
        if self.aug_mode == 'none':
            n_aug1, n_aug2 = 4, 4
            data1 = self.augmentations[n_aug1](data1, self.aug_ratio)
            data2 = self.augmentations[n_aug2](data2, self.aug_ratio)
        elif self.aug_mode == 'uniform':
            n_aug = np.random.choice(25, 1)[0]
            n_aug1, n_aug2 = n_aug // 5, n_aug % 5
            data1 = self.augmentations[n_aug1](data1, self.aug_ratio)
            data2 = self.augmentations[n_aug2](data2, self.aug_ratio)
        elif self.aug_mode == 'sample':
            n_aug = np.random.choice(25, 1, p=self.aug_prob)[0]
            n_aug1, n_aug2 = n_aug // 5, n_aug % 5
            data1 = self.augmentations[n_aug1](data1, self.aug_ratio)
            data2 = self.augmentations[n_aug2](data2, self.aug_ratio)

        return data, data1, data2





def node_drop(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)
    idx_nondrop = idx_perm[drop_num:].tolist()
    idx_nondrop.sort()

    edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.__num_nodes__, _ = data.x.shape
    return data


def subgraph(data, aug_ratio):
    G = tg_utils.to_networkx(data)

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * (1 - aug_ratio))

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

    while len(idx_sub) <= sub_num:
        if len(idx_neigh) == 0:
            idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
            idx_neigh = set([np.random.choice(idx_unsub)])
        sample_node = np.random.choice(list(idx_neigh))

        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

    idx_nondrop = idx_sub
    idx_nondrop.sort()

    edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.__num_nodes__, _ = data.x.shape
    return data


def edge_pert(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    pert_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index[:, np.random.choice(edge_num, (edge_num - pert_num), replace=False)]

    idx_add = np.random.choice(node_num, (2, pert_num))
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_add[0], idx_add[1]] = 1
    adj[np.arange(node_num), np.arange(node_num)] = 0
    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index
    return data


def attr_mask(data, aug_ratio):
    node_num, _ = data.x.size()
    mask_num = int(node_num * aug_ratio)
    _x = data.x.clone()

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)

    _x[idx_mask] = token
    data.x = _x
    return data


def custom_collate(data_list):
    batch = Batch.from_data_list([d[0] for d in data_list], follow_batch=['edge_index', 'edge_index_neg'])
    batch_1 = Batch.from_data_list([d[1] for d in data_list])
    batch_2 = Batch.from_data_list([d[2] for d in data_list])
    return batch, batch_1, batch_2


class PreDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True,
                 **kwargs):
        super(PreDataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=custom_collate, **kwargs)
