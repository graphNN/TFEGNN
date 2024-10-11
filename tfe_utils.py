import dgl
import torch
import scipy.sparse as sp
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from dgl import ops

import os
import sys
import pickle as pkl
import networkx as nx
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def eval_acc(y_pred, y_true):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def consis_loss(logps, tem, lam):
    logps = torch.exp(logps)
    sharp_logps = (torch.pow(logps, 1. / tem) / torch.sum(torch.pow(logps, 1. / tem), dim=1, keepdim=True)).detach()
    loss = torch.mean((logps - sharp_logps).pow(2).sum(1)) * lam

    return loss


def propagate_adj(adj, filter, left_power_number, right_power_number):
    D1 = np.array(adj.sum(axis=1)) ** (left_power_number)
    D2 = np.array(adj.sum(axis=0)) ** (right_power_number)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)

    if filter == 'high':
        A = sp.eye(adj.shape[0]) - A
    if filter == 'full':
        A = sp.eye(adj.shape[0])
    if filter == 'low':
        A = A

    A = sparse_mx_to_torch_sparse_tensor(A)

    return A


def random_walk_adj(adj, filter, power_number):
    D1 = np.array(adj.sum(axis=1), dtype=float) ** (power_number)
    D1 = sp.diags(D1[:, 0], format='csr')

    A = adj.dot(D1)

    if filter == 'high':
        A = sp.eye(adj.shape[0]) - A
    if filter == 'full':
        A = sp.eye(adj.shape[0])
    if filter == 'low':
        A = A

    A = sparse_mx_to_torch_sparse_tensor(A)

    return A


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_splits(labels, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,labels.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=len(labels))
    val_mask = index_to_mask(val_idx,size=len(labels))
    test_mask = index_to_mask(test_idx,size=len(labels))
    return train_mask, val_mask, test_mask


def fixed_splits(labels, num_classes, percls_trn, val_lb, name, seed=42):
    if name in ["Chameleon","Squirrel", "Actor"]:
        seed = 1941488137
    index=[i for i in range(0,labels.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=len(labels))
    val_mask = index_to_mask(val_idx,size=len(labels))
    test_mask = index_to_mask(test_idx,size=len(labels))

    return train_mask, val_mask, test_mask


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    np.random.seed(42)
    torch.manual_seed(42)

    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


file_dir_citation = os.getcwd() + '/data'
def load_data_citation(dataset_str='cora'):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(file_dir_citation, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(file_dir_citation, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features_norm = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])
    adj = sp.csr_matrix(adj)

    D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    # features = torch.FloatTensor(np.array(features.todense()))
    features_norm = torch.FloatTensor(np.array(features_norm.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(A)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, features_norm, labels, idx_train, idx_val, idx_test, adj


from load_geom import load_geom
from dgl.data import CoauthorCSDataset, CoraFullDataset, CoauthorPhysicsDataset
SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539,
             3212139042, 2424918363]
def load_data(dataset, full, random_split, train_rate, val_rate, i):
    if dataset in {'cora', 'citeseer', 'pubmed'}:
        _, features, labels, train_mask, val_mask, test_mask, adj = load_data_citation(dataset)
        percls_trn = int(round(train_rate * len(labels) / int(labels.max() + 1)))
        val_lb = int(round(val_rate * len(labels)))
        if full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
        # else:
        #     train_mask, val_mask, test_mask = fixed_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
        #                                                    dataset)
    if dataset in {'texas', 'cornell', "chameleon", 'squirrel', 'wisconsin'}:
        path = os.getcwd()
        dataset_split = path + '/splits/' + f'{dataset}_split_0.6_0.2_{i}.npz'
        g, features, labels, train_mask, val_mask, test_mask = load_geom(dataset, dataset_split,
                                                                         train_percentage=None, val_percentage=None,
                                                                         embedding_mode=None, embedding_method=None,
                                                                         embedding_method_graph=None,
                                                                         embedding_method_space=None)
        percls_trn = int(round(train_rate * len(labels) / int(labels.max() + 1)))
        val_lb = int(round(val_rate * len(labels)))
        g = dgl.add_self_loop(g)
        adj = g.adj(scipy_fmt='csr')
        if full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
        if not full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
        if not full and not random_split:
            train_mask, val_mask, test_mask = fixed_splits(labels, int(labels.max()) + 1, percls_trn, val_lb, dataset)
    if dataset in {'cora-full', 'cs', 'physics'}:
        if dataset == 'cora-full':
            data = CoraFullDataset()
        if dataset == 'cs':
            data = CoauthorCSDataset()
        if dataset == 'physics':
            data = CoauthorPhysicsDataset()

        g = data[0]  # .to(device)
        g = dgl.add_self_loop(g)
        adj = g.adj(scipy_fmt='csr')
        features = g.ndata['feat']
        labels = g.ndata['label']

        percls_trn = int(round(train_rate * len(labels) / int(labels.max() + 1)))
        val_lb = int(round(val_rate * len(labels)))
        if full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
        if not full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
        if not full and not random_split:
            train_mask, val_mask, test_mask = fixed_splits(labels, int(labels.max()) + 1, percls_trn, val_lb, dataset)

    return adj, features, labels, train_mask, val_mask, test_mask



class Dataset:
    def __init__(self, name, add_self_loops=False, device='cpu', use_sgc_features=False, use_identity_features=False,
                 use_adjacency_features=False, do_not_use_original_features=False):

        if do_not_use_original_features and not any([use_sgc_features, use_identity_features, use_adjacency_features]):
            raise ValueError('If original node features are not used, at least one of the arguments '
                             'use_sgc_features, use_identity_features, use_adjacency_features should be used.')

        print('Preparing data...')
        data = np.load(os.path.join('new_data2', f'{name.replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])

        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)

        if 'directed' not in name:
            graph = dgl.to_bidirected(graph)

        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        node_features = self.augment_node_features(graph=graph,
                                                   node_features=node_features,
                                                   use_sgc_features=use_sgc_features,
                                                   use_identity_features=use_identity_features,
                                                   use_adjacency_features=use_adjacency_features,
                                                   do_not_use_original_features=do_not_use_original_features)

        self.name = name
        self.device = device

        self.graph = graph.to(device)
        self.node_features = node_features.to(device)
        self.labels = labels.to(device)

        self.train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        self.val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        self.test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'

    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits

    def compute_metrics(self, logits):
        if self.num_targets == 1:
            train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
                                         y_score=logits[self.train_idx].cpu().numpy()).item()

            val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
                                       y_score=logits[self.val_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
                                        y_score=logits[self.test_idx].cpu().numpy()).item()

        else:
            preds = logits.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics

    @staticmethod
    def augment_node_features(graph, node_features, use_sgc_features, use_identity_features, use_adjacency_features,
                              do_not_use_original_features):

        n = graph.num_nodes()
        original_node_features = node_features

        if do_not_use_original_features:
            node_features = torch.tensor([[] for _ in range(n)])

        if use_sgc_features:
            sgc_features = Dataset.compute_sgc_features(graph, original_node_features)
            node_features = torch.cat([node_features, sgc_features], axis=1)

        if use_identity_features:
            node_features = torch.cat([node_features, torch.eye(n)], axis=1)

        if use_adjacency_features:
            graph_without_self_loops = dgl.remove_self_loop(graph)
            adj_matrix = graph_without_self_loops.adjacency_matrix().to_dense()
            node_features = torch.cat([node_features, adj_matrix], axis=1)

        return node_features

    @staticmethod
    def compute_sgc_features(graph, node_features, num_props=5):
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        for _ in range(num_props):
            node_features = ops.u_mul_e_sum(graph, node_features, norm_coefs)

        return node_features


from sklearn.preprocessing import label_binarize
from ogb.nodeproppred import NodePropPredDataset
from os import path
import scipy
import gdown
import pandas as pd
from tfe_load_data import load_twitch, load_fb100, load_twitch_gamer, DATAPATH, even_quantile_labels

class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}


def load_fixed_splits(dataset, sub_dataset):
    """ loads saved fixed splits for dataset
    """
    name = dataset
    if sub_dataset and sub_dataset != 'None':
        name += f'-{sub_dataset}'

    if not os.path.exists(f'./new_data2/splits/{name}-splits.npy'):
        assert dataset in splits_drive_url.keys()
        gdown.download(
            id=splits_drive_url[dataset], \
            output=f'./new_data2/splits/{name}-splits.npy', quiet=False)

    splits_lst = np.load(f'./new_data2/splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


def load_nc_dataset(dataname, sub_dataname=''):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname == 'twitch-e':
        # twitch-explicit graph
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'DE'
        dataset = load_twitch_dataset(sub_dataname)
    elif dataname == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(sub_dataname)
    elif dataname == 'ogbn-proteins':
        dataset = load_proteins_dataset()
    elif dataname == 'deezer-europe':
        dataset = load_deezer_dataset()
    elif dataname == 'arxiv-year':
        dataset = load_arxiv_year_dataset()
    elif dataname == 'pokec':
        dataset = load_pokec_mat()
    elif dataname == 'snap-patents':
        dataset = load_snap_patents_mat()
    elif dataname == 'yelp-chi':
        dataset = load_yelpchi_dataset()
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(dataname)
    elif dataname in ('Cora', 'CiteSeer', 'PubMed'):
        dataset = load_planetoid_dataset(dataname)
    elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset = load_geom_gcn_dataset(dataname)
    elif dataname == "genius":
        dataset = load_genius()
    elif dataname == "twitch-gamer":
        dataset = load_twitch_gamer_dataset()
    elif dataname == "wiki":
        dataset = load_wiki()
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    metadata = metadata.astype(np.int32)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_arxiv_year_dataset(nclass=5):
    filename = 'arxiv-year'
    dataset = NCDataset(filename)
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    label = even_quantile_labels(
        dataset.graph['node_year'].flatten(), nclass, verbose=False)
    dataset.label = torch.as_tensor(label).reshape(-1, 1)
    return dataset


def load_proteins_dataset():
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins')
    dataset = NCDataset('ogbn-proteins')

    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}

    dataset.get_idx_split = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels

    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)
    return dataset


def load_ogb_dataset(name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name)
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx

    dataset.get_idx_split = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


def load_pokec_mat():
    """ requires pokec.mat
    """
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'], \
                       output=f'{DATAPATH}pokec.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_snap_patents_mat(nclass=5):
    if not path.exists(f'{DATAPATH}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=dataset_drive_url['snap-patents'], \
                       output=f'{DATAPATH}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')

    dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_yelpchi_dataset():
    if not path.exists(f'{DATAPATH}YelpChi.mat'):
        gdown.download(id=dataset_drive_url['yelp-chi'], \
                       output=f'{DATAPATH}YelpChi.mat', quiet=False)
    fulldata = scipy.io.loadmat(f'{DATAPATH}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_planetoid_dataset(name):
    torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                              name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    def planetoid_orig_split(**kwargs):
        return {'train': torch.as_tensor(dataset.train_idx),
                'valid': torch.as_tensor(dataset.valid_idx),
                'test': torch.as_tensor(dataset.test_idx)}

    dataset.get_idx_split = planetoid_orig_split
    dataset.label = label

    return dataset


def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATAPATH}/{name}.mat')
    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_genius():
    filename = 'genius'
    dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'new_data2/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
                       output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
                       output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)

    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_wiki():
    if not path.exists(f'{DATAPATH}wiki_features2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_features'], \
                       output=f'{DATAPATH}wiki_features2M.pt', quiet=False)

    if not path.exists(f'{DATAPATH}wiki_edges2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_edges'], \
                       output=f'{DATAPATH}wiki_edges2M.pt', quiet=False)

    if not path.exists(f'{DATAPATH}wiki_views2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_views'], \
                       output=f'{DATAPATH}wiki_views2M.pt', quiet=False)

    dataset = NCDataset("wiki")
    features = torch.load(f'{DATAPATH}wiki_features2M.pt')
    edges = torch.load(f'{DATAPATH}wiki_edges2M.pt').T
    row, col = edges
    print(f"edges shape: {edges.shape}")
    label = torch.load(f'{DATAPATH}wiki_views2M.pt')
    num_nodes = label.shape[0]

    print(f"features shape: {features.shape[0]}")
    print(f"Label shape: {label.shape[0]}")
    dataset.graph = {"edge_index": edges,
                     "edge_feat": None,
                     "node_feat": features,
                     "num_nodes": num_nodes}
    dataset.label = label
    return dataset

