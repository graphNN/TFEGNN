import dgl
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import scipy

import time
import argparse
import os
from tqdm import tqdm
from tfe_utils import Dataset

from tfe_models import TFE_GNN, TFE_GNN_large
from tfe_utils import propagate_adj, set_seed, accuracy, load_data, random_walk_adj, consis_loss, load_nc_dataset, \
    load_fixed_splits, eval_rocauc, eval_acc, normalize


def train(model, optimizer, adj_hp, adj_lp, x, y, mask):
    model.train()
    optimizer.zero_grad()
    out = model(adj_hp, adj_lp, x)

    if args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
        if y.shape[1] == 1:
            true_label = F.one_hot(y, y.max() + 1).squeeze(1)
        else:
            true_label = y
        #print(out.sum())
        loss = loss_func(out[mask[0]], true_label.squeeze(1)[mask[0]].to(torch.float))
    elif args.dataset in ('arxiv-year', 'fb100'):
        logits = F.log_softmax(out, dim=1)
        loss = loss_func(logits[mask[0]], y.squeeze(1)[mask[0]].long())
    else:
        out = F.log_softmax(out, dim=1)
        loss = F.cross_entropy(out[mask[0]], y.squeeze(1)[mask[0]].long())

    if args.dataset in {'citeseer'} and not args.full:
        cos_loss = consis_loss(out, 0.5, 0.9)
        (loss+cos_loss).backward()
    else:
        loss.backward()
    optimizer.step()
    del out


def test(model, adj_hp, adj_lp, x, y, mask):
    model.eval()
    logits, accs, losses = model(adj_hp, adj_lp, x), [], []

    for i in range(3):
        # acc = accuracy(logits[mask[i]], y[mask[i]])
        if args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            #print(logits.sum())
            loss = loss_func(logits[mask[0]], true_label.squeeze(1)[mask[0]].to(torch.float))
            acc = eval_rocauc(y[mask[i]], logits[mask[i]])
        elif args.dataset in ('arxiv-year', 'fb100'):
            logits = F.log_softmax(logits, dim=1)
            loss = loss_func(logits[mask[i]], y.squeeze(1)[mask[i]].long())
            acc = eval_acc(logits[mask[i]], y[mask[i]])
        else:
            logits = F.log_softmax(logits, dim=1)
            loss = F.cross_entropy(logits[mask[i]], y.squeeze(1)[mask[i]].long())
            acc = eval_acc(logits[mask[i]], y[mask[i]])

        accs.append(acc)
        losses.append(loss)

    return accs, losses, logits


def run(args, dataset, optimi, full, random_split, i):
    if args.random_split:
        set_seed(args.seed)
    else:
        set_seed(i)
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    if args.dataset in {'roman-empire', 'amazon-ratings', 'chameleon-filtered', 'squirrel-filtered'}:
        adj = dataset_att.graph.adj(scipy_fmt='csr')
        features = dataset_att.node_features.to(device)
        labels = dataset_att.labels.to(device)
        train_mask, val_mask, test_mask = dataset_att.train_idx, dataset_att.val_idx, dataset_att.test_idx
    elif args.dataset in {'fb100', 'genius', 'arxiv-year'}:
        num_nodes = dataset_new.graph['num_nodes']
        features = dataset_new.graph['node_feat']
        features = torch.Tensor(normalize(features)).to(device)
        labels = dataset_new.label.to(device)
        idx_list = load_fixed_splits(args.dataset, args.sub_dataset)[i]
        train_mask, val_mask, test_mask = idx_list['train'], idx_list['valid'], idx_list['test']
        index = dataset_new.graph['edge_index']
        adj = scipy.sparse.csr_matrix((np.ones(len(index[0])), (index[0], index[1])), shape=[num_nodes, num_nodes])+scipy.sparse.eye(num_nodes)
    else:
        adj, features, labels, train_mask, val_mask, test_mask = load_data(dataset, full, random_split, args.train_rate, args.val_rate, i)
        features = features.clone().detach().to(device)
        labels = labels.clone().detach().to(device).unsqueeze(1)
        train_mask = train_mask.clone().detach().to(device)
        val_mask = val_mask.clone().detach().to(device)
        test_mask = test_mask.clone().detach().to(device)

    if args.dataset in {'physics', 'cora-full', 'roman-empire', 'amazon-ratings', 'genius', 'fb100', 'arxiv-year'}:
        model = TFE_GNN_large(features.shape[1], args.hidden,max(labels.max().item() + 1, labels.shape[1]), args.layers, args.dropout,
                                   args.activation, args.hop, args.combine)
    else:
        model = TFE_GNN(features.shape[1], args.hidden, int(max(labels)) + 1, args.layers, args.dropout,
                               args.activation, args.hop, args.combine)
    if optimi == 'Adam':
        optimizer = optim.Adam(
            [{'params': model.adaptive, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
             {'params': model.adaptive_lp, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
             {'params': model.layers.parameters(), 'weight_decay': args.wd_lin, 'lr': args.lr_lin},
             {'params': model.ense_coe, 'weight_decay': args.wd_adaptive2, 'lr': args.lr_adaptive2},
             ])
    if optimi == "RMSprop":
        optimizer = optim.RMSprop(
            [{'params': model.adaptive, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
             {'params': model.adaptive_lp, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
             {'params': model.layers.parameters(), 'weight_decay': args.wd_lin, 'lr': args.lr_lin},
             {'params': model.ense_coe, 'weight_decay': args.wd_adaptive2, 'lr': args.lr_adaptive2},
             ])
    if optimi == "AdamW":
        optimizer = optim.AdamW(
            [{'params': model.adaptive, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
             {'params': model.adaptive_lp, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
             {'params': model.layers.parameters(), 'weight_decay': args.wd_lin, 'lr': args.lr_lin},
             {'params': model.ense_coe, 'weight_decay': args.wd_adaptive2, 'lr': args.lr_adaptive2},
             ])
    model.to(device)
    mask = [train_mask.to(device), val_mask.to(device), test_mask.to(device)]
    #print(mask)

    if args.gf == 'sym':
        adj_lp = propagate_adj(adj, 'low', -0.5, -0.5).to(device)
        adj_hp = propagate_adj(adj, 'high', args.eta, args.eta).to(device)

    elif args.gf == 'rw':
        adj_lp = random_walk_adj(adj, 'low', -1.).to(device)
        adj_hp = random_walk_adj(adj, 'high', -1.).to(device)
    else:
        print("Unsupported Graph Filter Forms")

    best_acc, best_val_acc, test_acc, best_val_loss = 0, 0, 0, float("inf")
    train_losses = []
    val_losses = []
    run_time = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train(model, optimizer, adj_hp, adj_lp, features, labels, mask)
        run_time.append(time.time()-t0)
        [train_acc, val_acc, tmp_test_acc], [train_loss, val_loss, tmp_test_loss], logits = test(model, adj_hp, adj_lp, features, labels, mask)
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        # print('--------------',val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            bad_epoch = 0

            ada = model.adaptive.data.cpu()
            ada_lp = model.adaptive_lp.data.cpu()
        else:
            bad_epoch += 1
        if bad_epoch == args.patience:
            break
    if args.dataset in {'roman-empire', 'amazon-ratings',  'chameleon-filtered', 'squirrel-filtered'}:
        dataset_att.next_data_split()  # turn to next split

    return test_acc, best_val_loss, ada, ada_lp, run_time


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='genius', help='texas, cornell, wisconsin, chameleon, squirrel, cora'
                                            'citeseer, pubmed, cora-full, cs, physics, roman-empire, amazon-ratings，chameleon-filtered'
                    'fb100: Penn94,. genius, arxiv-year, pokec')
parser.add_argument('--sub_dataset', type=str, default='', help='fb100:Penn94, other: ')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--layers', type=int, default=2, help='')  #
parser.add_argument('--device', type=int, default=0, help='GPU device.')
parser.add_argument('--runs', type=int, default=5, help='number of runs.')

parser.add_argument('--optimizer', type=str, default='AdamW', help="Adam, RMSprop, AdamW")
parser.add_argument('--hop_lp', type=int, default=7, help='K_lp in our paper')
parser.add_argument('--hop_hp', type=int, default=0, help='K_hp in our paper')
parser.add_argument('--pro_dropout', type=float, default=0.9, help='Dropout rate (1 - keep probability) of propagation.')
parser.add_argument('--lin_dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability) of linear.')
parser.add_argument('--eta', type=float, default=-0.5, help='exponent of H_hp')

parser.add_argument('--lr_adaptive', type=float, default=0.01, help='Initial learning rate of coefficients.')
parser.add_argument('--wd_adaptive', type=float, default=0.0005, help='Weight decay (L2 loss on parameters) of coefficients.')
parser.add_argument('--lr_adaptive2', type=float, default=0.01, help='Initial learning rate of coefficients.')
parser.add_argument('--wd_adaptive2', type=float, default=0.000, help='Weight decay (L2 loss on parameters) of coefficients.')
parser.add_argument('--lr_lin', type=float, default=0.01, help='Initial learning rate of linear.')
parser.add_argument('--wd_lin', type=float, default=0.1, help='Weight decay (L2 loss on parameters) of linear.')

parser.add_argument('--gf', type=str, default='sym', help="H_hp, H_lp: sym, rw")
parser.add_argument('--activation', type=bool, default=True)
parser.add_argument('--full', type=bool, default=True, help='Whether full-supervised')
parser.add_argument('--random_split', type=bool, default=True, help='Whether random split')
parser.add_argument('--combine', type=str, default='sum', help='sum, con, lp, hp')

args = parser.parse_args()
print(args)

args.dropout = [args.pro_dropout, args.lin_dropout]
args.hop = [args.hop_lp, args.hop_hp]

if args.full:
    args.train_rate = 0.6
    args.val_rate = 0.2
else:
    args.train_rate = 0.025
    args.val_rate = 0.025

if args.dataset in {'roman-empire', 'amazon-ratings', 'chameleon-filtered', 'squirrel-filtered'}:
    dataset_att = Dataset(name=args.dataset,
                      add_self_loops=True,
                      device=args.device,
                      use_sgc_features=False,
                      use_identity_features=False,
                      use_adjacency_features=False,
                      do_not_use_original_features=False)
    if len(dataset_att.labels.shape) == 1:
        dataset_att.labels = dataset_att.labels.unsqueeze(1)
elif args.dataset in {'fb100', 'genius', 'arxiv-year'}:
    dataset_new = load_nc_dataset(args.dataset, args.sub_dataset)
    #print(dataset_new.label.shape)
    if len(dataset_new.label.shape) == 1:
        dataset_new.label = dataset_new.label.unsqueeze(1)
    #print(dataset_new.label)
else:
    args.dataset = args.dataset

if args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
    loss_func = torch.nn.BCEWithLogitsLoss()
if args.dataset in ('arxiv-year', 'fb100'):
    loss_func = torch.nn.NLLLoss()
#print(dataset_new.graph['node_feat'])

results = []
time_results=[]
all_test_accs = []

for i in tqdm(range(args.runs)):
    test_acc, best_val_loss, ada, ada_lp, run_time = run(args, args.dataset, args.optimizer, args.full, args.random_split, i)
    time_results.append(run_time)
    results.append([ada, ada_lp])
    all_test_accs.append(test_acc)
    print(f'run_{str(i+1)} \t test_acc: {test_acc:.4f}')
run_sum=0
epochsss=0
for i in time_results:
    run_sum+=sum(i)
    epochsss+=len(i)
#print(results)
print("each run avg_time:",run_sum/(args.runs),"s")
print("each epoch avg_time:",1000*run_sum/epochsss,"ms")
print('test acc mean (%) =', np.mean(all_test_accs)*100, np.std(all_test_accs)*100)
