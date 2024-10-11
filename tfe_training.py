import dgl
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

import time
import argparse
import os
from tqdm import tqdm

from tfe_models import TFE_GNN, TFE_GNN_large
from tfe_utils import propagate_adj, set_seed, accuracy, load_data, random_walk_adj, consis_loss


def train(model, optimizer, adj_hp, adj_lp, x, y, mask):
    model.train()
    optimizer.zero_grad()
    out = model(adj_hp, adj_lp, x)
    out = F.log_softmax(out, dim=1)
    loss = F.cross_entropy(out[mask[0]], y[mask[0]])
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
    logits = F.log_softmax(logits, dim=1)
    for i in range(3):
        acc = accuracy(logits[mask[i]], y[mask[i]])
        loss = F.cross_entropy(logits[mask[i]], y[mask[i]])
        accs.append(acc)
        losses.append(loss)

    return accs, losses, logits


def run(args, dataset, optimi, full, random_split, i):
    if args.random_split:
        set_seed(args.seed)
    else:
        set_seed(i)
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    adj, features, labels, train_mask, val_mask, test_mask = load_data(dataset, full, random_split, args.train_rate, args.val_rate, i)
    if args.dataset in {'physics', 'cora-full'}:
        model = TFE_GNN_large(features.shape[1], args.hidden, int(max(labels)) + 1, args.layers, args.dropout,
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


    model.to(device)
    features = features.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    train_mask = train_mask.clone().detach().to(device)
    val_mask = val_mask.clone().detach().to(device)
    test_mask = test_mask.clone().detach().to(device)
    mask = [train_mask, val_mask, test_mask]

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
    #torch.save(train_losses, f"/home/user/duan/four_mix_ChebNet_1/eigenvalue/{args.dataset}_tfe_train_loss_mlrrm{i}.pkl")
    #torch.save(val_losses, f"/home/user/duan/four_mix_ChebNet_1/eigenvalue/{args.dataset}_tfe_val_loss_mlrrm{i}.pkl")

    return test_acc, best_val_loss, ada, ada_lp, run_time


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='squirrel', help='texas, cornell, wisconsin, chameleon, squirrel, cora'
                                                                    'citeseer, pubmed, cora-full, cs, physics')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
parser.add_argument('--layers', type=int, default=2, help='')
parser.add_argument('--device', type=int, default=0, help='GPU device.')
parser.add_argument('--runs', type=int, default=10, help='number of runs.')

parser.add_argument('--optimizer', type=str, default='Adam', help="Adam, RMSprop")
parser.add_argument('--hop_lp', type=int, default=6, help='K_lp in our paper')
parser.add_argument('--hop_hp', type=int, default=5, help='K_hp in our paper')
parser.add_argument('--pro_dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability) of propagation.')
parser.add_argument('--lin_dropout', type=float, default=0., help='Dropout rate (1 - keep probability) of linear.')
parser.add_argument('--eta', type=float, default=-0.3, help='exponent of H_hp')

parser.add_argument('--lr_adaptive', type=float, default=0.1, help='Initial learning rate of coefficients.')
parser.add_argument('--wd_adaptive', type=float, default=0.05, help='Weight decay (L2 loss on parameters) of coefficients.')
parser.add_argument('--lr_adaptive2', type=float, default=0.00, help='Initial learning rate of coefficients.')
parser.add_argument('--wd_adaptive2', type=float, default=0.000, help='Weight decay (L2 loss on parameters) of coefficients.')
parser.add_argument('--lr_lin', type=float, default=0.005, help='Initial learning rate of linear.')
parser.add_argument('--wd_lin', type=float, default=0.0, help='Weight decay (L2 loss on parameters) of linear.')

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

results = []
time_results=[]
all_test_accs = []
adas = []
adas_lp = []

for i in tqdm(range(args.runs)):
    test_acc, best_val_loss, ada, ada_lp, run_time = run(args, args.dataset, args.optimizer, args.full, args.random_split, i)
    time_results.append(run_time)
    results.append([ada, ada_lp])
    all_test_accs.append(test_acc.item())
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
