import math
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv


class deep_gcn(nn.Module):
    def __init__(self, input_dim, hidden, classes, num_layer, dropout, activation):
        super(deep_gcn, self).__init__()
        self.num_layer = num_layer
        self.layer = nn.ModuleList()
        self.drop = dropout
        self.acti = activation

        for i in range(num_layer):
            in_feat = input_dim if i == 0 else hidden
            out_feat = hidden if i < num_layer-1 else classes
            self.layer.append(GraphConv(in_feats=in_feat, out_feats=out_feat, allow_zero_in_degree=True, bias=False))

    def forward(self, g_list, features):
        g = g_list
        x = features
        h = F.dropout(x, self.drop[0], self.training)

        for j in range(self.num_layer):
            h = self.layer[j](g, h)
            if j < self.num_layer - 1:
                if self.acti:
                    h = F.relu(h)
                h = F.dropout(h, self.drop[1], self.training)

        return h


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, dropout, activation, bn):
        super(MLP, self).__init__()
        self.nl = num_layers
        self.drop = dropout
        self.acti = activation
        self.bn = bn

        self.bnlayer = nn.ModuleList()
        if bn:
            for i in range(num_layers):
                if i == 0:
                    self.bnlayer.append(nn.BatchNorm1d(input_dim))
                if 0 < i < num_layers:
                    self.bnlayer.append(nn.BatchNorm1d(hidden_dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            if 0 < i < num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, out_dim))

        self.init_parameter()

    def init_parameter(self):
        for i in range(self.nl):
            stdv = 1. / math.sqrt(self.layers[i].weight.size(1))
            self.layers[i].weight.data.normal_(-stdv, stdv)

    def forward(self, x):
        if self.bn:
            x = self.bnlayer[0](x)
        h = F.dropout(x, self.drop[0], self.training)
        for i in range(self.nl):
            h = self.layers[i](h)
            if i < self.nl - 1:
                if self.acti:
                    h = F.relu(h)
                    # h = F.leaky_relu(h)
                if self.bn and i+1 < self.nl:
                    h = self.bnlayer[i+1](h)
                h = F.dropout(h, self.drop[1], self.training)

        return h


class GCNBase(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers):
        super(GCNBase, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, out_dim, num_layers, [0,0], True, False)


    def forward(self, adj, x):
        x = torch.mm(adj, x)
        x = torch.mm(adj, x)

        h = self.mlp(x)

        return h


class TFE_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, dropout, activation, hop, combine):
        super(TFE_GNN, self).__init__()
        self.nl = num_layers
        self.drop = dropout
        self.acti = activation
        self.hop = hop
        self.combine = combine

        a = torch.Tensor(hop[1] + 1)
        aa = torch.Tensor(hop[0] + 1)
        self.adaptive = nn.Parameter(a)
        self.adaptive_lp = nn.Parameter(aa)

        if self.combine == 'sum' or 'con':
            self.ense_coe = nn.Parameter(torch.Tensor(2))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                if combine == 'con':
                    self.layers.append(nn.Linear(2*input_dim, hidden_dim, bias=False))
                else:
                    self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            if 0 < i < num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            if i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, out_dim, bias=False))

        self.init_parameter()

    def init_parameter(self):
        if self.combine == 'sum' or 'con':
           self.ense_coe.data.fill_(1.0)
        self.adaptive.data.fill_(0.5)
        self.adaptive_lp.data.fill_(0.5)

        for i in range(self.nl):
            stdv = 1. / math.sqrt(self.layers[i].weight.size(1))
            self.layers[i].weight.data.normal_(-stdv, stdv)
    def mix_prop(self, adj, x, coe):
        x0 = x.clone()
        xx = x.clone() #* coe[0]
        del x
        for i in range(1, len(coe)):
            x0 = torch.mm(adj, x0)
            xx = xx + coe[i] * x0

        return xx

    def forward(self, adj_hp, adj_lp, h0):
        coe_tmp = self.adaptive
        coe_tmp_lp = self.adaptive_lp
        if self.combine == 'sum' or 'con':
           ense_coe = self.ense_coe

        if self.drop[0] > 0:
            h0 = F.dropout(h0, self.drop[0], self.training)

        x = self.mix_prop(adj_hp, h0, coe_tmp)
        xx = self.mix_prop(adj_lp, h0, coe_tmp_lp)

        h_lp, h_hp = xx, x
        del x, xx

        if self.combine == 'sum':
            h = ense_coe[1]*h_hp + ense_coe[0]*h_lp
        if self.combine == 'con':
            h = torch.cat([ense_coe[0]*h_lp, ense_coe[1]*h_hp], dim=1)

        if self.combine == 'lp':
            h = h_lp
        if self.combine == 'hp':
            h = h_hp

        for i in range(self.nl):
            h = self.layers[i](h)

            if i < self.nl - 1:
                if self.acti:
                    h = F.relu(h)
                h = F.dropout(h, self.drop[1], self.training)

        return h


class TFE_GNN_large(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, dropout, activation, hop, combine):
        super(TFE_GNN_large, self).__init__()
        self.nl = num_layers
        self.drop = dropout
        self.acti = activation
        self.hop = hop
        self.combine = combine

        a = torch.Tensor(hop[1] + 1)
        aa = torch.Tensor(hop[0] + 1)
        self.adaptive = nn.Parameter(a)
        self.adaptive_lp = nn.Parameter(aa)

        if self.combine == 'sum' or 'con':
            self.ense_coe = nn.Parameter(torch.Tensor(2))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                if combine == 'con':
                    self.layers.append(nn.Linear(2*input_dim, hidden_dim, bias=False))
                else:
                    self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            if 0 < i < num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            if i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, out_dim, bias=False))

        self.init_parameter()

    def init_parameter(self):
        if self.combine == 'sum' or 'con':
           self.ense_coe.data.fill_(1.0)
        self.adaptive.data.fill_(0.5)
        self.adaptive_lp.data.fill_(0.5)

        for i in range(self.nl):
            stdv = 1. / math.sqrt(self.layers[i].weight.size(1))
            self.layers[i].weight.data.normal_(-stdv, stdv)
    def mix_prop(self, adj, x, coe):
        x0 = x.clone()
        xx = x.clone() * coe[0]
        del x
        for i in range(1, len(coe)):
            x0 = torch.mm(adj, x0)
            xx = xx + coe[i] * x0

        return xx

    def forward(self, adj_hp, adj_lp, h1):
        coe_tmp = self.adaptive
        coe_tmp_lp = self.adaptive_lp
        if self.combine == 'sum' or 'con':
           ense_coe = self.ense_coe
        if self.combine == 'con':
            h1 = torch.cat([ense_coe[0]*h1, ense_coe[1]*h1], dim=1)

        for i in range(self.nl):
            h1 = self.layers[i](h1)

            if i < self.nl - 1:
                if self.acti:
                    h1 = F.relu(h1)
                h1 = F.dropout(h1, self.drop[1], self.training)
        h0 = h1

        if self.drop[0] > 0:
            h0 = F.dropout(h0, self.drop[0], self.training)

        x = self.mix_prop(adj_hp, h0, coe_tmp)
        xx = self.mix_prop(adj_lp, h0, coe_tmp_lp)

        h_lp, h_hp = xx, x
        del x, xx

        if self.combine == 'sum':
            h = ense_coe[1]*h_hp + ense_coe[0]*h_lp
        if self.combine == 'con':
            h = ense_coe[1] * h_hp + ense_coe[0] * h_lp
        if self.combine == 'lp':
            h = h_lp
        if self.combine == 'hp':
            h = h_hp

        return h

