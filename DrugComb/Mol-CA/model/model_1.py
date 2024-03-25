import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_scatter import scatter

from torch_scatter import scatter_add
from torch_geometric.utils import softmax
import math


class Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=6, aggr='sum'):
        super().__init__(aggr=aggr)
        self.aggr = aggr
        self.lin_neg = nn.Linear(in_channels + edge_dim, out_channels)
        self.lin_root = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x_adj = torch.cat([x[edge_index[1]], edge_attr], dim=1)
        # print('x_adj shape:',x_adj.shape)
        x_adj = F.tanh(self.lin_neg(x_adj))

        neg_sum = scatter(x_adj, edge_index[0], dim=0, reduce=self.aggr)

        x_out = F.tanh(self.lin_root(x)) + neg_sum
        # x_out = self.bn1(x_out)
        return x_out


"""
############### GlobalAttentaion###############
"""


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate1 = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate1 * x, batch, dim=0, dim_size=size)

        return out, gate - torch.mean(gate)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'nn={self.nn})')


class CCPGraph(torch.nn.Module):
    def __init__(self, input_bias=0):
        super().__init__()
        self.atom_type = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
        self.bond_type = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']
        self.d_model = 64
        self.d_model_bond = 6
        self.embeds = nn.Embedding(len(self.atom_type)*2, self.d_model)
        self.embeds_bond = nn.Embedding(len(self.bond_type)*2, self.d_model_bond)
        self.conv1 = Conv(self.d_model, 64, edge_dim=self.d_model_bond)
        self.gn1 = GraphNorm(64)
        self.conv2 = Conv(64, 16, edge_dim=self.d_model_bond)
        self.gn2 = GraphNorm(16)
        self.conv3 = Conv(32, 16, edge_dim=self.d_model_bond)
        self.gn3 = GraphNorm(16)
        self.output_bias = input_bias

        # pool
        gate_nn = nn.Sequential(nn.Linear(16, 64),
                                nn.ReLU(),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, 1))

        self.readout = GlobalAttention(gate_nn)
        self.lin1 = nn.Linear(16, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.15)
        self.lin2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=0.15)
        self.lin3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(p=0.15)
        self.lin = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.15)
        self.semi_layer = nn.Linear(16, 200, bias=False)
        self.bn_semi = nn.BatchNorm1d(200)
        self.dropout1 = nn.Dropout(0.05)
        self.semi_layer2 = nn.Linear(200, 64, bias=False)
        self.bn_semi2 = nn.BatchNorm1d(64)
        self.final_layer = nn.Linear(in_features=200, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        # self.init_parameter()

    def init_parameter(self):
        torch.nn.init.constant_(self.final_layer.bias.data, val=-1 * self.output_bias)
        self.final_layer.bias.requires_grad = False

    def forward(self, data):
        data_x = self.embeds(data.x)
        # data_edge_attr = self.embeds_bond(data.edge_attr)
        # data_x = data.x
        data_edge_attr = data.edge_attr
        x = self.conv1(data_x, data.edge_index, data_edge_attr)
        x = self.conv2(x, data.edge_index, data_edge_attr)
        # x = self.conv3(x, data.edge_index, data_edge_attr)

        embedding, att = self.readout(x, data.batch)
        # print('embedding shape',embedding.shape)

        out_1 = self.dropout(embedding)
        out_1 = self.bn_semi(self.semi_layer(out_1))
        out_1 = self.dropout1(out_1)
        # out_1 = self.bn_semi2(self.semi_layer2(out_1))
        out = self.final_layer(out_1)

        # embedding = self.dropout(embedding)
        # out = self.dp1(self.bn1(F.relu(self.lin1(embedding))))
        # out = self.dp2(self.bn2(F.relu(self.lin2(out))))
        # out = self.dp3(self.bn3(F.relu(self.lin3(out))))
        # out = self.final_layer(out)
        # out = self.lin(out)
        out = out.view(-1)

        return out, self.sigmoid(out), att, out_1