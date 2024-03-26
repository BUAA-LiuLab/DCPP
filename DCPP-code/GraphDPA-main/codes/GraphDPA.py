import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch.nn import Linear, Embedding
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv, TransformerConv


def global_select_concat(feature, batch, x):
    feature = feature[x==-1]
    batch_size = batch[-1].item() + 1
    return feature.view(batch_size, -1)


class GCN_Model(torch.nn.Module):
    def __init__(self, word_sizes, embed_dim,  n_output=2):
        super(GCN_Model, self).__init__()

        self.embed_dim = embed_dim
        #layers
        self.embedding = Embedding(word_sizes, self.embed_dim)
        self.gcn1 = GCNConv(embed_dim, 50)
        self.gcn2 = GCNConv(50, 50)
        self.gcn3 = GCNConv(50, 100)
        self.fc1 = Linear(200, 100)
        self.output = Linear(100, n_output)
        
    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        feature = [torch.zeros(self.embed_dim).to(x) if point==-1 else self.embedding(point) for point in x]
        feature = self.gcn1(torch.stack(feature), edge_index)
        feature = F.relu(feature)
        feature = self.gcn2(feature, edge_index)
        feature = F.relu(feature)
        feature = self.gcn3(feature, edge_index)
        feature = F.relu(feature)
        feature = global_select_concat(feature, batch, x)
        feature = self.fc1(feature)
        feature = F.dropout(feature, 0.2)
        out = self.output(feature)
        return out

