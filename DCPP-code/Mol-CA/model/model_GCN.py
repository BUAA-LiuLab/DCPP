import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchData
import torch
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import torch_sparse
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
import math


class GCNLayer(nn.Module):

    def __init__(self, V_shape, A_shape, no_filters, no_features_for_conv):
        super(GCNLayer, self).__init__()
        self.no_A = A_shape[2]
        self.no_feature = no_features_for_conv

        self.fc_A = nn.Linear(self.no_feature*self.no_A, no_filters)
        self.fc_V = nn.Linear(self.no_feature, no_filters)
        self.b = torch.nn.Parameter(torch.zeros(no_filters))
        self.register_parameter('b', self.b)
        self.init_weight()

    def init_weight(self):

        torch.nn.init.normal_(self.fc_A.weight, std=math.sqrt(1.0/(self.no_feature*(self.no_A+1))))
        torch.nn.init.normal_(self.fc_V.weight, std=math.sqrt(1/(self.no_feature*(self.no_A+1))))
        torch.nn.init.zeros_(self.b)

    def forward(self, input):
        A_reshape = torch.reshape(input[1], shape=[-1, input[1].shape[1]*self.no_A, input[1].shape[1]])
        A_conv = torch.matmul(A_reshape, input[0])
        A_conv = A_conv.resize(input[0].shape[0], input[0].shape[1], self.no_A * self.no_feature)
        A_conv = self.fc_A(A_conv)
        V_conv = self.fc_V(input[0])
        out = A_conv + V_conv + self.b

        return out


class GCNBlock(nn.Module):

    def __init__(self, U_shape, A_shape, V_shape, U_broadcast_shape, no_filters=64, flag=0):
        super(GCNBlock, self).__init__()
        if flag == 0:
            self.input_size_U = U_shape[-1]
            self.no_filters = no_filters
            self.no_features_for_conv = no_filters + V_shape[-1]
            self.fc_u1 = nn.Linear(self.input_size_U, no_filters)
            self.bn_u1 = nn.BatchNorm1d(no_filters)
            # self.concat_V1 = torch.cat([V, U_broadcast], dim=-1)
            ######## Graph Convolution #########
            self.graph_conv_1 = GCNLayer(V_shape, A_shape, self.no_filters, self.no_features_for_conv)
            self.bn_v1 = nn.BatchNorm1d(self.no_filters)
            self.bn_u2 = nn.BatchNorm1d(no_filters)
        elif flag == 1:
            self.input_size_U = no_filters
            self.no_filters = no_filters
            self.no_features_for_conv = no_filters + no_filters
            self.fc_u1 = nn.Linear(self.input_size_U, no_filters)
            self.bn_u1 = nn.BatchNorm1d(no_filters)
            # self.concat_V1 = torch.cat([V, U_broadcast], dim=-1)
            ######## Graph Convolution #########
            self.graph_conv_1 = GCNLayer(V_shape, A_shape, self.no_filters, self.no_features_for_conv)
            self.bn_v1 = nn.BatchNorm1d(self.no_filters)
            self.bn_u2 = nn.BatchNorm1d(no_filters)

    def init_weight(self):

        torch.nn.init.normal_(self.fc1, std=1.0/math.sqrt(self.input_size_U))

    def forward(self, input):

        U = self.fc_u1(input[0].reshape([-1, input[0].shape[-1]]))
        U = F.relu(self.bn_u1(U))
        U = U.reshape(-1, 2, self.no_filters)
        U_broadcast = self.fc_u1(input[3])
        V = torch.cat([input[2], U_broadcast], dim=-1)
        ######## Graph Convolution #########
        V = self.graph_conv_1([V, input[1], self.no_filters, self.no_features_for_conv])
        V_batch = V.reshape(-1, V.shape[-1])
        V = F.relu(self.bn_v1(V_batch)).reshape(V.shape)

        U = U.reshape(-1, self.no_filters)
        U = F.relu(self.bn_u2(U))
        # U = U.reshape(-1, self.no_filters * 2)

        return V, U, U_broadcast


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output


class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super(SelfAttention, self).__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return output


class GCNNet_With_Finetune(nn.Module):

    def __init__(self, U_shape, A_shape, V_shape, U_broadcast_shape, no_filters=64, n_head=5):
        super(GCNNet_With_Finetune, self).__init__()
        self.gcn1 = GCNBlock(U_shape, A_shape, V_shape, U_broadcast_shape, no_filters)
        self.gcn2 = GCNBlock(U_shape, A_shape, V_shape, U_broadcast_shape, no_filters, flag=1)
        self.gcn3 = GCNBlock(U_shape, A_shape, V_shape, U_broadcast_shape, no_filters, flag=1)
        self.attend1 = SelfAttention(n_head=n_head, d_k=no_filters, d_v=no_filters, d_x=no_filters, d_o=no_filters)
        # self.fc1 = nn.Linear(V.shape[-1]+U.shape[-1], 256)
        self.fc1 = nn.Linear((A_shape[1] + 2) * no_filters, 256)
        self.drop1 = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.fc_merge1 = nn.Linear(512*2, 512)
        self.drop_merge1 = nn.Dropout(p=0.3)
        self.fc_merge2 = nn.Linear(512, 256)
        self.drop_merge2 = nn.Dropout(p=0.3)
        self.bn_merge2 = nn.BatchNorm1d(256)
        self.fc_merge3 = nn.Linear(256, 64)
        # self.fc4 = nn.Linear(64, 2)
        self.fc4 = nn.Linear(64, 1)

    def init_weight(self):
        pass

    def forward(self, input):
        V, U, U_broadcast = self.gcn1(input[:-1])
        V, U, U_broadcast = self.gcn2([U, input[1], V, U_broadcast])
        V, U, _ = self.gcn3([U, input[1], V, U_broadcast])
        V = self.attend1(V)
        U = torch.reshape(U, shape=[-1, 2, U.shape[-1]])
        V = torch.cat([V, U], dim=1)
        V = V.reshape([V.shape[0], -1])
        V = F.relu(self.bn1(self.fc1(V)))
        V = F.relu(self.bn2((self.fc2(V))))
        V = F.relu(self.bn3(self.fc3(V)))

        V_merge1 = F.relu(self.drop_merge1(self.fc_merge1(V)))
        V_merge2 = F.relu(self.drop_merge2(self.bn_merge2(self.fc_merge2(V_merge1))))
        V_merge3 = F.relu(self.fc_merge3(V_merge2))
        V = F.sigmoid(self.fc4(V_merge3))

        return V

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter) or isinstance(param, torch.Tensor):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
