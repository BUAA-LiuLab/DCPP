import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNetForMolCLR(nn.Module):

    def __init__(self):
        super(FCNetForMolCLR, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.dp1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(p=0.1)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.dp3 = nn.Dropout(p=0.1)
        self.bn3 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):

        x1 = F.relu(self.fc1(x))
        x1 = self.dp1(x1)
        x1 = self.bn1(x1)
        x2 = F.relu(self.fc2(x1))
        x2 = self.dp2(x2)
        x2 = self.bn2(x2)
        x3 = F.relu(self.fc3(x2))
        x3 = self.bn3(x3)
        out = self.out(x3)

        return out


class FPN(nn.Module):
    def __init__(self, input_bias=0.):
        super(FPN, self).__init__()
        self.output_bias = input_bias
        self.morgan_dim = 2048
        self.maccs_dim = 334
        self.embed_dim = 1024
        self.hidden_dim_1 = 128
        self.hidden_dim2 = 64
        self.dropout = 0.1

        # morgan layer
        self.morgan_fc1 = nn.Linear(self.morgan_dim, self.hidden_dim_1)
        self.morgan_act_func = nn.ReLU()
        self.morgan_fc2 = nn.Linear(self.hidden_dim_1, self.hidden_dim2)
        self.morgan_dropout = nn.Dropout(p=self.dropout)
        # massc layer
        self.maccs_fc1 = nn.Linear(self.maccs_dim, self.hidden_dim_1)
        self.maccs_act_func = nn.ReLU()
        self.maccs_fc2 = nn.Linear(self.hidden_dim_1, self.hidden_dim2)
        self.maccs_dropout = nn.Dropout(p=self.dropout)
        # embed layer
        self.embed_fc1 = nn.Linear(self.embed_dim, self.hidden_dim_1)
        self.embed_act_func = nn.ReLU()
        self.embed_fc2 = nn.Linear(self.hidden_dim_1, self.hidden_dim2)
        self.embed_dropout = nn.Dropout(p=self.dropout)
        # merge layer
        self.merge_fc1 = nn.Linear(self.hidden_dim2*3, self.hidden_dim2)
        self.merge_act_func = nn.ReLU()
        self.merge_fc2 = nn.Linear(self.hidden_dim2, 1)
        self.merge_dropout = nn.Dropout(p=self.dropout)

        self.sigmoid = nn.Sigmoid()
        # self.init_parameter()

    def init_parameter(self):
        torch.nn.init.constant_(self.merge_fc2.bias.data, val=-1 * self.output_bias)
        self.merge_fc2.bias.requires_grad = False

    def forward(self, x):

        x_morgan = x[:,:self.morgan_dim]
        x_maccs = x[:,self.morgan_dim:self.morgan_dim+self.maccs_dim]
        x_embed = x[:, self.morgan_dim+self.maccs_dim:]

        x_morgan = self.morgan_fc1(x_morgan)
        x_morgan = self.morgan_dropout(x_morgan)
        x_morgan = self.morgan_act_func(x_morgan)
        x_morgan = self.morgan_fc2(x_morgan)

        x_maccs = self.maccs_fc1(x_maccs)
        x_maccs = self.maccs_dropout(x_maccs)
        x_maccs = self.maccs_act_func(x_maccs)
        x_maccs = self.maccs_fc2(x_maccs)
        #
        x_embed = self.embed_fc1(x_embed)
        x_embed = self.embed_dropout(x_embed)
        x_embed = self.embed_act_func(x_embed)
        x_embed = self.embed_fc2(x_embed)

        x_merge = torch.cat([x_morgan, x_maccs, x_embed], dim=1)
        x_merge = self.merge_fc1(x_merge)
        x_merge = self.merge_dropout(x_merge)
        # x_merge = self.merge_act_func(x_merge)
        x_out= self.merge_fc2(x_merge)
        x_out = x_out.view(-1)

        return x_out, self.sigmoid(x_out), x_merge
