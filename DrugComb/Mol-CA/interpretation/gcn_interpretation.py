from model import FCNetForMolCLR, FPN
from model_cnn_transformer import TransformerEncoder, TransformerEncoderMultiModel
from model_gcn_new import CCPGraph
# from functionTrainGCNNet import CFunctionTrainGCNNetWithFinetune
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torchsampler import ImbalancedDatasetSampler
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time

seed = 13
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def func_train_gcn(list_label, list_feature, feature_dic, pos_number, neg_number):
    # 分别训练每个子模型，并保存模型参数和子模型结果
    k = 10
    list_feature_graph = list_feature[0]
    # random.seed(1000)
    # random.shuffle(list_feature_graph)
    loader_Train, loader_Test = train_test_split(list_feature_graph, test_size=0.1)
    # loader_Train = list_feature_graph[len(list_feature_graph) // k:]
    # loader_Test = list_feature_graph[:len(list_feature_graph) // k]
    train_loader = DataLoader(loader_Train, batch_size=128, shuffle=False, drop_last=True)
    valid_loader = DataLoader(loader_Test, batch_size=128, shuffle=False)
    net_gcn = func_train_gcn_net(train_loader, valid_loader, pos_number, neg_number)

    net_gcn = CCPGraph()
    net_gcn.load_state_dict(torch.load('save/net1_params-8-0.9915930389455285-0.9905385735080059.pth'))


def func_train_gcn_net(train_tf, valid_tf, pos_num, neg_num, fold=0):
    n_epoch = 100
    # initial_bias = float(np.log2(pos_num / neg_num) * 3 + 0.5)
    net = CCPGraph()
    # optimizer = torch.optim.Adam(params=net.parameters(), lr=0.006, betas=(0.9, 0.98), eps=1e-9)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9)
    # optimizer = torch.optim.SGD(params=net.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=30, min_lr=0.001)
    softmax_cross_entropy = torch.nn.BCELoss(reduction='mean', reduce=True)

    def loss_function(real, pred_logit, sampleW=None):
        cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
        # cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real)
        return cross_ent.mean()

    weight_for_1 = torch.tensor(neg_num / pos_num, dtype=torch.float32)
    best_bacc = 0.
    best_i = 0
    best_index = 0
    for epoch in range(1, n_epoch):

        start = time.time()
        list_pred, list_real = [], []
        net.train()
        for (batch, data) in enumerate(train_tf):
            optimizer.zero_grad()
            output, y_pred, att, _ = net(data)
            loss = loss_function(data.y, output, sampleW=weight_for_1)
            # loss = softmax_cross_entropy(y_pred, data.y)
            # print(epoch, '--', batch, '--', loss)
            loss.backward()
            optimizer.step()
            # print(epoch, batch,loss)
            list_real.extend(list(data.y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        scheduler.step(epoch)
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        # print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Train:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_train_bacc = (tpr + tnr) / 2
        train_time = time.time()
        # print(train_time - start)

        list_pred, list_real = [], []
        net.eval()
        for (batch, data) in enumerate(valid_tf):
            optimizer.zero_grad()
            output, y_pred, att, _ = net(data)
            list_real.extend(list(data.y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        list_pred1 = np.array(list_pred)
        list_pred1[list_pred1 <= 0.5] = 0
        list_pred1[list_pred1 > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred1)
        # print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Valid:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_val_bacc = (tpr + tnr) / 2
        valid_time = time.time()
        # print(valid_time - train_time)
        if tmp_val_bacc > best_bacc:
            best_index = [tpr, tnr, tmp_val_bacc]
            # with open('save\score_gcn0813.txt', 'w')as f:
            #     best_bacc = tmp_val_bacc
            #     f.write(str(tmp_val_bacc)+'\n')
            #     for i in range(len(list_real)):
            #         f.write(str(list_real[i])+'\t'+str(list_pred[i])+'\n')
            torch.save(net.state_dict(),
                       'save/model/gcn_net_params0130-' + str(fold) +'_'+ str(tmp_train_bacc) + '-' + str(tmp_val_bacc) + '.pth')
    # with open('save\gcn_bacc.txt', 'a')as f:
    #     f.write(str(fold) + '\t' + '\t'.join([str(i) for i in best_index]) + '\n')
    return net


def func_interpretation_gcn(list_label, list_feature, feature_dic, pos_number, neg_number):
    # 分别训练每个子模型，并保存模型参数和子模型结果
    k = 10
    list_feature_graph = list_feature[0]
    # random.seed(1000)
    # random.shuffle(list_feature_graph)
    # loader_Train = list_feature_graph[len(list_feature_graph) // k:]
    # loader_Test = list_feature_graph[:len(list_feature_graph) // k]
    test_tf = DataLoader(list_feature_graph, batch_size=1, shuffle=False)

    net = CCPGraph()
    # net.load_state_dict(torch.load('save/model/gcn_net_params-0.9273314474640522-0.9216268109617733.pth'))
    net.load_state_dict(torch.load('save/model/gcn_net_params-0.9377767480177119-0.9379535558780843.pth'))
    net.eval()
    list_hidden = []
    for (batch, data) in enumerate(test_tf):
        _, _, interpretation, _ = net(data)
        interpretation = list(interpretation.detach().numpy())
        # print(epoch, batch,loss)
        list_hidden.append(interpretation)
    print(list_hidden)
    with open(r'F:\PycharmProjects\DeepTCM\save\interpretation\inter08121.txt', 'w')as f:
        for i in list_hidden:
            i = [str(float(j)) for j in i]
            f.write('\t'.join(i) + '\n')
    return list_hidden
