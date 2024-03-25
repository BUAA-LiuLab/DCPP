from model import FCNetForMolCLR, FPN
from model_cnn_transformer import TransformerEncoder, TransformerEncoderMultiModel
from model_gcn_new import CCPGraph
from interpretation.gcn_interpretation import func_train_gcn_net
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

def func_train_all(list_label, list_feature, feature_dic, pos_number, neg_number):

    # 分别训练每个子模型，并保存模型参数和子模型结果
    k = 10
    list_feature_graph = list_feature[0]
    random.seed(1000)
    random.shuffle(list_feature_graph)
    # loader_Train, loader_Test = train_test_split(list_feature_graph, test_size=0.1)
    loader_Train = list_feature_graph[len(list_feature_graph)//k:]
    loader_Test = list_feature_graph[:len(list_feature_graph)//k]
    train_loader = DataLoader(loader_Train, batch_size=128, shuffle=False, drop_last=True)
    valid_loader = DataLoader(loader_Test, batch_size=128, shuffle=False)
    net_gcn = func_train_gcn_net(train_loader, valid_loader, pos_number, neg_number)

    net_gcn = CCPGraph()
    net_gcn.load_state_dict(torch.load('save/net1_params-8-0.9915930389455285-0.9905385735080059.pth'))

    list_feature_finger = list_feature[2]
    list_feature_finger = [[list_feature_finger[i], list_label[i]] for i in range(len(list_feature_finger))]
    # random.seed(1000)
    random.shuffle(list_feature_finger)
    # loader_Train, loader_Test = train_test_split(list_feature_graph, test_size=0.1)
    loader_Train = list_feature_finger[len(list_feature_finger) // k:]
    loader_Test = list_feature_finger[:len(list_feature_finger) // k]
    loader_Train_x = [i[0] for i in loader_Train]
    loader_Train_y = [i[1] for i in loader_Train]
    # pos_number = loader_Train_y.count(1)
    # neg_number = loader_Train_y.count(0)
    loader_Test_x = [i[0] for i in loader_Test]
    loader_Test_y = [i[1] for i in loader_Test]
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(loader_Train_x), torch.tensor(loader_Train_y, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(loader_Test_x), torch.tensor(loader_Test_y, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=True)
    valid_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    net_fc = func_train_fingerprint(train_loader, valid_loader, pos_number, neg_number)

    # list_feature_finger = list_feature[1]
    # list_feature_finger = [[list_feature_finger[i], list_label[i]] for i in range(len(list_feature_finger))]

    train_x, valid_x, train_y, valid_y = train_test_split(list_feature[1], list_label, test_size=0.1)
    pos_number, neg_number = train_y.count(1), train_y.count(0)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.float32))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                               sampler=ImbalancedDatasetSampler(train_dataset))

    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y, dtype=torch.float32))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    func_train_cnn_transformer(train_loader, valid_loader, feature_dic, pos_number, neg_number)
    # 根据各子模型的中间特征，训练多模态混合模型
    return 0

def func_train_multimodel(list_label, list_feature, list_label_v, list_feature_v, feature_dic, pos_number, neg_number, fold=0):

    # 分别训练每个子模型，并保存模型参数和子模型结果
    # k = 10
    if fold != 0:
        print('fold: ', fold)
        num_split = len(list_feature_v[0])
        list_feature_graph, list_feature_graph_v = [[],[],[]], [[],[],[]]
        list_feature_graph = list_feature[0][0:num_split * (fold - 1)] + list_feature[0][num_split * fold:] + list_feature_v[0]
        list_feature_graph_v = list_feature[0][num_split * (fold - 1):num_split * fold]

    else:
        list_feature_graph = list_feature[0]
        list_feature_graph_v = list_feature_v[0]
    # loader_Train, loader_Test = train_test_split(list_feature_graph, test_size=0.1)
    loader_Train = list_feature_graph
    loader_Test = list_feature_graph_v
    train_loader = DataLoader(loader_Train, batch_size=128, shuffle=True, drop_last=True)
    valid_loader = DataLoader(loader_Test, batch_size=128, shuffle=False)
    net_gcn = func_train_gcn_net(train_loader, valid_loader, pos_number, neg_number, fold)  # 训练GCN模型

    loader_Test = list_feature_v[0]
    test_loader = DataLoader(loader_Test, batch_size=1, shuffle=False)
    net_gcn = CCPGraph()
    # net_gcn.load_state_dict(torch.load('F:\PycharmProjects\DeepTCM\save\model/gcn_net_params-0.937761637283604-0.9037768226829768.pth'))
    net_gcn.load_state_dict(torch.load('F:\PycharmProjects\DeepTCM\save\model/gcn_net_params0130-0_0.9725755040820794-0.9601429049905101.pth'))
    feature_graph = func_interpretation_gcn_net(test_loader, net_gcn)
    # feature_graph = func_hidden_gcn_net(test_loader, net_gcn)
    input('123')

    list_feature_finger = list_feature[2]
    list_feature_finger = [[list_feature_finger[i], list_label[i]] for i in range(len(list_feature_finger))]
    list_feature_finger_v = list_feature_v[2]
    list_feature_finger_v = [[list_feature_finger_v[i], list_label_v[i]] for i in range(len(list_feature_finger_v))]
    if fold != 0:
        print('fold: ', fold)
        num_split = len(list_feature_v[2])
        list_feature_finger1 = list_feature_finger[0:num_split * (fold - 1)] + list_feature_finger[num_split * fold:] + list_feature_finger_v
        list_feature_finger_v1 = list_feature_finger[num_split * (fold - 1):num_split * fold]
        list_feature_finger = list_feature_finger1
        list_feature_finger_v = list_feature_finger_v1

    # loader_Train, loader_Test = train_test_split(list_feature_graph, test_size=0.1)
    loader_Train = list_feature_finger
    loader_Test = list_feature_finger_v
    loader_Train_x = [i[0] for i in loader_Train]
    loader_Train_y = [i[1] for i in loader_Train]
    # pos_number = loader_Train_y.count(1)
    # neg_number = loader_Train_y.count(0)
    loader_Test_x = [i[0] for i in loader_Test]
    loader_Test_y = [i[1] for i in loader_Test]
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(loader_Train_x), torch.tensor(loader_Train_y, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(loader_Test_x), torch.tensor(loader_Test_y, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=True)
    valid_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    net_fc = func_train_fingerprint(train_loader, valid_loader, pos_number, neg_number, fold)  # 训练分子指纹模型
    #
    # input('123')
    #
    list_feature_finger = list_feature[2]
    loader_Test = [[list_feature_finger[i], list_label[i]] for i in range(len(list_feature_finger))]
    loader_Test_x = [i[0] for i in loader_Test]
    loader_Test_y = [i[1] for i in loader_Test]
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(loader_Test_x),
                                                  torch.tensor(loader_Test_y, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    # net_fc = FPN()
    # net_fc.load_state_dict(torch.load('F:\PycharmProjects\DeepTCM\save\model/finger_net_params-0.998301351658966-0.9276830554637195.pth'))
    feature_finger = func_hidden_finger_net(test_loader, net_fc)

    # train_x, valid_x, train_y, valid_y = train_test_split(list_feature[1], list_label, test_size=0.1)
    # pos_number, neg_number = train_y.count(1), train_y.count(0)
    # train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.float32))
    # # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, )
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
    #                                            sampler=ImbalancedDatasetSampler(train_dataset))
    #
    # valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y, dtype=torch.float32))
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    # func_train_cnn_transformer(train_loader, valid_loader, feature_dic, pos_number, neg_number)
    #

    # # 根据各子模型的中间特征，训练多模态混合模型
    list_index = list(range(len(list_label)))[:-6]
    train_index, valid_index = train_test_split(list_index, test_size=0.1)
    train_x = [list_feature[1][i] for i in train_index]
    valid_x = [list_feature[1][i] for i in valid_index]
    test_x = list_feature[1][-6:]
    train_y = [list_label[i] for i in train_index]
    valid_y = [list_label[i] for i in valid_index]
    test_y = list_label[-6:]
    train_feature_gcn, valid_feature_gcn, test_feature_gcn = feature_graph[train_index], feature_graph[valid_index], feature_graph[-6:]
    train_feature_finger, valid_feature_finger, test_feature_finger = feature_finger[train_index], feature_finger[valid_index], feature_finger[-6:]
    pos_number, neg_number = train_y.count(1), train_y.count(0)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x), train_feature_gcn, train_feature_finger,
                                                   torch.tensor(train_y, dtype=torch.float32))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                               sampler=ImbalancedDatasetSampler(train_dataset))

    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_x), valid_feature_gcn, valid_feature_finger,
                                                   torch.tensor(valid_y, dtype=torch.float32))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x), test_feature_gcn, test_feature_finger,
                                                   torch.tensor(test_y, dtype=torch.float32))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    func_train_cnn_transformer_multimodel(train_loader, valid_loader, test_loader, feature_dic, pos_number, neg_number)  # 训练多模态模型
    #
    # return 0


def func_train_multimodel_0807(list_label, list_feature, feature_dic, pos_number, neg_number):

    # 分别训练每个子模型，并保存模型参数和子模型结果
    k = 10
    # list_feature_graph = list_feature[0]
    # random.seed(1000)
    # random.shuffle(list_feature_graph)
    # # loader_Train, loader_Test = train_test_split(list_feature_graph, test_size=0.1)
    # loader_Train = list_feature_graph[len(list_feature_graph)//k:]
    # loader_Test = list_feature_graph[:len(list_feature_graph)//k]
    # train_loader = DataLoader(loader_Train, batch_size=128, shuffle=False, drop_last=True)
    # valid_loader = DataLoader(loader_Test, batch_size=128, shuffle=False)
    # net_gcn = func_train_gcn_net(train_loader, valid_loader, pos_number, neg_number)

    # loader_Test = list_feature[0]
    # test_loader = DataLoader(loader_Test, batch_size=128, shuffle=False)
    # net_gcn = CCPGraph()
    # net_gcn.load_state_dict(torch.load('F:\PycharmProjects\DeepTCM\save\model/gcn_net_params-0.937761637283604-0.9037768226829768.pth'))
    # feature_graph = func_hidden_gcn_net(test_loader, net_gcn)

    list_feature_finger = list_feature[2]
    list_feature_finger = [[list_feature_finger[i], list_label[i]] for i in range(len(list_feature_finger))]
    random.shuffle(list_feature_finger)
    # loader_Train, loader_Test = train_test_split(list_feature_graph, test_size=0.1)
    loader_Train = list_feature_finger[len(list_feature_finger) // k:]
    loader_Test = list_feature_finger[:len(list_feature_finger) // k]
    loader_Train_x = [i[0] for i in loader_Train]
    loader_Train_y = [i[1] for i in loader_Train]
    # pos_number = loader_Train_y.count(1)
    # neg_number = loader_Train_y.count(0)
    loader_Test_x = [i[0] for i in loader_Test]
    loader_Test_y = [i[1] for i in loader_Test]
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(loader_Train_x), torch.tensor(loader_Train_y, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(loader_Test_x), torch.tensor(loader_Test_y, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=True)
    valid_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    net_fc = func_train_fingerprint(train_loader, valid_loader, pos_number, neg_number)

    list_feature_finger = list_feature[2]
    loader_Test = [[list_feature_finger[i], list_label[i]] for i in range(len(list_feature_finger))]
    loader_Test_x = [i[0] for i in loader_Test]
    loader_Test_y = [i[1] for i in loader_Test]
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(loader_Test_x),
                                                  torch.tensor(loader_Test_y, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    net_fc = FPN()
    net_fc.load_state_dict(torch.load('F:\PycharmProjects\DeepTCM\save\model/finger_net_params-0.998301351658966-0.9276830554637195.pth'))
    feature_finger = func_hidden_finger_net(test_loader, net_fc)

    train_x, valid_x, train_y, valid_y = train_test_split(list_feature[1], list_label, test_size=0.1)
    pos_number, neg_number = train_y.count(1), train_y.count(0)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.float32))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                               sampler=ImbalancedDatasetSampler(train_dataset))

    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y, dtype=torch.float32))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    func_train_cnn_transformer(train_loader, valid_loader, feature_dic, pos_number, neg_number)

    # 根据各子模型的中间特征，训练多模态混合模型
    list_index = list(range(len(list_label)))[:-6]
    train_index, valid_index = train_test_split(list_index, test_size=0.1)
    train_x = [list_feature[1][i] for i in train_index]
    valid_x = [list_feature[1][i] for i in valid_index]
    test_x = list_feature[1][-6:]
    train_y = [list_label[i] for i in train_index]
    valid_y = [list_label[i] for i in valid_index]
    test_y = list_label[-6:]
    train_feature_gcn, valid_feature_gcn, test_feature_gcn = feature_graph[train_index], feature_graph[valid_index], feature_graph[-6:]
    train_feature_finger, valid_feature_finger, test_feature_finger = feature_finger[train_index], feature_finger[valid_index], feature_finger[-6:]
    pos_number, neg_number = train_y.count(1), train_y.count(0)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x), train_feature_gcn, train_feature_finger,
                                                   torch.tensor(train_y, dtype=torch.float32))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                               sampler=ImbalancedDatasetSampler(train_dataset))

    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_x), valid_feature_gcn, valid_feature_finger,
                                                   torch.tensor(valid_y, dtype=torch.float32))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x), test_feature_gcn, test_feature_finger,
                                                   torch.tensor(test_y, dtype=torch.float32))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    func_train_cnn_transformer_multimodel(train_loader, valid_loader, test_loader, feature_dic, pos_number, neg_number)

    return 0


def func_test_interpretation_gcn(list_label, list_feature, feature_dic, pos_number, neg_number):

    # 分别训练每个子模型，并保存模型参数和子模型结果
    # k = 10
    # list_feature_graph = list_feature[0]
    # random.seed(1)
    # list_index = list(range(len(list_label)))
    # random.shuffle(list_index)
    # # loader_Train, loader_Test = train_test_split(list_feature_graph, test_size=0.1)
    # list_feature_graph = [list_feature_graph[i] for i in list_index]
    # list_label = [list_label[i] for i in list_index]
    # loader_Train = list_feature_graph[len(list_feature_graph)//k:]
    # loader_Test = list_feature_graph[:len(list_feature_graph)//k]
    #
    # list_index = list(range(len(list_label[len(list_feature_graph)//k:])))
    # train_dataset = torch.utils.data.TensorDataset(torch.tensor(list_index), torch.tensor(list_label[len(list_feature_graph)//k:], dtype=torch.float32))
    # # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, )
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
    #                                            sampler=ImbalancedDatasetSampler(train_dataset))
    # loader_Train_new = []
    # for i_index, i_label in train_loader:
    #     loader_Train_new.append(loader_Train[i_index])
    # pos_number = list_label[len(list_feature_graph)//k:].count(1.)
    # neg_number = list_label[len(list_feature_graph)//k:].count(0.)
    # train_loader = DataLoader(loader_Train_new, batch_size=128, shuffle=False, drop_last=True)
    # valid_loader = DataLoader(loader_Test, batch_size=128, shuffle=False)
    # net_gcn = func_train_gcn_net(train_loader, valid_loader, pos_number, neg_number)
    # input('1231231313')

    loader_Test = list_feature[0]
    test_loader = DataLoader(loader_Test, batch_size=1, shuffle=False)
    net_gcn = CCPGraph()
    net_gcn.load_state_dict(torch.load('F:\PycharmProjects\DeepTCM\save\model/gcn_net_params-0.9407126875774894-0.9006662011985803.pth'))
    feature_graph = func_interpretation_gcn_net(test_loader, net_gcn)


# def loss_function(real, pred_logit, sampleW=None):
#     cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
#     return cross_ent.mean()

# def function_train(train_loader, valid_loader, pos_num, neg_num):
#
#     net = FCNetForMolCLR()
#     iteration = 100
#     l_r = 0.005
#
#     pos_weight = torch.tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=torch.float32)
#     softmax_cross_entropy = torch.nn.CrossEntropyLoss()  # 定义损失函数
#     optimizer = torch.optim.Adam(params=net.parameters(), lr=l_r, betas=(0.9, 0.98), eps=1e-9)
#     # optimizer = torch.optim.SGD(net.parameters(), lr=l_r)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
#
#     # train
#     for i_iter in range(iteration):
#         net.train()
#         for param in net.parameters():
#             param.requires_grad = True
#
#         for i_batch, batch_data in enumerate(train_loader):
#             optimizer.zero_grad()
#             batch_x, batch_y = batch_data
#             y_pre = net(batch_x)
#
#             # print(y_pre.shape, batch_y.shape)
#             loss = loss_function(y_pre, batch_y, pos_weight)
#             print(i_iter, i_batch, loss)
#             loss.backward(retain_graph=True)
#             # print(i_iter, i_batch, loss)
#             optimizer.step()
#             scheduler.step()
#
#         optimizer.zero_grad()


def func_train_gcn_net1(train_tf, valid_tf, pos_num, neg_num):

    n_epoch = 400
    initial_bias = float(np.log2(pos_num / neg_num)*3+0.5)
    net = CCPGraph(initial_bias)
    # optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.02)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=30, min_lr=0.00001)
    softmax_cross_entropy = torch.nn.BCELoss(reduction='mean', reduce=True)

    def loss_function(real, pred_logit, sampleW=None):
        cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
        # cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real)
        return cross_ent.mean()
    weight_for_1 = torch.tensor(neg_num/pos_num, dtype=torch.float32)

    for epoch in range(n_epoch):

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
        # scheduler.step(epoch)
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Train:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_train_bacc = (tpr + tnr) / 2
        train_time = time.time()
        print(train_time - start)

        list_pred, list_real = [], []
        net.eval()
        for (batch, data) in enumerate(valid_tf):
            optimizer.zero_grad()
            output, y_pred, att, _ = net(data)
            list_real.extend(list(data.y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Valid:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_val_bacc = (tpr + tnr) / 2
        valid_time = time.time()
        print(valid_time - train_time)

    torch.save(net.state_dict(),
               'save/model/gcn_net_params-' + str(tmp_train_bacc) + '-' + str(tmp_val_bacc) + '.pth')

    return net


def func_interpretation_gcn_net(test_tf, net):

    net.eval()
    list_hidden = []
    for (batch, data) in enumerate(test_tf):
        _, _, interpretation, _ = net(data)
        interpretation = list(interpretation.detach().numpy())
        # print(epoch, batch,loss)
        list_hidden.append(interpretation)

    # with open(r'F:\PycharmProjects\DeepTCM\save\interpretation\inter0720.txt', 'w')as f:
    with open(r'F:\PycharmProjects\DeepTCM\save\interpretation\inter0131.txt', 'w') as f:
        for i in list_hidden:
            i = [str(float(j)) for j in i]
            f.write('\t'.join(i) + '\n')
    return list_hidden


def func_hidden_gcn_net(test_tf, net):

    net.eval()
    list_hidden = []
    for (batch, data) in enumerate(test_tf):
        _, _, _, hidden = net(data)
        # print(epoch, batch,loss)
        if batch == 0:
            list_hidden = hidden
        else:
            list_hidden = torch.cat([list_hidden,hidden], dim=0)

    x_mean, x_std = torch.mean(list_hidden, dim=0), torch.std(list_hidden, dim=0)
    list_hidden = (list_hidden-x_mean) / (1e3 * x_std)
    list_hidden = torch.tensor(list_hidden, requires_grad=False)
    # BN = nn.BatchNorm1d(64)
    # list_hidden = BN(list_hidden)

    return list_hidden


def func_train_fingerprint(train_tf, valid_tf, pos_num, neg_num, fold=0):

    n_epoch = 55
    # initial_bias = float(np.log2(pos_num / neg_num)*3 + 0.5)
    net = FPN()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=30, min_lr=0.00001)
    softmax_cross_entropy = torch.nn.BCELoss(reduction='mean', reduce=True)

    def loss_function(real, pred_logit, sampleW=None):
        cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
        # cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real)
        return cross_ent.mean()

    weight_for_1 = torch.tensor(neg_num / pos_num, dtype=torch.float32)
    best_bacc = 0.

    for epoch in range(0, n_epoch, 1):

        start = time.time()
        list_pred, list_real = [], []
        net.train()
        for (batch, [x, y]) in enumerate(train_tf):
            optimizer.zero_grad()
            output, y_pred, _ = net(x)
            loss = loss_function(y, y_pred, sampleW=weight_for_1)
            # loss = softmax_cross_entropy(y, output)
            # print(epoch, '--', batch, '--', loss)
            loss.backward()
            optimizer.step()
            # print(epoch, batch,loss)
            list_real.extend(list(y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        # scheduler.step(epoch)
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Train:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_train_bacc = (tpr + tnr) / 2
        train_time = time.time()
        print(train_time - start)

        list_pred, list_real = [], []
        net.eval()
        for (batch, [x, y]) in enumerate(valid_tf):
            output, y_pred, _ = net(x)
            loss = loss_function(y_pred, y, sampleW=weight_for_1)
            list_real.extend(list(y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        list_pred1 = np.array(list_pred)
        list_pred1[list_pred1 <= 0.5] = 0
        list_pred1[list_pred1 > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred1)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Valid:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_val_bacc = (tpr + tnr) / 2
        valid_time = time.time()
        print(valid_time - train_time)

        if tmp_val_bacc > best_bacc:
            best_bacc = tmp_val_bacc
            best_index = [tpr, tnr, tmp_val_bacc]
    with open('save\score_fingerprint0813.txt', 'w')as f:
        for i in range(len(list_real)):
            f.write(str(list_real[i])+'\t'+str(list_pred[i])+'\n')
    # with open('save\\finger_bacc.txt', 'a')as f:
    #     f.write(str(fold) + '\t' + '\t'.join([str(i) for i in best_index]) + '\n')
    # torch.save(net.state_dict(),
    #            'save/model/finger_net_params-' + str(tmp_train_bacc) + '-' + str(
    #                tmp_val_bacc) + '.pth')

    return net


def func_hidden_finger_net(test_tf, net):

    net.eval()
    list_hidden = []
    for (batch, [x, _]) in enumerate(test_tf):
        _, _, hidden = net(x)
        if batch == 0:
            list_hidden = hidden
        else:
            list_hidden = torch.cat([list_hidden,hidden], dim=0)

    x_mean, x_std = torch.mean(list_hidden, dim=0), torch.std(list_hidden, dim=0)
    list_hidden = (list_hidden - x_mean) / (1e3 * x_std)
    list_hidden = torch.tensor(list_hidden, requires_grad=False)
    # BN = nn.BatchNorm1d(64)
    # list_hidden = BN(list_hidden)

    return list_hidden


def func_train_cnn_transformer_multimodel(train_tf, valid_tf, test_tf, feature_dic, pos_num, neg_num):

    def loss_function(real, pred_logit, sampleW=None):
        cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
        return cross_ent.mean()

    dicC2I = feature_dic

    num_layers = 2
    d_model = 100
    num_heads = 1
    dff = 1024
    seq_size = 200
    dropout_rate = 0.15
    epochs = 50

    print(pos_num / neg_num)
    initial_bias = float(np.log2(pos_num / neg_num) * 3 + 0.5)
    # weight_for_1 = torch.tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=torch.float32)
    weight_for_1 = torch.tensor(neg_num / pos_num, dtype=torch.float32)
    print(weight_for_1)
    net = TransformerEncoderMultiModel(num_layers, d_model, num_heads, dff, initial_bias, seq_size,
                             dropout_rate)
    learning_rate = 0.00005
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    bit_size = 1024  # circular fingerprint
    embeds = nn.Embedding(bit_size, d_model, padding_idx=0)
    nn.init.uniform(embeds.weight[1:], -1, 1)
    embeds.weight.requires_grad = False
    embeds.weight[1:].requires_grad = True

    tmp_train_bacc, tmp_val_bacc = 0, 0
    tmp_best_bacc = 0
    for epoch in range(epochs):
        start = time.time()

        list_pred, list_real = [], []
        net.train()
        for (batch, (X, X_GCN, X_Fin, Y)) in enumerate(train_tf):
            optimizer.zero_grad()
            mask = ~(X != 0)
            X = embeds(X)
            pred_logit_, y_pred = net(X, mask, X_GCN, X_Fin)
            loss = loss_function(Y, pred_logit_, sampleW=weight_for_1)
            # print(epoch, '--', batch, '--', loss)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # print(epoch, batch,loss)
            list_real.extend(list(Y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Train:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_train_bacc = (tpr + tnr) / 2
        train_time = time.time()
        print(train_time - start)

        list_pred, list_real = [], []
        net.eval()
        for (batch, (X, X_GCN, X_Fin, Y)) in enumerate(valid_tf):
            mask = ~(X != 0)
            X = embeds(X)
            pred_logit_, y_pred = net(X, mask, X_GCN, X_Fin)
            list_real.extend(list(Y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Valid:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_val_bacc = (tpr + tnr) / 2
        valid_time = time.time()
        print(valid_time - train_time)

        for (batch, (X, X_GCN, X_Fin, Y)) in enumerate(test_tf):
            mask = ~(X != 0)
            X = embeds(X)
            pred_logit_, y_pred = net(X, mask, X_GCN, X_Fin)
            print('test')
            print(pred_logit_)
            print(y_pred)
        if (tmp_train_bacc + tmp_val_bacc) > tmp_best_bacc:
            tmp_best_bacc = tmp_train_bacc + tmp_val_bacc
            torch.save(embeds.state_dict(), 'save/parameter_multi/embed1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(
                tmp_val_bacc) + '.pth')
            torch.save(net.state_dict(),
                       'save/parameter_multi/net1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(tmp_val_bacc) + '.pth')
            tmp_train_bacc, tmp_val_bacc = 0, 0


def func_train_cnn_transformer(train_tf, valid_tf, feature_dic, pos_num, neg_num):

    def loss_function(real, pred_logit, sampleW=None):
        cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
        return cross_ent.mean()

    dicC2I = feature_dic
    test_tf = makeDataForSmilesOnly(dicC2I)

    num_layers = 2
    d_model = 100
    num_heads = 1
    dff = 1024
    seq_size = 200
    dropout_rate = 0.15
    epochs = 50

    print(pos_num / neg_num)
    initial_bias = float(np.log2(pos_num / neg_num) * 3 + 0.5)
    # weight_for_1 = torch.tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=torch.float32)
    weight_for_1 = torch.tensor(neg_num / pos_num, dtype=torch.float32)
    print(weight_for_1)
    net = TransformerEncoder(num_layers, d_model, num_heads, dff, initial_bias, seq_size,
                             dropout_rate)
    learning_rate = 0.00002
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    bit_size = 1024  # circular fingerprint
    embeds = nn.Embedding(bit_size, d_model, padding_idx=0)
    nn.init.uniform(embeds.weight[1:], -1, 1)
    embeds.weight.requires_grad = False
    embeds.weight[1:].requires_grad = True

    tmp_train_bacc, tmp_val_bacc = 0, 0
    tmp_best_bacc = 0
    for epoch in range(epochs):
        start = time.time()

        list_pred, list_real = [], []
        net.train()
        for (batch, (X, Y)) in enumerate(train_tf):
            optimizer.zero_grad()
            mask = ~(X != 0)
            X = embeds(X)
            pred_logit_, y_pred = net(X, mask)
            loss = loss_function(Y, pred_logit_, sampleW=weight_for_1)
            # print(epoch, '--', batch, '--', loss)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # print(epoch, batch,loss)
            list_real.extend(list(Y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Train:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_train_bacc = (tpr + tnr) / 2
        train_time = time.time()
        print(train_time - start)

        list_pred, list_real = [], []
        net.eval()
        for (batch, (X, Y)) in enumerate(valid_tf):
            mask = ~(X != 0)
            X = embeds(X)
            pred_logit_, y_pred = net(X, mask)
            list_real.extend(list(Y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Valid:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_val_bacc = (tpr + tnr) / 2
        valid_time = time.time()
        print(valid_time - train_time)

        for (batch, (X, Y)) in enumerate(test_tf):
            mask = ~(X != 0)
            X = embeds(X)
            pred_logit_, y_pred = net(X, mask)
            print('test')
            print(pred_logit_)
            print(y_pred)
        if (tmp_train_bacc + tmp_val_bacc) > tmp_best_bacc:
            tmp_best_bacc = tmp_train_bacc + tmp_val_bacc
            torch.save(embeds.state_dict(), 'save/parameter/embed1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(
                tmp_val_bacc) + '.pth')
            torch.save(net.state_dict(),
                       'save/parameter/net1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(tmp_val_bacc) + '.pth')
            tmp_train_bacc, tmp_val_bacc = 0, 0

def makeDataForSmilesOnly(dicC2I_):
    def char2indices(listStr, dicC2I):
        listIndices = [0] * 200
        charlist = listStr
        for i, c in enumerate(charlist):
            if c not in dicC2I:
                dicC2I[c] = len(dicC2I)
                listIndices[i] = dicC2I[c] + 1
            else:
                listIndices[i] = dicC2I[c] + 1
        return listIndices
    listX_test, listY_test = [], []
    afile = r'F:\PycharmProjects\sa-mtl-master\sa-mtl-master\TOX21\NR-AR_wholetraining1_test.smiles'
    f = open(afile, "r")
    lines = f.readlines()
    cntTooLong = 0
    weirdButUseful = 0
    for line in lines:
        splitted = line.split(" ")
        i_smiles1, i_smiles2 = splitted[0].split('&')
        if i_smiles1 < i_smiles2:
            splitted[0] = i_smiles2 + '&' + i_smiles1
        if len(splitted[0]) >= 200:
            cntTooLong += 1
            if splitted[1] == "1":
                weirdButUseful += 1
            continue
        listX_test.append(char2indices(splitted[0], dicC2I_))  # length can vary
        listY_test.append(float(splitted[1]))
    f.close()
    print("how many weird cases exist?", cntTooLong, weirdButUseful)
    test_x, test_y = listX_test, listY_test

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    return test_loader

def func_train_cnn_transformer_old(train_tf, valid_tf, pos_num, neg_num):

    def loss_function(real, pred_logit, sampleW=None):
        cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
        return cross_ent.mean()

    num_layers = 2
    d_model = 100
    num_heads = 1
    dff = 1024
    seq_size = 200
    dropout_rate = 0.15
    epochs = 50

    print(pos_num / neg_num)
    initial_bias = float(np.log2(pos_num / neg_num) * 3 + 0.5)
    # weight_for_1 = torch.tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=torch.float32)
    weight_for_1 = torch.tensor(neg_num / pos_num, dtype=torch.float32)
    print(weight_for_1)
    net = TransformerEncoder(num_layers, d_model, num_heads, dff, initial_bias, seq_size,
                             dropout_rate)
    learning_rate = 0.00005
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

    bit_size = 1024  # circular fingerprint
    embeds = nn.Embedding(bit_size, d_model, padding_idx=0)
    nn.init.uniform(embeds.weight[1:], -1, 1)
    embeds.weight.requires_grad = False
    embeds.weight[1:].requires_grad = True

    tmp_train_bacc, tmp_val_bacc = 0, 0
    tmp_best_bacc = 0
    for epoch in range(epochs):
        start = time.time()

        list_pred, list_real = [], []
        net.train()
        for (batch, (X, Y)) in enumerate(train_tf):
            optimizer.zero_grad()
            mask = ~(X != 0)
            X = embeds(X)
            pred_logit_, y_pred = net(X, mask)
            loss = loss_function(Y, pred_logit_, sampleW=weight_for_1)
            # print(epoch, '--', batch, '--', loss)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # print(epoch, batch,loss)
            list_real.extend(list(Y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Train:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_train_bacc = (tpr + tnr) / 2
        train_time = time.time()
        print(train_time - start)

        list_pred, list_real = [], []
        net.eval()
        for (batch, (X, Y)) in enumerate(valid_tf):
            mask = ~(X != 0)
            X = embeds(X)
            pred_logit_, y_pred = net(X, mask)
            list_real.extend(list(Y.detach().numpy()))
            list_pred.extend(list(y_pred.detach().numpy()))
        list_pred = np.array(list_pred)
        list_pred[list_pred <= 0.5] = 0
        list_pred[list_pred > 0.5] = 1
        list_real = np.array(list_real)
        matrix = confusion_matrix(list_real, list_pred)
        print(matrix)
        tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        print('epoch:', epoch, '    Valid:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
        tmp_val_bacc = (tpr + tnr) / 2
        valid_time = time.time()
        print(valid_time - train_time)

        # for (batch, (X, Y)) in enumerate(test_tf):
        #     mask = ~(X != 0)
        #     X = embeds(X)
        #     pred_logit_, y_pred = net(X, mask)
        #     print('test')
        #     print(pred_logit_)
        #     print(y_pred)
        # if (tmp_train_bacc + tmp_val_bacc) > tmp_best_bacc:
        #     tmp_best_bacc = tmp_train_bacc + tmp_val_bacc
        #     torch.save(embeds.state_dict(), 'save/embed1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(
        #         tmp_val_bacc) + '.pth')
        #     torch.save(net.state_dict(),
        #                'save/net1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(tmp_val_bacc) + '.pth')
        #     tmp_train_bacc, tmp_val_bacc = 0, 0
