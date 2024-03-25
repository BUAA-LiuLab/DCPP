from model_GCN import GCNNet_With_Finetune

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random

def broadcast_global_state(subgraph_size: np.ndarray, global_state: np.ndarray, V_shape: list):

    no_global_state_feature = int(global_state.shape[-1]//2)
    global_broadcast = np.zeros(shape=(1, V_shape[1], no_global_state_feature))
    max_graph_size = V_shape[1]

    for i in range(global_state.shape[0]):
        subg1 = np.zeros(subgraph_size[i][0], dtype=np.int)
        subg2 = np.ones(subgraph_size[i][1], dtype=np.int)
        cated = np.concatenate([subg1, subg2], axis=0)
        i_global_state = global_state[i].reshape([2,-1])
        gathered_gs = i_global_state[cated]
        padding_num = max_graph_size - (subgraph_size[i][0] + subgraph_size[i][1])
        padding = np.pad(gathered_gs, pad_width=((0, padding_num), (0, 0)))
        padding = padding.reshape([1, max_graph_size, no_global_state_feature])
        global_broadcast = np.concatenate([global_broadcast, padding], axis=0)

    return global_broadcast[1:]

class CFunctionTrainGCNNetWithFinetune:

    def __init__(self, inputDP):

        self.dp = inputDP

    def __soliderCalAccuracy(self, list_y_pre: torch.Tensor, list_y: torch.Tensor, type_accuracy='accuracy'):

        if type_accuracy == 'accuracy':

            num_y = list_y_pre.shape[0]
            return torch.sum(list_y == list_y_pre) / num_y

        elif type_accuracy == 'TPR':

            num_tp_fn = torch.sum(list_y == 1)
            return torch.sum(list_y_pre * (list_y_pre == list_y)) / num_tp_fn

        elif type_accuracy == 'TNR':

            num_tn_fp = torch.sum(list_y == 0)
            return torch.sum((1-list_y_pre) * (list_y_pre == list_y)) / num_tn_fp

    def __captrainEval(self, net, data_loader, optimizer):

        net.eval()
        for param in net.parameters():
            param.requires_grad = False
        loss_all = 0.
        list_y_pre = []
        list_y = []
        softmax_cross_entropy = torch.nn.BCELoss(reduction='mean', reduce=True)  # 定义损失函数
        for i_batch, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            batch_U, batch_A, batch_V, batch_U_broadcast, batch_E, batch_y = batch_data
            y_pre = net([batch_U, batch_A, batch_V, batch_U_broadcast, batch_E])
            y_pre = y_pre.reshape(-1)
            loss = softmax_cross_entropy(y_pre, batch_y)
            print(i_batch, loss)
            loss_all += loss.data.numpy()

            if list_y_pre == []:
                list_y_pre = F.softmax(y_pre)
            else:
                list_y_pre = torch.cat([list_y_pre, F.softmax(y_pre)], dim=0)

            if list_y == []:
                list_y = batch_y
            else:
                list_y = torch.cat([list_y, batch_y], dim=0)

        loss = loss_all / len(data_loader)
        list_y_pre = torch.argmax(list_y_pre, dim=-1)
        acc = self.__soliderCalAccuracy(list_y_pre, list_y)
        tpr = self.__soliderCalAccuracy(list_y_pre, list_y, type_accuracy='TPR')
        tnr = self.__soliderCalAccuracy(list_y_pre, list_y, type_accuracy='TNR')
        bacc = (tpr + tnr) / 2

        print('loss', loss, 'acc: ', acc, 'TPR: ', tpr, 'TNR: ', tnr, 'BACC: ', bacc)

        return loss, acc, tpr, tnr, bacc

    def __captrainEvalSigmoid(self, net, data_loader, optimizer):

        net.eval()
        for param in net.parameters():
            param.requires_grad = False
        loss_all = 0.
        list_y_pre = []
        list_y = []
        softmax_cross_entropy = torch.nn.BCELoss(reduction='mean', reduce=True)  # 定义损失函数
        for i_batch, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            batch_U, batch_A, batch_V, batch_U_broadcast, batch_E, batch_y = batch_data
            y_pre = net([batch_U, batch_A, batch_V, batch_U_broadcast, batch_E])
            y_pre = y_pre.reshape(-1)
            loss = softmax_cross_entropy(y_pre, batch_y)
            print(i_batch, loss)
            loss_all += loss.data.numpy()

            if list_y_pre == []:
                list_y_pre = y_pre
            else:
                list_y_pre = torch.cat([list_y_pre, y_pre], dim=0)

            if list_y == []:
                list_y = batch_y
            else:
                list_y = torch.cat([list_y, batch_y], dim=0)

        loss = loss_all / len(data_loader)
        list_y_pre[list_y_pre>0.5] = 1.
        list_y_pre[list_y_pre<=0.5] = 0.
        # list_y_pre = torch.argmax(list_y_pre, dim=-1)
        acc = self.__soliderCalAccuracy(list_y_pre, list_y)
        tpr = self.__soliderCalAccuracy(list_y_pre, list_y, type_accuracy='TPR')
        tnr = self.__soliderCalAccuracy(list_y_pre, list_y, type_accuracy='TNR')
        bacc = (tpr + tnr) / 2

        print('loss', loss, 'acc: ', acc, 'TPR: ', tpr, 'TNR: ', tnr, 'BACC: ', bacc)

        return loss, acc, tpr, tnr, bacc

    def train(self):

        # subgraph_size = self.dp.myFeature_FINETURE.MATRIX_SG
        # U = self.dp.myFeature_FINETURE.MATRIX_G
        # V = self.dp.myFeature_FINETURE.MATRIX_V
        # A = self.dp.myFeature_FINETURE.MATRIX_A
        # E = self.dp.myFeature_FINETURE.MATRIX_E
        # U_broadcast = broadcast_global_state(subgraph_size, U, V.shape)
        # Y = self.dp.myFeature_FINETURE.VECTOR_LABEL
        # U = U.reshape([U.shape[0], 2, -1])
        # U = torch.Tensor(U)
        # V = torch.Tensor(V)
        # A = torch.Tensor(A)
        # E = torch.tensor(E)
        # U_broadcast = torch.Tensor(U_broadcast)
        # Y = torch.tensor(Y, dtype=torch.long)
        # data = TensorDataset(U, A, V, U_broadcast, E, Y)
        #
        # import pickle
        # with open(r'F:\PycharmProjects\UltraTCM\python_script\CoAggregators\out\out1_FeatureFinetune_preprocess_torch.pkl', 'wb') as f:
        #
        #     pickle.dump(data,
        #                 f, protocol=4)
        import pickle
        with open(r'F:\PycharmProjects\UltraTCM\python_script\CoAggregators\out\out1_FeatureFinetune_preprocess_torch.pkl', 'rb') as f:

            data = pickle.load(f)
        U = data[:][0]
        A = data[:][1]
        V = data[:][2]
        U_broadcast = data[:][3]
        E = data[:][4]
        Y = data[:][5]

        print('train')
        data_loader = DataLoader(data, batch_size=64, shuffle=True)
        net = GCNNet_With_Finetune(U.shape, A.shape, V.shape, U_broadcast.shape)
        iteration = 100
        l_r = 0.01
        softmax_cross_entropy = torch.nn.CrossEntropyLoss()  # 定义损失函数
        # optimizer = torch.optim.Adam(net.parameters(), lr=l_r)
        optimizer = torch.optim.SGD(net.parameters(), lr=l_r)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
        # train

        for i_iter in range(iteration):
            net.train()
            for param in net.parameters():
                param.requires_grad = True

            for i_batch, batch_data in enumerate(data_loader):

                optimizer.zero_grad()
                batch_U, batch_A, batch_V, batch_U_broadcast, batch_E, batch_y = batch_data
                y_pre = net([batch_U, batch_A, batch_V, batch_U_broadcast, batch_E])
                loss = softmax_cross_entropy(y_pre, batch_y)
                print(i_iter, i_batch, loss, end='\t')
                loss.backward(retain_graph=True)
                print(i_iter, i_batch, loss)
                optimizer.step()
                scheduler.step()

            optimizer.zero_grad()

            self.__captrainEval(net, data_loader, optimizer)

    def cross_validation(self, k=10):

        # subgraph_size = self.dp.myFeature_FINETURE.MATRIX_SG
        # U = self.dp.myFeature_FINETURE.MATRIX_G
        # V = self.dp.myFeature_FINETURE.MATRIX_V
        # A = self.dp.myFeature_FINETURE.MATRIX_A
        # E = self.dp.myFeature_FINETURE.MATRIX_E
        # U_broadcast = broadcast_global_state(subgraph_size, U, V.shape)
        # Y = self.dp.myFeature_FINETURE.VECTOR_LABEL
        # U = U.reshape([U.shape[0], 2, -1])
        # U = torch.Tensor(U)
        # V = torch.Tensor(V)
        # A = torch.Tensor(A)
        # E = torch.tensor(E)
        # U_broadcast = torch.Tensor(U_broadcast)
        # Y = torch.tensor(Y, dtype=torch.long)
        # data = TensorDataset(U, A, V, U_broadcast, E, Y)
        #
        # import pickle
        # with open(r'F:\PycharmProjects\UltraTCM\python_script\CoAggregators\out\out1_FeatureFinetune_preprocess_torch.pkl', 'wb') as f:
        #
        #     pickle.dump(data,
        #                 f, protocol=4)
        import pickle
        with open(r'F:\PycharmProjects\UltraTCM\python_script\CoAggregators\out\out1_FeatureFinetune_preprocess_torch.pkl', 'rb') as f:

            data = pickle.load(f)
        # U = data[:][0][:-1142]
        # A = data[:][1][:-1142]
        # V = data[:][2][:-1142]
        # U_broadcast = data[:][3][:-1142]
        # E = data[:][4][:-1142]
        # Y = data[:][5][:-1142]
        # print(U.shape)
        # print('train')
        # list_shuffle_index = list(range(len(data)-1142))
        # random.seed(1)
        # random.shuffle(list_shuffle_index)
        # len_val = (len(data)-1142) // k

        # U = data[:][0][:-1142]
        # A = data[:][1][:-1142]
        # V = data[:][2][:-1142]
        # U_broadcast = data[:][3][:-1142]
        # E = data[:][4][:-1142]
        # Y = data[:][5][:-1142]
        # print(U.shape, A.shape, V.shape, E.shape, U_broadcast.shape)
        # print('train')
        # list_shuffle_index = list(range(len(data) - 1142))
        # random.seed(1)
        # random.shuffle(list_shuffle_index)
        # len_val = (len(data) - 1142) // k

        U = data[:][0]
        A = data[:][1]
        V = data[:][2]
        U_broadcast = data[:][3]
        E = data[:][4]
        Y = data[:][5]
        Y = torch.tensor(Y, dtype=torch.float)
        print(U.shape)
        print('train')
        list_shuffle_index = list(range(len(data)))
        random.seed(1)
        random.shuffle(list_shuffle_index)
        len_val = len(data) // k

        list_train_index = list_shuffle_index[len_val:]
        list_val_index = list_shuffle_index[:len_val]
        data_train = TensorDataset(U[list_train_index], A[list_train_index], V[list_train_index], U_broadcast[list_train_index], E[list_train_index], Y[list_train_index])
        data_val = TensorDataset(U[list_val_index], A[list_val_index], V[list_val_index], U_broadcast[list_val_index], E[list_val_index], Y[list_val_index])
        data_loader_train = DataLoader(data_train, batch_size=64, sampler=ImbalancedDatasetSampler(data_train))
        # data_loader_train = DataLoader(data_train, batch_size=64, shuffle=True)
        data_loader_val = DataLoader(data_val, batch_size=64, shuffle=True)

        random_seed = 123
        torch.manual_seed(random_seed)
        net = GCNNet_With_Finetune(U.shape, A.shape, V.shape, U_broadcast.shape)
        net_save = GCNNet_With_Finetune(U.shape, A.shape, V.shape, U_broadcast.shape)
        iteration = 100
        l_r = 0.01
        count_class1 = torch.sum(Y[list_train_index])
        count_class0 = len(Y[list_train_index]) - count_class1
        weight_CE = torch.FloatTensor([count_class1/count_class0, 1.])
        print(weight_CE)
        # softmax_cross_entropy = torch.nn.CrossEntropyLoss(weight=weight_CE)  # 定义损失函数
        softmax_cross_entropy = torch.nn.BCELoss(reduction='mean', reduce=True)
        # optimizer = torch.optim.Adam(net.parameters(), lr=l_r)
        optimizer = torch.optim.SGD(net.parameters(), lr=l_r, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
        # iteration train
        _, _, _, _, _ = self.__captrainEvalSigmoid(net, data_loader_train, optimizer)
        _, _, _, _, _ = self.__captrainEvalSigmoid(net, data_loader_val, optimizer)
        best_train_bacc, best_val_bacc = 0, 0

        for i_iter in range(iteration):
            net.train()
            for param in net.parameters():
                param.requires_grad = True

            for i_batch, batch_data in enumerate(data_loader_train):

                optimizer.zero_grad()
                batch_U, batch_A, batch_V, batch_U_broadcast, batch_E, batch_y = batch_data
                y_pre = net([batch_U, batch_A, batch_V, batch_U_broadcast, batch_E])
                y_pre = y_pre.reshape(-1)
                # print(y_pre.shape, batch_y.shape)
                loss = softmax_cross_entropy(y_pre, batch_y)
                loss.backward(retain_graph=True)
                print(i_iter, i_batch, loss)
                optimizer.step()
                scheduler.step()

            optimizer.zero_grad()
            # evaluation trainDataset and valDataset every iteration
            tmp_train_loss, tmp_train_acc, tmp_train_tpr, tmp_train_tnr, tmp_train_bacc = self.__captrainEvalSigmoid(net, data_loader_train, optimizer)
            tmp_val_loss, tmp_val_acc, tmp_val_tpr, tmp_val_tnr, tmp_val_bacc = self.__captrainEvalSigmoid(net, data_loader_val, optimizer)
            # save best model param
            if tmp_val_bacc > best_val_bacc:
                best_val_bacc = tmp_val_bacc
                tmp_best_state_dict = net.state_dict()
                net_save.load_state_dict(tmp_best_state_dict)
            if i_iter % 10 == 0:
                print(i_iter, '1114model(val bacc:{:s}) save...'.format(str(best_val_bacc)))
                torch.save(net_save.state_dict(), IO_PATH_SAVE_MODEL + '\\sigmoid_new_' + str(i_iter) + 'model_' + str(best_val_bacc) + '.pth')

    def predict1(self, k=10):

        # subgraph_size = self.dp.myFeature_FINETURE.MATRIX_SG
        # U = self.dp.myFeature_FINETURE.MATRIX_G
        # V = self.dp.myFeature_FINETURE.MATRIX_V
        # A = self.dp.myFeature_FINETURE.MATRIX_A
        # E = self.dp.myFeature_FINETURE.MATRIX_E
        # U_broadcast = broadcast_global_state(subgraph_size, U, V.shape)
        # Y = self.dp.myFeature_FINETURE.VECTOR_LABEL
        # U = U.reshape([U.shape[0], 2, -1])
        # U = torch.Tensor(U)
        # V = torch.Tensor(V)
        # A = torch.Tensor(A)
        # E = torch.tensor(E)
        # U_broadcast = torch.Tensor(U_broadcast)
        # Y = torch.tensor(Y, dtype=torch.long)
        # data = TensorDataset(U, A, V, U_broadcast, E, Y)
        #
        # import pickle
        # with open(r'F:\PycharmProjects\UltraTCM\python_script\CoAggregators\out\out1_FeatureFinetune_preprocess_torch.pkl', 'wb') as f:
        #
        #     pickle.dump(data,
        #                 f, protocol=4)
        import pickle
        with open(r'F:\PycharmProjects\UltraTCM\python_script\CoAggregators\out\out1_FeatureFinetune_preprocess_torch.pkl', 'rb') as f:

            data = pickle.load(f)
        # U = data[:][0][:-1142]
        # A = data[:][1][:-1142]
        # V = data[:][2][:-1142]
        # U_broadcast = data[:][3][:-1142]
        # E = data[:][4][:-1142]
        # Y = data[:][5][:-1142]
        # print(U.shape)
        # print('train')
        # list_shuffle_index = list(range(len(data)-1142))
        # random.seed(1)
        # random.shuffle(list_shuffle_index)
        # len_val = (len(data)-1142) // k

        # U = data[:][0][:-1142]
        # A = data[:][1][:-1142]
        # V = data[:][2][:-1142]
        # U_broadcast = data[:][3][:-1142]
        # E = data[:][4][:-1142]
        # Y = data[:][5][:-1142]
        # print(U.shape, A.shape, V.shape, E.shape, U_broadcast.shape)
        # print('train')
        # list_shuffle_index = list(range(len(data) - 1142))
        # random.seed(1)
        # random.shuffle(list_shuffle_index)
        # len_val = (len(data) - 1142) // k

        U = data[:][0]
        A = data[:][1]
        V = data[:][2]
        U_broadcast = data[:][3]
        E = data[:][4]
        Y = data[:][5]
        Y = torch.tensor(Y, dtype=torch.float)
        print(U.shape)
        print('train')

        data_test = TensorDataset(U, A, V, U_broadcast, E, Y)
        data_loader_test = DataLoader(data_test, batch_size=1)
        # data_loader_train = DataLoader(data_train, batch_size=64, shuffle=True)

        random_seed = 123
        torch.manual_seed(random_seed)
        net = GCNNet_With_Finetune(U.shape, A.shape, V.shape, U_broadcast.shape)
        checkpoint = torch.load(IO_PATH_SAVE_MODEL + '\\sigmoid_new_90model_tensor(0.9709).pth')
        net.load_state_dict(checkpoint)

        # iteration train
        list_y_pre = []
        list_y = []
        # list_h_feature = []
        for i_iter in range(1):
            net.eval()

            for i_batch, batch_data in enumerate(data_loader_test):

                batch_U, batch_A, batch_V, batch_U_broadcast, batch_E, batch_y = batch_data
                y_pre = net([batch_U, batch_A, batch_V, batch_U_broadcast, batch_E])
                y_pre = y_pre.reshape(-1)
                # print(y_pre.shape, batch_y.shape)
                list_y_pre.append(y_pre.item())
                list_y.append(batch_y.item())
                # y_h_feature = y_h_feature.detach().numpy()
                # y_h_feature = y_h_feature.reshape(-1,)
                # y_h_feature = [str(float(i_feature)) for i_feature in y_h_feature]
                # list_h_feature.append(y_h_feature)

        list_y = [str(i) for i in list_y]
        list_y_pre = [str(i) for i in list_y_pre]

        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
        tmp_val_loss, tmp_val_acc, tmp_val_tpr, tmp_val_tnr, tmp_val_bacc = self.__captrainEvalSigmoid(net,
                                                                                                       data_loader_test,
                                                                                                       optimizer)
        print(tmp_val_loss, tmp_val_acc, tmp_val_tpr, tmp_val_tnr, tmp_val_bacc)
        # with open(r'F:\PycharmProjects\UltraTCM\python_script\test\predict_20221125_train_dataset_97_feature.txt', 'w') as f:
        #     # f.write('---------------------\n')
        #     f.write('\t'.join(['y_pre', 'y', 'feature']) + '\n')
        #     for i in range(len(list_y_pre)):
        #         f.write(list_y[i] + '\t' + list_y_pre[i] + '\t')
        #         f.write('\t'.join(list_h_feature[i]) + '\n')



    def predict(self):

        subgraph_size = self.dp.myFeature_FINETURE.MATRIX_SG
        U = self.dp.myFeature_FINETURE.MATRIX_G
        V = self.dp.myFeature_FINETURE.MATRIX_V
        A = self.dp.myFeature_FINETURE.MATRIX_A
        E = self.dp.myFeature_FINETURE.MATRIX_E
        U_broadcast = broadcast_global_state(subgraph_size, U, V.shape)
        Y = self.dp.myFeature_FINETURE.VECTOR_LABEL
        U = U.reshape([U.shape[0], 2, -1])
        U = torch.Tensor(U)
        V = torch.Tensor(V)
        A = torch.Tensor(A)
        E = torch.tensor(E)
        U_broadcast = torch.Tensor(U_broadcast)
        Y = torch.tensor(Y, dtype=torch.long)
        data = TensorDataset(U, A, V, U_broadcast, E, Y)
        data_loader = DataLoader(data, batch_size=1, shuffle=False)

        random_seed = 123
        torch.manual_seed(random_seed)
        net = GCNNet_With_Finetune([8866,2,12], [8866,242,4,242], [8866,242,34], [8866,242,12])
        checkpoint = torch.load(IO_PATH_SAVE_MODEL + '\\50model_tensor(0.9685).pth')
        net.load_state_dict(checkpoint)

        net.eval()
        for param in net.parameters():
            param.requires_grad = False
        softmax = nn.Softmax(dim=0)
        for i_batch, batch_data in enumerate(data_loader):

            batch_U, batch_A, batch_V, batch_U_broadcast, batch_E, batch_y = batch_data
            print(torch.sum(batch_A), torch.sum(batch_U), torch.sum(batch_V), torch.sum(batch_U_broadcast), torch.sum(batch_E))
            y_pre = net([batch_U, batch_A, batch_V, batch_U_broadcast, batch_E])[0]
            print(y_pre.data.numpy())
            out = softmax(y_pre)[1]
            print(out.data.numpy())

    def trainByFL(self, k=10):

        import pickle
        with open(
                r'F:\PycharmProjects\UltraTCM\python_script\CoAggregators\out\out1_FeatureFinetune_preprocess_torch.pkl',
                'rb') as f:

            data = pickle.load(f)
        # U = data[:][0][:-1142]
        # A = data[:][1][:-1142]
        # V = data[:][2][:-1142]
        # U_broadcast = data[:][3][:-1142]
        # E = data[:][4][:-1142]
        # Y = data[:][5][:-1142]
        # print(U.shape)
        # print('train')
        # list_shuffle_index = list(range(len(data)-1142))
        # random.seed(1)
        # random.shuffle(list_shuffle_index)
        # len_val = (len(data)-1142) // k
        U = data[:][0]
        A = data[:][1]
        V = data[:][2]
        U_broadcast = data[:][3]
        E = data[:][4]
        Y = data[:][5]
        print(U.shape, A.shape, V.shape, U_broadcast.shape)
        print('train')
        list_shuffle_index = list(range(len(data)))
        random.seed(1)
        random.shuffle(list_shuffle_index)
        len_val = (len(data)) // k
        list_train_index = list_shuffle_index[len_val:]
        list_val_index = list_shuffle_index[:len_val]
        data_train = TensorDataset(U[list_train_index], A[list_train_index], V[list_train_index],
                                   U_broadcast[list_train_index], E[list_train_index], Y[list_train_index])
        data_val = TensorDataset(U[list_val_index], A[list_val_index], V[list_val_index], U_broadcast[list_val_index],
                                 E[list_val_index], Y[list_val_index])
        data_loader_train = DataLoader(data_train, batch_size=U.shape[0], sampler=ImbalancedDatasetSampler(data_train))
        data_loader_train_test = DataLoader(data_train, batch_size=64, shuffle=True)
        data_loader_val = DataLoader(data_val, batch_size=64, shuffle=True)
        net = GCNNet_With_Finetune(U.shape, A.shape, V.shape, U_broadcast.shape)
        # net_save = GCNNet_With_Finetune(U.shape, A.shape, V.shape, U_broadcast.shape)
        iteration = 3000
        l_r = 0.01
        softmax_cross_entropy = torch.nn.CrossEntropyLoss()  # 定义损失函数
        # optimizer = torch.optim.Adam(net.parameters(), lr=l_r)
        optimizer = torch.optim.SGD(net.parameters(), lr=l_r, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
        # 为节点和服务器划分数据集
        num_workers = 20
        defence = "fedavg"
        t = 0
        batch_size = 64
        server_data, each_worker_data = self.__captainAssignData(data_loader_train, 0.5, num_workers=num_workers)

        # train
        grad_list = []
        step_count = 0

        for e in range(iteration):
            print('epoch: ', e)
            train_loss = []
            global_weights = [x.clone() for x in net.parameters()]

            param_groups = optimizer.param_groups.copy()
            state = optimizer.state.copy()

            for i in range(num_workers):
                for j, (param) in enumerate(net.parameters()):
                    param.requires_grad = False
                    param_size = 1
                    for i_ in range(len(param.data.size())):
                        param_size *= param.data.size()[i_]
                    param.set_(global_weights[j].clone().data)
                # if False:
                if defence == Defence.SAMPLE.value:
                    optimizer.state = state.copy()
                    optimizer.param_groups = param_groups.copy()

                    if t < 10000:
                        n = 5
                    else:
                        n = 1
                    for k in range(n):

                        worker_sample_num = len(each_worker_data[i])
                        mini_index_all = np.arange(worker_sample_num)
                        np.random.shuffle(mini_index_all)
                        for mini_index in range(0, worker_sample_num, batch_size):
                            optimizer.zero_grad()
                            net.zero_grad()
                            for param in net.parameters():
                                param.requires_grad = True
                            minibatch = mini_index_all[mini_index:min(mini_index + batch_size, worker_sample_num)]
                            x = each_worker_data[i][minibatch][:-1]
                            y = each_worker_data[i][minibatch][-1]
                            output = net(x)
                            loss = softmax_cross_entropy(output, y)
                            train_loss.append(loss)
                            loss.backward()

                            updates_v = [param.grad.clone() for param in net.parameters()]
                            updates_v = torch.cat([xx.reshape((-1, 1)) for xx in updates_v], dim=0)
                            p = [x.clone() for x in net.parameters()]
                            self.__soliderUpdate(net, updates_v, p, lr=0.125)

                    grad_new = []
                    for g, (param) in enumerate(net.parameters()):
                        # param.requires_grad = False
                        p_ = global_weights[g] - param
                        # p_ = param
                        grad_new.append(p_.data.clone())
                    grad_list.append(grad_new)

                    for j, (param) in enumerate(net.parameters()):
                        param.requires_grad = False
                        param_size = 1
                        for i_ in range(len(param.data.size())):
                            param_size *= param.data.size()[i_]
                        param.set_(global_weights[j].clone().data)

                    optimizer.state = state.copy()
                    optimizer.param_groups = param_groups.copy()

                else:
                    for k in range(1):
                        for param in net.parameters():
                            param.requires_grad = True
                        net.zero_grad()
                        minibatch = np.random.choice(list(range(len(each_worker_data[i]))), size=batch_size,
                                                     replace=True)
                        x = each_worker_data[i][minibatch][:-1]
                        y = each_worker_data[i][minibatch][-1]
                        output = net(x)
                        loss = softmax_cross_entropy(output, y)
                        # train_loss += loss
                        print(loss, end='\t')
                        train_loss.append(loss)
                        loss.backward()

                    grad_list.append([param.grad.clone() for param in net.parameters()])
                    print(end='\n')
                    a = 1
                    a += 1

            if True:

                for param in net.parameters():
                    param.requires_grad = True
                net.zero_grad()
                output = net(server_data[:][:-1])
                loss = softmax_cross_entropy(output, server_data[:][-1])

                loss.backward()
                grad_list.append([param.grad.clone() for param in net.parameters()])
                # perform the aggregation
                optimizer.zero_grad()
                optimizer = self.__captainAVG(grad_list, net, defence, optimizer)
                if defence != Defence.SAMPLE.value:
                    optimizer.step()
                    scheduler.step()
                # lr_scheduler.step()
                t += 1

            del grad_list
            grad_list = []
            optimizer.zero_grad()
            # evaluation trainDataset and valDataset every iteration
            if e % 10 == 0:
                print('validation:')
                tmp_train_loss, tmp_train_acc, tmp_train_tpr, tmp_train_tnr, tmp_train_bacc = self.__captrainEval(net,
                                                                                                                  data_loader_train_test,
                                                                                                                  optimizer)
                tmp_val_loss, tmp_val_acc, tmp_val_tpr, tmp_val_tnr, tmp_val_bacc = self.__captrainEval(net,
                                                                                                        data_loader_val,
                                                                                                        optimizer)
            # if (e + 1) % 100 == 0:
            l_r = self.__soliderLRDecrease(learning_rate=l_r, epoch=e)
            print("learning rate : %s" % str(optimizer.param_groups[0]['lr']))

    def __captainAVG(self, gradients, net, defence_type=Defence.FEDAVG.value, optimizer=None):
        """
        gradients: list of gradients. The last one is the server update.
        net: model parameters.
        lr: learning rate.
        f: number of malicious clients. The first f clients are malicious.
        byz: attack type.
        """
        param_list = []
        for x in gradients:
            x = torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0)
            param_list.append(x)
        # param_list = [torch.cat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
        # let the malicious clients (last f clients) perform the byzantine attack
        n = len(param_list) - 1

        if defence_type == Defence.FEDAVG.value:
            global_update = self.__soliderAVG(param_list)
        self.__soliderParamUpdata(net, global_update)

        del param_list
        return optimizer

    def __soliderAVG(self, grad_li, drop=0.5):

        global_update = torch.mean(torch.cat(grad_li, dim=1), axis=-1)
        return global_update

    def __soliderParamUpdata(self, net, global_update, ):
        idx = 0
        # optimizer.zero_grad()
        for j, (param) in enumerate(net.parameters()):
            # param.requires_grad = False
            param_size = 1

            for i in range(len(param.data.size())):
                param_size *= param.data.size()[i]
            # param.set_(param.data - lr * global_update[idx: idx + param_size].reshape(param.data.shape))
            # param.set_()

            with torch.no_grad():
                param.grad = global_update[idx: idx + param_size].reshape(param.data.shape)
            idx += param_size

    def __captainAssignData(self, train_data, bias, num_labels=2, num_workers=2, server_pc=20, batch_size=16, seed=1):

        # assign training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_data_loader = [[] for _ in range(num_workers)]
        each_worker_data_u = [[] for _ in range(num_workers)]
        each_worker_data_a = [[] for _ in range(num_workers)]
        each_worker_data_v = [[] for _ in range(num_workers)]
        each_worker_data_u_broadcast = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]
        each_worker_data_e = [[] for _ in range(num_workers)]
        server_data_u = []
        server_data_a = []
        server_data_v = []
        server_data_u_broadcast = []
        server_data_e = []
        server_label = []
        server_data_load = []


        # randomly assign the data points base on the labels
        server_counter = [0 for _ in range(num_labels)]
        num_labels_server = server_pc / num_labels
        num_labels_server_count = {}
        server_sample_index = []

        for data_U, data_A, data_V, data_U_broadcast, data_E, label in train_data:
            for i in range(data_U.shape[0]):
                l = label[i].cpu().numpy().reshape(-1)[0]  # label
                if l not in num_labels_server_count:
                    num_labels_server_count[l] = 0
                if len(num_labels_server_count.keys()) == num_labels:
                    if sum(num_labels_server_count.values()) >= server_pc:
                        break
                if num_labels_server_count[l] < num_labels_server:
                    num_labels_server_count[l] += 1
                    server_sample_index.append(i)
                else:
                    continue

            server_data_u = data_U[server_sample_index]
            server_data_a = data_A[server_sample_index]
            server_data_v = data_V[server_sample_index]
            server_data_u_broadcast = data_U_broadcast[server_sample_index]
            server_data_e = data_E[server_sample_index]
            server_label = label[server_sample_index]
            server_data = TensorDataset(server_data_u, server_data_a, server_data_v, server_data_u_broadcast,
                                        server_data_e, server_label)
            # server_data_loader = DataLoader(server_data, batch_size=batch_size, shuffle=False)

            # server_data = data[:server_pc]
            # server_label = label[:server_pc]
            client_index = list(set(list(range(data_U.shape[0]))).difference(set(server_sample_index)))
            client_data_u = data_U[client_index]
            client_data_a = data_A[client_index]
            client_data_v = data_V[client_index]
            client_data_u_broadcast = data_U_broadcast[client_index]
            client_data_e = data_E[client_index]
            client_label = label[client_index]

            # data_index = self.__soliderNonIID(client_label[:-num_workers * batch_size], num_workers, bias, num_labels, seed=seed)
            # last_index = client_label.shape[0]
            # for i in range(len(data_index)):
            #     item_index = data_index[i] + list(range(last_index - num_workers * batch_size + i * batch_size,
            #                                             last_index - num_workers * batch_size + (i + 1) * batch_size))
            data_index = self.__soliderNonIID(client_label, num_workers, bias, num_labels,
                                              seed=seed)
            for i in range(len(data_index)):
                item_index = data_index[i]
                each_worker_data_u[i] = client_data_u[item_index]
                each_worker_data_a[i] = client_data_a[item_index]
                each_worker_data_v[i] = client_data_v[item_index]
                each_worker_data_u_broadcast[i] = client_data_u_broadcast[item_index]
                each_worker_data_e[i] = client_data_e[item_index]
                each_worker_label[i] = client_label[item_index]
                each_worker_data_i = TensorDataset(each_worker_data_u[i], each_worker_data_a[i], each_worker_data_v[i],
                                                   each_worker_data_u_broadcast[i], each_worker_data_e[i], each_worker_label[i])
                each_worker_data[i] = each_worker_data_i
                # each_worker_data_loader[i] = DataLoader(each_worker_data_i, batch_size=batch_size, shuffle=False)


        return server_data, each_worker_data

    def __soliderNonIID(self, labels, num_users, beta, num_class, seed):
        # print('num_class', num_class)
        client_idx_map = {i: {} for i in range(num_users)}
        client_size_map = {i: {} for i in range(num_users)}
        label_distributions = np.random.dirichlet(np.repeat(beta, num_users),
                                                  num_class)  # 返回一个num_class, num_users的二维矩阵
        dict_users = []
        for y in range(num_class):
            label_y_idx = np.where(labels.numpy() == y)[0]
            label_y_size = len(label_y_idx)
            sample_size = (label_distributions[y] * label_y_size).astype(np.int)
            sample_size[num_users - 1] += len(label_y_idx) - np.sum(sample_size)
            for i in range(num_users):
                client_size_map[i][y] = sample_size[i]

            np.random.shuffle(label_y_idx)
            sample_interval = np.cumsum(sample_size)
            for i in range(num_users):
                up_bound = sample_interval[i]
                client_idx_map[i][y] = label_y_idx[(sample_interval[i - 1] if i > 0 else 0): up_bound]
        for i in range(num_users):
            users_list = []
            for j in range(num_class):
                users_list += list(client_idx_map[i][j])
            dict_users.append(users_list)

        return dict_users

    def __soliderUpdate(self, net, updates, p, lr=0.125):
        idx = 0
        for j, (param) in enumerate(net.parameters()):
            param.requires_grad = False
            param_size = 1
            for i in range(len(param.data.size())):
                param_size *= param.data.size()[i]
            param.set_(p[j].data - lr * updates[idx: idx + param_size].reshape(param.data.shape))

            # with torch.no_grad():
            #    param.grad = updates[idx: idx + param_size].reshape(param.data.shape)
            idx += param_size

    def __soliderLRDecrease(self, learning_rate, epoch):

        learning_rate = max(learning_rate * 1 / (1 + 0.0001 * epoch), 0.0000000001)
        return learning_rate
