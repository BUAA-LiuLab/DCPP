import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import random

def function_split_train_val(k, list_label, list_feature):

    number_feature = len(list_label)
    list_shuffle_index = list(range(number_feature))
    random.seed(1)
    random.shuffle(list_shuffle_index)
    len_val = number_feature // k

    list_train_index = list_shuffle_index[len_val:]
    list_val_index = list_shuffle_index[:len_val]

    list_feature = torch.tensor(np.array(list_feature))
    list_label = torch.tensor(list_label, dtype=torch.float)
    list_label = torch.unsqueeze(list_label, 1)
    train_list_feature = list_feature[list_train_index]
    train_list_label = list_label[list_train_index]
    valid_list_feature = list_feature[list_val_index]
    valid_list_label = list_label[list_val_index]
    data_train = TensorDataset(train_list_feature, train_list_label)
    data_valid = TensorDataset(valid_list_feature, valid_list_label)
    # data_val = TensorDataset(U[list_val_index], A[list_val_index], V[list_val_index], U_broadcast[list_val_index],
    #                          E[list_val_index], Y[list_val_index])
    data_loader_train = DataLoader(data_train, batch_size=64, shuffle=True)
    data_loader_valid = DataLoader(data_valid, batch_size=64, shuffle=False)
    # data_loader_train = DataLoader(data_train, batch_size=64, shuffle=True)
    # data_loader_val = DataLoader(data_val, batch_size=64, shuffle=True)

    return data_loader_train, data_loader_valid