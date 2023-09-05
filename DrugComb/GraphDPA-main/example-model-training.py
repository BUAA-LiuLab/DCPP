import pickle
import pandas as pd
import numpy as np
import os
import sys
import time
from sklearn.model_selection import train_test_split
from codes.GraphDPA import GCN_Model
from codes.utils import *
import torch.nn.functional as F

print('Start Time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
embed_dim = 11

if not os.path.exists('example_saved_data/model_tmp'):
    os.mkdir('example_saved_data/model_tmp')

with open('example_saved_data/entities2id.pkl', 'rb') as file:
    entities2id = pickle.load(file)

with open('example_saved_data/graphs/graph_map.pkl', 'rb') as file:
    graph_map = pickle.load(file)
print('123')
train_indexs, test_indexs = train_test_split(range(len(graph_map)), test_size=0.1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GCN_Model(len(entities2id), embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
best_model_file = 'example_saved_data/model_tmp/{}+{}+GCNConv.pt'.format(embed_dim, 'None')
early_stopping = EarlyStopping(file=best_model_file, patience=5)
print('123')
train_dataset = MyOwnDataset('example_saved_data', train_indexs, graph_map)
test_dataset = MyOwnDataset('example_saved_data', test_indexs, graph_map)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100)
print('123')
# if not os.path.exists(best_model_file):
if 1:
    model.train()
    for epoch in range(1000):
        time_start = time.time()
        print(epoch)
        model.train()

        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)

            loss = F.cross_entropy(logits, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            break

        if epoch % 1 == 0:
            model.eval()
            test_loss = 0
            for data in test_loader:
                data = data.to(device)
                logits = model(data)
                # print(data.y.cpu().tolist())
                # print(logits.cpu()[:, 1].tolist())
                # test_loss = roc_auc_score(data.y.cpu().tolist(), logits.cpu()[:, 1].tolist())
                test_loss = 0.
                break

            print('{} Epoch {}: train loss {:6}, test ROC {}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch, train_loss, test_loss))
            early_stopping(-test_loss, model)
            if early_stopping.early_stop:
                print('Early Stopping')
                break
        print(time.time() - time_start)