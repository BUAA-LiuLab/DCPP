import pickle
import pandas as pd
import numpy as np
import os
import sys
import time
from sklearn.model_selection import train_test_split
from codes.GraphDPA import GCN_Model
from codes.utils import *
# from prefetch_generator import BackgroundGenerator
import torch.nn.functional as F

print('Start Time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

embed_dim = 11
num = 15
conv_type = 'GCNConv'
substructure_type = 'molecular graph'

if not os.path.exists('example_saved_data/model_tmp'):
    os.mkdir('example_saved_data/model_tmp')
if not os.path.exists('example_saved_data/results'):
    os.mkdir('example_saved_data/results')

with open('example_saved_data/entities2id.pkl'.format(substructure_type), 'rb') as file:
    entities2id = pickle.load(file)

with open('example_saved_data/graphs/graph_map.pkl', 'rb') as file:
    graph_map = pickle.load(file)

res = {}
for i in range(num):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # best_model_file = 'saved_models/{}+{}+{}+{}.pt'.format(substructure_type, embed_dim, i, conv_type)
    best_model_file = 'saved_models/' + r'molecular graph+11+' + str(i) + '+GCNConv.pt'
    val_dataset = MyOwnDataset('example_saved_data', list(range(len(graph_map))), graph_map)
    # val_loader = BackgroundGenerator(DataLoader(val_dataset, batch_size=1000))
    val_loader = DataLoader(val_dataset, batch_size=1000)

    if not os.path.exists(best_model_file):
        print('ERROR-----The saved model file does not exist！！！！')

    model = torch.load(best_model_file, map_location=device)
    labels = []
    predicts = []

    model1 = GCN_Model(len(entities2id), embed_dim)
    model1.embedding.weight = model.embedding.weight
    model1.fc1 = model.fc1
    model1.gcn1.bias = model.gcn1.bias
    model1.gcn1.lin.weight.data = model.gcn1.weight.data.T
    model1.gcn2.bias = model.gcn2.bias
    model1.gcn2.lin.weight.data = model.gcn2.weight.data.T
    model1.gcn3.bias = model.gcn3.bias
    model1.gcn3.lin.weight.data = model.gcn3.weight.data.T
    model1.output = model.output
    model = model1

    for data in val_loader:
        print(i, '11')
        data = data.to(device)
        logits = model(data)
        labels.extend(data.y.cpu().tolist())
        predicts.extend(F.softmax(logits, dim=1).cpu()[:, 1].tolist())
    res['label'] = labels
    res[str(i)] = predicts
# res = pd.DataFrame(res)
# res['mean'] = res.iloc[:, 1:].mean(axis=1)
# res.to_csv('example_saved_data/results/{}_res1.csv'.format(conv_type), index=False)

res = pd.DataFrame(res)
res['mean'] = res.iloc[:,1:].mean(axis=1)
res.to_csv('example_saved_data/results/{}_res_0821_test.csv'.format(conv_type), index=False)
print(evaluate(res['label'], res['mean'], 'None'))