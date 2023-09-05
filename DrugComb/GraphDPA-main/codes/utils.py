import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data, DataLoader, Dataset
import pickle

def evaluate(label, predict, fold):
    roc = roc_auc_score(label, predict)
    pr = average_precision_score(label, predict)
    return {'Fold':fold, 'ROC AUC':roc, 'PR AUC':pr}


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, file='finish_model.pkl'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.file = file

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        print('save', self.file)
        torch.save(model, self.file)                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss

        
class MyOwnDataset(Dataset):
    def __init__(self, root, indexs, graph_map, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.indexs = indexs
        self.graph_map = graph_map
            
    def len(self):
        return len(self.indexs)

    def get(self, idx):
        drug, pathway, label = self.graph_map[self.indexs[idx]]
        with open('{}/graphs/{}+{}+{}.pkl'.format(self.root, drug, pathway, label), 'rb') as file:
            data = pickle.load(file)
        return data