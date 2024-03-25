from train.model_train import func_train_all, func_train_multimodel, func_test_interpretation_gcn, func_train_multimodel_0807
from feature.get_feature import func_get_feature_multimodel, tool_LoadFeature_multimodel
from feature.split_train_val import function_split_train_val
from interpretation.gcn_interpretation import func_train_gcn, func_interpretation_gcn
from utils.read_data import func_read_smiles
import pickle

if __name__ == '__main__':

    path_smiles_train = r'F:\PycharmProjects\DeepTCM\data\train_smiles.txt'
    path_smiles_valid = r'F:\PycharmProjects\DeepTCM\data\valid_smiles.txt'
    # read data
    list_smiles_data_train = func_read_smiles(path_smiles_train)
    list_smiles_data_valid = func_read_smiles(path_smiles_valid)
    print('the all number of smiles pair:', len(list_smiles_data_train), len(list_smiles_data_valid))
    # get feature of smiles pair
    dicC2I_ = {}

    list_label, list_feature, dicC2I_ = func_get_feature_multimodel(list_smiles_data_train,'0129_train', dicC2I_)
    list_label_v, list_feature_v, feature_dic = func_get_feature_multimodel(list_smiles_data_valid, '0129_valid', dicC2I_)
    # list_label, list_feature, _ = tool_LoadFeature_multimodel('0129_train')
    # list_label_v, list_feature_v, feature_dic = tool_LoadFeature_multimodel('0129_valid')
    # list_label, list_feature, _ = tool_LoadFeature_multimodel('0129_train_finger_1')
    # list_label_v, list_feature_v, feature_dic = tool_LoadFeature_multimodel('0129_valid_finger_1')
    # list_label, list_feature, _ = tool_LoadFeature_multimodel('0811_train_r')
    # list_label_v, list_feature_v, feature_dic = tool_LoadFeature_multimodel('0811_valid_r')
    # divide train dataset and validation dataset
    k = 10
    list_label_count = list_label+list_label_v
    pos_number = list_label.count(1)
    neg_number = list_label.count(0)
    # train_loader, valid_loader = function_split_train_val(k, list_label, list_feature)
    # train
    # func_test_interpretation_gcn(list_label, list_feature, feature_dic, pos_number, neg_number)
    for i in range(0,1):
        # func_train_gcn(list_label, list_feature, feature_dic, pos_number, neg_number)
        func_train_multimodel(list_label, list_feature, list_label_v, list_feature_v, feature_dic, pos_number, neg_number, i)

    # 测试用的
    # path_smiles = r'F:\PycharmProjects\DeepTCM\data\top10.txt'
    # # read data
    # list_smiles_data = func_read_smiles(path_smiles)
    # print('the all number of smiles pair:', len(list_smiles_data))
    # # get feature of smiles pair
    # list_label, list_feature, _ = func_get_feature_multimodel(list_smiles_data,'interpret')
    # # list_label, list_feature, _ = tool_LoadFeature_multimodel('interpret')
    # with open(r'save\feature_save_all\\'+ '0811_valid_r2' + 'out_feature_dic.pkl',
    #           'rb') as f:
    #     feature_dic = pickle.load(f)
    # # divide train dataset and validation dataset
    # k = 10
    # pos_number = list_label.count(1)
    # neg_number = list_label.count(0)
    # # train_loader, valid_loader = function_split_train_val(k, list_label, list_feature)
    # # train
    # func_interpretation_gcn(list_label, list_feature, feature_dic, pos_number, neg_number)
    # func_train_gcn(list_label, list_feature, feature_dic, pos_number, neg_number)