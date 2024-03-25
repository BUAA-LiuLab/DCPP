from model_mol2vec import GCN_Finetune
from functionMolecular import ClassCoformer, ClassCocrystal, ClassCocrystalByOne
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
import torch
from torch_geometric.data import Data
import numpy as np
import os


class CFunctionExtractFeatureConvForFinetune:

    def __init__(self):

        self.dic_embedding = {}

        self.ATOM_LIST = list(range(1, 119))
        self.CHIRALITY_LIST = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]
        self.BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
        self.BONDDIR_LIST = [
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ]

    def __captainCalGraphFeature(self, inputSmile):

        mol = Chem.MolFromSmiles(inputSmile)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(self.ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(self.CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                self.BOND_LIST.index(bond.GetBondType()),
                self.BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                self.BOND_LIST.index(bond.GetBondType()),
                self.BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

    def calculateEmbedding(self, list_smiles):

        # 首先加载预训练模型
        model = GCN_Finetune()
        checkpoints_folder = os.path.join('./lib/pretrained_gcn', 'checkpoints')
        state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))
        model.load_my_state_dict(state_dict)
        model.eval()
        # 对每个smile分子，计算embedding向量
        for i, tmp_smile in enumerate(list_smiles):
            if i % 100 == 0:
                print(i)
            if tmp_smile not in self.dic_embedding:
                try:
                    tmp_feature = self.__captainCalGraphFeature(tmp_smile)
                    tmp_embedding = model(tmp_feature)
                    self.dic_embedding[tmp_smile] = tmp_embedding
                except:
                    info = r"can get finetune feature for molecular {:s}: ".format(tmp_smile)
                    print(info)
                    # logToUser(info)

    def extract(self, smile1, smile2):

        embed1 = self.dic_embedding[smile1]
        embed2 = self.dic_embedding[smile2]
        embed_conncet = torch.cat([embed1, embed2], dim=-1)

        return embed_conncet


class CFunctionExtractFeature:

    def __init__(self):

        self.C2_USED_DESC = 1
        self.C3_A_TYPE = 'OnlyCovalentBond'

    def extractBaseFilename(self, filename1, filename2):

        c1 = ClassCoformer(filename1)
        c2 = ClassCoformer(filename2)
        cc = ClassCocrystal(c1, c2)

        dic_feature = {}
        subgraph_size = np.array([c1.atom_number, c2.atom_number])
        dic_feature.setdefault('subgraph_size', subgraph_size)

        if self.C2_USED_DESC:  # 分子描述符
            desc = cc.captainDescriptors(includeSandP=True, charge_model='eem2015bm')
            dic_feature.setdefault('global_state', desc)

        if self.C3_A_TYPE:  # 共价键
            A = cc.captainCCGraphTensor(t_type=self.C3_A_TYPE)
            V = cc.VertexMatrix.captainFeatureMatrix()
            dic_feature.setdefault('A', A)
            dic_feature.setdefault('V', V)

        return dic_feature

    def extractBaseSmile(self, smile1, smile2):

        c1 = ClassCoformer(smile1, mol_type=0)
        c2 = ClassCoformer(smile2, mol_type=0)
        cc = ClassCocrystal(c1, c2)

        dic_feature = {}
        subgraph_size = np.array([c1.atom_number, c2.atom_number])
        dic_feature.setdefault('subgraph_size', subgraph_size)

        if self.C2_USED_DESC:  # 分子描述符
            desc = cc.captainDescriptors(includeSandP=True, charge_model='eem2015bm')
            dic_feature.setdefault('global_state', desc)

        if self.C3_A_TYPE:  # 共价键
            A = cc.captainCCGraphTensor(t_type=self.C3_A_TYPE)
            V = cc.VertexMatrix.captainFeatureMatrix()
            dic_feature.setdefault('A', A)
            dic_feature.setdefault('V', V)

        return dic_feature

    def extractOneBaseSmile(self, smile1):

        c1 = ClassCoformer(smile1, mol_type=0)
        cc = ClassCocrystalByOne(c1)

        dic_feature = {}
        subgraph_size = c1.atom_number
        dic_feature.setdefault('subgraph_size', subgraph_size)

        if self.C2_USED_DESC:  # 分子描述符
            desc = cc.captainDescriptors(includeSandP=True, charge_model='eem2015bm')
            dic_feature.setdefault('global_state', desc)

        if self.C3_A_TYPE:  # 共价键
            A = cc.captainCCGraphTensor(t_type=self.C3_A_TYPE)
            V = cc.VertexMatrix.captainFeatureMatrix()
            dic_feature.setdefault('A', A)
            dic_feature.setdefault('V', V)

        return dic_feature

    def mergeSmilePair(self, dic_feature1, dic_feature2):

        dic_feature = {}

        subgraph_size = np.array([dic_feature1['subgraph_size'], dic_feature2['subgraph_size']])
        dic_feature.setdefault('subgraph_size', subgraph_size)

        if self.C2_USED_DESC:  # 分子描述符
            desc = np.append(dic_feature1['global_state'], dic_feature2['global_state'], axis=0)
            dic_feature.setdefault('global_state', desc)

        if self.C3_A_TYPE:  # 共价键
            num_nodes_pair = dic_feature1['subgraph_size'] + dic_feature2['subgraph_size']
            A = np.zeros([num_nodes_pair, 4, num_nodes_pair])
            A[:dic_feature1['subgraph_size'], :, :dic_feature1['subgraph_size']] = dic_feature1['A']
            A[-dic_feature2['subgraph_size']:, :, -dic_feature2['subgraph_size']:] = dic_feature2['A']
            V = np.concatenate([dic_feature1['V'], dic_feature2['V']], axis=0)

            dic_feature.setdefault('A', A)
            dic_feature.setdefault('V', V)

        return dic_feature
