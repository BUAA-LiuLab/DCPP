import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import os
from rdkit.Chem.rdchem import BondType as BT
import torch

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


att_dtype = np.float32

PeriodicTable = Chem.GetPeriodicTable()
try:
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('/RDKit file path**/RDKit/Data/','BaseFeatures.fdef')  #The 'RDKit file path**' is the installation path of RDKit.
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
possible_atom_type = ['H','B','C','N','O','F','Si','P','S','Cl','Br','I', 'Virtual']  # virtual means virtual atom node
possible_hybridization = ['S','SP','SP2', 'SP3', 'SP3D','SP3D2', 'UNSPECIFIED']
possible_bond_type = ['SINGLE','DOUBLE','TRIPLE','AROMATIC', 'Virtual']  # virtual means virtual bond

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        print(x,type(x))
        raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def donor_acceptor(rd_mol):
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)
    feats = factory.GetFeaturesForMol(rd_mol)
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            for u in feats[i].GetAtomIds():
                is_donor[u] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            for u in feats[i].GetAtomIds():
                is_acceptor[u] = 1
    return is_donor, is_acceptor

def AtomAttributes(rd_atom, is_donor, is_acceptor, extra_attributes=[]):
    
    rd_idx = rd_atom.GetIdx()
    #Inititalize
    attributes = []
    #Add atimic number
    attributes += one_of_k_encoding(rd_atom.GetSymbol(), possible_atom_type)
    #Add heavy neighbor count
    attributes += one_of_k_encoding(len(rd_atom.GetNeighbors()), [0, 1, 2, 3, 4, 5, 6])
    #Add neighbor hydrogen count
    attributes += one_of_k_encoding(rd_atom.GetTotalNumHs(includeNeighbors=True), [0, 1, 2, 3, 4])
    #Add hybridization type
    attributes += one_of_k_encoding(rd_atom.GetHybridization().__str__(), possible_hybridization)
    #Add boolean if chiral
    attributes += one_of_k_encoding(int(rd_atom.GetChiralTag()), [0, 1, 2, 3])
    # Add boolean if in ring
    attributes.append(rd_atom.IsInRing())
    # Add boolean if aromatic atom
    attributes.append(rd_atom.GetIsAromatic())
    #Add boolean if donor
    # attributes.append(is_donor[rd_idx])
    # #Add boolean if acceptor
    # attributes.append(is_acceptor[rd_idx])
    
    attributes += extra_attributes
    return np.array(attributes, dtype=att_dtype)


def atom_featurizer(rd_mol):
    
    is_donor, is_acceptor = donor_acceptor(rd_mol)
    
    #### add atoms descriptors####
    V = []
    for k, atom in enumerate(rd_mol.GetAtoms()):
        all_atom_attr = AtomAttributes(atom, is_donor, is_acceptor) 
        V.append(all_atom_attr)
    return np.array(V, dtype=att_dtype)


def atom_featurizer_embed(rd_mol):

    #### add atoms descriptors####
    V = []
    for k, atom in enumerate(rd_mol.GetAtoms()):
        atom_attr = possible_atom_type.index(atom.GetSymbol())
        # if atom.GetIsAromatic():
        #     atom_attr = possible_atom_type.index(atom.GetSymbol())+len(possible_atom_type)
        # else:
        #     atom_attr = possible_atom_type.index(atom.GetSymbol())
        V.append(atom_attr)

    return np.array(V, dtype=np.int)

def atom_featurizer_embed_with_nerb(rd_mol):

    #### add atoms descriptors####
    V = []
    for k, atom in enumerate(rd_mol.GetAtoms()):
        atom_attr = possible_atom_type.index(atom.GetSymbol())
        atom_attr2 = [0, 1, 2, 3, 4, 5, 6].index(len(atom.GetNeighbors()))
        atom_attr3 = [0, 1, 2, 3, 4].index(atom.GetTotalNumHs(includeNeighbors=True))
        atom_attr4 = possible_hybridization.index(atom.GetHybridization().__str__())
        atom_attr5 = [0, 1, 2, 3].index(int(atom.GetChiralTag()))
        atom_attr6 = atom.IsInRing()
        atom_attr7 = atom.GetIsAromatic()
        V.append([atom_attr, atom_attr2, atom_attr3, atom_attr4, atom_attr5, atom_attr6, atom_attr7])

    return np.array(V, dtype=np.int)
    
def bond_featurizer(mol):
    #conf = mol.GetConformer()
    bond_idx, bond_feats = [], []
    for b in mol.GetBonds():
        start = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        b_type = one_of_k_encoding(b.GetBondType().__str__(), possible_bond_type)
        #start_coor = [i for i in conf.GetAtomPosition(start)]
        #end_coor = [i for i in conf.GetAtomPosition(end)]
        #b_length = np.linalg.norm(np.array(end_coor)-np.array(start_coor))
        #b_type.insert(0, b_length)
        b_type.insert(0, b.GetIsConjugated())
        b_type.insert(0, b.GetIsAromatic())
        b_type.insert(0, b.IsInRing())
        bond_idx.append([start, end])
        bond_idx.append([end, start])
        bond_feats.append(b_type)
        bond_feats.append(b_type)
    e_sorted_idx = sorted(range(len(bond_idx)), key=lambda k:bond_idx[k])
    bond_idx = np.array(bond_idx)[e_sorted_idx]
    bond_feats = np.array(bond_feats, dtype=np.float32)[e_sorted_idx]
    return bond_idx.astype(np.int64), bond_feats.astype(np.float32) #bond_idx.astype(np.int64).T

def bond_featurizer_embed(mol):
    #conf = mol.GetConformer()
    bond_idx, bond_feats = [], []
    for b in mol.GetBonds():
        start = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        # b_type = one_of_k_encoding(b.GetBondType().__str__(), possible_bond_type)
        if b.IsInRing():
            b_type = possible_bond_type.index(b.GetBondType().__str__())+len(possible_bond_type)
        else:
            b_type = possible_bond_type.index(b.GetBondType().__str__())
        #start_coor = [i for i in conf.GetAtomPosition(start)]
        #end_coor = [i for i in conf.GetAtomPosition(end)]
        #b_length = np.linalg.norm(np.array(end_coor)-np.array(start_coor))
        #b_type.insert(0, b_length)
        # b_type.insert(0, b.GetIsConjugated())
        # b_type.insert(0, b.IsInRing())
        bond_idx.append([start, end])
        bond_idx.append([end, start])
        bond_feats.append(b_type)
        bond_feats.append(b_type)
    e_sorted_idx = sorted(range(len(bond_idx)), key=lambda k:bond_idx[k])
    bond_idx = np.array(bond_idx)[e_sorted_idx]
    bond_feats = np.array(bond_feats, dtype=np.int)[e_sorted_idx]
    return bond_idx.astype(np.int64), bond_feats.astype(np.int) #bond_idx.astype(np.int64).T

class Mol2Graph(object):
    def __init__(self, mol, **kwargs):
        # self.x = atom_featurizer(mol)
        self.x = atom_featurizer_embed(mol)
        # self.x = atom_featurizer_embed_with_nerb(mol)
        # self.edge_idx, self.edge_feats = bond_featurizer(mol)
        self.edge_idx, self.edge_feats = bond_featurizer_embed(mol)
        self.node_num = self.x.shape[0]
        #self.tag = mol.GetProp('_Name')
        for k in kwargs:
            self.__dict__[k] = kwargs[k]

class VirtualMol2Graph(object):

    def __init__(self, g1: Mol2Graph, g2: Mol2Graph, flag_embed=0,**kwargs):
        x = np.concatenate([g1.x, g2.x], axis=0)
        if not flag_embed:
            x_virtual = np.zeros(shape=(1, g1.x.shape[1]))
            self.x = np.concatenate([x, x_virtual], axis=0, dtype=att_dtype)
        else:
            x_virtual = np.array(possible_atom_type.index('Virtual')).reshape(1,)
            self.x = np.concatenate([x, x_virtual], axis=0, dtype=np.long).reshape(x.shape[0]+1,-1)
        edge_feats = np.concatenate([g1.edge_feats, g2.edge_feats], axis=0)
        e_idx2 = g2.edge_idx + g1.node_num
        edge_index = np.concatenate([g1.edge_idx, e_idx2], axis=0)
        if flag_embed:
            edge_feats_virtual = np.ones(shape=(2 * (g1.node_num + g2.node_num),)) * possible_bond_type.index('Virtual')
        else:
            edge_feats_virtual = np.zeros(shape=(2 * (g1.node_num + g2.node_num), edge_feats.shape[1]))
        edge_feats = np.concatenate([edge_feats, edge_feats_virtual], axis=0)
        bond_idx = [list(i) for i in edge_index]
        index_virtual = g1.node_num + g2.node_num
        for i in range(x.shape[0]-1):
            bond_idx.append([i, index_virtual])
            bond_idx.append([index_virtual, i])
        e_sorted_idx = sorted(range(len(bond_idx)), key=lambda k: bond_idx[k])
        self.edge_index = np.array(bond_idx)[e_sorted_idx]
        if not flag_embed:
            self.edge_feats = np.array(edge_feats, dtype=np.int)[e_sorted_idx].reshape(len(bond_idx),-1)
        else:
            self.edge_feats = np.array(edge_feats, dtype=np.long)[e_sorted_idx].reshape(len(bond_idx), -1)
        self.node_num = self.x.shape[0]
        for k in kwargs:
            self.__dict__[k] = kwargs[k]



        # def __init__(self, mol):
    #
    #     N = mol.GetNumAtoms()
    #     M = mol.GetNumBonds()
    #     atoms = mol.GetAtoms()
    #     bonds = mol.GetBonds()
    #
    #     #########################
    #     # Get the molecule info #
    #     #########################
    #     type_idx = []
    #     chirality_idx = []
    #     atomic_number = []
    #     for atom in mol.GetAtoms():
    #         type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
    #         chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
    #         atomic_number.append(atom.GetAtomicNum())
    #
    #     x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    #     x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    #     x = torch.cat([x1, x2], dim=-1)
    #
    #     row, col, edge_feat = [], [], []
    #     for bond in mol.GetBonds():
    #         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    #         row += [start, end]
    #         col += [end, start]
    #         edge_feat.append([
    #             BOND_LIST.index(bond.GetBondType()),
    #             BONDDIR_LIST.index(bond.GetBondDir())
    #         ])
    #         edge_feat.append([
    #             BOND_LIST.index(bond.GetBondType()),
    #             BONDDIR_LIST.index(bond.GetBondDir())
    #         ])
    #
    #     edge_index = torch.tensor([row, col], dtype=torch.long)
    #     edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    #
    #     self.x = x
    #     self.edge_idx = edge_index
    #     self.edge_feats = edge_attr
    #     self.node_num = self.x.shape[0]