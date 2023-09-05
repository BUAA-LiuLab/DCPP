import pickle
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, KFold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

##节点编码
with open('example_base_data/drug2smile.pkl', 'rb') as file:
    drug2smile = pickle.load(file)
with open('example_base_data/pathway2genes.pkl', 'rb') as file:
    pathway2genes = pickle.load(file)
pathway2genes1 = dict()
for i in pathway2genes:
    value = pathway2genes[i]
    i = i.replace(':', '_')
    pathway2genes1[i] = value
pathway2genes = pathway2genes1
drugs, smiles = zip(*drug2smile.items())
pathways, genes = zip(*pathway2genes.items())
drugs = list(drugs)
smiles = list(smiles)
pathways = list(pathways)
genes = list(set(np.concatenate(list(genes), axis=0)))
print(len(drugs), len(smiles), len(pathways), len(genes))
input('123')
atom_tags = set()
for smile in smiles:
    mol = Chem.MolFromSmiles(smile)
    for atom in mol.GetAtoms():
        name = []
        name.append(str(atom.GetSymbol()))
        name.append(str(atom.GetDegree()))
        name.append(str(atom.GetImplicitValence()))
        name.append(str(atom.GetExplicitValence()))
        name.append(str(atom.GetIsAromatic()))
        name.append(str(atom.IsInRing()))
        name.append(str(atom.GetHybridization()))
        name.append(str(atom.GetTotalDegree()))
        atom_tags.add('-'.join(name))
atom_tags = list(atom_tags)

entities = []
entities.extend(drugs)
entities.extend(atom_tags)
entities.extend(pathways)
entities.extend(genes)

if not os.path.exists('molecular graph'):
    os.mkdir('molecular graph')
entities2id = dict(zip(entities, range(len(entities))))
with open('example_saved_data/entities2id.pkl', 'wb') as file:
    pickle.dump(entities2id, file)

pathway2gene_ids = {}
for pathway, genes in pathway2genes.items():
    pathway2gene_ids[pathway] = [entities2id[gene] for gene in genes]
with open('example_saved_data/pathway2gene_ids.pkl', 'wb') as file:
    pickle.dump(pathway2gene_ids, file)

# 构建药物通路对的拓扑图
with open('example_base_data/drug2smile.pkl', 'rb') as file:
    drug2smile = pickle.load(file)
with open('example_saved_data/entities2id.pkl'.format('molecular graph'), 'rb') as file:
    entities2id = pickle.load(file)
with open('example_saved_data/pathway2gene_ids.pkl'.format('molecular graph'), 'rb') as file:
    pathway2gene_ids = pickle.load(file)


def construction_graph(drug, pathway, label):
    smile = drug2smile[drug.lower()]
    mol = Chem.MolFromSmiles(smile)
    structure_atoms = []
    for atom in mol.GetAtoms():
        name = []
        name.append(str(atom.GetSymbol()))
        name.append(str(atom.GetDegree()))
        name.append(str(atom.GetImplicitValence()))
        name.append(str(atom.GetExplicitValence()))
        name.append(str(atom.GetIsAromatic()))
        name.append(str(atom.IsInRing()))
        name.append(str(atom.GetHybridization()))
        name.append(str(atom.GetTotalDegree()))
        structure_atoms.append(entities2id['-'.join(name)])

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_from = []
    edge_to = []
    for e1, e2 in g.edges:
        edge_from.append(e1 + 2)
        edge_to.append(e2 + 2)

    genes = pathway2gene_ids[pathway]
    x = [-1, -1] + structure_atoms + genes
    edge_from = edge_from + list(range(2, 2 + len(structure_atoms))) + list(
        range(2 + len(structure_atoms), 2 + len(structure_atoms) + len(genes)))
    edge_to = edge_to + np.zeros(len(structure_atoms), dtype=int).tolist() + np.ones(len(genes), dtype=int).tolist()
    return Data(x=torch.LongTensor(x), edge_index=torch.LongTensor([edge_from, edge_to]), y=torch.LongTensor([label]))


if not os.path.exists('example_saved_data/graphs'):
    os.mkdir('example_saved_data/graphs')

drug_pathway_associations = pd.read_csv('example_base_data/extra_val_set.csv')
drug_pathway_associations['label'] = 1
drugs = np.array(list(set(drug_pathway_associations['ChemicalName'].apply(lambda name: name.lower()).tolist())))
pathways = np.array(list(set(drug_pathway_associations['PathwayID'].tolist())))
pos_nums = drug_pathway_associations.shape[0]
while drug_pathway_associations.shape[0] < 2 * pos_nums:
    nega_drug_ind = np.random.randint(0, len(drugs), pos_nums)
    nega_pathway_ind = np.random.randint(0, len(pathways), pos_nums)
    nega_drugs = drugs[nega_drug_ind]
    nega_pathways = pathways[nega_pathway_ind]
    drug_pathway_associations = drug_pathway_associations.append(
        pd.DataFrame(list(zip(nega_drugs, nega_pathways, np.zeros(pos_nums, dtype=int))),
                     columns=drug_pathway_associations.columns), ignore_index=True)
    drug_pathway_associations = drug_pathway_associations.drop_duplicates(subset=['ChemicalName', 'PathwayID'],
                                                                          keep='first', ignore_index=True)
drug_pathway_associations = drug_pathway_associations[0:2 * pos_nums]
drug_pathway_associations.to_csv('example_saved_data/undersample_data_set_0821.csv', index=False)
graph_map = []
for drug, pathway, label in drug_pathway_associations.apply(lambda row: tuple(row), axis=1).values.tolist():

    graph_map.append((drug, pathway, label))
    graph = construction_graph(drug, pathway, label)
    file_name = r'example_saved_data/graphs/{}+{}+{}.pkl'.format(drug, pathway, label)
    with open(file_name, 'wb') as file:
        pickle.dump(graph, file)
with open('example_saved_data/graphs/graph_map_0821.pkl', 'wb') as file:
    pickle.dump(graph_map, file)