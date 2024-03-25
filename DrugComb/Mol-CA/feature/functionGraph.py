import numpy as np

element_symbol_list = ['Cl', 'N', 'P', 'Br', 'B', 'S', 'I', 'F', 'C', 'O', 'H']
possible_hybridization_types = ['SP2', 'SP3', 'SP', 'S', 'SP3D', 'SP3D2']
possible_chirality_types = ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER']

reference_dic = {
    'symbol': element_symbol_list,
    'hybridization': possible_hybridization_types,
    'chirality': possible_chirality_types,
}

atom_feat_to_use = ['symbol','hybridization', 'chirality', 'is_chiral', 'is_cyclic',
                    'is_aromatic', 'is_donor', 'is_acceptor', 'degree','vdw_radius','explicitvalence',
                    'implicitvalence', 'totalnumHs', 'formalcharge','radical_electrons','atomic_number']

def toolOneHotEncoding(inputValue, inputListValue):

    if inputValue not in inputListValue:
        print("MCTool.py, toolOneHotEncoding, inputValue: {0} is not in inputListValue: {1}"
                    .format(inputValue, inputListValue))
        exit()

    return list(map(lambda s: inputValue == s, inputListValue))

class ClassAdjacentTensor:

    def __init__(self, atoms, edges, atom_number):

        self.atoms = atoms
        self.edges = edges
        self.atom_number = atom_number

    def OnlyCovalentBond(self):

        A = np.zeros([self.atom_number, 4, self.atom_number])
        for e in self.edges:
            e_feature = self.edges[e]
            e_type = e_feature.type_num
            A[e[0], e_type-1, e[1]] = 1
            A[e[1], e_type-1, e[0]] = 1

        return A


class ClassVertexMatrix:

    def __init__(self, inputAtoms):

        self.nodes = inputAtoms
        self.node_number = len(inputAtoms)

    def captainFeatureMatrix(self, atom_feat=atom_feat_to_use):

        V = []
        for node_idx in range(self.node_number):
            result = []
            node_feature = self.nodes[node_idx].feature.__dict__
            result.extend(toolOneHotEncoding(node_feature['symbol'], reference_dic['symbol']))
            result.extend(toolOneHotEncoding(node_feature['hybridization'], reference_dic['hybridization']))
            result.extend(toolOneHotEncoding(node_feature['chirality'], reference_dic['chirality']))

            rest_feature = [float(node_feature[feat_key]) for feat_key in atom_feat[3:]]
            result.extend(rest_feature)
            V.append(result)

        return np.array(V)
