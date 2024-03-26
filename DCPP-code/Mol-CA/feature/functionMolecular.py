import pybel
from rdkit import Chem
import numpy as np
from functionGraph import ClassAdjacentTensor, ClassVertexMatrix
from utils.MCToolMolecular import toolCalcuDescriptors

pt = Chem.GetPeriodicTable()


class ClassAtomFeat(object):

    def __init__(self, Atom):
        self.coordinates = Atom.ob_coor
        self.symbol = Atom.rdkit_atom.GetSymbol()
        self.hybridization = Atom.rdkit_atom.GetHybridization().__str__()  # 获得原子杂化方式
        self.chirality = Atom.rdkit_atom.GetChiralTag().__str__()  # 原子的手性
        self.is_chiral = Atom.ob_atom.IsChiral()
        self.explicitvalence = Atom.rdkit_atom.GetExplicitValence()
        self.implicitvalence = Atom.rdkit_atom.GetImplicitValence()
        self.totalnumHs = Atom.rdkit_atom.GetTotalNumHs()
        self.formalcharge = Atom.rdkit_atom.GetFormalCharge()
        self.radical_electrons = Atom.rdkit_atom.GetNumRadicalElectrons()
        self.is_aromatic = Atom.rdkit_atom.GetIsAromatic()
        self.is_acceptor = Atom.ob_atom.IsHbondAcceptor()
        self.is_donor = Atom.ob_atom.IsHbondDonor()
        self.is_cyclic = Atom.rdkit_atom.IsInRing()
        self.is_metal = Atom.ob_atom.IsMetal
        self.atomic_weight = Atom.rdkit_atom.GetMass()
        self.atomic_number = Atom.rdkit_atom.GetAtomicNum()
        self.vdw_radius = pt.GetRvdw(self.symbol)
        self.sybyl_type = Atom.ob_atom.GetType()
        self.degree = Atom.rdkit_atom.GetDegree()


class ClassAtom:

    def __init__(self, rd_atom, ob_atom):

        self.rdkit_atom = rd_atom
        self.ob_atom = ob_atom
        self.idx = self.rdkit_atom.GetIdx()
        self.ob_coor = np.array([self.ob_atom.GetX(), self.ob_atom.GetY(), self.ob_atom.GetZ()])

    @property
    def feature(self):
        return ClassAtomFeat(self)


class ClassBond(object):

    def __init__(self, rdkit_bond):

        self.begin_atom_idx = rdkit_bond.GetBeginAtomIdx()
        self.end_atom_idx = rdkit_bond.GetEndAtomIdx()
        self.begin_atom_coor = [i for i in rdkit_bond.GetOwningMol().GetConformer().GetAtomPosition(self.begin_atom_idx)]
        self.end_atom_coor = [i for i in rdkit_bond.GetOwningMol().GetConformer().GetAtomPosition(self.end_atom_idx)]
        bond_type_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        self.bond_type = rdkit_bond.GetBondType().__str__()

        try:
            self.type_num = bond_type_list.index(self.bond_type) + 1
        except:
            logToUser("bond type is out of range {}".format(self.bond_type))
            self.type_num = 5

        self.length = np.linalg.norm(np.array(self.end_atom_coor) - np.array(self.begin_atom_coor))
        self.is_ring = rdkit_bond.IsInRing()
        self.is_conjugated = rdkit_bond.GetIsConjugated()


class ClassCoformer:

    def __init__(self, mol_file, mol_type=0, fmt=None, removeh=False):

        if mol_type == 0:
            # print('mol', mol_file)
            if fmt == None:
                format = mol_file.split('.')[-1]
            else:
                format = fmt

            self.removeh = removeh
            self.ob_mol = pybel.readstring('smi', mol_file)
            self.ob_mol.make3D()
            self.rdkit_mol = Chem.MolFromMolBlock(self.ob_mol.write('mol'), removeHs=removeh)
            self.ob_mol = self.ob_mol.OBMol
            self.mol_name = mol_file

        elif mol_type == 1:

            if fmt == None:
                format = mol_file.split('.')[-1]
            else:
                format = fmt
            self.removeh = removeh
            self.ob_mol = pybel.readfile(format, mol_file).__next__()
            self.rdkit_mol = Chem.MolFromMolBlock(self.ob_mol.write('mol'), removeh=removeh)
            self.ob_mol = self.ob_mol.OBMol
            self.mol_name = mol_file.split('\n')[0].strip()

        if self.rdkit_mol == None:
            raise ValueError('rdkit can not read:\n{}'.format(mol_file))

        if self.removeh:
            self.ob_mol.DeleteHydrogens()()
            self.rdkit_mol = Chem.RemoveHs(self.rdkit_mol)
        if self.ob_mol.NumAtoms() != len(self.rdkit_mol.GetAtoms()):
            raise ValueError('len(rdkit_mol.GetAtoms()) != len(ob_mol.atoms):\n{}'.format(mol_file))

        rd_atoms = {}
        ob_atoms = {}
        for i in range(self.ob_mol.NumAtoms()):
            rdcoord = tuple([round(j,4) for j in self.rdkit_mol.GetAtomWithIdx(i).GetOwningMol().GetConformer().GetAtomPosition(i)])
            rd_atoms.setdefault(rdcoord, self.rdkit_mol.GetAtomWithIdx(i))
            ob_atom = self.ob_mol.GetAtomById(i)
            obcoord = (round(ob_atom.GetX(),4), round(ob_atom.GetY(),4), round(ob_atom.GetZ(), 4))
            ob_atoms.setdefault(obcoord, ob_atom)
        self.atoms = {}
        for key in rd_atoms:
            ix = rd_atoms[key].GetIdx()
            self.atoms.setdefault(ix, ClassAtom(rd_atoms[key], ob_atoms[key]))
        self.atom_number = len(self.atoms)

    @property
    def get_edges(self):
        '''
        得到分子对象的所有键（边）的信息
        :return: 返回一个存放分子所有键的字典，key是键连接着的两个原子，value是Bond对象
        '''
        rdkit_mol = self.rdkit_mol
        edges = {}
        for b in rdkit_mol.GetBonds():
            start = b.GetBeginAtomIdx()
            end = b.GetEndAtomIdx()
            edges.setdefault((start, end), ClassBond(b))

        return edges

    def descriptors(self, includeSandP=True, charge_model='eem2015bm'):  # 计算药物分子描述符

        return toolCalcuDescriptors(self, includeSandP, charge_model)


class ClassCocrystal:

    def __init__(self, inputCoformer1: ClassCoformer, inputCoformer2: ClassCoformer):

        self.conformer1 = inputCoformer1
        self.conformer2 = inputCoformer2
        self.node_num1 = len(self.conformer1.atoms)
        self.node_num2 = len(self.conformer2.atoms)
        self.node_number = self.node_num1 + self.node_num2

        if self.conformer1.mol_name != None and self.conformer2.mol_name != None:
            self.name = self.conformer1.mol_name + '&' + self.conformer2.mol_name
        else:
            self.name = None

    @property
    def get_nodes(self):  # 返回混合物的原子信息字典（相当于图的节点信息），键是混合物分子的原子序数，值是rdkit和pybel的原子对象
        self.nodes = {}
        for atom_idx1 in self.conformer1.atoms:
            self.nodes.setdefault(atom_idx1, self.conformer1.atoms[atom_idx1])
        for atom_idx2 in self.conformer2.atoms:
            self.nodes.setdefault(atom_idx2+self.node_num1, self.conformer2.atoms[atom_idx2])

        return self.nodes

    @property
    def get_edges(self):
        self.edges = {}
        edges1 = self.conformer1.get_edges
        for k1 in edges1:
            self.edges.setdefault(k1, edges1[k1])
        edges2 = self.conformer2.get_edges
        for k2 in edges2:
            self.edges.setdefault((k2[0]+self.node_num1, k2[1]+self.node_num1), edges2[k2])

        return self.edges

    @property
    def VertexMatrix(self):

        return ClassVertexMatrix(self.get_nodes)

    def captainDescriptors(self, includeSandP=True, charge_model='eem2015bm'):  # 计算药物分子对的描述符

        desc1 = self.conformer1.descriptors(includeSandP, charge_model)
        desc2 = self.conformer2.descriptors(includeSandP, charge_model)

        return np.append(desc1, desc2, axis=0)

    def captainCCGraphTensor(self, t_type='OnlyCovalentBond', ):

        t_type_set = {'allfeature', 'allfeaturebin', 'isringandconjugated', 'onlycovalentbond',
                      'withbinbistancematrix', 'withbondlenth', 'withdistancematrix'}
        t_type = t_type.lower()
        if t_type not in t_type_set:
            raise ValueError('t_type is case insensitive and should be one of the list :{}'.format(str(list(t_type_set))))

        CCG_ins = ClassAdjacentTensor(self.get_nodes, self.get_edges, self.node_number)
        methods = [i for i in dir(CCG_ins) if '__' not in i]
        methods_dict = {}
        for method in methods:
            methods_dict[method.lower()] = getattr(CCG_ins, method)
        CCG = methods_dict[t_type]()

        return CCG


class ClassCocrystalByOne:

    def __init__(self, inputCoformer1: ClassCoformer):

        self.conformer1 = inputCoformer1
        self.node_num1 = len(self.conformer1.atoms)
        self.node_num2 = 0
        self.node_number = self.node_num1 + self.node_num2

        if self.conformer1.mol_name != None:
            self.name = self.conformer1.mol_name
        else:
            self.name = None

    @property
    def get_nodes(self):  # 返回混合物的原子信息字典（相当于图的节点信息），键是混合物分子的原子序数，值是rdkit和pybel的原子对象
        self.nodes = {}
        for atom_idx1 in self.conformer1.atoms:
            self.nodes.setdefault(atom_idx1, self.conformer1.atoms[atom_idx1])

        return self.nodes

    @property
    def get_edges(self):
        self.edges = {}
        edges1 = self.conformer1.get_edges
        for k1 in edges1:
            self.edges.setdefault(k1, edges1[k1])

        return self.edges

    @property
    def VertexMatrix(self):

        return ClassVertexMatrix(self.get_nodes)

    def captainDescriptors(self, includeSandP=True, charge_model='eem2015bm'):  # 计算药物分子对的描述符

        desc1 = self.conformer1.descriptors(includeSandP, charge_model)

        return desc1

    def captainCCGraphTensor(self, t_type='OnlyCovalentBond', ):

        t_type_set = {'allfeature', 'allfeaturebin', 'isringandconjugated', 'onlycovalentbond',
                      'withbinbistancematrix', 'withbondlenth', 'withdistancematrix'}
        t_type = t_type.lower()
        if t_type not in t_type_set:
            raise ValueError('t_type is case insensitive and should be one of the list :{}'.format(str(list(t_type_set))))

        CCG_ins = ClassAdjacentTensor(self.get_nodes, self.get_edges, self.node_number)
        methods = [i for i in dir(CCG_ins) if '__' not in i]
        methods_dict = {}
        for method in methods:
            methods_dict[method.lower()] = getattr(CCG_ins, method)
        CCG = methods_dict[t_type]()

        return CCG