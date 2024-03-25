from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import math
import numpy as np

class CFunctionExtractFeatureDesc:

    def __init__(self):

        self.descr = Descriptors._descList
        self.calc = [x[1] for x in self.descr]

    def __captainCalFeature(self, inputSmile: str):

        input_mol = Chem.MolFromSmiles(inputSmile)
        # fp = AllChem.GetMorganFingerprintAsBitVect(input_mol, 4, nBits=2048)
        # fp_list = []
        # fp_list.extend(fp.ToBitString())
        # fp_expl = [float(x) for x in fp_list]
        # ds_n = []
        # for d in self.calc:
        #     v = d(input_mol)
        #     if v > np.finfo(
        #             np.float32
        #     ).max:  # postprocess descriptors for freak large values
        #         ds_n.append(np.finfo(np.float32).max)
        #     elif math.isnan(v):
        #         ds_n.append(np.float32(0.0))
        #     else:
        #         ds_n.append(np.float32(v))
        # fp_expl.extend(list(ds_n))

        fp = []
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(input_mol, 2, nBits=1024)
        fp.extend(fp_morgan)
        fp_maccs = AllChem.GetMACCSKeysFingerprint(input_mol)
        fp.extend(fp_maccs)

        return fp

    def extract(self, smile1, smile2):

        desc_feature1 = self.__captainCalFeature(smile1)
        desc_feature2 = self.__captainCalFeature(smile2)

        desc_feature1.extend(desc_feature2)

        return desc_feature1
