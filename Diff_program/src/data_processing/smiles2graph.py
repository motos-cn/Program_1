from rdkit import Chem
from rdkit.Chem import rdchem


class smilesToGraphConverter:
    def __init__(self):
        # 0 reserved for virtual node, real categories start from 1, except num_h
        self.hybridization_map = {
            rdchem.HybridizationType.OTHER: 1,
            rdchem.HybridizationType.S: 2,
            rdchem.HybridizationType.SP: 3,
            rdchem.HybridizationType.SP2: 4,
            rdchem.HybridizationType.SP2D: 5,
            rdchem.HybridizationType.SP3: 6,
            rdchem.HybridizationType.SP3D: 7,
            rdchem.HybridizationType.SP3D2: 8,
            rdchem.HybridizationType.UNSPECIFIED: 9,
        }
        self.bond_type_map = {
            rdchem.BondType.SINGLE: 1,
            rdchem.BondType.DOUBLE: 2,
            rdchem.BondType.TRIPLE: 3,
            rdchem.BondType.AROMATIC: 4,
            rdchem.BondType.UNSPECIFIED: 5,
        }
        self.stereo_map = {
            rdchem.BondStereo.STEREONONE: 1,
            rdchem.BondStereo.STEREOZ: 2,
            rdchem.BondStereo.STEREOE: 3,
            rdchem.BondStereo.STEREOCIS: 4,
            rdchem.BondStereo.STEREOTRANS: 5,
            rdchem.BondStereo.STEREOATROPCCW: 6,
            rdchem.BondStereo.STEREOATROPCW: 7,
            rdchem.BondStereo.STEREOANY: 8,
        }

    def smiles2graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        node_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()  
            hybridization = self.hybridization_map.get(atom.GetHybridization(), 9)
            aromatic = 2 if atom.GetIsAromatic() else 1 
            in_ring = 2 if atom.IsInRing() else 1  
            degree = atom.GetTotalDegree()
            charge = atom.GetFormalCharge() + 2
            num_h = atom.GetTotalNumHs()
            node_features.append([atomic_num, hybridization, aromatic, in_ring, degree, charge, num_h])
        edge_index = [[], []]
        edge_attr = []
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            edge_index[0].append(start_idx)
            edge_index[1].append(end_idx)
            edge_index[0].append(end_idx)
            edge_index[1].append(start_idx)
            bond_type = self.bond_type_map.get(bond.GetBondType(), 5)
            stereo = self.stereo_map.get(bond.GetStereo(), 8)
            conjugated = 2 if bond.GetIsConjugated() else 1
            edge_attr.append([bond_type, stereo, conjugated])
            edge_attr.append([bond_type, stereo, conjugated])
        return node_features, edge_index, edge_attr
