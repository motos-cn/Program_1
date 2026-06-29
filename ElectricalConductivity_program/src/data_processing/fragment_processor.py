import numpy as np
from rdkit import Chem
from .smiles2graph import smilesToGraphConverter


class FragmentProcessor:

    def __init__(self):
        self.fragment_names = [
            # Core cation skeletons
            '[P]+', '[Im13]+', '[N]+', '[Pyr11]+', '[Py1]+', '[Pipz]+', '[ABN]+',
            # Connecting groups (C bonded to core N/P)
            'P-CH2', 'aN-CH3', 'aN-CH2', 'N-CH2', 'N-CH3', 'cN-CH3', 'cN-CH2',
            # Alkyl groups
            'CH3', 'CH2',
            # Functional groups
            'CH2O', 'CH3O', 'N(CH3)2', 'OH', 'CH3COO',
        ]
        self.core_smarts = {
            '[P]+': '[P+]',
            '[Im13]+': 'c1ncc[n+]1',
            '[N]+': '[N+;!R;!a]',
            '[Pyr11]+': '[N+]1(-CCCC1)',
            '[Py1]+': 'c1cc[n+]cc1',
            '[Pipz]+': '[N+]1(-CCN(-CC1))',
            '[ABN]+': '[N+]1(-CC2CCC(CC2)C1)',
        }
        self.core_patterns = {}
        for name, smarts in self.core_smarts.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat:
                self.core_patterns[name] = pat

        self.core_type_map = {
            '[P]+': 'P', '[Im13]+': 'aN', '[N]+': 'N',
            '[Pyr11]+': 'cN', '[Py1]+': 'aN', '[Pipz]+': 'cN', '[ABN]+': 'cN',
        }
        self.graph_converter = smilesToGraphConverter()

    def find_fragments(self, cation_smiles):
        mol = Chem.MolFromSmiles(cation_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {cation_smiles}")

        frag_pos = {name: [] for name in self.fragment_names}
        assigned = set()

        # Step 1: core skeletons
        core_atoms = {}
        for name, pattern in self.core_patterns.items():
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                frag_pos[name] = [tuple(m) for m in matches]
                core_type = self.core_type_map[name]
                for match in matches:
                    assigned.update(match)
                    for atom_idx in match:
                        atom = mol.GetAtomWithIdx(atom_idx)
                        if atom.GetAtomicNum() in (7, 15):  # N或P
                            core_atoms[atom_idx] = core_type

        # Step 2: connecting groups (C directly bonded to core N/P)
        connecting = {'P-CH2': [], 'aN-CH3': [], 'aN-CH2': [],
                      'N-CH2': [], 'N-CH3': [], 'cN-CH3': [], 'cN-CH2': []}
        for core_idx, core_type in core_atoms.items():
            for nbr in mol.GetAtomWithIdx(core_idx).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx in assigned or nbr.GetAtomicNum() != 6:
                    continue
                if nbr.GetIsAromatic():
                    continue
                if core_type == 'cN' and nbr.IsInRing():
                    continue
                is_ch3 = (nbr.GetTotalNumHs() == 3 and nbr.GetDegree() == 1)
                is_ch2 = (nbr.GetTotalNumHs() == 2 and nbr.GetDegree() == 2)
                key = None
                if core_type == 'P' and (is_ch2 or is_ch3):
                    key = 'P-CH2'
                elif core_type == 'aN':
                    key = 'aN-CH3' if is_ch3 else ('aN-CH2' if is_ch2 else None)
                elif core_type == 'N':
                    key = 'N-CH3' if is_ch3 else ('N-CH2' if is_ch2 else None)
                elif core_type == 'cN':
                    key = 'cN-CH3' if is_ch3 else ('cN-CH2' if is_ch2 else None)
                if key:
                    connecting[key].append((nbr_idx,))
                    assigned.add(nbr_idx)
        frag_pos.update(connecting)

        # Step 3: functional groups
        func = {'CH2O': [], 'CH3O': [], 'N(CH3)2': [], 'OH': [], 'CH3COO': []}

        # OH: terminal O-H
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 8 or atom.GetIdx() in assigned:
                continue
            if atom.GetTotalNumHs() == 1 and atom.GetDegree() == 1:
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 6:
                        func['OH'].append((atom.GetIdx(),))
                        assigned.add(atom.GetIdx())
                        break

        # CH3O / CH2O
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 8 or atom.GetIdx() in assigned or atom.GetTotalNumHs() > 0:
                continue
            c_nbrs = [nbr for nbr in atom.GetNeighbors()
                       if nbr.GetAtomicNum() == 6 and nbr.GetIdx() not in assigned]
            for c in c_nbrs:
                c_idx = c.GetIdx()
                if c.GetTotalNumHs() == 3 and c.GetDegree() == 1:
                    func['CH3O'].append((c_idx, atom.GetIdx()))
                    assigned.update([c_idx, atom.GetIdx()])
                    break
                elif c.GetTotalNumHs() == 2 and c.GetDegree() == 2:
                    func['CH2O'].append((c_idx, atom.GetIdx()))
                    assigned.update([c_idx, atom.GetIdx()])
                    break

        # N(CH3)2
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 7 or atom.GetIdx() in assigned or atom.GetFormalCharge() != 0:
                continue
            ch3s = [nbr for nbr in atom.GetNeighbors()
                    if nbr.GetAtomicNum() == 6 and nbr.GetIdx() not in assigned
                    and nbr.GetTotalNumHs() == 3 and nbr.GetDegree() == 1]
            if len(ch3s) == 2:
                group = (atom.GetIdx(), ch3s[0].GetIdx(), ch3s[1].GetIdx())
                func['N(CH3)2'].append(group)
                assigned.update(group)

        # CH3COO
        acetoxy = Chem.MolFromSmarts('CC(=O)O')
        if acetoxy:
            for match in mol.GetSubstructMatches(acetoxy):
                if any(idx in assigned for idx in match):
                    continue
                c1, c2 = mol.GetAtomWithIdx(match[0]), mol.GetAtomWithIdx(match[1])
                if c1.GetTotalNumHs() == 3 and c2.GetDegree() == 3:
                    func['CH3COO'].append(match)
                    assigned.update(match)
        frag_pos.update(func)

        # Step 4: remaining alkyl CH3/CH2
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6 or atom.GetIdx() in assigned or atom.GetIsAromatic():
                continue
            if atom.GetTotalNumHs() == 3 and atom.GetDegree() == 1:
                frag_pos['CH3'].append((atom.GetIdx(),))
            elif atom.GetTotalNumHs() == 2 and atom.GetDegree() == 2:
                frag_pos['CH2'].append((atom.GetIdx(),))

        return frag_pos

    def split_cation_anion(self, il_smiles):
        idx = il_smiles.index('.')
        return il_smiles[:idx], il_smiles[idx+1:]

    def process_ionic_liquid(self, il_smiles, temperature, t_mean=None, t_std=None):
        cation_smiles, _ = self.split_cation_anion(il_smiles)
        graph = self.graph_converter.smiles2graph(il_smiles)
        fragments = self.find_fragments(cation_smiles)
        num_bonds = len(graph[2]) // 2
        if t_mean is not None and t_std is not None and t_std > 0:
            temperature = (temperature - t_mean) / t_std
        return [graph, fragments, temperature, num_bonds]
