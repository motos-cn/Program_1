import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from rdkit.Chem import rdFingerprintGenerator
import rdkit.Chem as Chem

def smiles_to_fingerprint(smiles, radius=3, fp_size=2048):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fp = fpg.GetFingerprint(mol)
    return list(fp)

def batch_smiles_to_fingerprints(smiles_list, radius=3, fp_size=2048):
    fingerprints = []
    for smiles in smiles_list:
        fp = smiles_to_fingerprint(smiles, radius=radius, fp_size=fp_size)
        fingerprints.append(fp)
    return fingerprints

def main():
    dataset = pd.read_csv('../../data/raw/whole.csv')
    dataset_new = dataset.drop_duplicates(subset=['IL_smiles'])

    fingerprints = batch_smiles_to_fingerprints(dataset_new['IL_smiles'])
    fp_columns = [f'fp_{i}' for i in range(len(fingerprints[0]))]
    df = pd.DataFrame(fingerprints, columns=fp_columns, index=dataset_new['IL_smiles'])

    output_dir = '../../data/processed/fingerprint'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'fp_sigma.csv'))

if __name__ == '__main__':
    main()
