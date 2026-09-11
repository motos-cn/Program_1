import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from rdkit.Chem import rdFingerprintGenerator
import rdkit.Chem as Chem

def il_smiles_to_concat_fingerprint(il_smiles, radius=3, fp_size=1024):
    """Compute Morgan fingerprint for cation and anion separately, then concatenate."""
    parts = il_smiles.split('.')
    cation_smiles, anion_smiles = parts[0], parts[1]
    
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    
    cat_mol = Chem.MolFromSmiles(cation_smiles)
    cat_mol = Chem.AddHs(cat_mol)
    cat_fp = list(fpg.GetFingerprint(cat_mol))
    
    ani_mol = Chem.MolFromSmiles(anion_smiles)
    ani_mol = Chem.AddHs(ani_mol)
    ani_fp = list(fpg.GetFingerprint(ani_mol))
    
    return cat_fp + ani_fp

def batch_il_smiles_to_concat_fingerprints(smiles_list, radius=3, fp_size=1024):
    fingerprints = []
    for smiles in smiles_list:
        fp = il_smiles_to_concat_fingerprint(smiles, radius=radius, fp_size=fp_size)
        fingerprints.append(fp)
    return fingerprints

def main():
    dataset = pd.read_csv('./data/raw/whole.csv')
    dataset_new = dataset.drop_duplicates(subset=['IL_smiles'])

    fingerprints = batch_il_smiles_to_concat_fingerprints(dataset_new['IL_smiles'])
    fp_columns = [f'fp_{i}' for i in range(len(fingerprints[0]))]
    df = pd.DataFrame(fingerprints, columns=fp_columns, index=dataset_new['IL_smiles'])

    output_dir = './data/processed/fingerprint'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'fp_lambda.csv'))

if __name__ == '__main__':
    main()
