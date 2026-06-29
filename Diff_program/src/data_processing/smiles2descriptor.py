import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

def generate_descriptors(smiles_list):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(x[0] for x in Descriptors._descList)
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors, desc_names

def remove_correlated_features(descriptors, threshold=0.9):
    correlated_matrix = descriptors.corr().abs()
    upper_triangle = correlated_matrix.where(
        np.triu(np.ones(correlated_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] >= threshold)
    ]
    return descriptors.drop(columns=to_drop)

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]].copy()

def fill_missing_with_median(df):
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df

def main():
    dataset = pd.read_csv('../../data/raw/whole_D-.csv')
    dataset_new = dataset.drop_duplicates(subset=['IL_smiles'])
    
    Mol_descriptors, desc_names = generate_descriptors(dataset_new['IL_smiles'])
    df = pd.DataFrame(Mol_descriptors, columns=desc_names)
    df = remove_correlated_features(df)
    X = remove_low_variance(df)
    X = fill_missing_with_median(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df = pd.DataFrame(X_scaled, columns=X.columns, index=dataset_new['IL_smiles'])
    
    output_dir = '../../data/processed/descriptor'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'descriptor_D-.csv'))
    
if __name__ == '__main__':
    main()
