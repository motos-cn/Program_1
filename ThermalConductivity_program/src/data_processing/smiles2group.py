import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

def main():
    groups_df = pd.read_csv('../../data/processed/groups/groups.csv')
    group_cols = [col for col in groups_df.columns if col not in ['Ion', 'SMILES']]
    smiles_to_groups = groups_df.set_index('SMILES')[group_cols]

    dataset = pd.read_csv('../../data/raw/whole.csv')
    dataset_new = dataset.drop_duplicates(subset=['IL_smiles'])

    features = []
    for IL_smiles in dataset_new['IL_smiles']:
        parts = IL_smiles.split('.')
        row = [0] * len(group_cols)
        for smi in parts:
            if smi in smiles_to_groups.index:
                row = [a + b for a, b in zip(row, smiles_to_groups.loc[smi].tolist())]
        features.append(row)

    df = pd.DataFrame(features, columns=group_cols, index=dataset_new['IL_smiles'])

    output_dir = '../../data/processed/groups'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'gc_lambda.csv'))

if __name__ == '__main__':
    main()
