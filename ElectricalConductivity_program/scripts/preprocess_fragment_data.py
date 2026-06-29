import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.data_processing.fragment_processor import FragmentProcessor


def process_frag_data(input_csv, output_npy, label_col=None, label_npy=None):
    processor = FragmentProcessor()
    df = pd.read_csv(input_csv)

    t_mean, t_std = df['T'].mean(), df['T'].std(ddof=0)

    data_list = []
    for i, row in df.iterrows():
        if i % 200 == 0:
            print(f"Processing row {i}/{len(df)}")
        try:
            result = processor.process_ionic_liquid(row['IL_smiles'], row['T'], t_mean, t_std)
            data_list.append(result)
        except Exception as e:
            print(f"Error row {i} [{row['IL_smiles']}]: {e}")

    np.save(output_npy, np.array(data_list, dtype=object), allow_pickle=True)
    print(f"Processed {len(data_list)} ILs -> {output_npy}")
    print(f"T Z-score: mean={t_mean:.2f}, std={t_std:.2f}")

    if label_col is not None:
        np.save(label_npy, df[label_col].values, allow_pickle=True)
        print(f"Labels saved -> {label_npy}")


if __name__ == '__main__':
    process_frag_data('../data/raw/whole.csv', '../data/processed/fragments/data_frag_sigma.npy',
                      'sigma', '../data/processed/fragments/label_frag_sigma.npy')
