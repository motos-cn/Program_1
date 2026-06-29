import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.data_processing.graph_processor import GraphProcessor


def process_graph_data(input_csv, output_npy, label_col=None, label_npy=None):
    processor = GraphProcessor()
    df = pd.read_csv(input_csv)

    t_mean, t_std = df['T'].mean(), df['T'].std(ddof=0)

    # Pressure: log-transform then standardize
    log_p = np.log(df['P'].values)
    p_mean, p_std = log_p.mean(), log_p.std(ddof=0)

    data_list = []
    for i, row in df.iterrows():
        if i % 200 == 0:
            print(f"Processing row {i}/{len(df)}")
        try:
            result = processor.process_ionic_liquid(
                row['IL_smiles'], row['T'], row['P'],
                t_mean, t_std, p_mean, p_std
            )
            data_list.append(result)
        except Exception as e:
            print(f"Error row {i} [{row['IL_smiles']}]: {e}")

    np.save(output_npy, np.array(data_list, dtype=object), allow_pickle=True)
    print(f"Processed {len(data_list)} ILs -> {output_npy}")
    print(f"T Z-score: mean={t_mean:.2f}, std={t_std:.2f}")
    print(f"P log Z-score: mean={p_mean:.4f}, std={p_std:.4f}")

    if label_col and label_npy:
        np.save(label_npy, df[label_col].values, allow_pickle=True)
        print(f"Labels saved -> {label_npy}")


if __name__ == '__main__':
    process_graph_data('../data/raw/whole.csv', '../data/processed/graphs/data_lambda.npy',
                       'lambda', '../data/processed/graphs/label_lambda.npy')
