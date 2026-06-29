import numpy as np
import pandas as pd

def load_feature_vectors(feature_path, data_path):
    """Load feature vectors from processed CSV and merge with raw data.

    Args:
        feature_path: Path to processed feature CSV (IL_smiles as first column)
        data_path: Path to raw data CSV (whole_D+.csv or whole_D-.csv)

    Returns:
        features: numpy array with features + T column
        labels: numpy array
        feature_names: list of feature column names + 'T'
    """
    feature_df = pd.read_csv(feature_path, index_col=0)
    data_df = pd.read_csv(data_path)

    feature_names = list(feature_df.columns) + ['T']

    t_values = data_df['T'].values
    t_mean, t_std = t_values.mean(), t_values.std()

    features = []
    labels = []
    for _, row in data_df.iterrows():
        IL_smiles = row['IL_smiles']
        if IL_smiles in feature_df.index:
            feat = feature_df.loc[IL_smiles].tolist()
            feat.append((row['T'] - t_mean) / t_std)
            features.append(feat)
            labels.append(row.iloc[5])

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32), feature_names
