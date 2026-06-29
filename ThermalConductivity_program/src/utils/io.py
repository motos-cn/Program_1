import numpy as np
import pandas as pd

def load_feature_vectors(feature_path, data_path):
    """Load feature vectors from processed CSV and merge with raw data.

    Args:
        feature_path: Path to processed feature CSV (IL_smiles as first column)
        data_path: Path to raw data CSV (whole.csv)

    Returns:
        features: numpy array with features + T + P columns
        labels: numpy array
        feature_names: list of feature column names + 'T' + 'P'
    """
    feature_df = pd.read_csv(feature_path, index_col=0)
    data_df = pd.read_csv(data_path)

    feature_names = list(feature_df.columns) + ['T', 'P']

    t_values = data_df['T'].values
    t_mean, t_std = t_values.mean(), t_values.std()

    # Pressure: log-transform then standardize
    p_values = data_df['P'].values
    log_p = np.log(p_values)
    p_mean, p_std = log_p.mean(), log_p.std()

    features = []
    labels = []
    for _, row in data_df.iterrows():
        IL_smiles = row['IL_smiles']
        if IL_smiles in feature_df.index:
            feat = feature_df.loc[IL_smiles].tolist()
            feat.append((row['T'] - t_mean) / t_std)
            feat.append((np.log(row['P']) - p_mean) / p_std)
            features.append(feat)
            labels.append(row['lambda'])

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32), feature_names
