from collections import defaultdict
from sklearn.model_selection import train_test_split
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def random_split_data(features, labels, test_size=0.2, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

# Generate scaffolds for a list of SMILES strings
def generate_scaffolds(smiles_list, log_every_n=1000):
    scaffold_map = defaultdict(list)
    total_count = len(smiles_list)
    print("\nAbout to generate scaffolds")
    for idx, smiles in enumerate(smiles_list):
        if idx % log_every_n == 0:
            print(f"Generating scaffold {idx}/{total_count}")
        scaffold_smiles = MurckoScaffoldSmiles(smiles)
        scaffold_map[scaffold_smiles].append(idx)
    sorted_scaffold_items = sorted(scaffold_map.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    sorted_groups = []
    for scaffold_smiles, group in sorted_scaffold_items:
        sorted_groups.append(group)
    return sorted_groups

# Split a dataset by scaffold
def scaffold_split(smiles_list, valid_size, test_size):
    scaffold_groups = generate_scaffolds(smiles_list)
    n_total = len(smiles_list)
    train_limit = int((1.0 - valid_size - test_size) * n_total)
    valid_limit = int((1.0 - test_size) * n_total)
    train_idx, valid_idx, test_idx = [], [], []
    print("\nAbout to split dataset by scaffold")
    for group in scaffold_groups:
        if len(train_idx) + len(group) <= train_limit:
            train_idx.extend(group)
        elif len(train_idx) + len(valid_idx) + len(group) <= valid_limit:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)
    return train_idx, valid_idx, test_idx
