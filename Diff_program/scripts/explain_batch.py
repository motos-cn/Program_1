import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from src.models.gnn.gin import GIN
from src.evaluation.explainer import GNNExplainer
from src.evaluation.occlusion_explainer import OcclusionExplainer
from src.evaluation.visualization import plot_fragment_scores
from src.data_processing.fragment_processor import FragmentProcessor
from src.datasets.graph_dataset import add_global_node

# ============ Config ============
method = 'occlusion'       # 'GNNExplainer' or 'occlusion'
config_path = '../configs/gin.yaml'
checkpoint_path = '../results/checkpoints/GIN_D+/model.pth'
data_path = '../data/processed/fragments/data_frag_D+.npy'
output_dir = '../results/fragment_explain/' + method
seed = 42
# Reliability assessment
d_threshold = 0.8          # Cohen's d threshold (0.5=medium, 0.8=large)
ci_level = 95              # Bootstrap confidence interval level (90 or 95)
# GNNExplainer only
epochs = 100
num_runs = 5
# ================================

torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
model = GIN(**config['model']['params']).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Load data
data = np.load(data_path, allow_pickle=True)
print(f"Loaded {len(data)} samples from {data_path}")

fragment_names = FragmentProcessor().fragment_names
Frag_importance = {name: [] for name in fragment_names}

# Explain each sample
if method == 'GNNExplainer':
    explainer = GNNExplainer(model, epochs=epochs, lr=0.01, num_runs=num_runs, seed=seed)

    for i in tqdm(range(len(data)), desc='GNN explaining'):
        graph_data, fragments, temperature, num_bonds = data[i]
        node_features, edge_index, edge_attr = graph_data
        num_atoms = len(node_features)

        graph = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )
        graph = add_global_node(graph)
        graph = Batch.from_data_list([graph]).to(device)
        cond = torch.tensor([temperature], dtype=torch.float32).to(device)

        try:
            edge_mask = explainer.explain_graph(graph, cond)
            real_edge_mask = edge_mask[:num_bonds * 2]
            real_edge_index = graph.edge_index[:, :num_bonds * 2]

            atom_importance = torch.zeros(num_atoms)
            for atom_idx in range(num_atoms):
                connected = (real_edge_index[0] == atom_idx) | (real_edge_index[1] == atom_idx)
                if connected.any():
                    atom_importance[atom_idx] = real_edge_mask[connected].mean()

            mean_score = atom_importance.mean().item()
            for frag_name in fragment_names:
                if frag_name not in fragments or len(fragments[frag_name]) == 0:
                    continue
                for piece in fragments[frag_name]:
                    piece_scores = [atom_importance[idx].item() for idx in piece]
                    if piece_scores:
                        Frag_importance[frag_name].append(np.mean(piece_scores) - mean_score)
        except Exception as e:
            print(f"Error row {i}: {e}")

else:  # occlusion
    explainer = OcclusionExplainer(model, device=device)

    for i in tqdm(range(len(data)), desc='Occlusion explaining'):
        graph_data, fragments, temperature, num_bonds = data[i]
        node_features, edge_index, edge_attr = graph_data

        try:
            fragment_scores = explainer.explain_graph(
                node_features, edge_index, edge_attr, temperature, fragments)
            for frag_name in fragment_names:
                if frag_name in fragment_scores and len(fragment_scores[frag_name]) > 0:
                    Frag_importance[frag_name].extend(fragment_scores[frag_name])
        except Exception as e:
            print(f"Error row {i}: {e}")

# Compute statistics (Cohen's d + Bootstrap CI)
def compute_reliability(scores, n_bootstrap=1000, ci_level=90, d_threshold=0.5):
    """Compute effect size and confidence interval for fragment importance."""
    n = len(scores)
    mean_val = float(np.mean(scores)) if n > 0 else 0.0
    std_val = float(np.std(scores)) if n > 0 else 0.0

    if n < 2:
        return {'mean': mean_val, 'cohen_d': 0.0, 'ci_low': 0.0,
                'ci_high': 0.0, 'reliable': False, 'n': n}

    cohen_d = mean_val / (std_val + 1e-10)

    # Bootstrap confidence interval
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha = (100 - ci_level) / 2
    ci_low = float(np.percentile(boot_means, alpha))
    ci_high = float(np.percentile(boot_means, 100 - alpha))

    # Reliable: effect is substantial and direction is certain
    ci_not_cross_zero = (ci_low > 0) or (ci_high < 0)
    reliable = abs(cohen_d) >= d_threshold and ci_not_cross_zero

    return {'mean': mean_val, 'cohen_d': cohen_d, 'ci_low': ci_low,
            'ci_high': ci_high, 'reliable': reliable, 'n': n}


stats = {}
for frag_name in fragment_names:
    scores = Frag_importance[frag_name]
    stats[frag_name] = compute_reliability(scores, ci_level=ci_level,
                                            d_threshold=d_threshold)

# Save results
save_dir = output_dir
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, 'frag_importance_raw.npy'),
        np.array(Frag_importance, dtype=object), allow_pickle=True)
np.save(os.path.join(save_dir, 'frag_importance_stats.npy'),
        np.array(stats, dtype=object), allow_pickle=True)

# Print summary
print(f"\nFragment Importance ({method}, mean delta, d>={d_threshold}, {ci_level}% CI):")
print("{:15s} | {:>10s} | {:>8s} | {:>18s} | {:>5s} | {:>8s}".format(
    'Fragment', 'Mean', 'Cohen_d', f'{ci_level}% CI', 'N', 'Reliable'))
print('-' * 80)
for frag_name in sorted(stats, key=lambda x: -abs(stats[x]['mean'])):
    s = stats[frag_name]
    ci_str = "[{:+.4f}, {:+.4f}]".format(s['ci_low'], s['ci_high'])
    reliable = 'Yes' if s['reliable'] else 'No'
    print("{:15s} | {:>+10.6f} | {:>+8.2f} | {:>18s} | {:5d} | {:>8s}".format(
        frag_name, s['mean'], s['cohen_d'], ci_str, s['n'], reliable))

# Visualization
plot_fragment_scores(stats, os.path.join(save_dir, 'fragment_score.png'),
                     d_threshold=d_threshold, ci_level=ci_level)
print(f"\nSaved results to {save_dir}/")
