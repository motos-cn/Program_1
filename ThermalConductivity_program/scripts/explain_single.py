import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from torch_geometric.data import Data, Batch

from src.models.gnn.gin import GIN
from src.evaluation.explainer import GNNExplainer
from src.evaluation.occlusion_explainer import OcclusionExplainer
from src.evaluation.visualization import plot_atom_heatmap
from src.datasets.graph_dataset import add_global_node

# ============ Config ============
method = 'occlusion'       # 'GNNExplainer' or 'occlusion'
idx = 440                  # sample index
config_path = '../configs/gin.yaml'
checkpoint_path = '../results/checkpoints/GIN_lambda/model.pth'
data_path = '../data/processed/fragments/data_frag_lambda.npy'
output_dir = '../results/fragment_explain/' + method
seed = 42
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
graph_data, fragments, temperature, pressure, num_bonds = data[idx]
node_features, edge_index, edge_attr = graph_data
num_atoms = len(node_features)

print(f"Sample {idx}: {num_atoms} atoms, method={method}")

# Explain
if method == 'GNNExplainer':
    graph = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    graph = add_global_node(graph)
    graph = Batch.from_data_list([graph]).to(device)
    cond = torch.tensor([[temperature, pressure]], dtype=torch.float32).to(device)

    explainer = GNNExplainer(model, epochs=epochs, lr=0.01, num_runs=num_runs, seed=seed)
    edge_mask = explainer.explain_graph(graph, cond)

    virtual_edge_mask = edge_mask[num_bonds * 2:]
    atom_importance = np.zeros(num_atoms)
    for i in range(num_atoms):
        atom_importance[i] = (virtual_edge_mask[2 * i] + virtual_edge_mask[2 * i + 1]).item() / 2
else:
    explainer = OcclusionExplainer(model, device=device)
    atom_importance = explainer.explain_atoms(
        node_features, edge_index, edge_attr, temperature, pressure)

# Save heatmap
save_dir = output_dir
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'single_IL_importance_idx{idx}.png')
plot_atom_heatmap(atom_importance, edge_index, num_atoms, save_path)
print(f"Saved to {save_path}")
