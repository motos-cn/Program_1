"""Compare atom importance for [BMIM][BF4] between ILExplainer and VNEO."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from torch_geometric.data import Data, Batch

from src.models.gnn.gin import GIN
from src.evaluation.explainer import GNNExplainer
from src.evaluation.occlusion_explainer import OcclusionExplainer
from src.datasets.graph_dataset import add_global_node

# ============ Config ============
idx = 251  # [BMIM][BF4] sample index
config_path = './configs/gin.yaml'
checkpoint_path = './results/checkpoints/GIN_D+/model.pth'
data_path = './data/processed/fragments/data_frag_D+.npy'
seed = 42
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
graph_data, fragments, temperature, num_bonds = data[idx]
node_features, edge_index, edge_attr = graph_data
# Convert to numpy if needed
node_features = np.array(node_features)
edge_index = np.array(edge_index)
edge_attr = np.array(edge_attr)
num_atoms = len(node_features)

print(f"Sample {idx}: {num_atoms} atoms, T={temperature}")
print(f"Node features shape: {node_features.shape}")
print(f"Fragments: {fragments}")

# Identify BF4- atoms
# BF4- SMILES: F[B-](F)(F)F -> atoms: F, B-, F, F, F (5 atoms)
# In [BMIM][BF4], the BF4- part should be at the end of the atom list
# Let's find the B atom and F atoms by checking node features
print(f"\n=== Node features ===")
for i in range(num_atoms):
    print(f"  Atom {i}: {node_features[i]}")

print(f"\n=== Edge index ===")
for j in range(edge_index.shape[1]):
    print(f"  Edge {j}: {edge_index[0][j]} -> {edge_index[1][j]}, attr={edge_attr[j]}")

# ============ VNEO (Occlusion) ============
print(f"\n{'='*60}")
print("VNEO (Occlusion) Atom Importance")
print(f"{'='*60}")
explainer_occlusion = OcclusionExplainer(model, device=device)
atom_imp_occlusion = explainer_occlusion.explain_atoms(
    node_features, edge_index, edge_attr, temperature)
print(f"Atom importance (occlusion): {atom_imp_occlusion}")
for i, v in enumerate(atom_imp_occlusion):
    print(f"  Atom {i}: {v:.6f}")

# ============ ILExplainer (GNNExplainer) ============
print(f"\n{'='*60}")
print("ILExplainer (GNNExplainer) Atom Importance")
print(f"{'='*60}")
graph = Data(
    x=torch.tensor(node_features, dtype=torch.float32),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
)
graph = add_global_node(graph)
graph_batch = Batch.from_data_list([graph]).to(device)
cond = torch.tensor([temperature], dtype=torch.float32).to(device)

explainer_gnn = GNNExplainer(model, epochs=epochs, lr=0.01, num_runs=num_runs, seed=seed)
edge_mask = explainer_gnn.explain_graph(graph_batch, cond)

virtual_edge_mask = edge_mask[num_bonds * 2:]
atom_imp_gnn = np.zeros(num_atoms)
for i in range(num_atoms):
    atom_imp_gnn[i] = (virtual_edge_mask[2 * i].item() + virtual_edge_mask[2 * i + 1].item()) / 2
atom_imp_gnn_centered = atom_imp_gnn - atom_imp_gnn.mean()

print(f"Virtual edge mask (ILExplainer): {virtual_edge_mask.cpu().numpy()}")
print(f"Atom importance (ILExplainer, raw): {atom_imp_gnn}")
print(f"Atom importance (ILExplainer, centered): {atom_imp_gnn_centered}")
for i, v in enumerate(atom_imp_gnn_centered):
    print(f"  Atom {i}: {v:.6f}")

# ============ Compare symmetry ============
print(f"\n{'='*60}")
print("Symmetry Analysis for BF4- F atoms")
print(f"{'='*60}")
# BF4- atoms identified from node features: 
# Atom 10, 12, 13, 14 are F (atomic number 9), Atom 11 is B (atomic number 5)
f_atom_indices = [10, 12, 13, 14]
b_atom_index = 11

print(f"Inferred B atom index: {b_atom_index}")
print(f"Inferred F atom indices: {f_atom_indices}")

print(f"\nVNEO F atom importance values:")
f_values_occlusion = [atom_imp_occlusion[i] for i in f_atom_indices]
print(f"  {f_values_occlusion}")
print(f"  std: {np.std(f_values_occlusion):.8f}")
print(f"  range: {max(f_values_occlusion) - min(f_values_occlusion):.8f}")

print(f"\nILExplainer F atom importance values (centered):")
f_values_gnn = [atom_imp_gnn_centered[i] for i in f_atom_indices]
print(f"  {f_values_gnn}")
print(f"  std: {np.std(f_values_gnn):.8f}")
print(f"  range: {max(f_values_gnn) - min(f_values_gnn):.8f}")

# Also check B atom
print(f"\nB atom importance:")
print(f"  VNEO: {atom_imp_occlusion[b_atom_index]:.6f}")
print(f"  ILExplainer: {atom_imp_gnn_centered[b_atom_index]:.6f}")
