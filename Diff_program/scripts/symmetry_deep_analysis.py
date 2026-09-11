"""Deep symmetry analysis: run ILExplainer multiple times, compare with VNEO."""
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

idx = 251
config_path = './configs/gin.yaml'
checkpoint_path = './results/checkpoints/GIN_D+/model.pth'
data_path = './data/processed/fragments/data_frag_D+.npy'
seed = 42
epochs = 100
num_runs = 5

torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
model = GIN(**config['model']['params']).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

data = np.load(data_path, allow_pickle=True)
graph_data, fragments, temperature, num_bonds = data[idx]
node_features = np.array(graph_data[0])
edge_index = np.array(graph_data[1])
edge_attr = np.array(graph_data[2])
num_atoms = len(node_features)

# ============ VNEO ============
explainer_occlusion = OcclusionExplainer(model, device=device)
atom_imp_vneo = explainer_occlusion.explain_atoms(
    node_features, edge_index, edge_attr, temperature)

# ============ ILExplainer multiple independent runs ============
graph = Data(
    x=torch.tensor(node_features, dtype=torch.float32),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
)
graph = add_global_node(graph)
cond = torch.tensor([temperature], dtype=torch.float32).to(device)

n_trials = 10
all_f_importances = []  # shape: (n_trials, 4)
all_atom_imps = []

for trial in range(n_trials):
    trial_seed = seed + trial * 100
    explainer_gnn = GNNExplainer(model, epochs=epochs, lr=0.01, num_runs=num_runs, seed=trial_seed)
    graph_batch = Batch.from_data_list([graph]).to(device)
    edge_mask = explainer_gnn.explain_graph(graph_batch, cond)
    
    virtual_edge_mask = edge_mask[num_bonds * 2:]
    atom_imp = np.zeros(num_atoms)
    for i in range(num_atoms):
        atom_imp[i] = (virtual_edge_mask[2*i].item() + virtual_edge_mask[2*i+1].item()) / 2
    atom_imp_centered = atom_imp - atom_imp.mean()
    all_atom_imps.append(atom_imp_centered)
    
    # F atom importances (atoms 10, 12, 13, 14)
    f_imp = [atom_imp_centered[i] for i in [10, 12, 13, 14]]
    all_f_importances.append(f_imp)

all_f_importances = np.array(all_f_importances)  # (n_trials, 4)
all_atom_imps = np.array(all_atom_imps)  # (n_trials, num_atoms)

# ============ Print results ============
print("=" * 70)
print("VNEO Atom Importance for [BMIM][BF4] (D+)")
print("=" * 70)
print(f"\nFull atom importance:")
labels = ['C(CH3)', 'C(CH2)', 'C(CH2)', 'C(CH2)', 'N(ring)', 'C(ring)', 'C(ring)', 'N(ring)', 'C(ring)', 'C(CH3)', 'F', 'B', 'F', 'F', 'F']
for i in range(num_atoms):
    print(f"  Atom {i:2d} ({labels[i]:8s}): {atom_imp_vneo[i]:+.6f}")

print(f"\nBF4- detail:")
print(f"  B  (atom 11): {atom_imp_vneo[11]:+.6f}")
print(f"  F1 (atom 10): {atom_imp_vneo[10]:+.6f}")
print(f"  F2 (atom 12): {atom_imp_vneo[12]:+.6f}")
print(f"  F3 (atom 13): {atom_imp_vneo[13]:+.6f}")
print(f"  F4 (atom 14): {atom_imp_vneo[14]:+.6f}")
print(f"  F std: {np.std([atom_imp_vneo[i] for i in [10,12,13,14]]):.8f}")

print(f"\n{'=' * 70}")
print(f"ILExplainer Atom Importance (10 trials with different seeds)")
print(f"{'=' * 70}")

print(f"\nBF4- F atom importance across trials:")
print(f"{'Trial':>5s} | {'F1(10)':>10s} | {'F2(12)':>10s} | {'F3(13)':>10s} | {'F4(14)':>10s} | {'std':>10s} | {'range':>10s}")
print("-" * 80)
for t in range(n_trials):
    f_vals = all_f_importances[t]
    print(f"{t:5d} | {f_vals[0]:+10.6f} | {f_vals[1]:+10.6f} | {f_vals[2]:+10.6f} | {f_vals[3]:+10.6f} | {np.std(f_vals):10.6f} | {np.ptp(f_vals):10.6f}")

print(f"\nSummary statistics across trials:")
print(f"  Mean F std:  {np.mean([np.std(all_f_importances[t]) for t in range(n_trials)]):.6f}")
print(f"  Max F std:   {np.max([np.std(all_f_importances[t]) for t in range(n_trials)]):.6f}")
print(f"  Mean F range: {np.mean([np.ptp(all_f_importances[t]) for t in range(n_trials)]):.6f}")
print(f"  Max F range:  {np.max([np.ptp(all_f_importances[t]) for t in range(n_trials)]):.6f}")

# Per-atom mean and std across trials
print(f"\nPer-F-atom importance (mean +/- std across trials):")
for fi, ai in enumerate([10, 12, 13, 14]):
    vals = all_f_importances[:, fi]
    print(f"  F{fi+1} (atom {ai}): {np.mean(vals):+.6f} +/- {np.std(vals):.6f}")

# Compare with VNEO
print(f"\n{'=' * 70}")
print(f"Symmetry Preservation Comparison")
print(f"{'=' * 70}")
vneo_f_std = np.std([atom_imp_vneo[i] for i in [10,12,13,14]])
il_f_stds = [np.std(all_f_importances[t]) for t in range(n_trials)]
print(f"  VNEO F-atom std:          {vneo_f_std:.8f} (exact symmetry)")
print(f"  ILExplainer F-atom std:   {np.mean(il_f_stds):.6f} +/- {np.std(il_f_stds):.6f} (symmetry broken)")
print(f"  Ratio (IL/VNEO):          {'inf' if vneo_f_std == 0 else np.mean(il_f_stds)/vneo_f_std:.1f}x")

# Also check B atom consistency
il_b_vals = [all_atom_imps[t][11] for t in range(n_trials)]
print(f"\n  B atom importance:")
print(f"    VNEO:       {atom_imp_vneo[11]:+.6f} (deterministic)")
print(f"    ILExplainer: {np.mean(il_b_vals):+.6f} +/- {np.std(il_b_vals):.6f} (varies across trials)")
