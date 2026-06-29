import torch
import numpy as np
from torch_geometric.data import Data, Batch


class OcclusionExplainer:
    """Fragment-level occlusion explainer for GNN regression models."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def explain_graph(self, node_features, edge_index, edge_attr,
                      temperature, pressure, fragments):
        """Compute fragment-level occlusion importance by cutting fragment-global edges.

        Instead of zeroing node features, this method removes the bidirectional
        edges between target fragment atoms and the virtual global node. This
        preserves atom-atom message passing (inductive/conjugation effects) while
        blocking the fragment's contribution to the global readout.
        """
        self.model.eval()

        # Flatten all fragment instances into a single list
        occlusion_instances = []
        for frag_name, pieces in fragments.items():
            for piece in pieces:
                if len(piece) > 0:
                    occlusion_instances.append((frag_name, tuple(piece)))

        # Build original graph with global node
        original_graph = self._build_graph(node_features, edge_index, edge_attr)
        global_idx = original_graph.x.shape[0] - 1

        # Build [original, masked_1, masked_2, ...] for batched inference
        all_graphs = [original_graph]
        for _, atom_indices in occlusion_instances:
            masked_graph = self._cut_global_edges(original_graph, atom_indices, global_idx)
            all_graphs.append(masked_graph)

        # Single batched forward pass
        batched = Batch.from_data_list(all_graphs).to(self.device)
        # Condition: [T, P] for each graph
        cond = torch.tensor([[temperature, pressure]] * len(all_graphs),
                            dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(batched, cond).flatten().cpu().numpy()

        original_pred = preds[0]
        deltas = original_pred - preds[1:]  # positive = fragment raises prediction

        # Aggregate per-fragment scores
        fragment_scores = {name: [] for name in fragments}
        for i, (frag_name, _) in enumerate(occlusion_instances):
            fragment_scores[frag_name].append(float(deltas[i]))

        return fragment_scores

    def explain_atoms(self, node_features, edge_index, edge_attr, temperature, pressure):
        """Compute per-atom importance by cutting each atom's edges to the global node."""
        self.model.eval()
        num_atoms = len(node_features)

        # Build original graph with global node
        original_graph = self._build_graph(node_features, edge_index, edge_attr)
        global_idx = original_graph.x.shape[0] - 1

        # Build [original, atom0_cut, atom1_cut, ...]
        all_graphs = [original_graph]
        for atom_idx in range(num_atoms):
            cut_graph = self._cut_global_edges(original_graph, (atom_idx,), global_idx)
            all_graphs.append(cut_graph)

        # Single batched forward pass
        batched = Batch.from_data_list(all_graphs).to(self.device)
        cond = torch.tensor([[temperature, pressure]] * len(all_graphs),
                            dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(batched, cond).flatten().cpu().numpy()

        original_pred = preds[0]
        atom_importance = original_pred - preds[1:]  # positive = atom raises prediction
        return atom_importance

    def _build_graph(self, node_features, edge_index, edge_attr):
        """Build a PyG Data object with a virtual global node appended."""
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return self._add_global_node(graph)

    @staticmethod
    def _cut_global_edges(graph, atom_indices, global_idx):
        """Remove bidirectional edges between target atoms and the global node."""
        atom_set = set(atom_indices)
        keep = []
        for i in range(graph.edge_index.shape[1]):
            src = graph.edge_index[0, i].item()
            dst = graph.edge_index[1, i].item()
            # Remove edge if one end is global and the other is in the fragment
            if (src == global_idx and dst in atom_set) or \
                (dst == global_idx and src in atom_set):
                keep.append(False)
            else:
                keep.append(True)

        keep = torch.tensor(keep, dtype=torch.bool)
        return Data(
            x=graph.x.clone(),
            edge_index=graph.edge_index[:, keep],
            edge_attr=graph.edge_attr[keep],
        )

    @staticmethod
    def _add_global_node(graph):
        """Append a zero-feature virtual node connected bidirectionally to all real nodes."""
        global_node = torch.zeros(7, dtype=torch.float32).reshape(1, -1)
        x = torch.cat([graph.x, global_node], dim=0)

        num_nodes = x.shape[0] - 1
        g_idx = x.shape[0] - 1
        srcs, dsts, edge_attr = [], [], []
        for i in range(num_nodes):
            srcs.extend([i, g_idx])
            dsts.extend([g_idx, i])
            edge_attr.extend([torch.zeros(3, dtype=torch.float32),
                        torch.zeros(3, dtype=torch.float32)])

        new_edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        new_edge_attr = torch.stack(edge_attr)
        edge_index = torch.cat([graph.edge_index, new_edge_index], dim=1)
        edge_attr = torch.cat([graph.edge_attr, new_edge_attr], dim=0)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
