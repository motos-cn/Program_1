from math import sqrt

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from inspect import signature

EPS = 1e-15


class GNNExplainer(torch.nn.Module):
    """GNNExplainer-based edge mask explainer for ionic liquid GNN models."""
    coeffs = {
        'edge_size': 0.05,   # L1 regularization on mask magnitude
        'edge_ent': 0.1,     # Entropy regularization for binary-like masks
    }

    def __init__(self, model, epochs=100, lr=0.01, num_runs=5, seed=42, log=True):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.num_runs = num_runs
        self.seed = seed
        self.log = log

    def __set_masks__(self, x, edge_index, init="normal"):
        """Initialize learnable edge mask and inject into MessagePassing layers."""
        (N, F), E = x.size(), edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        self.loop_mask = edge_index[0] != edge_index[1]

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = True
                module._edge_mask = self.edge_mask
                module._loop_mask = self.loop_mask

    def __clear_masks__(self):
        """Remove edge masks from all MessagePassing layers."""
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = False
                module._edge_mask = None
                module._loop_mask = None
        self.edge_mask = None
        self.loop_mask = None

    def __num_hops__(self):
        """Count the number of message-passing hops in the model."""
        num_hops = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                num_hops += 1
        return num_hops

    def __flow__(self):
        """Return the message-passing flow direction."""
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        """Extract the k-hop subgraph around node_idx, or the full graph if None."""
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        if node_idx is not None:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
                num_nodes=num_nodes, flow=self.__flow__())
            x = x[subset]
        else:
            subset = torch.arange(num_nodes)
            row, col = edge_index
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
            edge_mask[:] = True
            mapping = None

        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def __graph_loss__(self, pred, pred_original):
        """Prediction deviation + L1 and entropy regularization on edge mask."""
        loss = (pred - pred_original).pow(2).mean()

        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def explain_graph(self, graph, cond):
        """Explain a graph-level prediction by optimizing edge masks."""
        self.model.eval()
        self.__clear_masks__()

        x, edge_index = graph.x, graph.edge_index

        x, edge_index, _, hard_edge_mask, _ = self.__subgraph__(
            node_idx=None, x=x, edge_index=edge_index)

        with torch.no_grad():
            pred_original = self.model(graph, cond)

        edge_masks = []
        for run in range(self.num_runs):
            torch.manual_seed(self.seed + run)
            self.__set_masks__(x, edge_index)
            self.to(x.device)

            optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

            for epoch in range(1, self.epochs + 1):
                optimizer.zero_grad()
                pred = self.model(graph, cond)
                loss = self.__graph_loss__(pred, pred_original)
                loss.backward()
                optimizer.step()
                scheduler.step()

            edge_masks.append(self.edge_mask.detach().sigmoid())
            self.__clear_masks__()

        edge_mask = torch.stack(edge_masks).mean(dim=0)
        return edge_mask

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        """Explain a node-level prediction by optimizing edge masks."""
        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)

        x, edge_index, mapping, hard_edge_mask, kwargs = self.__subgraph__(
            node_idx, x, edge_index, **kwargs)

        with torch.no_grad():
            log_logits = self.model(x, edge_index)
            probs_Y = torch.softmax(log_logits, 1)
            pred_label = probs_Y.argmax(dim=-1)

        torch.manual_seed(self.seed)
        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.epochs, desc=f'Explain node {node_idx}')

        for _ in range(self.epochs):
            optimizer.zero_grad()
            log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
            pred = torch.softmax(log_logits, 1)
            loss = self.__graph_loss__(pred, pred_label, node_idx=mapping)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return edge_mask

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, **kwargs):
        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx is not None:
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            edge_mask = edge_mask[hard_edge_mask]
            subset = subset.tolist()
            if y is None:
                y = torch.zeros(edge_index.max().item() + 1,
                                device=edge_index.device)
            else:
                y = y[subset].to(torch.float) / y.max().item()
                y = y.tolist()
        else:
            subset = []
            for index, mask in enumerate(edge_mask):
                node_a = edge_index[0, index]
                node_b = edge_index[1, index]
                if node_a not in subset:
                    subset.append(node_a.item())
                if node_b not in subset:
                    subset.append(node_b.item())
            if y is None:
                y = [0.0 for i in range(len(subset))]
            else:
                y = y.tolist()

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=len(y)).to('cpu')
        G = to_networkx(data, edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset)}
        G = nx.relabel_nodes(G, mapping)

        with_labels = kwargs.get('with_labels', True)
        font_size = kwargs.get('font_size', 10)
        node_size = kwargs.get('node_size', 800)
        cmap = kwargs.get('cmap', 'cool')

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(node_size) / 2.0,
                    shrinkB=sqrt(node_size) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        nx.draw_networkx_nodes(G, pos, node_color=y, node_size=node_size, cmap=cmap)
        if with_labels:
            nx.draw_networkx_labels(G, pos, font_size=font_size)

        return ax, G

    def __repr__(self):
        return f'{self.__class__.__name__}()'


if __name__ == '__main__':
    pass
