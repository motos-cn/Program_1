import numpy as np
import torch
from torch_geometric.data import Data, Batch

config = {
    'add_global_node': True,
    'bi_direction': True,
}

def add_global_node(graph):
    global_node = torch.zeros(7, dtype=torch.float32).reshape(1, -1)
    x = torch.cat([graph.x, global_node], dim=0)

    num_nodes = x.shape[0] - 1
    global_node_idx = x.shape[0] - 1
    start_nodes = []
    end_nodes = []
    edge_attr = []
    for i in range(num_nodes):
        start_nodes.append(i)
        end_nodes.append(global_node_idx)
        edge_attr.append(torch.zeros(3, dtype=torch.float32))
        if config['bi_direction']:
            start_nodes.append(global_node_idx)
            end_nodes.append(i)
            edge_attr.append(torch.zeros(3, dtype=torch.float32))
    start_nodes = torch.tensor(start_nodes, dtype=torch.long).reshape(1, -1)
    end_nodes = torch.tensor(end_nodes, dtype=torch.long).reshape(1, -1)
    new_edges = torch.cat([start_nodes, end_nodes], dim=0)
    new_edge_attr = torch.stack(edge_attr)
    edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
    edge_attr = torch.cat([graph.edge_attr, new_edge_attr], dim=0)

    new_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return new_graph


class ILDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path):
        super(ILDataset, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.data = np.load(self.data_path, allow_pickle=True)
        self.label = np.load(self.label_path, allow_pickle=True)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        IL_data = self.data[idx][0]
        T = self.data[idx][1]
        label = self.label[idx]
        IL_graph = self._mol_to_graph(IL_data)
        if config['add_global_node']:
            IL_graph = add_global_node(IL_graph)
        condition = torch.tensor(T, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return IL_graph, condition, label

    def _mol_to_graph(self, mol_data):
        node_features = torch.tensor(mol_data[0], dtype=torch.float32)
        edge_index = torch.tensor(mol_data[1], dtype=torch.long)
        edge_attr = torch.tensor(mol_data[2], dtype=torch.float32)
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return graph

    @staticmethod
    def collate_fn(batch):
        graphs, conds, labels = zip(*batch)
        batch_graph = Batch.from_data_list(graphs)
        batch_cond = torch.stack(conds)
        batch_label = torch.stack(labels)
        return batch_graph, batch_cond, batch_label
