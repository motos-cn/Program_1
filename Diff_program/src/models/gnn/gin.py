import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# Node features: atomic_num, hybridization, is_aromatic, is_in_ring, total_degree, charge, num_H
# 0 reserved for virtual node, real categories start from 1, except num_H
num_atom_type = 119
num_hybridization = 10
num_aromatic = 3
num_in_ring = 3
num_degree = 7
num_charge = 4
num_numH = 4

# Edge features: bond_type, stereo, conjugated
num_bond_type = 6
num_bond_stereo = 9
num_bond_isConjugated = 3


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, eps=0., train_eps=True):
        super(GINEConv, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.bond_type_emb = nn.Embedding(num_bond_type, emb_dim)
        self.stereo_emb = nn.Embedding(num_bond_stereo, emb_dim)
        self.is_conjugated_emb = nn.Embedding(num_bond_isConjugated, emb_dim)
        
        for embedding in [self.bond_type_emb, self.stereo_emb, self.is_conjugated_emb]:
            nn.init.xavier_uniform_(embedding.weight.data)
        
        if train_eps:
            self.eps = nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.eps.data.fill_(eps)
            
    def forward(self, x, edge_index, edge_attr):
        edge_attr = edge_attr.long()
        edge_emb = self.bond_type_emb(edge_attr[:, 0]) + \
                          self.stereo_emb(edge_attr[:, 1]) + \
                          self.is_conjugated_emb(edge_attr[:, 2])
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        out = out + (1 + self.eps) * x
        return self.mlp(out)
    
    def message(self, x_j, edge_attr):
        return (x_j + edge_attr).relu()
    
    
class GIN(nn.Module):
    def __init__(self, num_layers=5, emb_dim=128, feat_dim=640, drop_ratio=0.2):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        
        self.atom_type_emb = nn.Embedding(num_atom_type, self.emb_dim)
        self.hybridization_emb = nn.Embedding(num_hybridization, self.emb_dim)
        self.aromatic_emb = nn.Embedding(num_aromatic, self.emb_dim)
        self.in_ring_emb = nn.Embedding(num_in_ring, self.emb_dim)
        self.degree_emb = nn.Embedding(num_degree, self.emb_dim)
        self.charge_emb = nn.Embedding(num_charge, self.emb_dim)
        self.numH_emb = nn.Embedding(num_numH, self.emb_dim)
        
        for embedding in [self.atom_type_emb, self.hybridization_emb, self.aromatic_emb,
                          self.in_ring_emb, self.degree_emb, self.charge_emb, self.numH_emb]:
            nn.init.xavier_uniform_(embedding.weight.data)
        
        self.conv_layers = nn.ModuleList(
            GINEConv(self.emb_dim) for _ in range(self.num_layers)
        )
        
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(self.emb_dim) for _ in range(self.num_layers)
        )
        
        self.feature_projection = nn.Linear(self.num_layers * self.emb_dim, self.feat_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(self.feat_dim + 1, self.feat_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_ratio),
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.drop_ratio),
            nn.Linear(self.feat_dim // 2, 1)
        )
        
    def extract(self, x, batch):
        _, counts = torch.unique(batch, return_counts=True)
        cumsum = torch.cumsum(counts, dim=0)
        global_node_idx = cumsum - 1
        global_node = x[global_node_idx]
        return global_node
    
    def forward(self, graph, cond):
        x = graph.x.long()
        embs = [
            self.atom_type_emb(x[:, 0]),
            self.hybridization_emb(x[:, 1]),
            self.aromatic_emb(x[:, 2]),
            self.in_ring_emb(x[:, 3]),
            self.degree_emb(x[:, 4]),
            self.charge_emb(x[:, 5]),
            self.numH_emb(x[:, 6]),
        ]
        h = sum(embs)
                        
        for layer in range(self.num_layers):
            h = self.conv_layers[layer](h, graph.edge_index, graph.edge_attr)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
            if layer == 0:
                layer_outputs = h.unsqueeze(1)
            else:
                layer_outputs = torch.cat([layer_outputs, h.unsqueeze(1)], dim=1)

        h = layer_outputs.view(layer_outputs.size(0), -1)

        h = self.feature_projection(h)
        h = self.extract(h, graph.batch)
        h = torch.cat([h, cond.unsqueeze(1)], dim=1)
        h = self.fc(h)
        
        return h
