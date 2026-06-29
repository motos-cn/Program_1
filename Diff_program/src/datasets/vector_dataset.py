import torch


class VectorDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.x = torch.tensor(features) if not isinstance(features, torch.Tensor) else features
        self.y = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]