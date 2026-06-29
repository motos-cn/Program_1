import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np


class MLPTrainer:
    def __init__(self, model, lr=0.001, weight_decay=0, scheduler_t_max=250, scheduler_eta_min=1e-6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=scheduler_t_max, eta_min=scheduler_eta_min)

    def load_pretrained(self, checkpoint_path):
        print("\nLoading pretrained model checkpoint...")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Finish loading pretrained model!")

    def save_checkpoint(self, save_path):
        print("Saving model checkpoint...")
        torch.save(self.model.state_dict(), save_path)

    def train(self, train_loader, epochs, warmup=50):
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                outputs = self.model(batch_features)
                loss = self.criterion(outputs.squeeze(), batch_labels)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch >= warmup:
                self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg_loss = train_loss / len(train_loader)
                print(f"Epoch [{epoch + 1}/{epochs}], Train loss: {avg_loss:.6f}")

    def evaluate(self, data_loader):
        self.model.eval()
        y_pred = []
        y_true = []
        total_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                outputs = self.model(batch_features)
                loss = self.criterion(outputs.squeeze(), batch_labels)
                total_loss += loss.item()
                y_pred.extend(outputs.detach().cpu().numpy().flatten())
                y_true.extend(batch_labels.cpu().numpy().flatten())
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        avg_loss = total_loss / len(data_loader)
        print(f"R2: {r2:.4f}, MAE: {mae:.4f}, Loss: {avg_loss:.6f}")
        return y_true, y_pred, {'r2': r2, 'mae': mae, 'loss': avg_loss}


class SklearnTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X, y, grid_search=True):
        self.model.fit(X, y, grid_search=grid_search)
        return self.model

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        print(f"R2: {r2:.4f}, MAE: {mae:.4f}")
        return y, y_pred, {'r2': r2, 'mae': mae}
