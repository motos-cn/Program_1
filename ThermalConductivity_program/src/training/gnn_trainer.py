import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error


class GNNTrainer:
    def __init__(self, model, lr=0.001, weight_decay=0, scheduler_t_max=250, scheduler_eta_min=1e-6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nRunning on: {self.device}")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=scheduler_t_max, eta_min=scheduler_eta_min)

    def load_pretrained(self, checkpoint_path):
        print("\nLoading pretrained model...")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Finish loading pretrained model!")

    def save_checkpoint(self, save_path):
        print("Saving model checkpoint...")
        torch.save(self.model.state_dict(), save_path)

    def train(self, train_loader, epochs=300, warmup=50):
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            batch_bar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}',
                             dynamic_ncols=True, leave=False)
            for graph, cond, label in train_loader:
                graph = graph.to(self.device)
                cond = cond.to(self.device)
                label = label.to(self.device)
                y = self.model(graph, cond)
                loss = self.criterion(y.flatten(), label.flatten())
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_bar.update()
            batch_bar.close()

            if epoch >= warmup:
                self.scheduler.step()

            avg_loss = train_loss / len(train_loader)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Train loss: {avg_loss:.6f}, LR: {lr:.8f}")

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        y_pred = []
        y_true = []
        batch_bar = tqdm(total=len(data_loader), desc='Evaluating',
                         dynamic_ncols=True, leave=False)
        with torch.no_grad():
            for graph, cond, label in data_loader:
                graph = graph.to(self.device)
                cond = cond.to(self.device)
                label = label.to(self.device)
                y = self.model(graph, cond)
                loss = self.criterion(y.flatten(), label.flatten())
                total_loss += loss.item()

                y_pred.extend(y.detach().cpu().numpy().flatten())
                y_true.extend(label.detach().cpu().numpy().flatten())
                batch_bar.update()
            batch_bar.close()

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        avg_loss = total_loss / len(data_loader)
        print(f"R2: {r2:.4f}, MAE: {mae:.4f}, Loss: {avg_loss:.6f}")
        return y_true, y_pred, {'r2': r2, 'mae': mae, 'loss': avg_loss}
