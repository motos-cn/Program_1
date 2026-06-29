import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import random_split, DataLoader

from src.models.gnn.gcn import GCN
from src.models.gnn.gat import GAT
from src.models.gnn.gin import GIN
from src.datasets.graph_dataset import ILDataset
from src.training.gnn_trainer import GNNTrainer
from src.evaluation.visualization import plot_true_vs_predicted, plot_error_distribution

MODEL_REGISTRY = {
    'gcn': GCN,
    'gat': GAT,
    'gin': GIN,
}

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/gcn.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg = config['training']
    output_cfg = config['output']
    
    dataset = ILDataset(data_path=data_cfg['data_path'], label_path=data_cfg['label_path'])
    print(f"Dataset size: {len(dataset)}")
    
    train_dataset,  test_dataset = random_split(
        dataset, [1 - train_cfg['test_size'], train_cfg['test_size']],
        generator=torch.Generator().manual_seed(train_cfg['random_seed'])
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg['batch_size'], shuffle=True,
        collate_fn=ILDataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_cfg['batch_size'], shuffle=False,
        collate_fn=ILDataset.collate_fn
    )
    
    model_class = MODEL_REGISTRY[model_cfg['name']]
    model = model_class(**model_cfg['params'])
    
    trainer = GNNTrainer(
        model, lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'],
        scheduler_t_max=train_cfg['scheduler_t_max'],
        scheduler_eta_min=train_cfg['scheduler_eta_min']
    )
    
    if train_cfg['train_mode']:
        trainer.train(train_loader, epochs=train_cfg['epochs'], warmup=train_cfg['warmup'])
        checkpoint_dir = os.path.join(output_cfg['save_dir'], 'checkpoints', output_cfg['model_name'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        trainer.save_checkpoint(os.path.join(checkpoint_dir, 'model.pth'))
    else:
        trainer.load_pretrained(train_cfg['load_checkpoint'])
        
    print("\nTest:")
    test_true, test_pred, test_metrics = trainer.evaluate(test_loader)
    print("\nTrain:")
    train_true, train_pred, train_metrics = trainer.evaluate(train_loader)
    
    fig_dir = os.path.join(output_cfg['save_dir'], 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    plot_true_vs_predicted(
        train_true, train_pred, test_true, test_pred,
        os.path.join(fig_dir, f"{output_cfg['model_name']}.png")
    )
    
    plot_error_distribution(
        train_true, train_pred, test_true, test_pred,
        os.path.join(fig_dir, f"{output_cfg['model_name']}_error.png")
    )

if __name__ == '__main__':
    main()
