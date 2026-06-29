import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.mlp import MLP
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from src.datasets.vector_dataset import VectorDataset
from src.training.ml_trainer import MLPTrainer, SklearnTrainer
from src.evaluation.visualization import plot_true_vs_predicted, plot_error_distribution
from src.utils.io import load_feature_vectors
from src.utils.split import random_split_data

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/rf_fp.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg = config['training']
    output_cfg = config['output']

    features, labels, feature_names = load_feature_vectors(data_cfg['feature_path'], data_cfg['data_path'])
    feature_type = data_cfg.get('feature_type', '')

    model_name = model_cfg['name']

    if model_name == 'mlp':
        train_data, test_data, train_labels, test_labels = random_split_data(
            features, labels, test_size=train_cfg['test_size'], random_state=train_cfg['random_seed']
        )

        train_dataset = VectorDataset(train_data, train_labels)
        test_dataset = VectorDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

        n_features = features.shape[1]
        model = MLP(n_features=n_features, **model_cfg['params'])

        trainer = MLPTrainer(
            model, lr=train_cfg['lr'], scheduler_t_max=train_cfg['scheduler_t_max'], scheduler_eta_min=train_cfg['scheduler_eta_min']
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

        if feature_names is not None and feature_type != 'fingerprint':
            from src.evaluation.shap_analysis import shap_analysis_deep
            shap_dir = os.path.join(output_cfg['save_dir'], 'figures', output_cfg['model_name'])
            os.makedirs(shap_dir, exist_ok=True)
            background_data = train_dataset.x
            explain_data = test_dataset.x
            model.cpu()
            shap_analysis_deep(model, background_data, explain_data, feature_names, shap_dir)

    elif model_name in ['random_forest', 'xgboost']:
        train_data, test_data, train_labels, test_labels = random_split_data(
            features, labels, test_size=train_cfg['test_size'], random_state=train_cfg['random_seed']
        )

        if model_name == 'random_forest':
            model = RandomForestModel(
                param_grid=model_cfg['params']['param_grid'], random_state=train_cfg['random_seed']
            )
        elif model_name == 'xgboost':
            model = XGBoostModel(
                param_grid=model_cfg['params']['param_grid'], random_state=train_cfg['random_seed']
            )

        trainer = SklearnTrainer(model)
        trainer.train(train_data, train_labels, grid_search=train_cfg['grid_search'])

        print("\nTest:")
        test_true, test_pred, test_metrics = trainer.evaluate(test_data, test_labels)
        print("\nTrain:")
        train_true, train_pred, train_metrics = trainer.evaluate(train_data, train_labels)

        if feature_names is not None and feature_type != 'fingerprint':
            from src.evaluation.shap_analysis import shap_analysis_tree
            shap_dir = os.path.join(output_cfg['save_dir'], 'figures', output_cfg['model_name'])
            os.makedirs(shap_dir, exist_ok=True)
            shap_analysis_tree(model.model, features, feature_names, shap_dir)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    fig_dir = os.path.join(output_cfg['save_dir'], 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    plot_true_vs_predicted(
        train_true, train_pred, test_true, test_pred, os.path.join(fig_dir, f"{output_cfg['model_name']}.png")
    )
    plot_error_distribution(
        train_true, train_pred, test_true, test_pred, os.path.join(fig_dir, f"{output_cfg['model_name']}_error.png")
    )


if __name__ == '__main__':
    main()
