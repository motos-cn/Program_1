from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    return metrics

def print_metrics(metrics, prefix=''):
    print(f"{prefix}R2: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f},"
          f"RMSE: {metrics['rmse']:.4f}")
