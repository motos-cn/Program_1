import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from scipy import stats


def plot_true_vs_predicted(train_true, train_pred, test_true, test_pred, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(train_true, train_pred, label='Train', c='#f57c6e')
    plt.scatter(test_true, test_pred, label='Test', c='#71b7ed')
    x_min = min(min(train_true), min(test_true))
    x_max = max(max(train_true), max(test_true))
    plt.plot([x_min, x_max], [x_min, x_max], 'k--', lw=2)
    plt.xlabel('True value')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_distribution(train_true, train_pred, test_true, test_pred, save_path):
    train_error = np.array(train_pred) - np.array(train_true)
    test_error = np.array(test_pred) - np.array(test_true)
    train_error = train_error.flatten()
    test_error = test_error.flatten()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(train_error, bins=30, color='#4c9bff', edgecolor='#666666',
             linewidth=1.5, alpha=0.7, density=True)
    ax1.axvline(x=0, color='#666666', linestyle='--', lw=1.5)
    kde_train = stats.gaussian_kde(train_error)
    x_range = np.linspace(train_error.min(), train_error.max(), 100)
    ax1.plot(x_range, kde_train(x_range), color='#666666', linewidth=2, linestyle='-', label='KDE')
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Density')
    ax1.set_title('Training Set')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.hist(test_error, bins=30, color='#ffe382', edgecolor='#666666',
             linewidth=1.5, alpha=0.7, density=True)
    ax2.axvline(x=0, color='#666666', linestyle='--', lw=1.5)
    kde_test = stats.gaussian_kde(test_error)
    x_range = np.linspace(test_error.min(), test_error.max(), 100)
    ax2.plot(x_range, kde_test(x_range), color='#666666', linewidth=2, linestyle='-', label='KDE')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Density')
    ax2.set_title('Testing Set')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_atom_heatmap(atom_importance, edge_index, num_atoms, save_path, layout_seed=400):
    """Render a single molecule with atom importance as a color heatmap."""
    g = nx.Graph()
    g.add_nodes_from(range(num_atoms))
    for j in range(len(edge_index[0])):
        a, b = edge_index[0][j], edge_index[1][j]
        if a < b:
            g.add_edge(a, b)

    pos = nx.spring_layout(g, seed=layout_seed)

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_edges(g, pos, alpha=0.4)
    node = nx.draw_networkx_nodes(
        g, pos, node_size=50, node_color=atom_importance, cmap=plt.cm.viridis)
    plt.colorbar(node)
    plt.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fragment_scores(stats, save_path, sort_by='abs_score', top_k=20, d_threshold=0.8, ci_level=95):
    """Bar chart of fragment importance with error bars for CI."""
    frag_names = list(stats.keys())

    # Sort
    scores = np.array([stats[n]['mean'] for n in frag_names])
    if sort_by == 'abs_score':
        sorted_idx = np.abs(scores).argsort()
    else:
        sorted_idx = scores.argsort()

    # Select top_k
    sorted_idx = sorted_idx[-top_k:]
    sorted_names = np.array(frag_names)[sorted_idx]
    sorted_scores = scores[sorted_idx]

    # Error bar data (asymmetric CI)
    xerr_low = np.array([sorted_scores[i] - stats[n]['ci_low']
                         for i, n in enumerate(sorted_names)])
    xerr_high = np.array([stats[n]['ci_high'] - sorted_scores[i]
                          for i, n in enumerate(sorted_names)])

    # Color and alpha by reliability
    colors = []
    alphas = []
    for name in sorted_names:
        if stats[name]['reliable']:
            colors.append('#2ca02c')
            alphas.append(0.9)
        else:
            colors.append('#bbbbbb')
            alphas.append(0.4)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(sorted_names))

    for i in range(len(sorted_names)):
        ax.barh(y_pos[i], sorted_scores[i], height=0.7,
                color=colors[i], alpha=alphas[i],
                xerr=[[xerr_low[i]], [xerr_high[i]]],
                error_kw=dict(ecolor='#555555', capsize=3, capthick=1.2, lw=1.2))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.axvline(x=0, color='#333333', linestyle='--', lw=1.0)

    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.9,
              label=f'Reliable (|d|>={d_threshold}, {ci_level}% CI)'),
        Patch(facecolor='#bbbbbb', alpha=0.4, label='Unreliable'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    ax.set_xlabel('Fragment importance (Δ)', fontsize=14)
    ax.set_ylabel('Fragment type', fontsize=14)
    ax.tick_params(direction='in', width=1.5, labelsize=11)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()