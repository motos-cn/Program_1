import os
import shap
import numpy as np
import matplotlib.pyplot as plt


def _save_shap_plots(shap_values, data, feature_names, save_dir, max_display=20):
    shap.summary_plot(shap_values, data, feature_names=feature_names,
                      plot_type='dot', show=False, max_display=max_display)
    plt.savefig(os.path.join(save_dir, 'shap_dot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values, data, feature_names=feature_names,
                      plot_type='bar', show=False, max_display=max_display)
    plt.savefig(os.path.join(save_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()


def shap_analysis_tree(model, data, feature_names, save_dir, max_display=20):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values.squeeze(2)
    _save_shap_plots(shap_values, data, feature_names, save_dir, max_display)


def shap_analysis_deep(model, background_data, explain_data, feature_names, save_dir, max_display=20):
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(explain_data)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values.squeeze(2)
    _save_shap_plots(shap_values, explain_data, feature_names, save_dir, max_display)
