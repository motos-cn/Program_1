import os, sys, importlib.util, warnings
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

warnings.filterwarnings("ignore")

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prediction conditions
TEMPS = [298.15, 323.15, 348.15, 368.15]   # K
P = 0.1                                     # MPa

# Z-score stats from training data
STATS = {
    "D+":     {"T_mean": 320.4308, "T_std": 26.1259},
    "D-":     {"T_mean": 322.4302, "T_std": 26.4232},
    "sigma":  {"T_mean": 320.6984, "T_std": 37.0722},
    "lambda": {"T_mean": 313.1379, "T_std": 32.46,
               "logP_mean": -0.6937, "logP_std": 2.5031},
}

# (project, checkpoint_dir, condition_dim)
MODELS = {
    "D+":     ("Diff_program",                   "GIN_D+",     1),
    "D-":     ("Diff_program",                   "GIN_D-",     1),
    "sigma":  ("ElectricalConductivity_program", "GIN_sigma",  1),
    "lambda": ("ThermalConductivity_program",    "GIN_lambda", 2),
}

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(ROOT, "Diff_program", "src"))
from data_processing.smiles2graph import smilesToGraphConverter
from datasets.graph_dataset import add_global_node
from models.gnn.gin import GIN as GIN_1cond

spec = importlib.util.spec_from_file_location(
    "gin_tc",
    os.path.join(ROOT, "ThermalConductivity_program", "src", "models", "gnn", "gin.py"),
)
_mod = importlib.util.module_from_spec(spec)
sys.modules["gin_tc"] = _mod
spec.loader.exec_module(_mod)
GIN_2cond = _mod.GIN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_converter = smilesToGraphConverter()

def smiles_to_graph(smiles):
    raw = _converter.smiles2graph(smiles)
    if raw is None:
        return None
    g = Data(
        x=torch.tensor(raw[0], dtype=torch.float32),
        edge_index=torch.tensor(raw[1], dtype=torch.long),
        edge_attr=torch.tensor(raw[2], dtype=torch.float32),
    )
    return add_global_node(g)


def norm_condition(name, t):
    s = STATS[name]
    t_norm = (t - s["T_mean"]) / s["T_std"]
    if name == "lambda":
        p_norm = (np.log(P) - s["logP_mean"]) / s["logP_std"]
        return torch.tensor([[t_norm, p_norm]], dtype=torch.float32)
    return torch.tensor([t_norm], dtype=torch.float32)


def load_model(project, ckpt_dir, cond_dim):
    GIN_cls = GIN_2cond if cond_dim == 2 else GIN_1cond
    path = os.path.join(ROOT, project, "results", "checkpoints", ckpt_dir, "model.pth")
    model = GIN_cls().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def batch_predict(model, graphs, cond, name):
    """Predict property for all valid graphs, return array aligned to input."""
    valid_idx = [i for i, g in enumerate(graphs) if g is not None]
    if not valid_idx:
        return np.full(len(graphs), np.nan)

    preds = np.full(len(graphs), np.nan)
    bs = 256

    for start in range(0, len(valid_idx), bs):
        end = min(start + bs, len(valid_idx))
        idxs = valid_idx[start:end]
        glist = [graphs[i] for i in idxs]
        batch_g = Batch.from_data_list(glist).to(DEVICE)
        n = len(glist)

        cond_b = cond.repeat(n, 1).to(DEVICE) if name == "lambda" \
                 else cond.repeat(n).to(DEVICE)

        with torch.no_grad():
            out = model(batch_g, cond_b).cpu().numpy().flatten()

        for k, idx in enumerate(idxs):
            preds[idx] = out[k]

    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Temperatures: {TEMPS} K  |  P: {P} MPa")

    # 1. Build cation × anion combinations
    cations = pd.read_csv(os.path.join(DIR, "Cation.csv"))
    anions = pd.read_csv(os.path.join(DIR, "Anion.csv")).drop_duplicates("anion_smiles")
    n_il = len(cations) * len(anions)
    print(f"ILs: {len(cations)} cations × {len(anions)} anions = {n_il}")

    il_smiles_list = []
    cat_smiles_list = []
    an_smiles_list = []
    for c in cations.to_dict("records"):
        for a in anions.to_dict("records"):
            il_smiles_list.append(f"{c['cation_smiles']}.{a['anion_smiles']}")
            cat_smiles_list.append(c["cation_smiles"])
            an_smiles_list.append(a["anion_smiles"])

    # 2. Convert SMILES to graphs (once, reused across temperatures)
    print("Converting SMILES to graphs...")
    graphs = [smiles_to_graph(s) for s in il_smiles_list]
    n_valid = sum(g is not None for g in graphs)
    n_failed = n_il - n_valid
    print(f"  {n_valid}/{n_il} valid  ({n_failed} failed)")
    if n_failed > 0:
        failed = [il_smiles_list[i] for i, g in enumerate(graphs) if g is None]
        print(f"  Failed SMILES (first 5): {failed[:5]}")

    # 3. Load all models
    loaded_models = {}
    for name, (proj, ckpt, cdim) in MODELS.items():
        loaded_models[name] = load_model(proj, ckpt, cdim)
        print(f"  Loaded {name} from {proj}/results/checkpoints/{ckpt}")

    # 4. Predict at each temperature — one row per (IL, T)
    rows = []
    for t in TEMPS:
        print(f"\nT = {t} K")
        # Predict all 4 properties for this temperature
        all_preds = {}
        for name, model in loaded_models.items():
            cond = norm_condition(name, t)
            all_preds[name] = batch_predict(model, graphs, cond, name)

        # Build one row per IL
        for i in range(n_il):
            d_plus = all_preds["D+"][i]
            d_minus = all_preds["D-"][i]
            sigma_val = all_preds["sigma"][i]
            lambda_val = all_preds["lambda"][i]

            # Skip ILs where ALL properties are NaN (graph conversion failed)
            if graphs[i] is None:
                continue

            delta_d = abs(d_plus - d_minus) if not (np.isnan(d_plus) or np.isnan(d_minus)) else np.nan

            rows.append({
                "IL_smiles": il_smiles_list[i],
                "cation_smiles": cat_smiles_list[i],
                "anion_smiles": an_smiles_list[i],
                "T": t,
                "D+": d_plus,
                "D-": d_minus,
                "|ΔD|": delta_d,
                "sigma": sigma_val,
                "lambda": lambda_val,
            })

    # 5. Save
    df = pd.DataFrame(rows)
    out_cols = ["IL_smiles", "cation_smiles", "anion_smiles", "T",
                "D+", "D-", "|ΔD|", "sigma", "lambda"]
    df[out_cols].to_csv(os.path.join(DIR, "virtual_IL_predictions.csv"), index=False)

    # 6. Summary
    print("\nPrediction summary:")
    for t in TEMPS:
        sub = df[df["T"] == t]
        n_complete = len(sub)
        n_any_nan = sub.isnull().any(axis=1).sum()
        print(f"  T={t}K: {n_complete} ILs  ({n_any_nan} with ≥1 NaN property)")
        for col in ["D+", "D-", "|ΔD|", "sigma", "lambda"]:
            v = sub[col].dropna()
            print(f"    {col}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

    print(f"\nSaved: virtual_IL_predictions.csv  ({len(df)} rows × {len(out_cols)} cols)")


if __name__ == "__main__":
    main()
