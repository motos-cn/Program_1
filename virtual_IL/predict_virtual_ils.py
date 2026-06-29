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
    """Z-score normalize T (and P for lambda)."""
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
    """Predict property for all valid graphs."""
    valid = [(i, g) for i, g in enumerate(graphs) if g is not None]
    if not valid:
        return np.full(len(graphs), np.nan)

    preds = np.full(len(graphs), np.nan)
    bs = 64

    for start in range(0, len(valid), bs):
        end = min(start + bs, len(valid))
        idxs = [valid[j][0] for j in range(start, end)]
        glist = [valid[j][1] for j in range(start, end)]
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

    pairs = [
        {"cation_smiles": c["cation_smiles"], "anion_smiles": a["anion_smiles"],
         "IL_smiles": f"{c['cation_smiles']}.{a['anion_smiles']}"}
        for c in cations.to_dict("records")
        for a in anions.to_dict("records")
    ]

    # 2. Convert SMILES to graphs (once, reused across temperatures)
    print("Converting SMILES to graphs...")
    graphs = [smiles_to_graph(p["IL_smiles"]) for p in pairs]
    n_valid = sum(g is not None for g in graphs)
    print(f"  {n_valid}/{len(graphs)} valid")

    # 3. Predict at each temperature
    rows = []
    loaded_models = {}
    for name, (proj, ckpt, cdim) in MODELS.items():
        loaded_models[name] = load_model(proj, ckpt, cdim)

    for t in TEMPS:
        print(f"\nT = {t} K")
        for name, model in loaded_models.items():
            cond = norm_condition(name, t)
            preds = batch_predict(model, graphs, cond, name)
            for i, p in enumerate(pairs):
                if not np.isnan(preds[i]):
                    row = {
                        "IL_smiles": p["IL_smiles"],
                        "cation_smiles": p["cation_smiles"],
                        "anion_smiles": p["anion_smiles"],
                        "T": t,
                        name: preds[i],
                    }
                    rows.append(row)

    # 4. Assemble into one DataFrame
    df = pd.DataFrame(rows)
    # Merge 4 property columns per (IL_smiles, T) — each row only has one property
    df = df.groupby(["IL_smiles", "cation_smiles", "anion_smiles", "T"], as_index=False).first()
    df["|ΔD|"] = (df["D+"] - df["D-"]).abs()

    out_cols = ["IL_smiles", "cation_smiles", "anion_smiles", "T",
                "D+", "D-", "|ΔD|", "sigma", "lambda"]
    df[out_cols].to_csv(os.path.join(DIR, "virtual_IL_predictions.csv"), index=False)

    # 5. Summary
    print("\nPrediction summary:")
    for t in TEMPS:
        sub = df[df["T"] == t]
        print(f"  T={t}K: {len(sub)} ILs")
        for col in ["D+", "D-", "|ΔD|", "sigma", "lambda"]:
            v = sub[col].dropna()
            print(f"    {col}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

    print(f"\nSaved: virtual_IL_predictions.csv  ({len(df)} rows × {len(out_cols)} cols)")


if __name__ == "__main__":
    main()
