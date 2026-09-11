# Thermal_program

Predicting thermal conductivity (**λ**) of ionic liquids.

> Structure, installation, usage, and configuration: see the [root README](../README.md).

## Condition Preprocessing

Unlike the other two programs, the thermal conductivity dataset includes two thermodynamic conditions:

- **Temperature (T)**: Z-score standardization: `(T - mean) / std`.
- **Pressure (P)**: Log-transformed first (`log(P)`), then Z-score standardized. Pressure data follows a highly skewed distribution (most values near atmospheric ~0.1 MPa, with some up to 65 MPa).

Both conditions are concatenated with molecular features and fed into the models.

## Raw Data Format

`data/raw/whole.csv`:

| Column | Description |
|---|---|
| `IL` | Ionic liquid ID |
| `Full Name` | Full name of the ionic liquid |
| `IL_smiles` | SMILES string (cation.anion) |
| `Cation` | Cation SMILES |
| `Anion` | Anion SMILES |
| `T` | Temperature (K) |
| `P` | Pressure (MPa) |
| `lambda` | Target thermal conductivity |

## License

This project is licensed under the [MIT License](../LICENSE).
