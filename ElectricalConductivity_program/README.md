# Conductivity_program

Predicting electrical conductivity (**σ**) of ionic liquids.

> Structure, installation, usage, and configuration: see the [root README](../README.md).

## Raw Data Format

`data/raw/whole.csv`:

| Column | Description |
|---|---|
| `IL` | Ionic liquid ID |
| `IL_smiles` | SMILES string (cation.anion) |
| `Cation` | Cation SMILES |
| `Anion` | Anion SMILES |
| `T` | Temperature (K) |
| `sigma` | Target electrical conductivity |

## License

This project is licensed under the [MIT License](../LICENSE).
