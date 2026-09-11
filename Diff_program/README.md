# Diff_program

Predicting cation/anion self-diffusion coefficients (**D⁺** and **D⁻**) of ionic liquids.

> Structure, installation, usage, and configuration: see the [root README](../README.md).

## Raw Data Format

`data/raw/whole_D+.csv`, `data/raw/whole_D-.csv`:

| Column | Description |
|---|---|
| `IL` | Ionic liquid abbreviation |
| `IL_smiles` | SMILES string (cation.anion) |
| `T` | Temperature (K) |
| `D+` / `D-` | Target diffusion coefficient |

## License

This project is licensed under the [MIT License](../LICENSE).
