import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

ion_groups = {
    1: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 3},
    2: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 6},
    3: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 10},
    4: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 14},
    5: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 7},
    6: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 12},
    7: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 1, 'CH2O': 1},
    8: {'[P]+': 1, 'P-CH2': 4, 'CH3': 3, 'CH2': 1, 'CH3O': 1},
    9: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 7, 'CH2O': 1},
    10: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 2},
    11: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1},
    12: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 2},
    13: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 4},
    14: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 6},
    15: {'[Im13]+': 1, 'aN-CH3': 2},
    16: {'[N]+': 1, 'N-CH2': 4, 'CH3': 4, 'CH2': 3},
    17: {'[N]+': 1, 'N-CH3': 2, 'N-CH2': 2, 'CH3': 2, 'CH2': 3},
    18: {'[N]+': 1, 'N-CH3': 2, 'N-CH2': 2, 'CH3': 2, 'CH2': 5},
    19: {'[N]+': 1, 'N-CH3': 2, 'N-CH2': 2, 'CH3': 2, 'CH2': 2},
    20: {'[N]+': 1, 'N-CH3': 2, 'N-CH2': 2, 'CH3': 2, 'CH2': 1, 'N(CH3)2': 1},
    21: {'[N]+': 1, 'N-CH3': 3, 'N-CH2': 1, 'CH2': 1, 'CH3O': 1},
    22: {'[N]+': 1, 'N-CH3': 1, 'N-CH2': 3, 'CH3': 3, 'CH2': 2},
    23: {'[N]+': 1, 'N-CH3': 3, 'N-CH2': 1, 'CH2': 1, 'OH': 1},
    24: {'[N]+': 1, 'N-CH3': 2, 'N-CH2': 2, 'CH3': 1, 'CH2': 1, 'CH3COO': 1},
    25: {'[N]+': 1, 'N-CH3': 2, 'N-CH2': 2, 'CH3': 1, 'CH2': 2, 'CH3O': 1, 'CH2O': 1},
    26: {'[Pyr11]+': 1, 'cN-CH3': 1, 'cN-CH2': 1, 'CH3': 1, 'CH2': 2},
    27: {'[Pyr11]+': 1, 'cN-CH3': 1, 'cN-CH2': 1, 'CH3': 1, 'CH2': 1},
    28: {'[Py1]+': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 2},
    29: {'[Pipz]+': 1, 'cN-CH3': 2, 'cN-CH2': 1, 'CH3': 1},
    30: {'[ABN]+': 1, 'cN-CH3': 1, 'cN-CH2': 1, 'CH3': 1, 'CH2': 1},
    31: {'[NTf2]-': 1},
    32: {'[CF3COO]-]': 1},
    33: {'[CH3SO3]-': 1},
    34: {'[CF3SO3]-': 1},
    35: {'[TCM]-': 1},
    36: {'[Pz]-': 1, 'aC-CF3': 1},
    37: {'[Pz]-': 1, 'aC-CF3': 1, 'aC-CH3': 1},
    38: {'[Ind]-': 1},
    39: {'[TCB]-': 1},
    40: {'[FAP]-': 1},
    41: {'[PF6]-': 1},
    42: {'[BF4]-': 1},
    43: {'[Pyr]-': 1, 'aC-CN': 1},
    44: {'[4triz]-': 1},
    45: {'[3triz]-': 1},
    46: {'[Pz]-': 1, 'aC-NOO': 1},
    47: {'[Pz]-': 1, 'aC-NOO': 1},
    48: {'[Tz]-': 1},
    49: {'[CH3SO4]-': 1},
    50: {'[N(SO2CF2CF3)2]-': 1},
    51: {'[CF3COO]-': 1},
    52: {'[N(SO2F)]-': 1},
    53: {'[B]-': 1, 'B-CF2': 1, 'B-F': 3, 'CF2': 1, 'CF3O': 1},
    54: {'[B]-': 1, 'B-CF2': 1, 'B-F': 3, 'CF3': 1, 'CF2': 1},
    55: {'[B]-': 1, 'B-CF2': 1, 'B-F': 3, 'CF3': 1, 'CF2': 2},
    56: {'[N(SO2CF3)SO2CF2]-': 1, 'CF3': 1, 'CF2': 2}
}

def add_groups_to_smiles_csv(smiles_csv_path, output_path):
    df = pd.read_csv(smiles_csv_path)

    all_groups = set()
    for groups in ion_groups.values():
        all_groups.update(groups.keys())
    all_groups = sorted(all_groups)

    for group in all_groups:
        df[group] = 0
    for idx, groups in ion_groups.items():
        row_idx = idx - 1
        for group, count in groups.items():
            df.at[row_idx, group] = count

    df.to_csv(output_path, index=False)

def main():
    input_path = '../../data/raw/smiles.csv'
    output_dir = '../../data/processed/groups'
    os.makedirs(output_dir, exist_ok=True)
    add_groups_to_smiles_csv(input_path, os.path.join(output_dir, 'groups.csv'))

if __name__ == '__main__':
    main()
