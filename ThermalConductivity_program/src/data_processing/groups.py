import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

ion_groups = {
    1: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 8},
    2: {'[Py14]+': 1, 'aN-CH2': 1, 'aC-CH3': 1, 'CH3': 1, 'CH2': 2},
    3: {'[P]+': 1, 'P-CH3': 1, 'P-CH2': 3, 'CH3': 3, 'CH2': 6},
    4: {'[Py134]+': 1, 'aN-CH2': 1, 'aC-CH3': 1, 'aC-N': 1, 'CH3': 3, 'CH2': 4},
    5: {'[Im13]+': 1 ,'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1},
    6: {'[Py14]+': 1, 'aN-CH2': 1, 'aC-N': 1, 'CH3': 3, 'CH2': 2},
    7: {'[N]+': 1, 'N-CH3': 1, 'N-CH2': 3, 'CH3': 3, 'CH2': 6},
    8: {'[Im13]+': 1, 'aN-CH2': 2, 'CH3': 2, 'CH2': 4},
    9: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 4},
    10: {'[Py1]+': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 2},
    11: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 4},
    12: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1},
    13: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 6},
    14: {'[NH3]+': 1, 'N-CH2': 1, 'CH3': 1},
    15: {'[Py1]+': 1, 'aN-CH3': 1},
    16: {'[Im123]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'aC-CH3': 1, 'CH3': 1, 'CH2': 1},
    17: {'[Im13]+': 1, 'aN-CH3': 2},
    18: {'[Py13]+': 1, 'aN-CH2': 1, 'aC-CH3': 1, 'CH3': 1, 'CH2': 2},
    19: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 24},
    20: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 2},
    21: {'[Pyr11]+': 1, 'cN-CH3': 1, 'cN-CH2': 1, 'CH3': 1, 'CH2': 2},
    22: {'[N]+': 1, 'N-CH3': 1, 'N-CH2': 3, 'CH3': 3, 'CH2': 18},
    23: {'[S]+': 1, 'S-CH2': 3, 'CH3': 3},
    24: {'[Py1]+': 1, 'CH3': 1, 'CH2': 6},
    25: {'[Pip11]+': 1, 'cN-CH3': 1, 'cN-CH2': 1, 'CH3': 1, 'CH2': 2},
    26: {'[P]+': 1, 'P-CH2': 4, 'CH3': 4, 'CH2': 24},
    27: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 6},
    28: {'[Im13]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 8},
    29: {'[NH3]+': 1, 'N-CH2': 1, 'CH3': 1, 'CH2': 2},
    30: {'[Py14]+': 1, 'aN-CH2': 1, 'aC-N': 1, 'CH3': 3, 'CH2': 4},
    31: {'[P]+': 1, 'P-CH2': 4, 'CH3' :4, 'CH2': 8},
    32: {'[NH3]+': 1, 'N-CH2': 1, 'OH': 1, 'CH2': 1},
    33: {'[N]+': 1, 'N-CH2': 4, 'CH3': 4},
    34: {'[Im123]+': 1, 'aN-CH3': 1, 'aN-CH2': 1, 'aC-CH3': 1, 'CH3': 1, 'CH2': 2},
    35: {'[Py1]+': 1, 'aN-CH2': 1, 'CH3': 1, 'CH2': 4},
    36: {'[Pyr11]+': 1, 'cN-CH3': 1, 'cN-CH2': 1, 'CH3': 1, 'CH2': 1},
    37: {'[N]+': 1, 'N-CH3': 3, 'N-CH2': 1, 'CH3': 1, 'CH2': 2},
    38: {'[CF3COO]-': 1},
    39: {'[CH3COO]-': 1},
    40: {'[NTf2]-': 1},
    41: {'[NO3]-': 1},
    42: {'[SbF6]-': 1},
    43: {'[CH2SO4]-': 1, 'CH3': 1},
    44: {'[TCB]-': 1},
    45: {'[Br]-': 1},
    46: {'[CH2COO]-': 1, 'CH3': 1, 'CH2': 7},
    47: {'[CF3SO3]-': 1},
    48: {'[CH3SO3]-': 1},
    49: {'[CH2SO3]-': 1, 'NH2': 1, 'CH2': 1},
    50: {'[NO3]-': 1},
    51: {'[DCA]-': 1},
    52: {'[Cl]-': 1},
    53: {'[PF6]-': 1},
    54: {'[(CH2)2PO4]-': 1, 'CH3': 2},
    55: {'[CH2COO]-': 1, 'CH3': 1, 'CH2': 1},
    56: {'[CH3COO]-': 1},
    57: {'[BF4]-': 1},
    58: {'[CH2SO4]-': 1, 'CH3': 1, 'CH2': 6},
    59: {'[CHCOO]-': 1, 'OH': 1, 'CH3': 1},
    60: {'[CH2COO]-': 1, 'CH3': 1, 'CH2': 3},
    61: {'[CH3HPO3]-': 1},
    62: {'[CH3SO4]-': 1},
    63: {'[SCN]-': 1},
    64: {'[CH2COO]-': 1, 'CH3': 1, 'CH2': 5},
    65: {'[C(CN)3]-': 1},
    66: {'[FAP]-': 1},
    67: {'[(CH3)2PO4]-': 1}
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
