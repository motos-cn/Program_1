import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['axes.unicode_minus'] = False

DIR = r'c:\Users\zuomengcheng\Desktop\program\virtual_IL'
df = pd.read_csv(os.path.join(DIR, 'virtual_IL_predictions.csv'))
cats = pd.read_csv(os.path.join(DIR, 'Cation.csv'))
ans = pd.read_csv(os.path.join(DIR, 'Anion.csv'))

sc = MinMaxScaler()
df[['Damb_n', 'sigma_n', 'lambda_n']] = sc.fit_transform(df[['Damb', 'sigma', 'lambda']])

df = df.merge(cats[['cation_smiles', 'cation_category']], on='cation_smiles', how='left')
df = df.merge(ans[['anion_smiles', 'anion_name']], on='anion_smiles', how='left')

def topsis(sub_df):
    ideal = np.array([sub_df['Damb_n'].max(), sub_df['sigma_n'].max(), sub_df['lambda_n'].min()])
    neg_ideal = np.array([sub_df['Damb_n'].min(), sub_df['sigma_n'].min(), sub_df['lambda_n'].max()])
    pts = sub_df[['Damb_n', 'sigma_n', 'lambda_n']].values
    d_pos = np.linalg.norm(pts - ideal, axis=1)
    d_neg = np.linalg.norm(pts - neg_ideal, axis=1)
    return d_neg / (d_pos + d_neg)

TOP_N = 50
temperatures = [298.15, 323.15, 348.15, 368.15]
t_labels = ['298', '323', '348', '368']

top_data = {}
for t in temperatures:
    sub = df[df['T'] == t].copy()
    sub['topsis'] = topsis(sub)
    top_data[t] = sub.nlargest(TOP_N, 'topsis')

# Anion grouping: use plain text keys for matching, LaTeX labels for display
anion_group_map = {
    'trifluoridotris(heptafluoropro': 'FAP',
    'tetracarbonylcobaltate': 'CoCO4',
    'tetracyanoborate': 'BCN4',
    'dicyanamide': 'NCN2',
    'bis(fluorosulfonyl)imide': 'FSI',
    'tricyanomethane': 'CCN3',
    'thiocyanate': 'SCN',
    'trifluoridotris(pentafluoroeth': 'FPE',
    'tris(trifluoromethylsulfonyl)m': 'TFSM',
}

anion_labels = {
    'FAP': '[FAP]$^-$',
    'BCN4': '[B(CN)$_4$]$^-$',
    'CoCO4': '[Co(CO)$_4$]$^-$',
    'FSI': '[FSI]$^-$',
    'NCN2': '[N(CN)$_2$]$^-$',
    'CCN3': '[C(CN)$_3$]$^-$',
    'SCN': '[SCN]$^-$',
    'FPE': '[FPE]$^-$',
    'TFSM': '[TFSM]$^-$',
    'Other': 'Other',
}

anion_colors = {
    'FAP': '#4e79a7', 'BCN4': '#e15759', 'CoCO4': '#f28e2b',
    'FSI': '#76b7b2', 'NCN2': '#59a14f', 'CCN3': '#edc948',
    'SCN': '#b07aa1', 'FPE': '#9c755f', 'TFSM': '#bab0ac', 'Other': '#d4d4d4',
}

cation_colors = {
    'Imidazolium': '#4e79a7', 'Sulfonium': '#f28e2b', 'Pyridinium': '#59a14f',
    'Ammonium': '#76b7b2', 'Pyrrolidinium': '#b07aa1', 'Other': '#d4d4d4',
}

for t in temperatures:
    top_data[t]['anion_group'] = top_data[t]['anion_name'].apply(
        lambda x: next((v for k, v in anion_group_map.items() if k in str(x)), 'Other'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), facecolor='white')

# Left: Anion composition
anion_types_all = []
for t in temperatures:
    anion_types_all.extend(top_data[t]['anion_group'].unique())
anion_order = [a for a in anion_colors if a in set(anion_types_all)]

bottom = np.zeros(4)
for an in anion_order:
    counts = []
    for t in temperatures:
        n = (top_data[t]['anion_group'] == an).sum()
        counts.append(n / TOP_N * 100)
    ax1.bar(t_labels, counts, bottom=bottom, label=anion_labels[an],
            color=anion_colors[an], edgecolor='white', linewidth=0.5, width=0.6)
    bottom += np.array(counts)

ax1.set_ylabel('Composition (%)', fontsize=11)
ax1.set_xlabel('Temperature (K)', fontsize=11)
ax1.set_ylim(0, 105)
ax1.set_yticks([0, 25, 50, 75, 100])
ax1.tick_params(labelsize=9, direction='in')
ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9, edgecolor='#ccc')
ax1.set_title('Anion', fontsize=11, fontweight='bold', pad=6)

# Right: Cation composition
cat_types_all = []
for t in temperatures:
    cat_types_all.extend(top_data[t]['cation_category'].unique())
cat_order = [c for c in cation_colors if c in set(cat_types_all)]

bottom = np.zeros(4)
for cat in cat_order:
    counts = []
    for t in temperatures:
        n = (top_data[t]['cation_category'] == cat).sum()
        counts.append(n / TOP_N * 100)
    ax2.bar(t_labels, counts, bottom=bottom, label=cat, color=cation_colors[cat],
            edgecolor='white', linewidth=0.5, width=0.6)
    bottom += np.array(counts)

ax2.set_ylabel('Composition (%)', fontsize=11)
ax2.set_xlabel('Temperature (K)', fontsize=11)
ax2.set_ylim(0, 105)
ax2.set_yticks([0, 25, 50, 75, 100])
ax2.tick_params(labelsize=9, direction='in')
ax2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9, edgecolor='#ccc')
ax2.set_title('Cation', fontsize=11, fontweight='bold', pad=6)

plt.tight_layout(w_pad=3)
fig.savefig(os.path.join(DIR, 'screening_composition.png'), dpi=300, facecolor='white')
plt.close()
print('Saved screening_composition.png')
