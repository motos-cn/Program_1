import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FuncFormatter
import os

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'c:\Users\zuomengcheng\Desktop\program\virtual_IL\virtual_IL_predictions.csv')

sc = MinMaxScaler()
df[['Damb_n', 'sigma_n', 'lambda_n']] = sc.fit_transform(df[['Damb', 'sigma', 'lambda']])

# TOPSIS
def topsis(sub_df):
    ideal = np.array([sub_df['Damb_n'].max(), sub_df['sigma_n'].max(), sub_df['lambda_n'].min()])
    neg_ideal = np.array([sub_df['Damb_n'].min(), sub_df['sigma_n'].min(), sub_df['lambda_n'].max()])
    pts = sub_df[['Damb_n', 'sigma_n', 'lambda_n']].values
    d_pos = np.linalg.norm(pts - ideal, axis=1)
    d_neg = np.linalg.norm(pts - neg_ideal, axis=1)
    return d_neg / (d_pos + d_neg)

df['topsis'] = 0.0
for t in df['T'].unique():
    mask = df['T'] == t
    df.loc[mask, 'topsis'] = topsis(df[mask])

damb_min, damb_max = df['Damb'].min(), df['Damb'].max()
sigma_min, sigma_max = df['sigma'].min(), df['sigma'].max()
lambda_min, lambda_max = df['lambda'].min(), df['lambda'].max()

temperatures = sorted(df['T'].unique())
out_dir = r'c:\Users\zuomengcheng\Desktop\program\virtual_IL'
TOP_N = 50

for t in temperatures:
    fig = plt.figure(figsize=(7, 6), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
        a.pane.fill = True
        a.pane.set_facecolor((1, 1, 1, 1))
        a.pane.set_edgecolor((0, 0, 0, 0))

    # Bottom edges: split into non-intersection (black) and plane-intersection (gray)
    ax.plot([0,1], [0,0], [0,0], color='black', lw=1.1)    # front
    ax.plot([1,1], [0,1], [0,0], color='black', lw=1.1)    # right
    ax.plot([0,1], [1,1], [0,0], color='#888', lw=1.1)     # back: XY ∩ XZ
    ax.plot([0,0], [0,1], [0,0], color='#888', lw=1.1)     # left: XY ∩ YZ
    # Pillars
    ax.plot([0,0], [0,0], [0,1], color='black', lw=1.1)    # left-back
    ax.plot([0,0], [1,1], [0,1], color='#888', lw=1.1)     # left-front: YZ ∩ XZ
    ax.plot([1,1], [1,1], [0,1], color='black', lw=1.1)    # right-back
    # Top edges
    ax.plot([0,0], [0,1], [1,1], color='black', lw=1.1)    # YZ top
    ax.plot([0,1], [1,1], [1,1], color='black', lw=1.1)    # XZ top

    for f in np.linspace(0.25, 0.75, 3):
        ax.plot([0,1], [f,f], [0,0], color='#bbb', ls='--', lw=0.5)
        ax.plot([f,f], [0,1], [0,0], color='#bbb', ls='--', lw=0.5)
        ax.plot([0,0], [f,f], [0,1], color='#bbb', ls='--', lw=0.5)
        ax.plot([0,0], [0,1], [f,f], color='#bbb', ls='--', lw=0.5)
        ax.plot([f,f], [1,1], [0,1], color='#bbb', ls='--', lw=0.5)
        ax.plot([0,1], [1,1], [f,f], color='#bbb', ls='--', lw=0.5)

    ax.grid(False)
    ax.set_box_aspect([1.0, 1.0, 0.9])
    ax.view_init(elev=16, azim=-62)

    sub = df[df['T'] == t].copy()
    top = sub.nlargest(TOP_N, 'topsis')
    rest = sub.drop(top.index)

    def project_lines(xs, ys, zs, **kwargs):
        n = len(xs)
        sx = np.empty(3 * n); sy = np.empty(3 * n); sz = np.empty(3 * n)
        sx[0::3] = xs; sx[1::3] = xs; sx[2::3] = np.nan
        sy[0::3] = ys; sy[1::3] = ys; sy[2::3] = np.nan
        sz[0::3] = 0;  sz[1::3] = zs; sz[2::3] = np.nan
        ax.plot(sx, sy, sz, **kwargs)

    # Gray background: projection lines + scatter
    project_lines(rest['Damb_n'].values, rest['sigma_n'].values, rest['lambda_n'].values,
                  color='#9a9a9a', ls='-', lw=0.15, alpha=0.1, zorder=1)
    ax.scatter(rest['Damb_n'], rest['sigma_n'], rest['lambda_n'],
               c='#6b8cae', s=16, marker='o', edgecolors='#4a6a8a',
               linewidths=0.25, depthshade=True, alpha=0.5, zorder=2)

    # Top N: red projection lines + scatter
    project_lines(top['Damb_n'].values, top['sigma_n'].values, top['lambda_n'].values,
                  color='#e63946', ls=':', lw=0.8, zorder=5, alpha=0.75)
    ax.scatter(top['Damb_n'], top['sigma_n'], top['lambda_n'],
               c='#e63946', s=16, marker='o', edgecolors='#a4161a',
               linewidths=0.4, depthshade=False, zorder=10)

    # Axis labels & ticks
    ax.set_xlabel(r'$D_{\mathrm{amb}}$ ($10^{-7}$ cm$^2$/s)', labelpad=8, fontsize=11)
    ax.set_ylabel(r'$\sigma$ (S/m)', labelpad=8, fontsize=11)
    ax.set_zlabel(r'$\lambda$ (W/(m$\cdot$K))', labelpad=6, fontsize=11)
    ax.set_title(r'$T = %g\ \mathrm{K}$' % t, fontsize=12, pad=2, fontweight='bold')

    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)

    def make_formatter(vmin, vmax, fmt):
        return FuncFormatter(lambda x, _: fmt.format(vmin + x * (vmax - vmin)))

    ax.xaxis.set_major_formatter(make_formatter(damb_min, damb_max, '{:.0f}'))
    ax.yaxis.set_major_formatter(make_formatter(sigma_min, sigma_max, '{:.2f}'))
    ax.zaxis.set_major_formatter(make_formatter(lambda_min, lambda_max, '{:.2f}'))

    ax.tick_params(axis='both', which='major', labelsize=9, pad=1, direction='in')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)

    t_str = str(int(t))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.savefig(os.path.join(out_dir, f'screening_3d_T{t_str}.png'), dpi=300, facecolor='white')
    plt.close()
    print(f'Saved T={t_str}  (Top {TOP_N} marked)')

# Save Top 50 per temperature
top_all = []
for t in temperatures:
    sub = df[df['T'] == t].copy()
    top_all.append(sub.nlargest(TOP_N, 'topsis'))
top_df = pd.concat(top_all, ignore_index=True)
top_df[['IL_smiles', 'cation_smiles', 'anion_smiles', 'T',
        'D+', 'D-', 'Damb', 'sigma', 'lambda', 'topsis']].to_csv(
    os.path.join(out_dir, 'topsis_top50_screening.csv'), index=False)
print(f'Saved top50_screening.csv  ({len(top_df)} candidates)')
