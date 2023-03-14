import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from pathlib import Path
import re

from scipy.stats import gaussian_kde
from .si_formatting import si_format
from .fig_utils import save_fig_formats, human_format, one_hot

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from adjustText import adjust_text
import umap
import pickle
import logomaker

from itertools import chain
from collections import Counter
from functools import partial

aa_alphabet = np.array([
    'A', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'K', 'L', 'M', 'N', 'P', 'Q',
    'R', 'S', 'T', 'V', 'W', 'Y'
])

####### THEME FOR ALL FIGURES #######

_new_black = '#000'
sns.set_theme(style='ticks', font_scale=0.75, rc={
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 9,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'axes.labelpad': 2,
    'axes.linewidth': 0.5,
    'axes.titlepad': 4,
    'lines.linewidth': 0.5,
    'legend.fontsize': 9,
    'legend.title_fontsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.major.size': 2,
    'xtick.major.pad': 1,
    'xtick.major.width': 0.5,
    'ytick.major.size': 2,
    'ytick.major.pad': 1,
    'ytick.major.width': 0.5,
    'xtick.minor.size': 2,
    'xtick.minor.pad': 1,
    'xtick.minor.width': 0.5,
    'ytick.minor.size': 2,
    'ytick.minor.pad': 1,
    'ytick.minor.width': 0.5,

    # Avoid black unless necessary
    'text.color': _new_black,
    'patch.edgecolor': _new_black,
    'patch.force_edgecolor': False, # Seaborn turns on edgecolors for histograms by default and I don't like it
    'hatch.color': _new_black,
    'axes.edgecolor': _new_black,
    # 'axes.titlecolor': _new_black # should fallback to text.color
    'axes.labelcolor': _new_black,
    'xtick.color': _new_black,
    'ytick.color': _new_black

    # Default colormap - personal preference
    # 'image.cmap': 'inferno'
})

####### ---- General functions ---- #######

# def save_fig_formats(fig, figname, fig_outdir, formats=['.png', '.svg', '.pdf']):
#     png_path = str(Path(fig_outdir) / 'PNGs' / (figname+'.png'))
#     svg_path = str(Path(fig_outdir) / 'SVGs' / (figname+'.svg'))
#     pdf_path = str(Path(fig_outdir) / 'PDFs' / (figname+'.pdf'))
    
#     all_paths = [png_path, svg_path, pdf_path]
    
#     for p in all_paths:
#         if any([form in p for form in formats]):
#             Path(p).parent.mkdir(parents=True, exist_ok=True)
#             fig.savefig(p, dpi=300)
            
#     return png_path


####### ---- Figs 1B,C + density kernel and correlation matrix support utils ---- #######

def compute_kernel(x, y):

    remove = (x == 0) & (y == 0)
    x = x[~remove]
    y = y[~remove]

    x_missing = x[y == 0]
    y_missing = y[x == 0]

    remove = ((x == 0) | (y == 0))
    x = np.log2(x[~remove])
    y = np.log2(y[~remove])

    kernel = gaussian_kde(np.vstack([x, y]))
    c = kernel(np.vstack([x, y]))
    return x, y, x_missing, y_missing, kernel, c

def compute_correlation_matrix_pulldown(dff, rep_cols):
    
    df = dff.copy()
    # correlation matrices
    
    cor_mat = np.zeros((3, 3))
    cor_mat[:] = np.nan
    missing_mat = np.copy(cor_mat)

    for i, j in zip(*np.tril_indices(3)):
        x = df[rep_cols[j]]
        y = df[rep_cols[i]]
        both_zero = ((x == 0) & (y == 0))
        x = x[~both_zero]
        y = y[~both_zero]

        remove = ((x == 0) | (y == 0))
        x = np.log2(x[~remove])
        y = np.log2(y[~remove])

        cor_mat[i, j] = np.corrcoef(x, y)[0, 1]
        missing_mat[i, j] = (len(x) / (len(x) + np.sum(remove)))
        
    return cor_mat, missing_mat 

def compute_correlation_matrix_invivo(dff, rep_cols_array): 
    # rep_cols_array should be a 2D array with different animals indexed by rows, and different technical replicates indexed by columns: e.g., [['B1-brain_1_RPM', 'B1-brain_2_RPM'], ['B2-brain_1_RPM', 'B2-brain_2_RPM']]
    
    df = dff.copy()
    
    cor_mat = np.zeros((2, 2))
    cor_mat[:] = np.nan
    missing_mat = np.copy(cor_mat)
    x1, x2 = rep_cols_array[0]
    y1, y2 = rep_cols_array[1]

    for i, j in zip(*np.tril_indices(2, k=-1)):
        x = (df[x1] + df[x2]) / 2
        y = (df[y1] + df[y2]) / 2
        both_zero = ((x == 0) & (y == 0))
        x = x[~both_zero]
        y = y[~both_zero]

        remove = ((x == 0) | (y == 0))
        x = np.log2(x[~remove])
        y = np.log2(y[~remove])

        cor_mat[i, j] = np.corrcoef(x, y)[0, 1]
        missing_mat[i, j] = (len(x) / (len(x) + np.sum(remove)))
    
    return cor_mat, missing_mat

####### Fig 1B #######

def plot_rep_correlations_pulldown(df_pulldown, fig_outdir='figures', figname='fig1B'):

    xlim = [-1.5, 15]
    bins = np.linspace(*xlim, 25)
    
    x_6a = df_pulldown['LY6A_1_RPM']
    y_6a = df_pulldown['LY6A_2_RPM']
    x_6c = df_pulldown['LY6C1_1_RPM']
    y_6c = df_pulldown['LY6C1_2_RPM']
    
    x_6a, y_6a, x_6a_missing, y_6a_missing, kernel_6a, c_6a = compute_kernel(x_6a, y_6a)
    x_6c, y_6c, x_6c_missing, y_6c_missing, kernel_6c, c_6c = compute_kernel(x_6c, y_6c)
    
    # Pulldown assay correlations
    fig = plt.figure(figsize=(1.8, 1), dpi=200)
    gs = fig.add_gridspec(1, 2, wspace=0.07, bottom=0.15, top=0.80, left=0.17, right=0.85)

    for i, x, y, c, x_missing, y_missing, title in zip(
            range(2),
            [x_6a, x_6c],
            [y_6a, y_6c],
            [c_6a, c_6c],
            [x_6a_missing, x_6c_missing],
            [y_6a_missing, y_6c_missing],
            ['LY6A-Fc', 'LY6C1-Fc']
        ):

        gs_sub = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, 
            width_ratios=[1, 5], height_ratios=[5, 1], hspace=0., wspace=0, subplot_spec=gs[0, i])

        ax = fig.add_subplot(gs_sub[0, 1])
        # ax.set_aspect('equal', 'box')
        if i < 2:
            ax.scatter(x, y, c=c, cmap=mpl.cm.inferno, s=0.5, edgecolor='none', rasterized=True)
        else:
            ax.scatter(x, y, c='k', s=0.5, edgecolor='none', rasterized=True)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(xlim)
        ax.text(0.1, 0.97, r'$\rho$ = {:.2f}'.format(np.corrcoef(x, y)[0, 1]),
               transform=ax.transAxes, ha='left', va='top', fontsize=6)

        if len(x) >= 1000:
            label = 'n={}'.format(si_format(len(x), precision=1, format_str='{value}{prefix}'))
        else:
            label = 'n={}'.format(len(x))

        ax.text(0.99, 0.03, label, transform=ax.transAxes, ha='right', va='bottom', fontsize=6)
        ax.set_title(title, pad=3)

        # Left Marginal
        ax = fig.add_subplot(gs_sub[0, 0])
        ax.hist(y_missing, bins=bins, edgecolor='none', orientation='horizontal', density=True, color='r')
        ax.set_ylim(xlim)
        ax.set_xticks([]); 
        ax.text(1., 0.99, 'n={}'.format(si_format(len(y_missing)), precision=1, format_str='{value}{prefix}',), 
                transform=ax.transAxes, ha='right', va='top', fontsize=6, rotation=90, color='r')
        if i == 0 or i == 2:
            ax.set_yticks([0, 5, 10, 15])
            #ax.set_ylabel('Rep 2', labelpad=0)
        else:
            ax.set_yticks([])

        # Bottom Marginal
        ax = fig.add_subplot(gs_sub[1, 1])
        ax.hist(x_missing, bins=bins, edgecolor='none', density=True, color='r')
        ax.set_xlim(xlim)
        ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([])
        ax.text(0.99, 0.8, 'n={}'.format(si_format(len(x_missing)), precision=1, format_str='{value}{prefix}',), 
                transform=ax.transAxes, ha='right', va='top', fontsize=6, color='r')
        # ax.set_xlabel('Rep 1')

        # Missing label
        if i == 0:
            ax = fig.add_subplot(gs_sub[1, 0])
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.8, 0.8, 'Missing', transform=ax.transAxes, color='r', 
                    ha='right', va='top', fontsize=7, clip_on=False)
    
    # Colorbar gridspecs
    gs_density_cbar = fig.add_gridspec(1, 1, left=0.87, right=0.89, top=0.80, bottom=0.26)
    ax = fig.add_subplot(gs_density_cbar[0, 0])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=1.0), 
            cmap=mpl.cm.inferno
        ), 
        cax=ax
    )
    cbar.set_ticks([])
    ax.yaxis.set_label_position('right')
    cbar.set_label('Density', rotation=-90, va='bottom')
    
    #fig.suptitle('Replicate 1 vs. 2')
    
    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path
    
####### Fig 1C #######

def plot_rep_correlations_invivo(df_invivo, fig_outdir='figures', figname='fig1C'):


    xlim = [-1.5, 15]
    bins = np.linspace(*xlim, 25)
    
    x_bc = (df_invivo['B1-brain_1_RPM'] + df_invivo['B1-brain_2_RPM']) / 2
    y_bc = (df_invivo['B2-brain_1_RPM'] + df_invivo['B2-brain_2_RPM']) / 2
    x_c57 = (df_invivo['C1-brain_1_RPM'] + df_invivo['C1-brain_2_RPM']) / 2
    y_c57 = (df_invivo['C2-brain_1_RPM'] + df_invivo['C2-brain_2_RPM']) / 2

    x_bc, y_bc, x_bc_missing, y_bc_missing, kernel_bc, c_bc = compute_kernel(x_bc, y_bc)
    x_c57, y_c57, x_c57_missing, y_c57_missing, kernel_c57, c_c57 =  compute_kernel(x_c57, y_c57)
    
    
    # In vivo assay correlations plot
    fig = plt.figure(figsize=(1.8, 1), dpi=200)
    gs = fig.add_gridspec(1, 2, wspace=0.07, bottom=0.15, top=0.80, left=0.17, right=0.85)
    for i, x, y, c, x_missing, y_missing, title in zip(
            range(2),
             [x_c57, x_bc],
             [y_c57, y_bc],
             [c_c57, c_bc],
             [x_c57_missing, x_bc_missing],
             [y_c57_missing, y_bc_missing],
             ['C57BL/6', 'Balb/C']
        ):

        gs_sub = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, 
            width_ratios=[1, 5], height_ratios=[5, 1], hspace=0., wspace=0, subplot_spec=gs[0, i])

        ax = fig.add_subplot(gs_sub[0, 1])
        ax.scatter(x, y, c='k', s=0.5, edgecolor='none', rasterized=True)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(xlim)
        ax.text(0.1, 0.97, r'$\rho$ = {:.2f}'.format(np.corrcoef(x, y)[0, 1]),
               transform=ax.transAxes, ha='left', va='top', fontsize=6)

        if len(x) >= 1000:
            label = 'n={}'.format(si_format(len(x), precision=1, format_str='{value}{prefix}'))
        else:
            label = 'n={}'.format(len(x))

        ax.text(0.99, 0.03, label, transform=ax.transAxes, ha='right', va='bottom', fontsize=6)
        ax.set_title(title, pad=3)

        # Left Marginal
        ax = fig.add_subplot(gs_sub[0, 0])
        ax.hist(y_missing, bins=bins, edgecolor='none', orientation='horizontal', density=True, color='r')
        ax.set_ylim(xlim)
        ax.set_xticks([]); 
        ax.text(1., 0.99, 'n={}'.format(si_format(len(y_missing)), precision=1, format_str='{value}{prefix}',), 
                transform=ax.transAxes, ha='right', va='top', fontsize=6, rotation=90, color='r')
        if i == 0 or i == 2:
            ax.set_yticks([0, 5, 10, 15])
            #ax.set_ylabel('Rep 2', labelpad=0)
        else:
            ax.set_yticks([])

        # Bottom Marginal
        ax = fig.add_subplot(gs_sub[1, 1])
        ax.hist(x_missing, bins=bins, edgecolor='none', density=True, color='r')
        ax.set_xlim(xlim)
        ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([])
        ax.text(0.99, 0.8, 'n={}'.format(si_format(len(x_missing)), precision=1, format_str='{value}{prefix}',), 
                transform=ax.transAxes, ha='right', va='top', fontsize=6, color='r')
        # ax.set_xlabel('Rep 1')

        # Missing label
        if i == 0:
            ax = fig.add_subplot(gs_sub[1, 0])
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.8, 0.8, 'Missing', transform=ax.transAxes, color='r', 
                    ha='right', va='top', fontsize=7, clip_on=False)
    
        # Colorbar gridspecs
    gs_density_cbar = fig.add_gridspec(1, 1, left=0.87, right=0.89, top=0.80, bottom=0.26)
    ax = fig.add_subplot(gs_density_cbar[0, 0])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=1.0), 
            cmap=mpl.cm.inferno
        ), 
        cax=ax
    )
    cbar.set_ticks([])
    ax.yaxis.set_label_position('right')
    cbar.set_label('Density', rotation=-90, va='bottom')
    
    #fig.suptitle('Animal 1 vs. 2')

    
    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path


####### ---- Figs 1D, 1S1A + density kernel and polygon drawing support utils ---- #######

# Precalculate kernels
def density_xy(_df, x_enr_col, x_mean_RPM_col, y_enr_col, y_mean_RPM_col):
    x = _df[x_enr_col]
    y = _df[y_enr_col]
    
    # Exclude all variants with 1 or less counts in both assays
    remove = (_df[x_mean_RPM_col] == 0) & (_df[y_mean_RPM_col] == 0)
    x = x[~remove]
    y = y[~remove]

    x_missing = x[_df[y_mean_RPM_col] == 0]
    y_missing = y[_df[x_mean_RPM_col] == 0]

    remove = ((_df[x_mean_RPM_col] == 0) | (_df[y_mean_RPM_col] == 0))
    x = x[~remove]
    y = y[~remove]

    n_sample = min([10000, len(x), len(y)])
    kernel = gaussian_kde(np.vstack([
        x.sample(n_sample, random_state=1), 
        y.sample(n_sample, random_state=1)
    ]))
    c = kernel(np.vstack([x, y]))
    
    return x, x_missing, y, y_missing, c


# Point inside convex polygon?
# https://towardsdatascience.com/is-the-point-inside-the-polygon-574b86472119

poly = [
    [-10, 0, 17, -10], # x
    [8,   8, 17,  17], # y
]
left_thresh = [8, 8]
poly_n = len(poly[0])

lines = []
for i in range(poly_n):
    x1 = poly[0][i]
    x2 = poly[0][i + 1 if i < poly_n-1 else 0]
    y1 = poly[1][i]
    y2 = poly[1][i + 1 if i < poly_n-1 else 0]
    lines.append((x1, x2, y1, y2))

def point_inside_polygon(xp, yp):
    res = []
    for x1, x2, y1, y2 in lines:
        res.append((yp - y1) * (x2-x1) - (xp - x1) * (y2 - y1))
    return all([r >= 0 for r in res]) or all([r <= 0 for r in res])

####### Figs 1D, 1S1A #######

def target_vs_fc_scatter(
        x_6a, x_6a_missing, y_6a, y_6a_missing, c_6a,
        x_6c, x_6c_missing, y_6c, y_6c_missing, c_6c,
        fig_outdir='figures',
        figname='fig1D'
    ):

    fig = plt.figure(figsize=(2.5,1.0), dpi=150)
    gs_all = fig.add_gridspec(1, 2, left=0.15, right=0.95, bottom=0.15, top=0.9, wspace=0.5)

    xlim = [-9, 16]
    bins = np.linspace(*xlim, 50)

    left_thresh = [8, 8]

    for i, x, y, c, x_missing, y_missing, ylabel in zip(
            range(2),
            [x_6a, x_6c], [y_6a, y_6c], [c_6a, c_6c],
            [x_6a_missing, x_6c_missing], [y_6a_missing, y_6c_missing],
            ['LY6A-Fc', 'LY6C1-Fc']
        ):

        gs = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, 
            width_ratios=[1, 4], height_ratios=[4, 1], hspace=0., wspace=0, subplot_spec=gs_all[0, i])
        ax = fig.add_subplot(gs[0, 1])

        #ax.set_aspect('equal', 'box')
        ax.scatter(x, y, c=c, cmap=mpl.cm.inferno, s=0.5, edgecolor='none', rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(xlim)
        ax.add_patch(mpl.patches.Polygon(
            np.transpose(np.vstack(poly)),
            facecolor=(75/255, 114/255, 175/255, 0.1), edgecolor='b', linewidth=0.5,
        ))


        # Left Marginal
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(y_missing, bins=bins, edgecolor='none', orientation='horizontal', density=True, color='r')
        ax.set_xticks([]); ax.set_yticks([-5, 0, 5, 10, 15])
        ax.set_ylim(xlim)
        ax.set_ylabel(ylabel, labelpad=-1)

        # Draw left marginal poly
        ax.add_patch(mpl.patches.Rectangle(
            (0, left_thresh[i]), 1, 12, transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
            facecolor='b', alpha=0.1
        ))
        ax.axhline(left_thresh[i], color='b', linewidth=0.5)

        # Bottom Marginal
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(x_missing, bins=bins, edgecolor='none', density=True, color='r')
        ax.set_xticks([-5, 0, 5, 10, 15]); ax.set_yticks([])
        ax.set_xlim(xlim)

        ax.text(0, -0.25, 'Fc control', transform=ax.transAxes, ha='right', va='top', fontsize=7)

        ax = fig.add_subplot(gs[1, 0])
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.8, 0.8, 'Missing', transform=ax.transAxes, color='r', ha='right', va='top', fontsize=7, clip_on=False)

    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path
    
def target_vs_fc_scatter_big(
        x_6a, x_6a_missing, y_6a, y_6a_missing, c_6a,
        x_6c, x_6c_missing, y_6c, y_6c_missing, c_6c,
        title,
        fig_outdir='figures',
        figname='fig1D_big'
    ):

    fig = plt.figure(figsize=(3, 1.25), dpi=150)
    gs_all = fig.add_gridspec(1, 2, left=0.15, right=0.95, bottom=0.15, top=0.9, wspace=0.5)

    xlim = [-9, 17]
    bins = np.linspace(*xlim, 50)

    left_thresh = [8, 8]


    for i, x, y, c, x_missing, y_missing, ylabel in zip(
            range(2),
            [x_6a, x_6c], [y_6a, y_6c], [c_6a, c_6c],
            [x_6a_missing, x_6c_missing], [y_6a_missing, y_6c_missing],
            ['LY6A-Fc', 'LY6C1-Fc']
        ):

        gs = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, 
            width_ratios=[1, 4], height_ratios=[4, 1], hspace=0., wspace=0, subplot_spec=gs_all[0, i])
        ax = fig.add_subplot(gs[0, 1])

        ax.scatter(x, y, c=c, cmap=mpl.cm.inferno, s=0.5, edgecolor='none', rasterized=True)
        
        ax.add_patch(mpl.patches.Polygon(
            np.transpose(np.vstack(poly)),
            facecolor=(75/255, 114/255, 175/255, 0.1), edgecolor='b', linewidth=0.5,
        ))
        
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(xlim)
        ax.text(0.03, 0.97, r'$\rho$ = {:.2f}'.format(np.corrcoef(x, y)[0, 1]),
               transform=ax.transAxes, ha='left', va='top', fontsize=5.5)

        ax.text(0.97, 0.03, 'n={}'.format(si_format(len(x)), precision=2, format_str='{value}{prefix}',), 
                transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5)

        if i == 0:
            ax.set_title(title)

        # Left Marginal
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(y_missing, bins=bins, edgecolor='none', orientation='horizontal', density=True, color='r')
        
        # Draw left marginal poly
        ax.add_patch(mpl.patches.Rectangle(
            (0, left_thresh[i]), 1, 12, transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
            facecolor='b', alpha=0.1
        ))
        ax.axhline(left_thresh[i], color='b', linewidth=0.5)
        
        ax.set_xticks([]); ax.set_yticks([-5, 0, 5, 10, 15])
        ax.set_ylim(xlim)
        ax.text(0.95, 0.95, 'n={}'.format(si_format(len(y_missing)), precision=2, format_str='{value}{prefix}',), 
                transform=ax.transAxes, ha='right', va='top', fontsize=5.5, rotation=90, color='r')
        ax.set_ylabel(ylabel, labelpad=-1)

        # Bottom Marginal
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(x_missing, bins=bins, edgecolor='none', density=True, color='r')
        ax.set_xticks([-5, 0, 5, 10, 15]); ax.set_yticks([])
        ax.set_xlim(xlim)
        ax.text(0.97, 0.85, 'n={}'.format(si_format(len(x_missing)), precision=2, format_str='{value}{prefix}',), 
                transform=ax.transAxes, ha='right', va='top', fontsize=5.5, color='r')

        # Missing label
    #     if i == 0:
        ax = fig.add_subplot(gs[1, 0])
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.8, 0.8, 'Missing', transform=ax.transAxes, color='r', ha='right', va='top', fontsize=7, clip_on=False)

    Path(fig_outdir).mkdir(parents=True, exist_ok=True) 
    filename = str(Path(fig_outdir) / figname)
    
    fig.savefig(filename + '.png', dpi=200)
    fig.savefig(filename + '.pdf', dpi=200)
    fig.savefig(filename + '.svg', dpi=200)

    plt.close()

    Image(filename + '.png')
    
    
####### ---- Fig 1E ---- #######

def assay_enrichment_heatmap(
        df, x_enr_col, x_mean_RPM_col, 
        ly6a_enr_col, ly6a_mean_RPM_col,
        ly6c_enr_col, ly6c_mean_RPM_col,
        AA_col, starter_mean_RPM_col,
        sample_cols,
        fig_outdir='figures',
        figname='fig1E',
        seed=2,
    ):
    
    # Check inside polys

    x = df[x_enr_col]
    y = df[ly6a_enr_col]

    inside_6a = (
        ((df[x_mean_RPM_col] == 0) & (y >= left_thresh[0])) |
        ((df[x_mean_RPM_col] > 0) & pd.Series(list(zip(x, y)), index=x.index).apply(lambda x: point_inside_polygon(*x)))
    )
    print(f'Inside 6A {inside_6a.sum()}')
    
    x = df[x_enr_col]
    y = df[ly6c_enr_col]

    inside_6c = (
        ((df[x_mean_RPM_col] == 0) & (y >= left_thresh[1])) |
        ((df[x_mean_RPM_col] > 0) & pd.Series(list(zip(x, y)), index=x.index).apply(lambda x: point_inside_polygon(*x)))
    )
    print(f'Inside 6C {inside_6c.sum()}')
    
    samples = ['Fc', 'LY6A', 'LY6C1']
    sample_titles = ['Fc', 'LY6A-Fc', 'LY6C1-Fc']

    n_reps = 3

    rep_cols = []
    col_names = []
    for sample, col in zip(samples, sample_cols):
        for rep in range(n_reps):
            rep_cols.append('{}_{}_RPM'.format(col, rep + 1))
            col_names.append('{} {}'.format(sample, rep + 1))

    dff_rpm = (
        df.loc[inside_6a | inside_6c, [AA_col] + rep_cols]
        .copy()
        .rename(columns=dict(zip(rep_cols, col_names)))
    )
    dff = dff_rpm.copy()
    dff.iloc[:, 1:] = np.log2((dff.iloc[:, 1:] + 0.01).divide(
        (df.loc[inside_6a | inside_6c, starter_mean_RPM_col]) + 0.01, axis=0))
    # dff.head()
    
    kmeans = KMeans(n_clusters=2, random_state=seed).fit(dff.iloc[:, 1:])

    dff['cluster'] = kmeans.labels_
    dff['LY6A'] = dff[['LY6A {}'.format(i + 1) for i in range(n_reps)]].mean(axis=1)
    dff['LY6C1'] = dff[['LY6C1 {}'.format(i + 1) for i in range(n_reps)]].mean(axis=1)
    cluster_cols = ['LY6A', 'LY6C1']
    dff['cluster_val'] = np.nan
    for i, row in dff.iterrows():
        dff.loc[i, 'cluster_val'] = row[cluster_cols[int(row['cluster'])]]
    dff = dff.sort_values(['cluster', 'cluster_val'], ascending=[True, False])
    
    # dff.head()
    
    # Mask missing
    dff_masked = dff.copy()
    dff_masked[dff_rpm.iloc[:, 1:] == 0] = np.nan
    mat = (
        dff_masked
        .drop(columns=['AA_sequence', 'cluster', 'cluster_val', 'LY6A', 'LY6C1'])
        .values
    )

    fig = plt.figure(figsize=(6, 0.7), dpi=200)
    gs = fig.add_gridspec(1, 1, left=0.1, right=0.94, top=0.8, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0])
    
    norm = mpl.colors.Normalize(vmin=-5, vmax=15)
    cmap = mpl.colors.ListedColormap(np.vstack([mpl.cm.bwr(np.linspace(0.5 * (2/3), 0.5, 50)), mpl.cm.bwr(np.linspace(0.5, 1.0, 150))]))

    ax.set_facecolor('#CCC')
    ax.imshow(
        mat.T, aspect='auto', interpolation='none',
        norm=norm, cmap=cmap
    )


    ax.set_yticks((np.arange(0, 3) * 3) + 1)
    ax.set_yticklabels(sample_titles, rotation=0, ha='right', va='center', fontsize=9)

    ax.set_xticks([])
    ax.set_xticklabels([])


    gs_cbar = fig.add_gridspec(1, 1, left=0.945, right=0.955, top=0.8, bottom=0.07)
    ax = fig.add_subplot(gs_cbar[0, 0])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cbar.set_ticks([-5, 0, 5, 10, 15])

    ax.text(1, 1.05, 'Log2 Enrichment', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)

    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path, dff, dff_rpm, dff_masked
    
    
####### ---- Figs 1F,G, 1S3A,B + UMAP support utils ---- #######

def prep_umap(dff):
    # Collapse by AA, average by non-log transformed enr, then re-log transform

    dffc = dff[['AA_sequence', 'cluster']].copy()

    dffc['LY6A'] = (2 ** dff[['LY6A 1', 'LY6A 2', 'LY6A 3']]).mean(axis=1)
    dffc['LY6C1'] = (2 ** dff[['LY6C1 1', 'LY6C1 2', 'LY6C1 3']]).mean(axis=1)
    dffc['Fc'] = (2 ** dff[['Fc 1', 'Fc 2', 'Fc 3']]).mean(axis=1)

    dffc = dffc.groupby(['AA_sequence', 'cluster'], as_index=False).agg(np.mean)
    dffc[['LY6A', 'LY6C1', 'Fc']] = dffc[['LY6A', 'LY6C1', 'Fc']].applymap(np.log2)

    return dffc

def fit_umap(l1_dff, l2_dff, umap_outdir='UMAPs'):
    
    l1_dffc = prep_umap(l1_dff)
    l2_dffc = prep_umap(l2_dff)
    
    ly6a = pd.concat([
    l1_dffc.loc[l1_dffc['cluster'] == 0].assign(dataset='library1'),
    l2_dffc.loc[l2_dffc['cluster'] == 0].assign(dataset='library2'),
        ], axis=0, ignore_index=True)
    
    ly6c1 = pd.concat([
    l1_dffc.loc[l1_dffc['cluster'] == 1].assign(dataset='library1'),
    l2_dffc.loc[l2_dffc['cluster'] == 1].assign(dataset='library2'),
        ], axis=0, ignore_index=True)
    
    seq_one_hot_6a = np.vstack(ly6a['AA_sequence'].apply(one_hot).values)
    seq_one_hot_6c1 = np.vstack(ly6c1['AA_sequence'].apply(one_hot).values)
    
    trans_6a = umap.UMAP(
        n_neighbors=200,
        min_dist=0.15,
        n_components=2,
        metric='euclidean',
        random_state=1,
        low_memory=True
    ).fit(seq_one_hot_6a)

    trans_6c1 = umap.UMAP(
        n_neighbors=200,
        min_dist=0.15,
        n_components=2,
        metric='euclidean',
        random_state=1,
        low_memory=True
    ).fit(seq_one_hot_6c1)
    
    Path(umap_outdir).mkdir(parents=True, exist_ok=True)
    
    with open(Path(umap_outdir) / 'LY6A_umap_l1_l2.pickle', 'wb') as fp:
        pickle.dump(trans_6a, fp)
    
    with open(Path(umap_outdir) / 'LY6C1_umap_l1_l2.pickle', 'wb') as fp:
        pickle.dump(trans_6c1, fp)

    ly6a['x1'] = trans_6a.embedding_[:, 0]
    ly6a['x2'] = trans_6a.embedding_[:, 1]

    ly6c1['x1'] = trans_6c1.embedding_[:, 0]
    ly6c1['x2'] = trans_6c1.embedding_[:, 1]
    
    ly6a.to_csv(Path(umap_outdir) / 'LY6A_joint_umap_l1_l2.csv', index=False)
    ly6c1.to_csv(Path(umap_outdir) / 'LY6C1_joint_umap_l1_l2.csv', index=False)
    
    return ly6a, ly6c1

####### Fig 1S3A #######

def plot_joint_umap(ly6a_umap, ly6c1_umap, fig_outdir='figures', figname='fig1S3A'):
    
    fig = plt.figure(figsize=(5.5, 2), dpi=200)
    gs = fig.add_gridspec(1, 2, wspace=0.15, hspace=0.3, left=0.05, right=0.75, top=0.9, bottom=0.1)

    for i, umap_df, quant_col, title, cmap, eps, min_samples in zip(
            range(2), 
            [ly6a_umap, ly6c1_umap],
            ['LY6A', 'LY6C1'],
            ['LY6A-Fc', 'LY6C1-Fc'],
            [sns.color_palette("crest", as_cmap=True), sns.color_palette("flare", as_cmap=True)],
            [0.17, 0.17], [20, 20]
        ):

        ax = fig.add_subplot(gs[0, i])

        sc = ax.scatter(umap_df['x1'], umap_df['x2'], c=umap_df['dataset'].map({'library1': 'r', 'library2': 'b'}),
                   s=0.5, edgecolor='none', rasterized=True, vmin=5, vmax=15)

        ax.set_xticks([]); ax.set_yticks([])

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        ax.set_title(title, loc='left')

        if i == 1:
            ax.legend(
                [
                    mpl.lines.Line2D((0,), (0,), marker='o', markerfacecolor='r', markeredgecolor='none', linewidth=0),
                    mpl.lines.Line2D((0,), (0,), marker='o', markerfacecolor='b', markeredgecolor='none', linewidth=0)
                ],
                ['Library 1'.format((umap_df['dataset']=='library1').sum()), 
                 'Library 2'.format((umap_df['dataset']=='library2').sum())],
                bbox_to_anchor=(1, 1), loc='upper left', frameon=False,
                handlelength=1, handletextpad=0.5,
            )
    
    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path
    
def make_umap_GM_clusters(ly6a_umap, ly6c1_umap, umap_outdir='UMAPs'):
    
    for i, umap_df, quant_col, title, cmap, n_clusters, random_state, cluster_prefix in zip(
            range(2), 
            [ly6a_umap, ly6c1_umap],
            ['LY6A', 'LY6C1'],
            ['LY6A-Fc', 'LY6C1-Fc'],
            [sns.color_palette("crest", as_cmap=True), sns.color_palette("flare", as_cmap=True)],
            [40, 40], [1, 1], ['A', 'C']
        ):

        gm = GaussianMixture(n_components=n_clusters, random_state=random_state, 
                             n_init=10, max_iter=1000)
        clustering = gm.fit_predict(umap_df[['x1', 'x2']].values)

        if i == 0:
            gm_ly6a = gm
        else:
            gm_ly6c1 = gm

        umap_df['cluster'] = clustering
     
    with open(Path(umap_outdir) / 'LY6A_gm_l1_l2.pickle', 'wb') as fp:
        pickle.dump(gm_ly6a, fp)
    
    with open(Path(umap_outdir) / 'LY6C1_gm_l1_l2.pickle', 'wb') as fp:
        pickle.dump(gm_ly6c1, fp)
    
    return gm_ly6a, gm_ly6c1

####### Figs 1F,G #######

def plot_umap_clusters_fewerlabels(ly6a_umap, ly6c1_umap, gm_ly6a, gm_ly6c1, umap_outdir='UMAPs', fig_outdir='figures', figname='fig1FG'):
    
    fig = plt.figure(figsize=(1.5, 2.5), dpi=200)
    gs = fig.add_gridspec(2, 1, wspace=0.03, hspace=0.1, 
                          left=0.13, right=0.95, top=0.92, bottom=0.05)
    
    new_dfs = []
    
    for i, umap_df, gm, cluster_colors, quant_col, title, cluster_prefix, cluster_inds in zip(
            range(2), [ly6a_umap, ly6c1_umap], [gm_ly6a, gm_ly6c1], [cluster_colors_6a, cluster_colors_6c],
            ['LY6A', 'LY6C1'], ['LY6A-Fc', 'LY6C1-Fc'], ['A', 'C'],
            [
                [
                    13, # ****PFR
                    18, # ***RPF*
                    21, # **F*PP*
                    4,  # ***F**V
                ],
                [
                    19, # ****GRW
                    31, # ***GS[VI]Y
                    20, # ***G[YF]AQ
                    4,  # ****GSS
                ]
            ]
        ):

        ax = fig.add_subplot(gs[i, 0]) 

        clustering = gm.predict(umap_df[['x1', 'x2']].values)


        umap_df['cluster'] = clustering

        umap_cluster_df = (
            umap_df
            .groupby('cluster')
            .agg(
                n=('AA_sequence', len),
                Ly6A=('LY6A', np.mean), Ly6C=('LY6C1', np.mean), Fc=('Fc', np.mean),
                x1=('x1', np.mean), x2=('x2', np.mean),
            )

        )

        sc = ax.scatter(umap_df['x1'], umap_df['x2'], 
                   c=umap_df['cluster'].apply(lambda x: cluster_colors[x]),
                   s=0.3, edgecolor='none', rasterized=True)


        cluster_labels = []

        for k in cluster_inds:
            cluster_labels.append(
                ax.text(
                    gm.means_[k-1][0], gm.means_[k-1][1],
                    f'{cluster_prefix}{k}', color=cluster_colors[k-1], fontsize=7))
        
        ax.set_xticks([]); ax.set_yticks([])
        if i == 1:
            ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        n_clusters = len(umap_cluster_df)
        ax.text(0.01, 1.01, f'k = {n_clusters} clusters', 
                transform=ax.transAxes, ha='left', va='bottom', fontsize=5)
        
        new_dfs.append(umap_df)
    
    Path(umap_outdir).mkdir(parents=True, exist_ok=True)
        
    with open(Path(umap_outdir) / 'LY6A_cluster_colors_l1_l2.pickle', 'wb') as fp:
        pickle.dump(cluster_colors_6a, fp)
    
    with open(Path(umap_outdir) / 'LY6C1_cluster_colors_l1_l2.pickle', 'wb') as fp:
        pickle.dump(cluster_colors_6c, fp)
    

    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path, new_dfs
    

# TAB 20 modified
cluster_colors = ((0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 (1.0, 0.4980392156862745, 0.054901960784313725),
 (1.0, 0.7333333333333333, 0.47058823529411764),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 (1.0, 0.596078431372549, 0.5882352941176471),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
 # (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
 # (0.7803921568627451, 0.7803921568627451, 0.7803921568627451), # GREY
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
 (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
 (0.6196078431372549, 0.8549019607843137, 0.8980392156862745))

cluster_colors = list(cluster_colors)

cluster_colors = cluster_colors * 3

# Shuffle colors
# Good seeds: 12
rng = np.random.RandomState(12)
rng.shuffle(cluster_colors)

cluster_colors_6a = [
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # 1
 (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
 (1.0, 0.7333333333333333, 0.47058823529411764),
 (1.0, 0.596078431372549, 0.5882352941176471),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), # 5
 (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
 (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
 (1.0, 0.7333333333333333, 0.47058823529411764), # 10
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
 (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), # 15
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), 
 (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), # 20
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 (0.8588235294117647, 0.8588235294117647, 0.5529411764705883), # 25
 (1.0, 0.4980392156862745, 0.054901960784313725),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
 (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
 (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
 (0.6196078431372549, 0.8549019607843137, 0.8980392156862745), # 30
 (1.0, 0.4980392156862745, 0.054901960784313725),
 (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), # 35
 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 (1.0, 0.596078431372549, 0.5882352941176471),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # 40
 (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
 (1.0, 0.596078431372549, 0.5882352941176471),
 (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
 (0.596078431372549, 0.8745098039215686, 0.5411764705882353), # 45
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (1.0, 0.7333333333333333, 0.47058823529411764),
 (1.0, 0.4980392156862745, 0.054901960784313725), # 50
 (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
 (0.7686274509803922, 0.611764705882353, 0.5803921568627451)
]

cluster_colors_6c = [
 (1.0, 0.7333333333333333, 0.47058823529411764),# 1
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), # 5
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
(0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
 (1.0, 0.4980392156862745, 0.054901960784313725),
 (0.596078431372549, 0.8745098039215686, 0.5411764705882353), # 10
 (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
 (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), 
 (1.0, 0.7333333333333333, 0.47058823529411764),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529), # 15
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (1.0, 0.596078431372549, 0.5882352941176471),
 (0.7725490196078432, 0.6901960784313725, 0.8352941176470589), # 20
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # 25
 (1.0, 0.4980392156862745, 0.054901960784313725),
 (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
 (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 (1.0, 0.596078431372549, 0.5882352941176471), # 30
 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
 (1.0, 0.4980392156862745, 0.054901960784313725),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353), # 35
 (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
 (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
 (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # 40
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
 (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
 (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
 (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
 (0.7686274509803922, 0.611764705882353, 0.5803921568627451), # 45
 (1.0, 0.7333333333333333, 0.47058823529411764),
 (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
 (1.0, 0.596078431372549, 0.5882352941176471),
 (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
 (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # 50
 (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
]


####### ---- Fig 1H,I + motifs support utils ---- #######

def collapse_motif(seqs, thresh=0.5):
    n_total = len(seqs)
    
    motif_seq = []
    
    for i in range(7):
        char_count = seqs.str.slice(i, i+1).value_counts().sort_values(ascending=False)
        top_frac = char_count[0] / n_total
        if top_frac > thresh:
            motif_seq.append(char_count.index[0])
        else:
            motif_seq.append('*')
    
    return ''.join(motif_seq)

def get_local_aa_counts(dff):
    # Get receptor-specific AA distribution?
    local_aa_counts = Counter(chain.from_iterable(dff['AA_sequence'].apply(list).values))
    local_aa_counts = {k: v / (len(dff) * 7) for k, v in local_aa_counts.most_common()}
    return local_aa_counts

def collapse_motif_flexible(seqs, local_aa_counts={}, global_thresh=4, local_thresh=0.4):
    n_total = len(seqs)
    
    motif_seq = []
    
    for i in range(7):
        candidates = []
        char_count = seqs.str.slice(i, i+1).value_counts()
        for aa, count in char_count.iteritems():
            if (count / n_total) > (global_thresh * local_aa_counts[aa]) or (count / n_total) > local_thresh:
                candidates.append(aa)
        motif_seq.append(candidates)
    
    return motif_seq

def assign_motifs(umap_df):
    
    local_aa_counts = get_local_aa_counts(umap_df)
    
    motif_df = (
        umap_df
        .loc[umap_df['cluster'] >= 0]
        .groupby('cluster')
        .agg(
            n=('AA_sequence', len),
            motif_seq=('AA_sequence', partial(collapse_motif, thresh=0.4)),
            flex_motif_seq=('AA_sequence', partial(
                collapse_motif_flexible, local_aa_counts=local_aa_counts, global_thresh=100, local_thresh=0.4
            )),
            LY6A=('LY6A', np.mean), LY6C1=('LY6C1', np.mean), Fc=('Fc', np.mean),
            x1=('x1', np.mean), x2=('x2', np.mean),
        )
        .sort_values('n', ascending=False)
    )

    return motif_df     

####### Figs 1H,I #######

def indiv_motif_plot(dff_masked, umap_df_6a, umap_df_6c, motif_df_6a, motif_df_6c, fig_outdir='figures', figname='fig1HI'):
    
    # Aggregate by AA sequence
    dff_masked_agg = (
        dff_masked
        .drop(columns=['cluster', 'LY6A', 'LY6C1', 'cluster_val'])
        .set_index('AA_sequence').applymap(lambda x: 2 ** x).reset_index() # need to exclude AA sequence column
        .groupby('AA_sequence') 
        .agg(np.mean)
        .applymap(np.log2)
    )
    
    
    fig = plt.figure(figsize=(4, 2.5), dpi=150)
    left = 0.15
    right = 0.87
    bottom = 0.1
    top = 0.95
    cbar_width = 0.015
    cbar_pad = 0.025
    cbar_vpad = 0.15
    gs = fig.add_gridspec(2, 4, wspace=1.2, hspace=0.25, left=left, right=right, bottom=bottom, top=top)
    gs_cbar = fig.add_gridspec(1, 1, wspace=1.2, hspace=0.25, left=right + cbar_pad, right=right + cbar_pad + cbar_width, 
                               bottom=bottom + cbar_vpad, top=top - cbar_vpad)

    n_seqs = 10
    
    cluster_inds = [
        [
            13, # ****PFR
            18, # ***RPF*
            21, # **F*PP*
            4,  # ***F**V
        ],
        [
            19, # ****GRW
            31, # ***GS[VI]Y
            20, # ***G[YF]AQ
            4,  # ****GSS
        ]
    ]
    
    consensus_motifs = [
        [
            [[], [], [], [], ['P'], ['F'], ['R']],
            [[], [], [], ['R'], ['P'], ['F'], []],
            [[], [], ['F'], [], ['P'], ['P'], []],
            [[], [], ['R', 'K'], ['F'], [], [], ['V']],
        ],
        [
            [[], [], [], [], ['G'], ['R'], ['W']],
            [[], [], [], ['G'], ['S'], ['V', 'I'], ['Y']],
            [[], [], [], ['G'], ['Y', 'F'], ['A'], ['Q']],
            [[], [], [], [], ['G'], ['S'], ['S']],
        ],
    ]

    plot_data = (
        (
            umap_df_6a, motif_df_6a, 'LY6A',
            cluster_inds[0], consensus_motifs[0]
        ),
        (
            umap_df_6c, motif_df_6c, 'LY6C1',
            cluster_inds[1], consensus_motifs[1]
        )
    )

    for k in range(2):
        umap_df = plot_data[k][0]
        motif_df = plot_data[k][1]
        col = plot_data[k][2]
        for i, cluster_ind in zip(range(4), plot_data[k][3]):
            ax = fig.add_subplot(gs[k, i])
            ax.set_facecolor('#CCC')

            cluster_df = (
                umap_df
                .loc[umap_df['cluster'] == cluster_ind - 1]
                # .sort_values(col, ascending=False)
                # .head(n_seqs)
            )

            # Filter out those that don't match the consensus-ish motif
            # consensus_motif = motif_df.at[cluster_ind - 1, 'flex_motif_seq']
            consensus_motif = plot_data[k][4][i]
            # print(consensus_motif)

            valid_cluster_seq = (
                umap_df['AA_sequence']
                .apply(list)
                .apply(lambda x: all([
                    (len(consensus_motif[l]) == 0) or (_x in consensus_motif[l])
                    for l, _x in enumerate(x)
                ]))
            )

            cluster_df = (
                cluster_df
                .loc[valid_cluster_seq]
                .sort_values(col, ascending=False)
                .head(n_seqs)
            )

            mat = (
                cluster_df
                .drop(columns=['dataset', 'LY6A', 'LY6C1', 'Fc', 'cluster', 'x1', 'x2'])
                .join(dff_masked_agg, on='AA_sequence', how='left')
                .head(n_seqs)
            ).iloc[:, 1:]
            

            norm = mpl.colors.Normalize(vmin=-15, vmax=15)
            cmap = mpl.cm.bwr

            im = ax.imshow(
                mat, aspect='auto', interpolation='none',
                norm=norm, cmap=cmap
            )

            # ax.set_xticks(np.arange(0, 9, 3) + 1)
            ax.set_xticks([])
            ax.tick_params(axis='x', length=0.5, pad=0.5)
            
            cluster_name = motif_df.at[cluster_ind - 1, 'motif_seq']

            ax.set_yticks(np.arange(0, n_seqs))
            #ax.set_yticklabels(cluster_df['AA_sequence'], fontsize=6, fontfamily='monospace')
            ax.set_yticklabels([''] * len(cluster_df))
            for sy, s in enumerate(cluster_df['AA_sequence'].values):
                for l, c in enumerate(list(s)):                
                    ax.text(0.0 - ((7-l)*0.14), sy, c,
                            fontsize=6, fontfamily='monospace',
                            color='#000' if c == cluster_name[l] else '#999', 
                            #fontweight='normal' if c == cluster_name[l] else 'light', 
                           transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData), 
                            ha='right', va='center')


            motif_x = -0.30
            motif_y = -0.05
            motif_gap = 0.25
            
            ax.text(motif_x - 0.1, motif_y, '{}{}:'.format('A' if k == 0 else 'C', cluster_ind), 
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)
            ax.text(motif_x, motif_y, cluster_name, color='#444',
                    transform=ax.transAxes, ha='left', va='top', fontfamily='monospace', fontsize=8)
    
    ax = fig.add_subplot(gs_cbar[0, 0])
    fig.colorbar(im, ticks=[-15, -10, -5, 0, 5, 10, 15], cax=ax)
    ax.set_ylabel('Log2 Enrichment')


    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path


####### ---- Fig 1S2 + replicability support utils ---- #######

def calculate_replicability(df, assays=['starter_virus', 'LY6A', 'LY6C1', 'Fc'], rep_pattern='_[1-3]_RPM'):
    
    plot_data = {}

    for k, assay in enumerate(assays):
        rep_cols = [col for col in df.columns if re.search(assay + rep_pattern, col)]
        #print(assay, rep_cols)

        for i, j in zip(*np.tril_indices(len(rep_cols), k=-1)):
            #print(i, j)

            x = df[rep_cols[j]]
            y = df[rep_cols[i]]
            remove = (x == 0) & (y == 0)
            x = x[~remove]
            y = y[~remove]

            x_missing = x[y == 0]
            y_missing = y[x == 0]

            remove = ((x == 0) | (y == 0))
            x = np.log2(x[~remove])
            y = np.log2(y[~remove])

            n_sample = min([10000, len(x), len(y)])

            kernel = gaussian_kde(np.vstack([
                x.sample(n_sample, random_state=1), y.sample(n_sample, random_state=1)
            ]))
            c = kernel(np.vstack([x, y]))

            plot_data[(assay, i, j)] = {}
            plot_data[(assay, i, j)]['x'] = x
            plot_data[(assay, i, j)]['y'] = y
            plot_data[(assay, i, j)]['x_missing'] = x_missing
            plot_data[(assay, i, j)]['y_missing'] = y_missing
            plot_data[(assay, i, j)]['c'] = c            
            
    return plot_data

def plot_replicability(df, plot_data, assays, assay_titles, rep_pattern='_[1-3]_RPM'):
    
    ncol = 4
    
    xlims = [ [-2, 17] for i in range(ncol)]

    
    fig = plt.figure(figsize=(2*ncol, 1.8), dpi=200)
    gs = mpl.gridspec.GridSpec(1, ncol, figure=fig, wspace=0.3, 
                              bottom=0.2, top=0.9, left=0.1*(6/(2*ncol)), right=1-0.05*(6/(2*ncol)))

    for k, assay in enumerate(assays):
        rep_cols = [col for col in df.columns if re.match(assay + rep_pattern, col)]
        # print(assay, rep_cols)

        gsa = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[k // ncol, k % ncol], hspace=0.075, wspace=0.075)

        for i, j in zip(*np.tril_indices(3, k=-1)):
            gsaa = mpl.gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=gsa[i-1, j],
                width_ratios=[1, 6], height_ratios=[6, 1], hspace=0., wspace=0
            )

            x = plot_data[(assay, i, j)]['x']
            y = plot_data[(assay, i, j)]['y']
            x_missing = plot_data[(assay, i, j)]['x_missing']
            y_missing = plot_data[(assay, i, j)]['y_missing']
            c = plot_data[(assay, i, j)]['c']

            ax = fig.add_subplot(gsaa[0, 1])
            # ax.set_aspect('equal', 'box')
            ax.scatter(x, y, c=c, cmap=mpl.cm.inferno, s=0.5, edgecolor='none', rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            xlim = xlims[k]
            bins = np.linspace(*xlim, 30)
            ax.set_xlim(xlim); ax.set_ylim(xlim)

            if i == 1:
                ax.set_title(assay_titles[k])

            ax.text(0.03, 0.97, r'$\rho$ = {:.3f}'.format(np.corrcoef(x, y)[0, 1]),
                   transform=ax.transAxes, ha='left', va='top', fontsize=5.5)

            ax.text(0.97, 0.03, 'n={}'.format(si_format(len(x)), precision=2, format_str='{value}{prefix}',), 
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5)

            # Left Marginal
            ax = fig.add_subplot(gsaa[0, 0])
            ax.hist(y_missing, bins=bins, edgecolor='none', orientation='horizontal', density=True, color='r')

            ax.set_xticks([])
            if j == 0:
                ax.set_yticks([0, 5, 10, 15])
                ax.set_ylabel('Rep {}'.format(i+1))
            else:
                ax.set_yticks([])

            ax.set_ylim(xlim)

            ax.text(0.97, 0.97, 'n={}'.format(si_format(len(y_missing)), precision=2, format_str='{value}{prefix}',), 
                    transform=ax.transAxes, ha='right', va='top', fontsize=5.5, rotation=90, color='r')

            # Bottom Marginal
            ax = fig.add_subplot(gsaa[1, 1])
            ax.hist(x_missing, bins=bins, edgecolor='none', density=True, color='r')

            if i == 2:
                ax.set_xticks([0, 5, 10, 15])
                ax.set_xlabel('Rep {}'.format(j+1))
            else:
                ax.set_xticks([])

            ax.set_xlim(xlim)

            ax.set_yticks([])
            ax.text(0.97, 0.8, 'n={}'.format(si_format(len(x_missing)), precision=2, format_str='{value}{prefix}',), 
                    transform=ax.transAxes, ha='right', va='top', fontsize=5.5, color='r')

            # Missing label
            ax = fig.add_subplot(gsaa[1, 0])
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])

            if i == 2 and j == 0:
                ax.text(0.8, 0.8, 'Missing', transform=ax.transAxes, color='r', ha='right', va='top', fontsize=7, clip_on=False)

                
        # Colorbar gridspecs
        gs_density_cbar = mpl.gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gsa[0, 1], width_ratios=[1, 10])
        ax = fig.add_subplot(gs_density_cbar[0, 0])
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=0, vmax=1.0), 
                cmap=mpl.cm.inferno
            ), 
            cax=ax
        )
        cbar.set_ticks([])
        ax.yaxis.set_label_position('right')
        cbar.set_label('Density', rotation=90, va='bottom', labelpad=8)

    return fig
        
####### Figs 1S2A-D, E-H #######

def plot_r1_replicability_pulldown(df, fig_outdir='figures', figname='fig1S2ABCD'):
    
    assays = ['starter_virus', 'LY6A', 'LY6C1', 'Fc']
    assay_titles = ['Virus', 'LY6A-Fc', 'LY6C1-Fc', 'Fc control']
    
    plot_data = calculate_replicability(df, assays=assays)

    fig = plot_replicability(df, plot_data, assays, assay_titles)
    
    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path
    
####### Figs 1S2I,J #######

def plot_r1_replicability_invivo(df, fig_outdir='figures', figname='fig1S2IJ'): 
    
    assays = ['B1-brain', 'B2-brain', 'C1-brain', 'C2-brain']
    assay_titles = ['BALB/c animal 1', 'BALB/c animal 2', 'C57BL/6 animal 1', 'C57BL/6 animal 2']
    
    plot_data_invivo = calculate_replicability(df, assays=assays)

    xlims = [ [-2, 17] for i in range(len(assays))]

    fig = plt.figure(figsize=(5, 1.25), dpi=150)
    gs = mpl.gridspec.GridSpec(1, 4, figure=fig, wspace=0.25, hspace=0.5,
                              bottom=0.25, top=0.85, left=0.1, right=0.95)

    for k, assay in enumerate(assays):
        rep_cols = [col for col in df.columns if re.match(assay + '_[1-3]_RPM', col)]

        gsa = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, k], hspace=0.1, wspace=0.1, width_ratios=[12, 1])

        for i, j in zip(*np.tril_indices(2, k=-1)):
            gsaa = mpl.gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=gsa[i-1, j],
                width_ratios=[1, 6], height_ratios=[6, 1], hspace=0., wspace=0
            )

            x = plot_data_invivo[(assay, i, j)]['x']
            y = plot_data_invivo[(assay, i, j)]['y']
            x_missing = plot_data_invivo[(assay, i, j)]['x_missing']
            y_missing = plot_data_invivo[(assay, i, j)]['y_missing']
            c = plot_data_invivo[(assay, i, j)]['c']

            ax = fig.add_subplot(gsaa[0, 1])
            ax.scatter(x, y, c=c, cmap=mpl.cm.inferno, s=0.5, edgecolor='none', rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            xlim = xlims[k]
            bins = np.linspace(*xlim, 25)
            ax.set_xlim(xlim); ax.set_ylim(xlim)

            if i == 1:
                ax.set_title(assay_titles[k])

            ax.text(0.03, 0.97, r'$\rho$ = {:.3f}'.format(np.corrcoef(x, y)[0, 1]),
                   transform=ax.transAxes, ha='left', va='top', fontsize=5.5)

            ax.text(0.97, 0.03, 'n={}'.format(si_format(len(x)), precision=2, format_str='{value}{prefix}',), 
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5)

            # Left Marginal
            ax = fig.add_subplot(gsaa[0, 0])
            ax.hist(y_missing, bins=bins, edgecolor='none', orientation='horizontal', density=True, color='r')

            ax.set_xticks([])
            if j == 0:
                ax.set_yticks([0, 5, 10, 15])
                if k == 0:
                    ax.set_ylabel('Rep {}'.format(i+1))
            else:
                ax.set_yticks([])

            ax.set_ylim(xlim)

            ax.text(0.97, 0.97, 'n={}'.format(si_format(len(y_missing)), precision=2, format_str='{value}{prefix}',), 
                    transform=ax.transAxes, ha='right', va='top', fontsize=5.5, rotation=90, color='r')

            # Bottom Marginal
            ax = fig.add_subplot(gsaa[1, 1])
            ax.hist(x_missing, bins=bins, edgecolor='none', density=True, color='r')

            ax.set_xticks([0, 5, 10, 15])
            ax.set_xlabel('Rep {}'.format(j+1))

            ax.set_xlim(xlim)

            ax.set_yticks([])
            ax.text(0.97, 0.8, 'n={}'.format(si_format(len(x_missing)), precision=2, format_str='{value}{prefix}',), 
                    transform=ax.transAxes, ha='right', va='top', fontsize=5.5, color='r')

            # Missing label
            ax = fig.add_subplot(gsaa[1, 0])
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])

            if k == 0:
                ax.text(0.8, 0.8, 'Missing', transform=ax.transAxes, color='r', 
                        ha='right', va='top', fontsize=7, clip_on=False)
    
       # Colorbar gridspecs
        if k == 1 or k == 3:
            ax = fig.add_subplot(gsa[0, 1])
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(vmin=0, vmax=1.0), 
                    cmap=mpl.cm.inferno
                ), 
                cax=ax
            )
            cbar.set_ticks([])
            ax.yaxis.set_label_position('right')
            cbar.set_label('Density', rotation=90, va='bottom', labelpad=8)
    
    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path 
    

####### ---- Fig 1S3B ---- #######

def plot_umap_clusters(ly6a_umap, ly6c1_umap, fig_outdir='figures', figname='fig1S3B'):
    
    fig = plt.figure(figsize=(5.5, 2), dpi=200)
    gs = fig.add_gridspec(1, 2, wspace=0.2, hspace=0.3, left=0.05, right=0.75, top=0.9, bottom=0.1)

    for i, umap_df, quant_col, title, n_clusters, random_state, cluster_prefix, cluster_colors in zip(
            range(2), 
            [ly6a_umap, ly6c1_umap], ['LY6A', 'LY6C1'], ['LY6A-Fc', 'LY6C1-Fc'],
            [40, 40], [1, 1], ['A', 'C'], [cluster_colors_6a, cluster_colors_6c]
        ):

        ax = fig.add_subplot(gs[0, i])

        sc = ax.scatter(umap_df['x1'], umap_df['x2'], 
                   c=umap_df['cluster'].apply(lambda x: cluster_colors[x] if x >= 0 else (0.7, 0.7, 0.7, 0.6)),
                   s=0.3, edgecolor='none', rasterized=True)

        ax.set_xticks([]); ax.set_yticks([])

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        ax.set_title(title, loc='left')

        ax.text(0.98, 0.01, 'k={}'.format(n_clusters), 
               transform=ax.transAxes, ha='right', va='bottom', fontsize=5)

        cluster_df = umap_df[['cluster', 'x1', 'x2']].groupby('cluster').agg(np.mean)
        cluster_labels = []
        for kk, (k, row) in enumerate(cluster_df.iterrows()):
            if k == -1:
                continue

            cluster_labels.append(
                ax.text(row['x1'], row['x2'], '{}{}'.format(cluster_prefix, k + 1), 
                        # color=cluster_colors[k] if k >= 0 else (0.7, 0.7, 0.7, 0.6), 
                        color='k',
                        fontsize=4)
            )

    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path
    
    
####### ---- Fig 1S3C + logos support utils ---- #######

# // ClustalX AA Colors
# // (by properties + conservation)
# // http://www.jalview.org/help/html/colourSchemes/clustal.html
clustalXAAColors = {
#   // Hydrophobic (Blue)
    'A': '#809df0',
    'I': '#809df0',
    'L': '#809df0',
    'M': '#809df0',
    'F': '#809df0',
    'W': '#809df0',
    'V': '#809df0',
    #   // Positive charge (Red)
    'K': '#ed000a',
    'R': '#ed000a',
    #   // Negative charge (Magenta)
    'D': '#be38bf',
    'E': '#be38bf',
    #   // Polar (Green)
    'N': '#29c417',
    'Q': '#29c417',
    'S': '#29c417',
    'T': '#29c417',
    #   // Cysteins (Pink)
    'C': '#ee7d80',
    #   // Glycines (Orange)
    'G': '#ef8f48',
    #   // Prolines (Yellow)
    'P': '#c1c204',
    #   // Aromatics (Cyan)
    'H': '#23a6a4',
    'Y': '#23a6a4',
    #   // STOP
    '_': '#FF0000',
    '*': '#AAAAAA'
}

def cluster_logos(umap_df, prefix, ncol=5):
    
    n_clusters = umap_df['cluster'].max() + 1
    
    #ncol = 4
    nrow = int(np.ceil(n_clusters / ncol))
    row_multiplier = 0.7
    # print(nrow * row_multiplier)
    
    fig = plt.figure(figsize=(6, nrow * row_multiplier), dpi=150)
    gs = fig.add_gridspec(nrow, ncol, wspace=0.25, hspace=0.5, left=0.07, right=0.98, bottom=0.05, top=0.95)
    
    # for i, k in enumerate(umap_df['cluster'].value_counts().index.values):
    for i, k in enumerate(range(n_clusters)):
        
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        
        logomaker.Logo(
            logomaker.alignment_to_matrix(umap_df.loc[umap_df['cluster'] == k, 'AA_sequence']), 
            color_scheme=clustalXAAColors,
            ax=ax
        )
        
        ax.set_title(f'k={prefix}{k + 1}, n={(umap_df["cluster"] == k).sum()}')
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
        
####### Fig 1S3C #######
        
def cluster_dataset_proportions(ly6a_umap, ly6c1_umap, fig_outdir, figname='fig1S3C'):
    
    fig = plt.figure(figsize=(6, 2.5), dpi=200)
    gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.6, left=0.1, right=0.95, bottom=0.2, top=0.9)

    for i, (cluster_df, ylim, title, cluster_prefix) in enumerate(zip(
            [ly6a_umap, ly6c1_umap], [[10, 1000], [10, 1000]], ['LY6A-Fc', 'LY6C1-Fc'], ['A', 'C']
        )):

        prop_df = (
            cluster_df[['cluster', 'dataset']]
            .groupby('cluster')
            .agg(
                n_total=('dataset', len),
                n_library1=('dataset', lambda x: (x == 'library1').sum()),
                n_library2=('dataset', lambda x: (x == 'library2').sum())
            )
            .sort_values('n_total', ascending=False)
        )

        ax = fig.add_subplot(gs[0, i])
        x = np.arange(0, len(prop_df))

        ax.plot(x, prop_df['n_library1'], '-', color='r', linewidth=1., label='Library 1', markersize=2.25)
        ax.plot(x, prop_df['n_library2'], '-', color='b', linewidth=1., label='Library 2', markersize=2.25)

        ax.set_xticks(x)
        x_labels = prop_df.index.values
        x_labels = [cluster_prefix + str(l+1) if l >= 0 else 'None' for l in x_labels]
        ax.set_xticklabels(x_labels, fontsize=5, rotation=90)
        ax.set_xlim(-1, len(prop_df))
        ax.set_xlabel('Cluster', labelpad=2)

        # ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_ylim(ylim)
        ax.set_yscale('log')
        ax.set_ylabel('Sequences\nper cluster')

        ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05), frameon=False, handlelength=1, borderaxespad=0, labelspacing=0.1, 
                  borderpad=0, fontsize=7, ncol=2, columnspacing=1, handletextpad=0.4)

        ax.set_title(title)

        ax = fig.add_subplot(gs[1, i])
        x = np.arange(0, len(prop_df))
        ax.bar(x, prop_df['n_library1'] / prop_df['n_total'], color='r', width=0.9,
               label='Library 1', edgecolor='none', linewidth=0)
        ax.bar(x, prop_df['n_library2'] / prop_df['n_total'], 
               bottom=prop_df['n_library1'] / prop_df['n_total'], color='b', width=0.9,
               label='Library 2', edgecolor='none', linewidth=0)

        ax.set_xticks(x)
        x_labels = prop_df.index.values
        x_labels = [cluster_prefix + str(l+1) if l >= 0 else 'None' for l in x_labels ]
        ax.set_xticklabels(x_labels, fontsize=5, rotation=90)
        ax.set_xlim(-1, len(prop_df))
        ax.set_xlabel('Cluster', labelpad=2)

        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([0, 20, 40, 60, 80, 100])
        ax.set_ylabel('% of cluster')

    fig.show()    

    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path
           

####### ---- Fig 1S4A,B ---- #######

def plot_all_motifs(umap_df, l1_umap_df, l1_motif_df, l1_dff_masked, 
                    l2_umap_df, l2_motif_df, l2_dff_masked, 
                    ncol=4, cluster_prefix='A', fontsize=4, fig_outdir='figures', figname='fig1S4'):
    ncol = 8

    # Aggregate by AA sequence
    dff_masked_aggs = []
    for k, dff_masked in zip(range(2), [l1_dff_masked, l2_dff_masked]):
        dff_masked_aggs.append(
            dff_masked
            .drop(columns=['cluster', 'LY6A', 'LY6C1', 'cluster_val'])
            .set_index('AA_sequence').applymap(lambda x: 2 ** x).reset_index() # need to exclude AA sequence column
            .groupby('AA_sequence') 
            .agg(np.mean)
            .applymap(np.log2)
        )

    n_clusters = len(l1_motif_df)

    nrow = int(np.ceil(n_clusters / ncol))
    row_multiplier = 0.85

    fig = plt.figure(figsize=(6, nrow * row_multiplier), dpi=150)
    gs = fig.add_gridspec(nrow, ncol, wspace=0.25, hspace=0.3, left=0.07, right=0.98, bottom=0.05, top=0.95)


    for i, (cluster_id, row) in enumerate(l1_motif_df.sort_index().iterrows()):

        # ax = fig.add_subplot(gs[i // ncol, i % ncol])
        
        gsa = mpl.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i // ncol, i % ncol], 
                                                   hspace=0.9, wspace=0.1, height_ratios=[3, 1.75, 1.75])
        ax = fig.add_subplot(gsa[0, 0])


        aa_seqs = umap_df.loc[umap_df['cluster'] == cluster_id, 'AA_sequence']

        logomaker.Logo(
            logomaker.alignment_to_matrix(aa_seqs), 
            color_scheme=clustalXAAColors,
            ax=ax
        )
        ax.set_title(f'Cluster={cluster_prefix}{i+1} | n={len(aa_seqs)}', fontsize=fontsize, pad=1)
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels(np.arange(7) + 1, fontsize=fontsize)
        # ax.set_yticks([])
        ax.tick_params(axis='x', length=1, pad=0.5)
        ax.tick_params(axis='y', length=1, pad=0.5, labelsize=fontsize)
        ax.set_xlabel('AA Position', fontsize=fontsize, labelpad=0)

        if i % ncol == 0:
            ax.set_ylabel('Frequency', fontsize=fontsize)

        
        for k, l_umap_df, dff_masked_agg in zip(
                range(2), 
                [l1_umap_df, l2_umap_df], dff_masked_aggs,
            ):
            ax = fig.add_subplot(gsa[k + 1, 0])

            # mat = dff_masked_agg.loc[aa_seqs].T
            mat = (
                l_umap_df
                .loc[l_umap_df['cluster'] == cluster_id]
                .sort_values('LY6A', ascending=False)
                .drop(columns=['LY6A', 'LY6C1', 'Fc', 'cluster', 'x1', 'x2', 'dataset'])
                .join(dff_masked_agg, on='AA_sequence', how='left')
                .iloc[:, 1:]
            ).T

            ax.set_facecolor('#888')
            ax.imshow(
                mat, aspect='auto', cmap=mpl.cm.bwr, interpolation='none',
                norm=mpl.colors.TwoSlopeNorm(vmin=-5., vcenter=0, vmax=15),
            )
            ax.tick_params(axis='x', length=1, pad=0.5, labelsize=fontsize)

            ax.set_yticks([1, 4, 7])
            ax.set_yticklabels(['Fc', '6A', '6C1'], fontsize=fontsize)
            if i % ncol == 0:
                ax.set_yticklabels(['Fc', 'LY6A-Fc', 'LY6C1-Fc'], fontsize=fontsize)
            # else:
            #     ax.set_yticklabels([])
            ax.tick_params(axis='y', length=1, pad=0, labelsize=fontsize)
            
            if k == 1:
                ax.set_xlabel('Cluster Sequences', fontsize=fontsize, labelpad=0)
                
            
            if i % ncol == 0:
                lib_label = f'Library {k+1}'
            else:
                lib_label = f'Lib {k+1}'
            
            ax.text(-0.0, 1, lib_label, transform=ax.transAxes, ha='right', va='bottom', 
                    fontsize=fontsize)


    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path


####### ---- Supplemental tables (related to motifs and clustering) ---- #######

def flex_motif_to_str(motif):
    out = ''
    for pos in motif:
        if len(pos) == 0:
            out += '*'
        else:
            out += '[' + ','.join(pos) + ']'
    return out

def make_cluster_df(motif_6a, motif_6c):
    cluster_df = pd.concat([
        (
            motif_6a
            .sort_index()
            .reset_index()
            .rename(columns={
                'n': 'n_seqs',
                'motif_seq': 'consensus_sequence',
                'flex_motif_seq': 'flexible_consensus',
                'LY6A': 'LY6A-Fc_mean_log2enr',
                'LY6C1': 'LY6C1-Fc_mean_log2enr',
                'Fc': 'Fc_mean_log2enr',
                'x1': 'UMAP1_centroid',
                'x2': 'UMAP2_centroid'
            })
            .assign(
                flexible_consensus=lambda x: x['flexible_consensus'].apply(flex_motif_to_str),
                receptor='LY6A',
                cluster=lambda x: x['cluster'] + 1
            )
            [[
                'receptor', 'cluster', 'n_seqs', 'consensus_sequence', 'flexible_consensus', 
                'UMAP1_centroid', 'UMAP2_centroid',
                'LY6A-Fc_mean_log2enr', 'LY6C1-Fc_mean_log2enr', 'Fc_mean_log2enr', 
            ]]
        ),
        (
            motif_6c
            .sort_index()
            .reset_index()
            .rename(columns={
                'n': 'n_seqs',
                'motif_seq': 'consensus_sequence',
                'flex_motif_seq': 'flexible_consensus',
                'LY6A': 'LY6A-Fc_mean_log2enr',
                'LY6C1': 'LY6C1-Fc_mean_log2enr',
                'Fc': 'Fc_mean_log2enr',
                'x1': 'UMAP1_centroid',
                'x2': 'UMAP2_centroid'
            })
            .assign(
                flexible_consensus=lambda x: x['flexible_consensus'].apply(flex_motif_to_str),
                receptor='LY6C1',
                cluster=lambda x: x['cluster'] + 1
            )
            [[
                'receptor', 'cluster', 'n_seqs', 'consensus_sequence', 'flexible_consensus', 
                'UMAP1_centroid', 'UMAP2_centroid',
                'LY6A-Fc_mean_log2enr', 'LY6C1-Fc_mean_log2enr', 'Fc_mean_log2enr', 
            ]]
        ),
    ], axis=0, ignore_index=True)
    
    return cluster_df

def make_cluster_seq_df(motif_df, umap_df):
    
    cluster_seq_df = (
        umap_df
        .rename(columns={
            'LY6A': 'LY6A-Fc_log2enr',
            'LY6C1': 'LY6C1-Fc_log2enr',
            'Fc': 'Fc_log2enr',
            'x1': 'UMAP1',
            'x2': 'UMAP2',
        })
        [[
            'AA_sequence',
            'cluster',
            'LY6A-Fc_log2enr','LY6C1-Fc_log2enr','Fc_log2enr',
            'UMAP1','UMAP2',
        ]]
        .assign(cluster=lambda x: x['cluster'] + 1)
    )
    
    return cluster_seq_df
    
    