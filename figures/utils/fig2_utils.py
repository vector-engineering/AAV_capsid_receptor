
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from pathlib import Path
import re

from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
import logomaker

from .si_formatting import *
from .fig_utils import *
from .fig1_utils import calculate_replicability, plot_replicability

def set_ticks_format():
    _new_black = '#000'
    sns.set_theme(style='ticks', font_scale=0.75, rc={
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'svg.fonttype': 'none',
        'text.usetex': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'axes.labelpad': 2,
        'axes.linewidth': 0.5,
        'axes.titlepad': 4,
        'lines.linewidth': 0.5,
        'legend.fontsize': 9,
        'legend.title_fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
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

def RGBtoHex(colortuple):
    return '#' + ''.join(f'{i:02x}' for i in colortuple)


####### ---- Fig 2A Bar ---- #######

def R2_library_composition_bar(df, fig_outdir='figures', figname='fig2A_bar'):
    
    labels = np.array(list(reversed([
              'LY6A-Fc', 
              'LY6C1-Fc', 
              'BALB/cJ',
              'C57BL/6J'
            ])))

    sizes = np.array(list(reversed([ 
             (df[df['SOURCE'] == 'LY6A-invivo'].shape[0]+  
             df[df['SOURCE'] == 'LY6A'].shape[0]),
             (df[df['SOURCE'] == 'LY6C1'].shape[0]+
             df[df['SOURCE'] == 'LY6C1-invivo'].shape[0]),
            df[df['invivo_source'] == 'BALB/cJ'].shape[0] +
            df[df['invivo_source'] == 'BALB/cJ C57BL/6J'].shape[0],
            df[df['invivo_source'] == 'C57BL/6J'].shape[0]
            ])))

    bottoms = [0]
    for i, size in enumerate(sizes[:-1]):
        bottoms.append(sum(sizes[:i+1]))


    barlabels = ['{:.1f}% ({})'.format(p*100,si_format(p * sum(sizes))) for p in sizes/sum(sizes)]
    print(barlabels)

    fig, ax1 = plt.subplots(figsize=(1.25,2.5),dpi=200) 
    plt.subplots_adjust(top=0.92, left=0.28, right=0.95, bottom=0.015)


    colors = [RGBtoHex((210,126,74)), '#E5CB25', RGBtoHex((120+30, 142+30, 171+30)), RGBtoHex((160, 197, 75))]

    containers_invitro = ax1.bar([0]*len(labels[2:]), sizes[2:], width=0.5, bottom=bottoms[2:], color=colors[2:])
    containers_balbc = ax1.bar([0]*len([labels[1]]), [sizes[1]], width=0.5, bottom=[bottoms[1]], color=colors[1]) 
    containers_c57 = ax1.bar([0]*len([labels[0]]), [sizes[0]], width=0.5, bottom=[bottoms[0]], color=colors[0]) 

    ax1.get_xaxis().set_visible(False)

    ax1.bar_label(containers_invitro, labels=labels[2:], label_type='center', padding=4)
    ax1.bar_label(containers_invitro, labels=barlabels[2:], padding=-5, label_type='center', fontsize=7)
    ax1.bar_label(containers_balbc, labels=[labels[1]], label_type='center', padding=4, bbox=dict(color='#E5CB25', alpha=0.6, edgecolor=None, mutation_aspect=0.5))
    ax1.bar_label(containers_balbc, labels=[barlabels[1]], padding=-5, label_type='center', fontsize=7,  bbox=dict(color='#E5CB25', alpha=0.6, edgecolor=None, mutation_aspect=0.1))
    ax1.bar_label(containers_c57, labels=[labels[0]], label_type='center', padding=0)
    ax1.bar_label(containers_c57, labels=[barlabels[0]], padding=-9, label_type='center', fontsize=7)

    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.tick_params(axis='y', labelsize=6)
    ax1.set_ylabel('# of unique DNA sequences', fontsize=6)
    ax1.yaxis.set_major_formatter(lambda x, pos: '{:.0f} k'.format(x/1000))

    ax1.set_title('R2 library\n composition', y=0.95)

    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path


####### ---- Fig 2B ---- #######

def sources_histogram(dff, fig_outdir='figures', figname='fig2B'):
    fig = plt.figure(figsize=(2.25, 1.5), dpi=300)
    gs = fig.add_gridspec(2, 2, left=0.3, right=0.93, bottom=0.2, top=0.97, wspace=0.25, hspace=0.3)

    # Same bins across all distributions for valid comparisons
    bins = np.linspace(-10, 10, 100)

    for i, col_title, quant_col, cutoff in zip(
            range(4), ['LY6A-Fc', 'LY6C1-Fc', 'C57BL/6J', 'BALB/cJ'], ['LY6A_log2enr', 'LY6C1_log2enr', 'C57BL/6_log2enr', 'BALB/c_log2enr'],
            [0, -2, 4, 4]
        ):

        gss = mpl.gridspec.GridSpecFromSubplotSpec(
            3, 1, 
            wspace=0.02, hspace=-0.5,
            subplot_spec=gs[i // 2, i % 2])

        for j, source, color, row_title in zip(
                range(3), 
                ['LY6A', 'LY6C1', 'Mouse-invivo'], ['g', 'b', 'orangered'],
                ['LY6A-Fc', 'LY6C1-Fc', 'in vivo']
            ):

            ax = fig.add_subplot(gss[j, 0])

            vals = dff.loc[dff['SOURCE'] == source, quant_col].values
            n_total = len(vals)

            # Set NaNs to some really low value and extend the bin range to count it,
            # but use xlim to only keep our plot within the original bin range

            # nan_placeholder_val = -100
            n_missing = np.sum(np.isnan(vals))
            # vals[np.isnan(vals)] = nan_placeholder_val
            xlim = [bins[0], bins[-1]]
            # new_bins = np.append(2 * nan_placeholder_val, bins)

            vals = vals[~np.isnan(vals)]
            new_bins = bins

            ax.hist(vals, bins=new_bins, density=True,
                color=color, edgecolor='none', alpha=0.7)
                    #rasterized=True)
            ax.set_xlim(xlim)
            ax.set_ylim([0, 0.5])

            # Transparent background
            ax.patch.set_alpha(0)

            # Turn off all spines except the bottom one
            for spine in ax.spines.keys():
                ax.spines[spine].set_visible(False)

            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['bottom'].set_color('#AAA')

            # Only show x ticks and label for bottom plot
            if j == 2:
                ax.spines['bottom'].set_color('#444')
                ax.set_xticks([-10, -5, 0, 5, 10])
                ax.tick_params(axis='x', labelsize=7)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            if j == 2 and i == 3:
                ax.set_xlabel('Log2 Enrichment', fontsize=7, x=-0.1)

            ax.set_yticks([])
            if j == 1 and (i == 1 or i == 3):
                ax.set_ylabel('Density', rotation=90, labelpad=2, fontsize=7, y=0.25)
                ax.yaxis.set_label_position('right')

            # % Not detected
            rect_width = 0.15
            rect_height = 0.5
            rect_padding = 0.01 # Padding from the axis
            ax.add_patch(mpl.patches.Rectangle(
                ((-1 * rect_width) - rect_padding, 0), rect_width, rect_height,
                transform=ax.transAxes, clip_on=False, facecolor=color, alpha=0.5
            ))
            ax.text((-1 * (rect_width / 2))-rect_padding + 0.01, 0.275, 
                '{:.0f}'.format((n_missing / n_total) * 100), 
                fontsize=6, transform=ax.transAxes, rotation=90, ha='center', va='center')

            if (i == 0 or i == 2) and j == 2:
                ax.text((-1 * (rect_width / 2))-rect_padding-0.3, -0.1, 
                '%ND', transform=ax.transAxes, ha='center', va='top', fontsize=7)

            # Label
            if i == 0 or i == 2:
                ax.text((-1 * rect_width) - rect_padding - 0.04, 0, row_title, color=color,
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8)

            # Column title
            if j == 0:
                ax.text(0 - rect_width - rect_padding, 0.6, col_title, 
                        transform=ax.transAxes, ha='left', va='bottom', fontsize=8)

            # Thresholds
            # ONLY SHOW WHERE THEY ARE APPLIED
            # LY6A (i == 0), LY6A (i == 0)
            # LY6C (i == 1), LY6C (i == 1)
            # IN VIVO (i >= 2)
            if (
                    (i == 0 and j == 0) or
                    (i == 1 and j == 1) or
                    (i >= 2)
                ):
                ax.add_line(mpl.lines.Line2D(
                    [cutoff, cutoff], [0, 0.5], 
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                    color='r', linewidth=0.5, clip_on=False))

    png_path = save_fig_formats(fig, figname, fig_outdir)
    
    plt.close()
    
    return png_path


####### ---- Fig 2C ---- #######

def print_counts(df):
    
    dff = df.assign(
        bc_animals=lambda x: (~x[[col for col in df.columns if 'BalbC-B_log2enr' in col]].isna()).sum(axis=1),
        c57_animals=lambda x: (~x[[col for col in df.columns if 'C57-B_log2enr' in col]].isna()).sum(axis=1)
    )
    print(f"LY6A")
    print(f"All:              n = {(dff['SOURCE'] == 'LY6A').sum()}")
    print(f"in vitro filter:  n = {((dff['SOURCE'] == 'LY6A') & (dff['LY6A_log2enr'] > 0)).sum()}")
    print(f"C57BL/6J:         n = {((dff['SOURCE'] == 'LY6A') & (dff['LY6A_log2enr'] > 0) & ((dff['C57BL/6_log2enr'] > 4) & (dff['c57_animals'] >= 2))).sum()}")
    print(f"BALB/cJ:          n = {((dff['SOURCE'] == 'LY6A') & (dff['LY6A_log2enr'] > 0) & ((dff['BALB/c_log2enr'] > 4) & (dff['bc_animals'] >= 2))).sum()}")

    print('')

    print(f"LY6C1")
    print(f"All:              n = {(dff['SOURCE'] == 'LY6C1').sum()}")
    print(f"in vitro filter:  n = {((dff['SOURCE'] == 'LY6C1') & (dff['LY6C1_log2enr'] > -2)).sum()}")
    print(f"C57BL/6J:         n = {((dff['SOURCE'] == 'LY6C1') & (dff['LY6C1_log2enr'] > -2) & ((dff['C57BL/6_log2enr'] > 4) & (dff['c57_animals'] >= 2))).sum()}")
    print(f"BALB/cJ:          n = {((dff['SOURCE'] == 'LY6C1') & (dff['LY6C1_log2enr'] > -2) & ((dff['BALB/c_log2enr'] > 4) & (dff['bc_animals'] >= 2))).sum()}")

    print('')

    print(f"Mouse in vivo")
    print(f"All:              n = {(dff['SOURCE'] == 'Mouse-invivo').sum()}")
    print(f"in vitro filter:  n = N/A") # Filter not applied
    print(f"C57BL/6J:         n = {((dff['SOURCE'] == 'Mouse-invivo') & ((dff['C57BL/6_log2enr'] > 4) & (dff['c57_animals'] >= 2))).sum()}")
    print(f"BALB/cJ:          n = {((dff['SOURCE'] == 'Mouse-invivo') & ((dff['BALB/c_log2enr'] > 4) & (dff['bc_animals'] >= 2))).sum()}")
    
    
####### ---- Fig 2D + heatmap matrix support utils ---- #######

def assign_animals(df):
    df = df.assign(
        bc_animals=lambda x: (~x[[col for col in df.columns if 'BalbC-B_log2enr' in col]].isna()).sum(axis=1),
        c57_animals=lambda x: (~x[[col for col in df.columns if 'C57-B_log2enr' in col]].isna()).sum(axis=1)
    )
    return df

def make_heatmap_df(dff, rankplot=False, BI30=False):
    
    dff_heatmap = dff.copy()
    
    if rankplot:
        clause = True
    else:
        clause = dff_heatmap['reference'] == 'AAV9_WT_SAQAQAQ'
        
    
    dff_heatmap = dff_heatmap.loc[
        dff_heatmap['SOURCE'].isin(['LY6C1', 'LY6A', 'LY6C1-invivo', 'LY6A-invivo', 'Mouse-invivo',]) |
        ((dff_heatmap['SOURCE'] == 'Reference') & clause)
    ]
    
    if BI30:
        heatmap_loc_filter = (
            (
                dff_heatmap['SOURCE'].isin(['Mouse-invivo'])
            ) &
            (
                ((dff_heatmap['BALB/c_log2enr'] > 2) | (dff_heatmap['C57BL/6_log2enr'] > 2))
            ) & 
            (
                (dff_heatmap['LY6A_log2enr'] < 0) & (dff_heatmap['LY6C1_log2enr'] < -2)
            )
        ) | (dff_heatmap['SOURCE'] == 'Reference')
    else:
        dff_heatmap = assign_animals(dff_heatmap)
        heatmap_loc_filter = (
                (
                    ((dff_heatmap['SOURCE'] == 'LY6A') & (dff_heatmap['LY6A_log2enr'] > 0)) |
                    ((dff_heatmap['SOURCE'] == 'LY6C1') & (dff_heatmap['LY6C1_log2enr'] > -2)) |
                    dff_heatmap['SOURCE'].isin(['Mouse-invivo', 'LY6A-invivo', 'LY6C1-invivo'])
                ) &
                (
                    ((dff_heatmap['BALB/c_log2enr'] > 4) | (dff_heatmap['C57BL/6_log2enr'] > 4)) &
                    ((dff_heatmap['bc_animals'] >= 2) | (dff_heatmap['c57_animals'] >= 2))

                )
            ) | (dff_heatmap['SOURCE'] == 'Reference')
    
    dff_heatmap = (
        dff_heatmap
        .loc[ heatmap_loc_filter
        ]
        .assign(source_orig=lambda x: x['SOURCE'])
        .assign(SOURCE=lambda x: x['SOURCE'].map({
            'Reference': 'zref',
            'LY6A': 'LY6A',
            'LY6C1': 'LY6C1',
            'Mouse-invivo': 'invivo',
            'LY6C1-invivo': 'invivo',
            'LY6A-invivo': 'invivo',
        }))
        .assign(invivo=lambda x: x['BALB/c_log2enr'].fillna(0) + x['C57BL/6_log2enr'].fillna(0))
        .sort_values(['SOURCE', 'invivo'], ascending=[True, False])
        .copy()
    )

    return dff_heatmap

def make_heatmap_matrix(dff_heatmap):
    
    invivo_cols = [col for col in dff_heatmap.columns if '-B_log2enr' in col]
    mat = dff_heatmap[['AA_sequence', 'SOURCE'] + invivo_cols]

    source_sort_map = {
        'zref': 0,
        'LY6A': 1,
        'LY6C1': 2,
        'invivo': 3,   
    }

    mat_all = (
        dff_heatmap
        .assign(source_sort=lambda x: x['SOURCE'].map(source_sort_map))
        .sort_values(['source_sort', 'invivo'], ascending=[True, False])
        [['AA_sequence', 'SOURCE', 'BALB/c_log2enr', 'C57BL/6_log2enr'] + invivo_cols]
    )

    return mat_all

####### Fig 2D #######

def rankplot_heatmap(df, fig_outdir='figures', figname='fig2D'):
    
    dff = assign_animals(df)
    
    dff_heatmap = make_heatmap_df(dff)
    mat_all = make_heatmap_matrix(dff_heatmap)
    
    datasets = ['WT', 'LY6A', 'LY6C1', 'invivo',]

    top=0.85
    bottom=0.25
    left=0.14
    image_right = 0.9
    cbar_padding=0.01
    cbar_width = 0.01

    cmap = mpl.cm.bwr

    vmin = -10
    vmax = 10

    fig = plt.figure(figsize=(6, 1.1), dpi=200)
    gs0 = fig.add_gridspec(2, 1, top=top, bottom=bottom, left=left, right=image_right, height_ratios=[4, 5], hspace=0.1)

    for i, col_inds, avg_col_inds, n_animals, animal_labels, row_title in zip(
            range(2), 
            [slice(8, 11), slice(4, 8)], [[3], [2]],
            [3, 4],
            [['F1', 'M1', 'M2'], ['F1', 'F2', 'M1', 'M2']],
            ['C57BL/6J', 'BALB/cJ']
            # [mat, mat]
        ):

        counts_per_dataset = mat_all['SOURCE'].value_counts(sort=False)
        counts_per_dataset = counts_per_dataset.reindex(['zref', 'LY6A', 'LY6C1', 'invivo'])
        dataset_breaks = np.append(0, counts_per_dataset.cumsum().values)

        width_ratios = counts_per_dataset.values
        # oversize AAV9
        width_ratios[0] = 10

        gs = mpl.gridspec.GridSpecFromSubplotSpec(
            2, len(datasets), 
            wspace=0.02, hspace=0.1,
            subplot_spec=gs0[i],
            height_ratios=[n_animals, 1], width_ratios=width_ratios)

        for k, _col_inds in zip(range(2), [col_inds, avg_col_inds]):

            for j, dataset, col_title, xticks in zip(
                    range(4),
                    datasets,
                    ['AAV9', 'LY6A-Fc', 'LY6C1-Fc', 'Mouse in vivo',],
                    [[], [0, 50, 100], [0, 100, 200, 300, 400], [0, 20]]
                ):

                ax = fig.add_subplot(gs[k, j])
                ax.set_facecolor('#CCC')
                ax.imshow(
                    mat_all.iloc[dataset_breaks[j]:dataset_breaks[j+1], _col_inds].values.T, 
                    aspect='auto', cmap=cmap, interpolation='none',
                    vmin=vmin, vmax=vmax
                )

                if k == 0 and i == 0:

                    if j == 3:
                        title_loc = 'right'
                    else:
                        title_loc = 'center'

                    ax.set_title(col_title, loc=title_loc, pad=0, fontsize=8)

                if i == 0 or k == 0 or j == 0:
                    ax.set_xticks([])
                elif xticks:
                    ax.set_xticks(xticks)

                if i == 1 and k == 1 and j == 2:
                    ax.set_xlabel('Variants, rank-sorted', fontsize=8, x=0.4)

                if k == 0:
                    ax.set_yticks(np.arange(0, n_animals))
                    if j > 0:
                        ax.set_yticklabels([])
                        ax.tick_params(axis='y', length=0)
                    else:
                        ax.set_yticklabels(animal_labels, fontsize=6)
                else:
                    ax.set_yticks([0])
                    if j > 0:
                        ax.tick_params(axis='y', length=0)
                        ax.set_yticklabels([])
                    else:
                        ax.set_yticklabels(['All'], fontsize=7)

                if k == 0 and j == 0:
                    ax.text(-3, 0.5, row_title, transform=ax.transAxes,
                            ha='right', va='center', fontsize=8)

                ax.tick_params(axis='x', labelsize=7)

    gs_cbar = fig.add_gridspec(1, 1, left=image_right + cbar_padding, right=image_right+cbar_padding+cbar_width, 
                               top=top, bottom=bottom + 0.14)
    ax = fig.add_subplot(gs_cbar[0, 0])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), 
            cmap=cmap
        ), 
        cax=ax
    )
    cbar.set_ticks([-10, -5, 0, 5, 10])
    cbar.set_label('Log2 Enrichment', rotation=90, labelpad=3, fontsize=8, y=0.4)
    ax.tick_params(axis='y', labelsize=7, pad=1)

    gs_nd = fig.add_gridspec(1, 1, left=image_right + cbar_padding, right=image_right+cbar_padding+cbar_width,
                             top = bottom + 0.07, bottom = bottom)
    ax = fig.add_subplot(gs_nd[0, 0])
    ax.set_facecolor('#CCC')
    ax.set_xticks([])
    ax.set_yticks([0.5])
    ax.set_yticklabels(['ND'], fontsize=7)
    ax.set_ylim([0, 1])
    ax.yaxis.set_ticks_position('right')
    ax.tick_params(axis='y', length=0, pad=2)

    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path, dff


####### ---- Fig 2E  ---- #######

def rankplots(dff, fig_outdir='figures', figname='fig2E'):
    
    dff_rankplot = make_heatmap_df(dff, rankplot=True)
    
    fig = plt.figure(figsize=(6, 1.), dpi=250)
    gs0 = fig.add_gridspec(1, 2, left=0.07, right=0.98, bottom=0.3, top=0.85,
                         wspace=0.1, hspace=0.05)

    scatter_params={
        'linewidth': 0.25,
        's': 2,
    }
    ticks = [-5, 0, 5, 10]

    for k, sort_col, xlim, col_title, invivo_cols in zip(
            range(2),
            ['C57BL/6_log2enr', 'BALB/c_log2enr'], [[0.8, 1000], [0.8, 700]], ['C57BL/6J', 'BALB/cJ'],
            [[col for col in dff_rankplot.columns if 'C57-B_log2enr' in col], [col for col in dff_rankplot.columns if 'BalbC-B_log2enr' in col]]
        ):

        gs = mpl.gridspec.GridSpecFromSubplotSpec(
            1, 3, wspace=0.1, hspace=0.1, subplot_spec=gs0[k])

        dff_animal = (
            dff_rankplot
            # Variants must be detected in at least 2 animals
            .loc[(~dff_rankplot[invivo_cols].isna()).apply(np.sum, axis=1) >= 2]
            .sort_values(sort_col, ascending=False)
            [['AA_sequence', 'source_orig', 'reference_sequence', sort_col]]
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={'index': 'rank'})
            .assign(rank=lambda x: x['rank'] + 1)
            .assign(reference_sequence=lambda x: x['reference_sequence'].apply(lambda y: np.nan if not y else y))
        )

        for i, source, color, row_title in zip(
                range(3), 
                ['LY6A', 'LY6C1', 'Mouse-invivo'], ['g', 'b', 'orangered'], 
                ['LY6A-Fc', 'LY6C1-Fc', 'in vivo']):

            ax = fig.add_subplot(gs[0, i])
            _dff = dff_animal.loc[dff_animal['source_orig'] == source]
            ax.scatter((_dff['rank']), _dff[sort_col], edgecolor=color, facecolor='none', 
                       rasterized=True,
                       **scatter_params)

            # IN VIVO + FC
            invivo_6a = dff_animal.loc[dff_animal['source_orig'] == 'LY6A-invivo']
            ax.scatter((invivo_6a['rank']), invivo_6a[sort_col], color='g', s=3, edgecolor='none')

            if k == 1 and i == 0:
                pass
            else:
                invivo_6c = dff_animal.loc[dff_animal['source_orig'] == 'LY6C1-invivo']
                ax.scatter((invivo_6c['rank']), invivo_6c[sort_col], color='b', s=3, edgecolor='none')

            # Plot WT AAV9
            label_inds = (dff_animal['reference_sequence'] == 'AAV9')
            x = dff_animal.loc[label_inds, 'rank'].values
            y = dff_animal.loc[label_inds, sort_col].values
            ax.scatter(x, y, s=8, marker='x', facecolor='#444', linewidth=0.5)

            if k == 0:
                wt_offset = (-150, 0.5)
            else:
                wt_offset = (-100, 0.5)
            ax.text(x + wt_offset[0], y + wt_offset[1], 'AAV9', color='#444', 
                    ha='right', va='center', fontsize=4.5)

            # LY6A references
            if i == 0:
                refs = ['PHP.B', 'PHP.eB']
                label_offsets = [(-10, -1), (10, 0.5)]
                alignments = [('right', 'top'), ('left', 'bottom')]
            elif i == 1:
                refs = ['AAVF', 'PHP.C1']
                if k == 0:
                    label_offsets = [(-10, -1.25), (20, 0.75)]
                    alignments = [('right', 'top'), ('left', 'bottom')]
                else:
                    label_offsets = [(0, 2.5), (5, -1)]
                    alignments = [('left', 'top'), ('right', 'top')]


            if i == 0 or i == 1:
                for ref, label_offset, alignment in zip(refs, label_offsets, alignments):
                    label_inds = (dff_animal['reference_sequence'] == ref)
                    x = dff_animal.loc[label_inds, 'rank'].values
                    y = dff_animal.loc[label_inds, sort_col].values

                    if np.isnan(x) or np.isnan(y) or len(x) == 0 or len(y) == 0:
                        continue

                    ax.scatter(x, y, s=8, marker='x', facecolor='#444', linewidth=0.5)
                    ax.text(x+label_offset[0], y+label_offset[1], ref, color='#444', 
                            ha=alignment[0], va=alignment[1], fontsize=4.5)


            ax.set_xscale('log')
            ax.tick_params(axis='x', which='minor', length=0)
            ax.tick_params(axis='x', which='major', length=2, pad=2, labelsize=7)
            ticks = [1, 10, 100, 1000]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks)

            if i == 1:
                ax.set_xlabel('Rank-sorted variants', fontsize=8)

            if k == 0:
                ax.set_xlim([0.6, 1000])
            else:
                ax.set_xlim([0.8, 700])

            ax.set_yticks([-5, 0, 5, 10])
            ax.tick_params(axis='y', labelsize=7)
            if i > 0:
                ax.set_yticklabels([])

            ax.set_ylim([-7, 11])
            if i == 0 and k == 0:
                ax.set_ylabel('Log2 Enrichment', fontsize=7)

            if i == 0:
                ax.set_title(f'{col_title} CNS Transduction', loc='left', fontsize=8, pad=2)

            if i == 0:
                ax.legend(
                    [
                        mpl.lines.Line2D([0], [0], linewidth=0, marker='o', 
                                         markeredgecolor='g', markerfacecolor='none', markeredgewidth=0.5),
                        mpl.lines.Line2D([0], [0], linewidth=0, marker='o', 
                                         markeredgecolor='g', markerfacecolor='g', markeredgewidth=0.5)
                    ], 
                    ['LY6A-Fc', f'LY6A-Fc+invivo (n={len(invivo_6a)})'],
                    loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=5,
                    handlelength=1., edgecolor='none', facecolor='none', 
                    handletextpad=0.15, markerscale=0.4, labelspacing=0.1, borderaxespad=0, borderpad=0
                )

            if i == 1:
                ax.legend(
                    [
                        mpl.lines.Line2D([0], [0], linewidth=0, marker='o', 
                                         markeredgecolor='b', markerfacecolor='none', markeredgewidth=0.5),
                        mpl.lines.Line2D([0], [0], linewidth=0, marker='o', 
                                         markeredgecolor='b', markerfacecolor='b', markeredgewidth=0.5)
                    ], 
                    ['LY6C1-Fc', f'LY6C1-Fc+invivo (n={len(invivo_6c)})'],
                    loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=5,
                    handlelength=1., edgecolor='none', facecolor='none', 
                    handletextpad=0.15, markerscale=0.4, labelspacing=0.1, borderaxespad=0, borderpad=0
                )

            if i == 2:
                ax.legend(
                    [
                        mpl.lines.Line2D([0], [0], linewidth=0, marker='o', 
                                         markeredgecolor='orangered', markerfacecolor='none', markeredgewidth=0.5),
                    ], 
                    [
                        'in vivo', 
                    ],
                    loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=5,
                    handlelength=1., edgecolor='none', facecolor='none', 
                    handletextpad=0.15, markerscale=0.4, labelspacing=0.1, borderaxespad=0, borderpad=0
                )

    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path, dff_rankplot, dff_animal


####### ---- Fig 2F + df support utils ---- #######

def sort_published_variants_df(dfr):
    
    # Re-sort dataframe

    """
    WT first
    CREATE: PHP.A, B, B2, B3
    eB
    mCREATE: rest of PHPs
    Voyager variants: 9P*
    """

    capsid_to_ind_map = {

        'WT': 0,

        # Deverman 2016 (CREATE)
        'PHP.A':  1,
        'PHP.B':  2,
        'PHP.B2': 3,
        'PHP.B3': 4,

        # Chan 2017
        'PHP.eB': 5,

        # Hanlon 2019
        'AAVF': 10,

        # Kumar 2020 (mCREATE)
        'PHP.B4': 20,
        'PHP.B5': 21,
        'PHP.B6': 22,
        'PHP.B7': 23,
        'PHP.B8': 24,

        'PHP.C1': 25,
        'PHP.C2': 26,
        'PHP.C3': 27,

        'PHP.N': 28,

        'PHP.V1': 29,
        'PHP.V2': 30,

        # Nonnenmacher 2021 (Voyager)
        '9P03': 101,
        '9P08': 102,
        '9P09': 103,
        '9P16': 104,
        '9P31': 105,
        '9P32': 106,
        '9P33': 107,
        '9P36': 108,
        '9P39': 109,
    }

    dfr = dfr.sort_values('Source', key=lambda x: x.map(capsid_to_ind_map))
    
    return dfr

def make_label_df(dfr):
    
    label_df = dfr.reset_index(drop=True)[['Source', 'full_AA_sequence', 'AA_pre', 'AA_sequence']].drop_duplicates(keep='first')
    label_df['Label'] = label_df['Source']

    return label_df
    
####### Fig 2F #######
    
def plot_published_variants(dfr, fig_outdir='figures', figname='fig2F'):
    
    cols = {
        'Production': [
            'starter_virus_1_Fc+biod_log2fitness','starter_virus_2_Fc+biod_log2fitness', 'starter_virus_3_Fc+biod_log2fitness',
            'starter_virus_1_trans_log2fitness', 'starter_virus_2_trans_log2fitness','starter_virus_3_trans_log2fitness',
        ],
        'LY6A-Fc': ['LY6A_1_log2enr','LY6A_2_log2enr', 'LY6A_3_log2enr', 'LY6A_4_log2enr',],
        'LY6C1-Fc': ['LY6C1_1_log2enr', 'LY6C1_2_log2enr', 'LY6C1_3_log2enr','LY6C1_4_log2enr',],
        'Brain Biod': [
            'biod_brain_m1_log2enr', 'biod_brain_m2_log2enr', 'biod_brain_m3_log2enr', 'biod_brain_m4_log2enr'
        ],
        'Brain Trans': [
            'trans_brain_m1_log2enr', 'trans_brain_m2_log2enr','trans_brain_m3_log2enr','trans_brain_m4_log2enr'
        ]
    }

    n_rows = len(dfr)
    all_cols = sum([c for c in cols.values()], [])
    n_cols = len(all_cols)
    col_x = np.array([len(c) for c in cols.values()])
    col_x = np.cumsum(col_x)
    # col_x = col_x - col_x[0]
    col_x = np.append(0, col_x)[:-1]
    col_labels = list(cols.keys())
    
    dfr = sort_published_variants_df(dfr)
    label_df = make_label_df(dfr)
    
    annots = [
        (0, 6, 'WT'),
        (6, 6+8, 'Deverman\net al. 2016'),
        (14, 14+4, 'Chan et al.\n2017'),
        (18, 18+2, 'Hanlon et al.\n2019'),
        (20, 20+(11*2), 'Kumar et al. 2020'),
        (42, 42+22, 'Nonnenmacher et al. 2021'),
    ]

    col_widths = []
    for start, end, _ in annots:
        col_widths.append(end-start)

    rows = [
        (0, 6, 'Production'),
        (6, 10, 'LY6A-Fc'),
        (10, 14, 'LY6C1-Fc'),
        (14, 18, 'Biodistribution'),
        (18, 22, 'Transduction')
    ]
    row_heights = []
    for start, end, _ in rows:
        row_heights.append(end-start)

    left = 0.17
    right = 0.9
    bottom = 0.22
    top = 0.75

    cbar_width = 0.01
    cbar_padding = 0.0075

    fig = plt.figure(figsize=(6, 1.5),dpi=150)
    gs = fig.add_gridspec(
        len(rows), len(annots), left=left, right=right, bottom=bottom, top=top,
        width_ratios=col_widths, height_ratios=row_heights, wspace=0.03, hspace=0.1
    )
    gs2 = fig.add_gridspec(1, 1, left=right + cbar_padding, right=right + cbar_padding + cbar_width, 
                           bottom=bottom, top=top)


    mat = dfr[all_cols].values
    bounds = (-10, 10)

    for i, (row_start, row_end, row_label) in enumerate(rows):
        for j, (col_start, col_end, annot_label) in enumerate(annots):

            ax = fig.add_subplot(gs[i, j])

            _mat = mat.T[slice(row_start, row_end), slice(col_start, col_end)]
            im = ax.imshow(
                _mat, aspect='auto', interpolation='none', cmap=mpl.cm.bwr, 
                vmin=bounds[0], vmax=bounds[1]
            )

            ax.set_facecolor('#CCC')

            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_label, rotation=0, y=0.5, va='center', ha='right')

            _label_df = label_df.loc[slice(col_start, col_end-1)]
            x_start = _label_df.index.values[0]

            label_x = _label_df.index.values - x_start
            label_offset = (np.diff(np.append(label_x, _mat.shape[1])) / 2) - 0.5
            label_txt = _label_df['Label'].values

            if i == 0:
                ax.tick_params(
                    bottom=False, top=True, labelbottom=False, labeltop=True,
                    length=1, pad=0
                )            
                ax.set_xticks(label_x + label_offset)
                ax.set_xticklabels(label_txt, fontsize=7)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('left')
            else:
                ax.set_xticks([])

            for x in label_x:
                ax.axvline(x - 0.5, color='#EEE', linewidth=0.5, alpha=0.8)

            if i == len(rows) - 1:

                if annot_label == 'WT':
                    continue

                annot_y = -0.25
                if j == 3:
                    annot_y = -1.1

                ax.text(0., annot_y, annot_label, 
                        transform=ax.transAxes,
                        ha='left', va='top', fontsize=6, color='#444'
                )

            ax.tick_params(axis='x', length=1.5, pad=-1)

    ax = fig.add_subplot(gs2[0, 0])
    cbar = mpl.colorbar.Colorbar(ax, im)
    cbar.set_label('Log2 Enrichment')
    cbar.set_ticks([-10, -5, 0, 5, 10])

    for spine in ax.spines.keys():
        ax.spines[spine].set_linewidth(0.5)

    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path, dfr, label_df
    

####### ---- Fig 2S1 ---- #######

def plot_replicability_between_animals(df, plot_data_invivo_mean, assays=['BalbC-B', 'C57-B'], assay_titles=['BALB/cJ', 'C57BL/6J'], rep_titles=[['BALB/cJ F1', 'BALB/cJ F2', 'BALB/cJ M1', 'BALB/cJ M2'], ['C57BL/6J F1', 'C57BL/6J M1', 'C57BL/6J M2']]):
    
    set_ticks_format()
    fig_theme()
    
    xlim = [-3, 18]

    fig = plt.figure(figsize=(6.5, 1.9*(3/2)), dpi=200)
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, wspace=0.3, 
                              bottom=0.15, top=0.94, left=0.1, right=0.95)

    for k, assay in enumerate(assays):
        rep_cols = [col for col in df.columns if re.match('[FM][1-2]-' + assay + '_mean_RPM', col)]

        gsa = mpl.gridspec.GridSpecFromSubplotSpec(
            # len(rep_cols)-1, len(rep_cols)-1, 
            3, 3,
            subplot_spec=gs[k // 2, k % 2], hspace=0.075, wspace=0.075)

        for i, j in zip(*np.tril_indices(len(rep_cols), k=-1)):
            gsaa = mpl.gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=gsa[i-1, j],
                # left=0.15, right=0.95, bottom=0.15, top=0.95, 
                width_ratios=[1, 6], height_ratios=[6, 1], hspace=0., wspace=0
            )

            x = plot_data_invivo_mean[(assay, i, j)]['x']
            y = plot_data_invivo_mean[(assay, i, j)]['y']
            x_missing = plot_data_invivo_mean[(assay, i, j)]['x_missing']
            y_missing = plot_data_invivo_mean[(assay, i, j)]['y_missing']
            c = plot_data_invivo_mean[(assay, i, j)]['c']

            ax = fig.add_subplot(gsaa[0, 1])
            # ax.set_aspect('equal', 'box')
            ax.scatter(x, y, c=c, cmap=mpl.cm.inferno, s=0.5, edgecolor='none', rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            # xlim = xlims[k]
            bins = np.linspace(*xlim, 25)
            ax.set_xlim(xlim); ax.set_ylim(xlim)

            if i == 1:
                ax.set_title(assay_titles[k])

            ax.text(0.03, 0.97, r'$\rho$ = {:.3f}'.format(np.corrcoef(x, y)[0, 1]),
                   transform=ax.transAxes, ha='left', va='top', fontsize=5.5)

            ax.text(0.97, 0.03, 'n={}'.format(human_format(len(x))), 
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5)

            # Left Marginal
            ax = fig.add_subplot(gsaa[0, 0])
            ax.hist(y_missing, bins=bins, edgecolor='none', orientation='horizontal', density=True, color='r')
        
            ax.set_xticks([])
            if j == 0:
                ax.set_yticks([0, 5, 10, 15])
                # ax.set_ylabel('Rep {}'.format(i+1))
                ax.set_ylabel(rep_titles[k][i], fontsize=8)
            else:
                ax.set_yticks([])

            ax.set_ylim(xlim)

            ax.text(0.97, 0.97, 'n={}'.format(human_format(len(y_missing))), 
                    transform=ax.transAxes, ha='right', va='top', fontsize=5.5, rotation=90, color='r')
            
            ax.tick_params(labelsize=7.5)

            # Bottom Marginal
            ax = fig.add_subplot(gsaa[1, 1])
            ax.hist(x_missing, bins=bins, edgecolor='none', density=True, color='r')

            if i == len(rep_cols)-1:
                ax.set_xticks([0, 5, 10, 15])
                # ax.set_xlabel('Animal {}'.format(j+1))
                ax.set_xlabel(rep_titles[k][j], fontsize=8)
            else:
                ax.set_xticks([])

            ax.set_xlim(xlim)
            
            ax.set_yticks([])
            ax.text(0.97, 0.8, 'n={}'.format(human_format(len(x_missing))), 
                    transform=ax.transAxes, ha='right', va='top', fontsize=5.5, color='r')
            
            ax.tick_params(labelsize=7.5)

            # Missing label
            ax = fig.add_subplot(gsaa[1, 0])
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
            


            if i == len(rep_cols)-1 and j == 0:
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
        cbar.set_label('Density', rotation=90, va='bottom', labelpad=10)
    
    
    return fig

####### Fig 2S1A-D #######

def plot_r2_replicability_pulldown(df, assays=['starter_virus', 'LY6A', 'LY6C1', 'Fc'], assay_titles=['Virus', 'LY6A-Fc', 'LY6C1-Fc', 'Fc control'], fig_outdir='figures', figname='fig2S1ABCD'):
    
    set_ticks_format()
    fig_theme()
    
    plot_data = calculate_replicability(df, assays=assays)

    fig = plot_replicability(df, plot_data, assays, assay_titles)

    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path

####### Fig 2S1E,F #######

def plot_r2_replicability_invivo(df, assays=['F1-BalbC-B', 'F2-BalbC-B', 'M1-BalbC-B', 'M2-BalbC-B'], assay_titles=['BALB/cJ F1', 'BALB/cJ F2', 'BALB/cJ M1', 'BALB/cJ M2'], fig_outdir='figures', figname='fig2SE'):
    
    set_ticks_format()
    fig_theme()

    plot_data = calculate_replicability(df, assays=assays, rep_pattern='[1-3]_RPM')

    fig = plot_replicability(df, plot_data, assays, assay_titles, rep_pattern='[1-3]_RPM')

    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path

####### Fig 2S1G,H #######

def plot_r2_replicability_animals(df, assays=['BalbC-B', 'C57-B'], assay_titles=['BALB/cJ', 'C57BL/6J'], fig_outdir='figures', figname='fig2SG'):
    
    set_ticks_format()
    fig_theme()

    plot_data = calculate_replicability(df, assays=assays, rep_pattern='_mean_RPM')

    fig = plot_replicability_between_animals(df, plot_data, assays, assay_titles)

    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path


####### ---- Fig 2S2 + codon aggregation and correlation support utils ---- #######

r2_assays = ['starter_virus', 'LY6A', 'LY6C1', 'Fc',
             'F1-BalbC-B', 'F2-BalbC-B', 'M1-BalbC-B', 'M2-BalbC-B',
             'F1-C57-B', 'M1-C57-B', 'M2-C57-B'
            ]

r2_assay_titles = ['Virus', 'LY6A-Fc', 'LY6C1-Fc', 'Fc control',
                   'BALB/cJ F1', 'BALB/cJ F2', 'BALB/cJ M1', 'BALB/cJ M2',
                   'C57BL/6J F1', 'C57BL/6J M1', 'C57BL/6J M2'
                  ]

def drop_zeros(x):
    return [_x for _x in x if _x > 0]

def drop_nans(x):
    return [_x for _x in x if np.isfinite(_x)]

def calculate_codon_replicability(df, assays, assay_cols, assay_cols_enr):
    
    set_ticks_format()
    fig_theme()

    df_codon_agg = df[['AA_sequence'] + assay_cols].groupby('AA_sequence').agg(list)
    df_codon_agg = df_codon_agg.applymap(drop_zeros)

    # Compute log2enr
    for k, (assay, assay_col, assay_col_enr) in enumerate(zip(assays, assay_cols, assay_cols_enr)):
        if assay_col == 'starter_virus_mean_RPM' or assay_col == 'DNA_mean_RPM':
            continue

        # In this case, we don't want to use a pseudocount because we will show missing value.  But other
        # plots use pseudocount values, so when we show the actual values, we want to include the pseudocount to be consistent.   So we filter out
        # values that are NaN and then apply the pseudocount.  In practice, adding the pseudocount here changes the plots very little.
        df[assay_col_enr] = 0*np.log2((df[assay_col])/(df['starter_virus_mean_RPM'])) + np.log2((df[assay_col]+0.01)/(df['starter_virus_mean_RPM']+0.01))

    df_codon_agg_enr = df[['AA_sequence'] + assay_cols_enr].groupby('AA_sequence').agg(list)
    df_codon_agg_enr[assay_cols_enr] = df_codon_agg_enr[assay_cols_enr].applymap(drop_nans)
    
    plot_data_codon_rep = {}

    for k, (assay, assay_col, assay_col_enr) in enumerate(zip(assays, assay_cols, assay_cols_enr)):    

        if assay_col == 'starter_virus_mean_RPM' or assay_col == 'DNA_mean_RPM':
            has_both = df_codon_agg[assay_col].apply(len) == 2
            has_one = df_codon_agg[assay_col].apply(len) == 1

            xy = df_codon_agg.loc[has_both, assay_col]
            x = xy.apply(lambda x: x[0])
            y = xy.apply(lambda x: x[1])
            x = np.log2(x)
            y = np.log2(y)

            x_missing = df_codon_agg.loc[has_one, assay_col].apply(lambda x: x[0])
            x_missing = np.log2(x_missing)

            if len(x) > 0 and len(y) > 0:
                n_sample = min([10000, len(x), len(y)])
                kernel = gaussian_kde(np.vstack([
                    x.sample(n_sample, random_state=1), y.sample(n_sample, random_state=1)
                ]))
                c = kernel(np.vstack([x, y]))
            else:
                c = []

            plot_data_codon_rep[assay] = {}
            plot_data_codon_rep[assay]['x'] = x
            plot_data_codon_rep[assay]['y'] = y
            plot_data_codon_rep[assay]['x_missing'] = x_missing
            plot_data_codon_rep[assay]['c'] = c

        else:
            #print(assay_col_enr, (has_one_rpm | has_one_starter).sum())
            
            enr = df_codon_agg_enr[assay_col_enr]
            has_both_enr = enr.apply(len) == 2
            has_one_enr = enr.apply(len) == 1
            
            mean_rpm = df_codon_agg[assay_col]
            has_both_rpm = mean_rpm.apply(len) == 2
            has_one_rpm = mean_rpm.apply(len) == 1
            
            starter = df_codon_agg['starter_virus_mean_RPM']
            has_both_starter = starter.apply(len) == 2
            has_one_starter = starter.apply(len) == 1
            
            xy = df_codon_agg_enr.loc[has_both_enr & has_both_rpm & has_both_starter, assay_col_enr]
            x = xy.apply(lambda _x: _x[0])
            y = xy.apply(lambda _x: _x[1])

            x_missing = df_codon_agg_enr.loc[(has_one_rpm | has_one_starter) & (has_one_enr | has_both_enr), 
                                            assay_col_enr].apply(lambda _x: _x[0])
            # x_missing = np.log2(x_missing)

            if len(x) > 0 and len(y) > 0:
                n_sample = min([10000, len(x), len(y)])
                kernel = gaussian_kde(np.vstack([
                    x.sample(n_sample, random_state=1), y.sample(n_sample, random_state=1)
                ]))
                c = kernel(np.vstack([x, y]))
            else:
                c = []

            plot_data_codon_rep[assay] = {}
            plot_data_codon_rep[assay]['x'] = x
            plot_data_codon_rep[assay]['y'] = y
            plot_data_codon_rep[assay]['x_missing'] = x_missing
            plot_data_codon_rep[assay]['c'] = c
        
    return plot_data_codon_rep, df_codon_agg
    
def plot_codon_replicability(df, assays=r2_assays, assay_titles=r2_assay_titles, DNA=True, fig_outdir='figures', figname='fig2S2'):
    
    set_ticks_format()
    fig_theme()

    if DNA:
        plot_assays = list(np.append(assays, 'DNA'))
        shift = 1
    else:
        plot_assays = assays
        shift = 0
    
    plot_assay_cols = []
    plot_assay_cols_enr = []
    for col in plot_assays:
        plot_assay_cols.append(col + '_mean_RPM')
        if col == 'starter_virus' or col == 'DNA':
            plot_assay_cols_enr.append(col + '_mean_RPM')
        else:
            plot_assay_cols_enr.append(col + '_log2enr')
    

    plot_data_codon_rep, df_codon_agg = calculate_codon_replicability(df, plot_assays, plot_assay_cols, plot_assay_cols_enr)

    xlims = [
        [-5, 10], # Virus
        [-4, 15], [-4, 15], # LY6A, LY6C1
        [-4, 15], [-4, 15], # Fc control
        [-2, 17], [-2, 17], # BALB/cJ F1, F2
        [-2, 17], [-2, 17], # BALB/cJ M1, M2
        [-4, 15], # [-3, 15],
        [-4, 15], [-4, 15],
    ]

    fig = plt.figure(figsize=(6, 5.25*((3+shift)/3)), dpi=200)
    nrow = 3 + shift
    ncol = 4
    gs = mpl.gridspec.GridSpec(nrow, ncol, figure=fig, wspace=0.45, hspace=0.55,
                              bottom=0.1, top=0.94, left=0.1, right=0.95,
                              width_ratios=[12, 12, 12, 13])
    if DNA:
        gsdna = mpl.gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=gs[0,0], hspace=0.0, wspace=0.1, height_ratios=[6, 1], width_ratios=[12,1])
        
        assay = 'DNA'
        
        x = plot_data_codon_rep[assay]['x']
        y = plot_data_codon_rep[assay]['y']
        x_missing = plot_data_codon_rep[assay]['x_missing']
        c = plot_data_codon_rep[assay]['c']

        ax = fig.add_subplot(gsdna[0, 0])
        # ax.set_aspect('equal', 'box')
        ax.scatter(x, y, c=c, cmap=mpl.cm.inferno, s=0.5, edgecolor='none', rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        # xlim = xlims[k]
        xlim = [-5.5, 17]
        bins = np.linspace(*xlim, 50)
        ax.set_title('DNA Library', fontsize=9)

        ax.set_yticks([-5, 0, 5, 10, 15])
        ax.set_ylabel('Codon 2', fontsize=9)

        ax.set_xlim(xlim); ax.set_ylim(xlim)

        ax.text(0.03, 0.97, r'$\rho$ = {:.3f}'.format(np.corrcoef(x, y)[0, 1]),
               transform=ax.transAxes, ha='left', va='top', fontsize=7)

        ax.text(0.97, 0.03, 'n={}'.format(human_format(len(x))), 
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7)

        ax.tick_params(axis='both', labelsize=8)

        # Bottom Marginal
        ax = fig.add_subplot(gsdna[1, 0])
        ax.hist(x_missing, bins=bins, edgecolor='none', density=True, color='r')

        ax.set_xticks([-5, 0, 5, 10, 15])
        ax.set_xlabel('Codon 1', fontsize=9)

        ax.set_xlim(xlim)

        ax.set_yticks([])
        ax.text(0.97, 0.8, 'n={}'.format(human_format(len(x_missing))), 
                transform=ax.transAxes, ha='right', va='top', fontsize=7, color='r')

        ax.text(-0.1, 0., 'Only one\ndetected', transform=ax.transAxes, ha='right', va='center', color='r', fontsize=7)

        ax.tick_params(axis='both', labelsize=8)
        
        # Add density colorbar
        # ax = fig.add_subplot(gsdna[0, 1])
        # cbar = fig.colorbar(
        #     mpl.cm.ScalarMappable(
        #         norm=mpl.colors.Normalize(vmin=0, vmax=1.0), 
        #         cmap=mpl.cm.inferno
        #     ), 
        #     cax=ax
        # )
        # cbar.set_ticks([])
        # ax.yaxis.set_label_position('right')
        # cbar.set_label('Density', rotation=90, va='bottom', labelpad=10)
    
    for k, assay in enumerate(assays):
        needs_density_bar = False
        first_in_row = k % ncol == 0

        if (k == 0 and shift == 1) or k % ncol == 3 or k == (len(assays) - 1):
            needs_density_bar = True
            first_in_row = False
            
        if k > 0 and k < 4:
            shift2 = -1
            if k == 1:
                first_in_row = True
        else:
            shift2 = 0

        if k == 0 and shift == 1:
            gsa = mpl.gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=gs[0, 1], hspace=0.0, wspace=0.1, height_ratios=[6, 1], width_ratios=[12,1])
            needs_density_bar = True
        elif needs_density_bar == True:
            gsa = mpl.gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=gs[k // ncol + shift, k % ncol + shift2], hspace=0.0, wspace=0.1, height_ratios=[6, 1], width_ratios=[12,1])
        else:
            gsa = mpl.gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[k // ncol + shift, k % ncol + shift2], hspace=0.0, wspace=0.1, height_ratios=[6, 1])


        x = plot_data_codon_rep[assay]['x']
        y = plot_data_codon_rep[assay]['y']
        x_missing = plot_data_codon_rep[assay]['x_missing']
        c = plot_data_codon_rep[assay]['c']

        ax = fig.add_subplot(gsa[0, 0])
        # ax.set_aspect('equal', 'box')
        ax.scatter(x, y, c=c, cmap=mpl.cm.inferno, s=0.5, edgecolor='none', rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        # xlim = xlims[k]
        
        if 'DNA' in assay or 'virus' in assay:
            xlim = [-5.5, 17]
            xticks = [-5, 0, 5, 10, 15]
        else:
            xlim = [-11, 12]
            xticks = [-10, -5, 0, 5, 10]
        bins = np.linspace(*xlim, 50)
        ax.set_title(assay_titles[k], fontsize=9)

        ax.set_yticks(xticks)
        ax.set_ylabel('Codon 2', fontsize=9)

        ax.set_xlim(xlim); ax.set_ylim(xlim)

        ax.text(0.03, 0.97, r'$\rho$ = {:.3f}'.format(np.corrcoef(x, y)[0, 1]),
               transform=ax.transAxes, ha='left', va='top', fontsize=7)

        ax.text(0.97, 0.03, 'n={}'.format(human_format(len(x))), 
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7)

        ax.tick_params(axis='both', labelsize=8)

        # Bottom Marginal
        ax = fig.add_subplot(gsa[1, 0])
        ax.hist(x_missing, bins=bins, edgecolor='none', density=True, color='r')

        ax.set_xticks(xticks)
        ax.set_xlabel('Codon 1', fontsize=9)

        ax.set_xlim(xlim)

        ax.set_yticks([])
        ax.text(0.97, 0.8, 'n={}'.format(human_format(len(x_missing))), 
                transform=ax.transAxes, ha='right', va='top', fontsize=7, color='r')

        if first_in_row:
            ax.text(-0.1, 0., 'Only one\ndetected', transform=ax.transAxes, ha='right', va='center', color='r', fontsize=7)

        ax.tick_params(axis='both', labelsize=8)
        
        # Add density colorbar to rightmost plot per row
        if needs_density_bar:
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
            cbar.set_label('Density', rotation=90, va='bottom', labelpad=10)

    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path


####### ---- Fig 2S3 + filtering, dendrogram, hamming and clustering support utils ---- #######

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def make_hamming_dist_mat(dff_heatmap, dff_cluster):
    
    hamming_dist_mat = np.zeros((len(dff_cluster), len(dff_cluster)))
    AAs = dff_heatmap['AA_sequence'].values
    for i, j in zip(*np.tril_indices(len(dff_cluster))):
        hamming_dist_mat[i, j] = hamming_xor(AAs[i], AAs[j])

    hamming_dist_mat[np.triu_indices(len(dff_cluster))] = hamming_dist_mat.T[np.triu_indices(len(dff_cluster))]
    
    return hamming_dist_mat
 
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
cluster_colors = cluster_colors * 100

cluster_colors_rgb = []
for (r,g,b) in cluster_colors:
    cluster_colors_rgb.append('#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)))    
    
####### Fig 2S3A #######

def histogram_thresholding(dff, fig_outdir='figures', figname='fig2S3A'):
    
    ncol = 2
    bins = np.linspace(-11, 5, 100)

    fig = plt.figure(figsize=(5.5, 1.25), dpi=250)
    gs = fig.add_gridspec(1, 4, wspace=0.3, hspace=0.5, bottom=0.25, top=0.8, left=0.1, right=0.95)

    for i, title, col, thresh, thresh_above in zip(
            range(4), 
            ['LY6A-Fc', 'LY6C1-Fc', 'C57BL/6J', 'BALB/cJ'], # title
            ['LY6A_log2enr', 'LY6C1_log2enr', 'C57BL/6_log2enr', 'BALB/c_log2enr'], # col
            [0, -2, 2, 2], [False, False, True, True]
        ):
        ax = fig.add_subplot(gs[0, i])

        _df = dff.loc[dff['SOURCE'].isin(['Mouse-invivo'])]
        ax.hist(_df[col], bins=bins, density=True, alpha=0.7, color='r', label='in vivo')

        if i == 0:
            other_df = dff.loc[dff['SOURCE'].isin(['LY6A', 'LY6A-invivo'])]
            other_label = 'LY6A-Fc'
        elif i == 1:
            other_df = dff.loc[dff['SOURCE'].isin(['LY6C1', 'LY6C1-invivo'])]
            other_label = 'LY6C1-Fc'
        else:
            other_df = dff.loc[dff['SOURCE'].isin(['LY6A', 'LY6C1', 'LY6A-invivo', 'LY6C1-invivo'])]
            other_label = 'LY6A-Fc + LY6C1-Fc'

        if i < 2:
            ax.hist(other_df[col], bins=bins, density=True, alpha=0.7, color='gray', label=other_label)

        ax.axvline(thresh, color='r', linewidth=0.5)


        if thresh_above:
            rect_xy = (thresh, 0)
            rect_width = 10
        else:
            rect_xy = (-15, 0)
            rect_width = thresh + 15

        ax.add_patch(mpl.patches.Rectangle(
            rect_xy, rect_width, 1, 
            edgecolor='r', facecolor='r', alpha=0.1, linewidth=0.5)
        )

        ax.set_xticks([-10, -5, 0, 5])

        ax.set_xlabel(title)

        if thresh_above:
            n_pass = (_df[col] > thresh).sum()
        else:
            n_pass = (_df[col] < thresh).sum()

        ax.text(thresh + 0.5, 0.95, f'n = {n_pass}', 
                transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                ha='left', va='top', rotation=90, fontsize=7, color='r')

        percent_missing = _df[col].isna().sum() / len(_df)
        ax.text(0.03, 0.07, f'{percent_missing:.2%} Missing', 
               transform=ax.transAxes, ha='left', va='bottom', rotation=90, fontsize=6, color='#666')

        if i == 0:
            ax.set_ylabel('Density')

        ax.legend(loc='lower left', bbox_to_anchor=(0, 1), borderaxespad=0, frameon=False, 
                  fontsize=7, labelspacing=0.2, handletextpad=0.3, handlelength=1, borderpad=0.1)

        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)


    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path

####### Fig 2S3B,C #######

def hierarchical_clustering_heatmap(df, fig_outdir='figures', figname='fig2S3BC'):
    
    dff_heatmap = make_heatmap_df(df, BI30=True)
    
    invivo_cols = [col for col in df.columns if 'B_log2enr' in col]
    invitro_cols = [col for col in df.columns if any(['_{}_log2enr'.format(i) in col for i in range(1,4)])]
    
    # Clustering for Fig 2S3B dendrogram 
    dff_cluster = dff_heatmap.loc[dff_heatmap['SOURCE'] == 'invivo']
    hamming_dist_mat = make_hamming_dist_mat(dff_heatmap, dff_cluster)
    clustering = AgglomerativeClustering(affinity='precomputed', linkage='average',
                                     distance_threshold=5, n_clusters=None).fit(hamming_dist_mat)
    dff_heatmap.loc[dff_heatmap['SOURCE'] == 'invivo', 'cluster'] = clustering.labels_
    
    # Heatmap matrix for Fig 2S3C
    mat_all = (
        dff_heatmap
        .sort_values(['SOURCE', 'BALB/c_log2enr'], ascending=[True, False])
        [['AA_sequence', 'SOURCE', 'BALB/c_log2enr', 'C57BL/6_log2enr', 'Fc_log2enr', 'LY6A_log2enr', 'LY6C1_log2enr'] + invitro_cols + invivo_cols]
    )
    
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [clustering.children_, clustering.distances_, counts]
    ).astype(float)
    
    datasets = ['invivo', 'WT']

    top=0.87
    bottom=0.2
    left=0.15
    image_right = 0.9
    cbar_padding=0.01
    cbar_width = 0.01

    cmap = mpl.cm.bwr

    vmin = -10
    vmax = 10

    fig = plt.figure(figsize=(6, 4), dpi=200)
    gs = fig.add_gridspec(2, 2, top=top, bottom=bottom, left=left, right=image_right, 
                          height_ratios=[1, 4], width_ratios=[31, 1], wspace=0., hspace=0.05)


    ax = fig.add_subplot(gs[0, 0])

    hierarchy.set_link_color_palette(cluster_colors_rgb)

    # Plot the corresponding dendrogram
    R = dendrogram(linkage_matrix, ax=ax, color_threshold=5, above_threshold_color='#AAA', no_labels=True)

    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_ylim([1, 7])
    ax.set_ylabel('Hamming\ndistance', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)


    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(
            5, 1, height_ratios=[4, 4, 4, 5, 4], subplot_spec=gs[1, :], hspace=0.05, wspace=0.01)

    for i, col_inds, avg_col_inds, n_animals, animal_labels, row_title in zip(
            range(5),
            [slice(7, 10), slice(10, 13), slice(13, 16), slice(16, 20), slice(20, 23)], 
            [[4], [5], [6], [2], [3]], [3, 3, 3, 4, 3],
            [['1', '2', '3'], ['1', '2', '3'],['1', '2', '3'],['F1','F2','M1','M2'],['F1','M1','M2']], 
            ['Fc-control', 'LY6A-Fc', 'LY6C1-Fc', 'BALB/cJ', 'C57BL/6J']
        ):

        counts_per_dataset = mat_all['SOURCE'].value_counts(sort=False)
        dataset_breaks = np.append(0, counts_per_dataset.cumsum().values)

        width_ratios = counts_per_dataset.values
        # oversize AAV9
        width_ratios[-1] = 4

        gs1 = mpl.gridspec.GridSpecFromSubplotSpec(
            2, len(datasets), 
            wspace=0.02, hspace=0.1,
            subplot_spec=gs0[i],
            height_ratios=[n_animals, 1], width_ratios=width_ratios)

        for k, _col_inds in zip(range(2), [col_inds, avg_col_inds]):

            for j, dataset, col_title, xticks in zip(
                    range(2),
                    datasets,
                    ['Mouse in vivo', 'AAV9',],
                    [[0, 25, 50, 75, 100, 125, 150, 175], []]
                ):

                if j == 0:
                    row_inds = R['leaves']
                else:
                    row_inds = slice(dataset_breaks[j], dataset_breaks[j+1])

                ax = fig.add_subplot(gs1[k, j])
                ax.set_facecolor('#CCC')
                ax.imshow(
                    mat_all.iloc[row_inds, _col_inds].values.T, 
                    aspect='auto', cmap=cmap, interpolation='none',
                    vmin=vmin, vmax=vmax
                )

                if i < 4 or k == 0 or j == 1:
                    ax.set_xticks([])
                elif xticks:
                    ax.set_xticks(xticks)

                if i == 4 and k == 1 and j == 0:
                    ax.set_xlabel('Variants, rank-sorted')

                if k == 0:
                    ax.set_yticks(np.arange(0, n_animals))
                    if j > 0:
                        ax.set_yticklabels([])
                        ax.tick_params(axis='y', length=0)
                    else:
                        ax.set_yticklabels(animal_labels)
                else:
                    ax.set_yticks([0])
                    if j > 0:
                        ax.tick_params(axis='y', length=0)
                        ax.set_yticklabels([])
                    else:
                        ax.set_yticklabels(['All'])
                ax.tick_params(axis='y', labelsize=7)

                if k == 0 and j == 0:
                    ax.text(-0.065, 0.5, row_title, transform=ax.transAxes,
                            ha='right', va='center')


    gs_cbar = fig.add_gridspec(1, 1, left=image_right + cbar_padding, right=image_right+cbar_padding+cbar_width, 
                               top=0.723, bottom=bottom + 0.045)
    ax = fig.add_subplot(gs_cbar[0, 0])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), 
            cmap=cmap
        ), 
        cax=ax
    )
    cbar.set_ticks([-10, -5, 0, 5, 10])
    cbar.set_label('Log2 Enrichment', rotation=90, labelpad=5, y=0.45)
    ax.tick_params(axis='y', labelsize=7, pad=1)
    

    gs_nd = fig.add_gridspec(1, 1, left=image_right + cbar_padding, right=image_right+cbar_padding+cbar_width,
                             top = bottom + 0.03, bottom = bottom)
    ax = fig.add_subplot(gs_nd[0, 0])
    ax.set_facecolor('#CCC')
    ax.set_xticks([])
    ax.set_yticks([0.5])
    ax.set_yticklabels(['ND'], fontsize=7)
    ax.set_ylim([0, 1])
    ax.yaxis.set_ticks_position('right')
    ax.tick_params(axis='y', length=0, pad=2)


    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path, dff_heatmap.drop(columns=['source_orig', 'invivo'])

####### Fig 2S3D #######

def bi30_logo(dff_heatmap, fig_outdir='figures', figname='fig2S3D'):
    
    dff_cluster = dff_heatmap.loc[dff_heatmap['SOURCE'] == 'invivo']

    fig = plt.figure(figsize=(3, 1.25), dpi=200)
    gs = fig.add_gridspec(1, 1, left=0.15, bottom=0.25, right=0.95)

    ax = fig.add_subplot(gs[0, 0])

    logomaker.Logo(
        logomaker.alignment_to_matrix(dff_cluster.loc[dff_cluster['cluster'] == 0, 'AA_sequence'].values), 
        color_scheme=clustalXAAColors, vpad=0.1,
        ax=ax
    )

    ax.set_xticks(np.arange(0, 7))
    ax.set_xticklabels(np.arange(0, 7) + 1)
    ax.set_xlabel('7-mer position')

    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel('Frequency', labelpad=4)
    
    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path