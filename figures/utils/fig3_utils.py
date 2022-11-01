
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from pathlib import Path
import re
import umap
import pickle

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering, OPTICS
from adjustText import adjust_text


from .si_formatting import *
from .fig_utils import save_fig_formats, human_format, one_hot, hamming_xor, clustalXAAColors


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

####### ---- Fig 3A + filtering, clustering and UMAP support utils---- #######

umap_pickle_paths = {'ly6a_umap': 'UMAPs/LY6A_umap_l1_l2.pickle',
                     'ly6c1_umap': 'UMAPs/LY6C1_umap_l1_l2.pickle',
                     'gm_ly6a': 'UMAPs/LY6A_gm_l1_l2.pickle',
                     'gm_ly6c1': 'UMAPs/LY6C1_gm_l1_l2.pickle',
                     'cluster_colors_6a': 'UMAPs/LY6A_cluster_colors_l1_l2.pickle',
                     'cluster_colors_6c': 'UMAPs/LY6C1_cluster_colors_l1_l2.pickle',
                    }

def get_ref_dff(dff, refs, bis, r1_trans, include_seqs):
    ref_dff = dff.loc[
        dff['reference_sequence'].isin(refs) | dff['AA_sequence'].isin(bis), 
        ['AA_sequence', 'reference_sequence']]

    ref_trans = r1_trans.transform(np.vstack(ref_dff['AA_sequence'].apply(one_hot).values))

    ref_dff['x1'] = ref_trans[:, 0]
    ref_dff['x2'] = ref_trans[:, 1]
    
    ref_dff['reference_sequence'] = ref_dff['reference_sequence'].apply(lambda x: np.nan if not x else x)
    ref_dff['name'] = ref_dff['AA_sequence'].map({v: k for k, v in include_seqs.items()})
    ref_dff['name'] = ref_dff['name'].combine_first(ref_dff['reference_sequence'])
    ref_dff = ref_dff.drop(columns=['reference_sequence'])
    
    ref_dff = ref_dff.reset_index(drop=True)
    
    return ref_dff

def prep_r2_UMAP(df, umap_pickle_paths=umap_pickle_paths):
    
    dff = df.loc[df['starter_virus_mean_RPM'] > 2**-2].copy()
    dff['reference_sequence'] = dff['reference_sequence'].fillna('')
    
    dff_filt = dff.loc[
        (dff['SOURCE'].isin(['LY6A', 'LY6A-invivo']) & (dff['LY6A_log2enr'] > 0))
        | (dff['SOURCE'].isin(['LY6C1', 'LY6C1-invivo']) & (dff['LY6C1_log2enr'] > -2))
    ]
    
    aa_seqs_6a = pd.Series(dff_filt.loc[dff_filt['SOURCE'].isin(['LY6A', 'LY6A-invivo'])]['AA_sequence'])
    aa_seqs_6c = pd.Series(dff_filt.loc[dff_filt['SOURCE'].isin(['LY6C1', 'LY6C1-invivo'])]['AA_sequence'])
    seq_one_hot_6a = np.vstack(aa_seqs_6a.apply(one_hot).values)
    seq_one_hot_6c = np.vstack(aa_seqs_6c.apply(one_hot).values)
    
    with open(umap_pickle_paths['ly6a_umap'], 'rb') as fp:
        r1_trans_6a = pickle.load(fp)

    with open(umap_pickle_paths['ly6c1_umap'], 'rb') as fp:
        r1_trans_6c = pickle.load(fp)
    
    r2_trans_6a = r1_trans_6a.transform(seq_one_hot_6a)
    r2_trans_6c = r1_trans_6c.transform(seq_one_hot_6c)
    
    umap_coord_df_6a = (
        dff_filt.loc[dff_filt['SOURCE'].isin(['LY6A', 'LY6A-invivo'])].merge(
            pd.DataFrame({
                'AA_7mer':aa_seqs_6a, 
                'x1': r2_trans_6a[:, 0], 
                'x2': r2_trans_6a[:, 1]
            }),
            left_on='AA_sequence', right_on='AA_7mer', how='inner'
        ).reset_index(drop=True).drop(columns=['AA_7mer'])
    )
    umap_coord_df_6c = (
        dff_filt.loc[dff_filt['SOURCE'].isin(['LY6C1', 'LY6C1-invivo'])].merge(
            pd.DataFrame({
                'AA_7mer':aa_seqs_6c, 
                'x1': r2_trans_6c[:, 0], 
                'x2': r2_trans_6c[:, 1]
            }),
            left_on='AA_sequence', right_on='AA_7mer', how='inner'
        ).reset_index(drop=True).drop(columns=['AA_7mer'])
    )
    
    with open(umap_pickle_paths['gm_ly6a'], 'rb') as fp:
        gm_ly6a = pickle.load(fp)

    with open(umap_pickle_paths['gm_ly6c1'], 'rb') as fp:
        gm_ly6c1 = pickle.load(fp)

    with open(umap_pickle_paths['cluster_colors_6a'], 'rb') as fp:
        cluster_colors_6a = pickle.load(fp)

    with open(umap_pickle_paths['cluster_colors_6c'], 'rb') as fp:
        cluster_colors_6c = pickle.load(fp)
        
    
    refs_6a = ['PHP.B']
    refs_6c = ['AAVF']

    include_seqs = {
        'BI48': 'TPQRPFI',
        'BI49': 'VDFVPPR',
        'BI28': 'KSVGSVY',
        'BI62': 'PKNDGRW',
        'BI65': 'IRTGYAQ'
    }

    bis_6a = ['BI48', 'BI49']
    bis_6c = ['BI28', 'BI62', 'BI65']

    bis_6a = [include_seqs[b] for b in bis_6a]
    bis_6c = [include_seqs[b] for b in bis_6c]
    
    ref_dff_6a = get_ref_dff(dff, refs_6a, bis_6a, r1_trans_6a, include_seqs)
    ref_dff_6c = get_ref_dff(dff, refs_6c, bis_6c, r1_trans_6c, include_seqs)
    
    return (umap_coord_df_6a, umap_coord_df_6c, 
            seq_one_hot_6a, seq_one_hot_6c, 
            ref_dff_6a, ref_dff_6c,
            gm_ly6a, gm_ly6c1,
            cluster_colors_6a, cluster_colors_6c)
    

####### Fig 3A,B #######

def plot_r2_UMAP(dff, umap_pickle_paths=umap_pickle_paths, fig_outdir='figures', figname='fig3AB'):
    
    (umap_coord_df_6a, umap_coord_df_6c, 
        seq_one_hot_6a, seq_one_hot_6c, 
        ref_dff_6a, ref_dff_6c,
        gm_ly6a, gm_ly6c1,
        cluster_colors_6a, cluster_colors_6c) = prep_r2_UMAP(dff, umap_pickle_paths=umap_pickle_paths)
    
    left = 0.15
    right = 0.95
    bottom = 0.08
    top = 0.9

    gap = 0.1
    col_width = ((right-left)-(gap))/3
    gs1_left = left
    gs1_right = left + col_width

    gs2_left = left + col_width + gap
    gs2_right = right

    fig = plt.figure(figsize=(6.0, 2.), dpi=200)
    gs1 = fig.add_gridspec(2, 1, wspace=0.03, hspace=0.05, left=gs1_left, right=gs1_right, bottom=bottom, top=top)
    gs2 = fig.add_gridspec(2, 2, wspace=0.03, hspace=0.05, left=gs2_left, right=gs2_right, bottom=bottom, top=top)

    bc_cols = [col for col in dff.columns if any(['BalbC-B{}_log2enr'.format(i) in col for i in range(1,4)])]

    c57_cols = [col for col in dff.columns if any(['C57-B{}_log2enr'.format(i) in col for i in range(1,4)])]

    for i, umap_df, seq_one_hot, ref_dff, invitro_col, invitro_thresh, title, cluster_prefix, gm, cluster_colors, n_clusters in zip(
            range(2), 
            [umap_coord_df_6a, umap_coord_df_6c], [seq_one_hot_6a, seq_one_hot_6c], [ref_dff_6a, ref_dff_6c], 
            ['LY6A_log2enr', 'LY6C1_log2enr'], [0, -2], ['LY6A-Fc', 'LY6C1-Fc'], ['A', 'C'],
            [gm_ly6a, gm_ly6c1], [cluster_colors_6a, cluster_colors_6c], [40, 40]
        ):

        xmin = umap_df['x1'].min()
        xmax = umap_df['x1'].max()
        xlim_padding = 2
        new_xlim = [xmin - xlim_padding, xmax + xlim_padding]
        xmid = xmin + ((xmax - xmin) / 2)

        ax = fig.add_subplot(gs1[i, 0])
        clustering = gm.predict(umap_df[['x1', 'x2']].values)
        umap_df['cluster'] = clustering

        umap_cluster_df = (
            umap_df
            .loc[umap_df['cluster'] >= 0]
            .groupby('cluster')
            .agg(
                n=('AA_sequence', len),
                LY6A=('LY6A_log2enr', np.mean), LY6C1=('LY6C1_log2enr', np.mean), Fc=('Fc_log2enr', np.mean),
                x1=('x1', np.mean), x2=('x2', np.mean),
            )
        )

        sc = ax.scatter(umap_df['x1'], umap_df['x2'], 
                   c=umap_df['cluster'].apply(lambda x: cluster_colors[x]),
                   s=0.25, edgecolor='none', rasterized=True)

        cluster_labels = []
        for k, row in enumerate(umap_cluster_df.iterrows()):
            cluster_labels.append(
                ax.text(
                    gm.means_[k][0], gm.means_[k][1], f'{cluster_prefix}{k+1}',
                    color='k', ha='center', va='center', fontsize=4)
            )

        # REFERENCES
        # Label points
        ax.scatter(ref_dff['x1'], ref_dff['x2'], s=5, 
                   facecolor='w', edgecolor='r', linewidth=0.5)

        ref_text_x = 1.03
        ref_text_y = 0.9
        ref_text_line_y = -0.15
        for k, row in ref_dff.iterrows():
            ax.text(row['x1'], row['x2'], row['name'], fontsize=6)

        # Parameters
        ax.text(0.03, 0.03, f'k = {n_clusters}', transform=ax.transAxes, ha='left', va='bottom', fontsize=5)
        ax.set_xticks([]); ax.set_yticks([])

        ax.text(-0.2, 0.5, title, transform=ax.transAxes, ha='right', va='center')
        if i == 0:
            # ax.set_ylabel('UMAP 2')
            ax.text(-0.02, 0, 'UMAP 2', rotation=90, transform=ax.transAxes, ha='right', va='center')
        if i == 1:
            ax.set_xlabel('UMAP 1')

        if i == 0:
            ax.set_title('Round 2 Library', loc='left')


        for j, invivo_col, invivo_thresh, invivo_title, rep_cols in zip(
                range(2), ['C57BL/6_log2enr', 'BALB/c_log2enr',], [4, 4], ['C57BL/6', 'BALB/c'], [c57_cols, bc_cols]
            ):


            ax = fig.add_subplot(gs2[i, j])

            nan_inds = umap_df[invivo_col].isna()
            sc_nan = ax.scatter(umap_df.loc[nan_inds, 'x1'], umap_df.loc[nan_inds, 'x2'],
                                s=1.0, marker='x', c='#DDD', alpha=0.5, rasterized=True, linewidths=0.2)

            high_invivo = (
                (umap_df[invivo_col] > invivo_thresh)
                & (umap_df[invitro_col] > invitro_thresh)
                & ((~umap_df[rep_cols].isna()).sum(axis=1) >= 3)
            )

            umap_df['high_' + invivo_col] = high_invivo

            sc1 = ax.scatter(umap_df.loc[~high_invivo, 'x1'], umap_df.loc[~high_invivo, 'x2'], s=0.25,
                            c='#CCCCCC', edgecolor='none', rasterized=True)
            sc2 = ax.scatter(umap_df.loc[high_invivo, 'x1'], umap_df.loc[high_invivo, 'x2'], s=1.,
                            c='#FF0000', edgecolor='none', rasterized=True)


            ax.set_xticks([]); ax.set_yticks([])
            #ax.set_xlim(new_xlim)

            if i == 1:
                ax.set_xlabel('UMAP 1')
            if i == 0 and j == 0:
                ax.text(-0.02, 0, 'UMAP 2', rotation=90, transform=ax.transAxes, ha='right', va='center')

            if i == 0:
                ax.text(0.12, 1.08, invivo_title, transform=ax.transAxes, clip_on=False,
                        ha='left', va='center', fontsize=7)
                ax.add_line(mpl.lines.Line2D(
                    (0.05, 0.05), (1.09, 1.09), transform=ax.transAxes, clip_on=False,
                    marker='o', markerfacecolor='r', markeredgecolor='none', markersize=5
                ))



    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path, umap_coord_df_6a, umap_coord_df_6c, ref_dff_6a, ref_dff_6c


####### ---- Fig 3C + reference sequence support utils ---- #######

def make_sequence_df(df):
    
    dff = df[['AA_sequence', 'reference_sequence',
              'DNA_mean_RPM', 'starter_virus_mean_RPM',
              'Fc_1_log2enr', 'Fc_2_log2enr', 'Fc_3_log2enr', 
              'LY6A_1_log2enr', 'LY6A_2_log2enr', 'LY6A_3_log2enr', 
              'LY6C1_1_log2enr', 'LY6C1_2_log2enr', 'LY6C1_3_log2enr',
              'F1-BalbC-B_log2enr', 'F2-BalbC-B_log2enr', 'M1-BalbC-B_log2enr', 'M2-BalbC-B_log2enr',
              'F1-C57-B_log2enr', 'M1-C57-B_log2enr', 'M2-C57-B_log2enr'
            ]].copy()

    
    include_references = [
        'AAV9',
        'PHP.eB',
        'AAVF',
    ]

    include_seqs = {
        '1BI48': 'TPQRPFI',
        '2BI49': 'VDFVPPR',
        '3BI28': 'KSVGSVY',
        '4BI62': 'PKNDGRW',
        '5BI65': 'IRTGYAQ'
    }

    # Filter for references, included sequences
    dff = (
        dff
        .loc[
            (dff['reference_sequence'].isin(include_references)) |
            (dff['AA_sequence'].isin(include_seqs.values()))
        ]
        # Map BI names on
        .assign(BI_name=lambda x: x['AA_sequence'].map({v: k for k, v in include_seqs.items()}))
        # Map reference order
        .assign(ref_order=lambda x: x['reference_sequence'].map({k: i for i, k in enumerate(include_references)}))
        # Sort by reference, then BI
        .sort_values(['ref_order', 'BI_name'], ascending=[True, True])
        # Drop sorting column
        .drop(columns=['ref_order'])
        # Calculate fitness
        .assign(log2fitness=lambda x: np.log2(x['starter_virus_mean_RPM'] / x['DNA_mean_RPM']))
    )
    
    # Create new name column
    dff.insert(0, 'name', dff['BI_name'].combine_first(dff['reference_sequence']))

    dff.loc[dff['name'] == 'AAV9', 'AA_sequence'] = ''
    
    return dff[['AA_sequence', 'name', 'log2fitness',
                'Fc_1_log2enr', 'Fc_2_log2enr', 'Fc_3_log2enr', 
                'LY6A_1_log2enr', 'LY6A_2_log2enr', 'LY6A_3_log2enr', 
                'LY6C1_1_log2enr', 'LY6C1_2_log2enr', 'LY6C1_3_log2enr',
                'F1-C57-B_log2enr', 'M1-C57-B_log2enr', 'M2-C57-B_log2enr',
                'F1-BalbC-B_log2enr', 'F2-BalbC-B_log2enr', 'M1-BalbC-B_log2enr', 'M2-BalbC-B_log2enr']]

def ax_mat(ax, mat, cmap, vmin, vmax):

    ax.set_facecolor('#CCC')

    _mat = np.copy(mat)
    _mat[_mat < -2] = -2

    ax.imshow(
        _mat, aspect='auto', cmap=cmap, interpolation='none', 
        norm=mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
    )
    ax.set_xticks([])
    ax.set_yticks([])

def ax_cbar(ax, fig, cmap, vmin, vmax):

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
            cmap=cmap
        ), 
        cax=ax, orientation='horizontal'
    )

    for tick in ax.get_yticklabels():
        tick.set_fontsize(5)
            
####### Fig 3C #######

def plot_reference_heatmap(df, fig_outdir='figures', figname='fig3C'):
    
    dff = make_sequence_df(df)
    
    fig = plt.figure(figsize=(3.5, 5.75), dpi=150)

    # Columns for: fitness, in vitro, in vivo

    mat_widths = [1, 9, 7]

    gs_width_ratios = [mat_widths[0], mat_widths[1], mat_widths[2]]

    left = 0.17
    right = 0.96
    bottom = 0.08
    cbar_height = 0.015
    cbar_pad = 0.01
    top = 0.92

    nrows = len(dff) // 2

    gs = fig.add_gridspec(nrows, 3, left=left, right=right, bottom=bottom, top=top,
                         width_ratios=gs_width_ratios, wspace=0.05, hspace=0.1,)
    gs_cbar = fig.add_gridspec(1, 2, left=left, right=right, bottom=bottom-(cbar_height+cbar_pad), top=bottom-cbar_pad,
                               wspace=0.2, width_ratios=[3, 5])

    assay_title_y = 1.05
    assay_x_offset = 0.25
    label_x = -0.4
    label_x_line_height = 0.7
    label_y_line_height = 0.7
    cmaps = [mpl.cm.bone, mpl.cm.bwr, mpl.cm.bwr]
    vlims = [
        (-2.5, 2.5),
        (-10, 10),
        (-10, 10)
    ]

    xlabel_fontsize = 7
    xlabel_rotation = 0

    for k in range(nrows):

        row_inds = slice(k*2, (k*2)+2)

        # FITNESS
        ax = fig.add_subplot(gs[k, 0])
        mat = dff.iloc[row_inds, [2]].values
        #print(mat)
        ax_mat(ax, mat, cmaps[0], *vlims[0])

        if k == 0:
            ax.text(0.5, assay_title_y, 'Fitness', 
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                    ha='right', va='bottom', rotation=xlabel_rotation, fontsize=xlabel_fontsize)

        aa_seq = dff.iat[k*2, 0]
        name = dff.iat[k*2, 1]

        if k > 2:
            name = name[1:]
            name = name[:2] + '-' + name[2:]

        label_y = 0.2
        if name == 'AAV9':
            label_y = 0.5

        ax.text(label_x, label_y, name, ha='right', va='center',
                transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                fontsize=9, color='k')
        ax.text(label_x, label_y + label_y_line_height, aa_seq, ha='right', va='center',
                transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                fontsize=8, fontfamily='monospace', color='#888')

        # IN VITRO
        ax = fig.add_subplot(gs[k, 1])
        mat = dff.iloc[row_inds, 3:12].values
        ax_mat(ax, mat, cmaps[1], *vlims[1])

        if k == 0:
            ax.text(1, assay_title_y, 'Fc-only\ncontrol', 
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                    ha='center', va='bottom', rotation=xlabel_rotation, fontsize=xlabel_fontsize)
            ax.text(4, assay_title_y, 'LY6A-Fc', 
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                    ha='center', va='bottom', rotation=xlabel_rotation, fontsize=xlabel_fontsize)
            ax.text(7, assay_title_y, 'LY6C1-Fc', 
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                    ha='center', va='bottom', rotation=xlabel_rotation, fontsize=xlabel_fontsize)

        for p in [2.5, 5.5]:
            ax.axvline(p, linewidth=0.5, color='k')

        # IN VIVO
        ax = fig.add_subplot(gs[k, 2])
        mat = dff.iloc[row_inds, 12:19].values
        ax_mat(ax, mat, cmaps[2], *vlims[2])

        if k == 0:
            ax.text(1, assay_title_y + 0.175, 'C57BL/6J', 
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                    ha='center', va='bottom', rotation=xlabel_rotation, fontsize=xlabel_fontsize)
            for i, animal_name in enumerate(['F1', 'F2', 'M1', 'M2']):
                ax.text(i, assay_title_y, animal_name, 
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                    ha='center', va='bottom', rotation=0, fontsize=5.5)

            ax.text(4.5, assay_title_y + 0.175, 'BALB/cJ', 
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                    ha='center', va='bottom', rotation=xlabel_rotation, fontsize=xlabel_fontsize)
            for i, animal_name in enumerate(['F1', 'M1', 'M2']):
                ax.text(4 + i, assay_title_y, animal_name, 
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), 
                    ha='center', va='bottom', rotation=0, fontsize=5.5)

        ax.axvline(2.5, linewidth=0.5, color='k')


    ax = fig.add_subplot(gs_cbar[:, 0])
    ax_cbar(ax, fig, cmaps[0], *vlims[0])
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.tick_params(axis='x', labelsize=8, pad=2)
    ax.set_xlabel('Log2 Fitness', fontsize=8)

    ax = fig.add_subplot(gs_cbar[:, 1])
    ax_cbar(ax, fig, cmaps[1], *vlims[1])
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.tick_params(axis='x', labelsize=8, pad=2)
    ax.set_xlabel('Log2 Enrichment', fontsize=8)


    png_path = save_fig_formats(fig, figname, fig_outdir, dpi=200)
    plt.close()
    
    return png_path

    



