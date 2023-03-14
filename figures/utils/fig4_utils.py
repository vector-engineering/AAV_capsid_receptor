
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from pathlib import Path
import re
import copy
import umap
import pickle
from scipy.stats import gaussian_kde

from .si_formatting import *
from .fig_utils import *
from .fig_utils import engFormat

def Fig4Theme():
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


    # Default colormap - personal preference
    # 'image.cmap': 'inferno'
    })


####### ---- Fig 4A ---- #######


####### ---- Fig 4B ---- #######




####### ---- Fig 4C ---- #######


    
####### ---- Fig 4D + PWM support utils ---- #######

def make_PWMs(r2_df, ss_df):
    
    r2_dff = r2_df[['AA_sequence', 'SOURCE', 'reference', 'reference_sequence'] +
                   [col for col in r2_df.columns if 'log2enr' in col]]
    vae_df = ss_df.copy()
    
    r1_ly6a = pd.read_csv('tables/tableS2_library1_ly6a_cluster_seqs.csv')
    r1_ly6c = pd.read_csv('tables/tableS3_library1_ly6c1_cluster_seqs.csv')
    
    r2_ly6a = r2_dff.copy()
    r2_ly6a = r2_ly6a.loc[
        r2_ly6a['SOURCE'].isin(['LY6A','LY6A-invivo']) &
        (r2_ly6a['LY6A_log2enr'] > 0)
    ]
    
    r2_ly6c = r2_dff.copy()
    r2_ly6c = r2_ly6c.loc[
        r2_ly6c['SOURCE'].isin(['LY6C1','LY6C1-invivo']) &
        (r2_ly6c['LY6C1_log2enr'] > -2)
    ]
    
    vae_ly6a = vae_df[(vae_df['Model'] == 'SVAE') & (vae_df['Source'] == '9K') & (vae_df['Receptor'] == 'LY6A')]
    vae_ly6c = vae_df[(vae_df['Model'] == 'SVAE') & (vae_df['Source'] == '9K') & (vae_df['Receptor'] == 'LY6C1')]
    naive_ly6a = vae_df[(vae_df['Model'] == 'SM') & (vae_df['Receptor'] == 'LY6A')]
    naive_ly6c = vae_df[(vae_df['Model'] == 'SM') & (vae_df['Receptor'] == 'LY6C1')]
    
    mats_ly6a = [r2_ly6a, naive_ly6a, vae_ly6a]
    mats_ly6a = [aa_to_matrix(_df['AA_sequence']) for _df in mats_ly6a]

    mats_ly6c = [r2_ly6c, naive_ly6c, vae_ly6c]
    mats_ly6c = [aa_to_matrix(_df['AA_sequence']) for _df in mats_ly6c]

    return mats_ly6a, mats_ly6c

####### Fig 4D #######

def plot_PWM(r2_df, ss_df, fig_outdir='figures', figname='fig4D'):
    
    mats_ly6a, mats_ly6c = make_PWMs(r2_df, ss_df)
    
    vmin = -0.1
    vmax = 0.1
    cmap = mpl.cm.coolwarm
    n_pos = 7

    bottom = 0.08
    top = 0.9
    left = 0.25
    heatmap_right = 0.81

    fig = plt.figure(figsize=(5, 1.5), dpi=250)
    gs = fig.add_gridspec(3, 2, left=left, right=heatmap_right, bottom=bottom, top=top,
                         wspace=0.1, hspace=0.05)

    for k, mats, row_title in zip(range(2), [mats_ly6a, mats_ly6c], ['LY6A-Fc', 'LY6C1-Fc']):

        for i, mat, col_title in zip(
                range(3),
                mats,
                ['Round 2', 'Saturation\nMutagenesis', 'SVAE']
            ):

            ax = fig.add_subplot(gs[i, k])

            # entropy = scipy.stats.entropy(mat, base=2, axis=0)
            # total_entropy = np.around(entropy.sum(), decimals=1)
            # print('Stat. entropy per column: {}'.format(entropy))
            # print('Total entropy (sum across columns): {}'.format(total_entropy))

            hm = sns.heatmap(
                mat.T,
                vmin=vmin, center=0, vmax=vmax,
                cmap=cmap, square=False, cbar=False, ax=ax
            )

            ax.set_xticks(np.arange(0, len(aa_alphabet)) + 0.5)
            if i == 2:
                ax.set_xticklabels(aa_alphabet, fontsize=5, fontfamily='monospace')
                for tick in ax.get_xticklabels():
                    tick.set_rotation(0)
            else:
                ax.set_xticklabels([])

            ax.set_yticks(np.arange(0, n_pos) + 0.5)
            ax.set_yticklabels([str(x) for x in np.arange(0, n_pos) + 1], fontsize=5)
            for tick in ax.get_yticklabels():
                tick.set_rotation(0)

            ax.tick_params(axis='x', pad=1, length=0)
            ax.tick_params(axis='y', pad=1, length=0)

            if i == 0:
                #ax.text(-0.25, 0.5, row_title, transform=ax.transAxes, ha='right', va='center', 
                #        rotation=0, fontsize=7)
                ax.set_title(row_title, fontsize=7, loc='left', pad=1)

            if k == 0:
                ax.text(-0.25, 0.5, col_title, transform=ax.transAxes, ha='right', va='center', 
                        rotation=0, fontsize=7)

            if k == 0 and i == 1:
                ax.set_ylabel('7-mer Position', fontsize=7)
                # ax.text(0, -0.08, '7-mer position', transform=ax.transAxes, ha='center', va='top', fontsize=7)

    cbar_pad = 0.02
    cbar_width = 0.015

    gs_cbar = fig.add_gridspec(1, 1, left=heatmap_right + cbar_pad, right=heatmap_right + cbar_pad + cbar_width, 
                               bottom=bottom + 0.02, top=top - 0.02)
    ax = fig.add_subplot(gs_cbar[0, 0])

    colornorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(
        ax, 
        cmap=cmap, norm=colornorm,
        values=np.linspace(vmin, vmax, 100),
        ticks=[-0.1, -0.05, 0, 0.05, 0.1],
    )
    cb1.set_label(
        'AA frequency relative to uniform', 
        rotation=90, labelpad=8, fontsize=7
    )
    # ax.set_title('AA frequency relative to expected', fontsize=7)
    # ax.yaxis.set_ticks_position('right')
    # ax.yaxis.set_label_position('right')

    ax.tick_params(axis='y', length=2, pad=1, labelsize=6)
    
    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path


####### ---- Fig 4E + UMAP and clustering support utils ---- #######

def prep_sequence_exploration_UMAP(r2_df, ss_df):
    
    print('Preparing UMAP clusters...')
    print('This will take on the order of 10 minutes.')
    
    vae_fitness_col = 'log2fitness'
    qhr2_fitness_col = 'log2fitness'

    # Filter on fitness
    fitness_cutoff = -9999 

    cluster_params = {       'LY6A': {'eps': {'SM': 0.5, 'SVAE': 0.2, 'R2': 0.2}, 'minn': {'SM': 5, 'SVAE': 5, 'R2': 5}},
                            'LY6C1': {'eps': {'SM': 0.5, 'SVAE': 0.2, 'R2': 0.2}, 'minn': {'SM': 5, 'SVAE': 5, 'R2': 5}},
        'Brain Transduction (LY6A)': {'eps': {'SM': 0.5, 'SVAE': 0.2, 'R2': 0.2}, 'minn': {'SM': 5, 'SVAE': 5, 'R2': 5}},
        'Brain Transduction (LY6C1)':{'eps': {'SM': 0.5, 'SVAE': 0.2, 'R2': 0.2}, 'minn': {'SM': 5, 'SVAE': 5, 'R2': 5}},
                     }


    umap_params =  {     'LY6A':   {'nn': {'SM': 5, 'SVAE': 5, 'R2': 5}, 'min_dist': {'SM': 0.01, 'SVAE': 0.01, 'R2': 0.01}},
                         'LY6C1':  {'nn': {'SM': 5, 'SVAE': 5, 'R2': 5}, 'min_dist': {'SM': 0.01, 'SVAE': 0.01, 'R2': 0.01}},
     'Brain Transduction (LY6A)':  {'nn': {'SM': 5, 'SVAE': 5, 'R2': 5}, 'min_dist': {'SM': 0.01, 'SVAE': 0.01, 'R2': 0.01}},
     'Brain Transduction (LY6C1)': {'nn': {'SM': 5, 'SVAE': 5, 'R2': 5}, 'min_dist': {'SM': 0.01, 'SVAE': 0.01, 'R2': 0.01}},
                     }
    
    overlap = get_overlap(ss_df, r2_df)
    
    get_calibrations = {}
    for assay in ['LY6A', 'LY6C1', 'brain_trans']:
        calib_output = prep_calibration(overlap, assay, hist=True)
        get_calibrations[assay] = calib_output[7]
    
    
    outdata = {}

    calibration = {}
    calibration['LY6C1'] = get_calibrations['LY6C1'] 
    calibration['Brain Transduction (LY6A)'] = get_calibrations['brain_trans']
    calibration['Brain Biodistribution'] = np.nan
    calibration['LY6A'] = get_calibrations['LY6A']
    calibration['Brain Transduction (LY6C1)'] = get_calibrations['brain_trans']
    
    receptor = {'LY6C1': 'LY6C1',
                'Brain Transduction (LY6A)': 'LY6A',
                'Brain Biodistribution': None,
                'LY6A': 'LY6A',
                'Brain Transduction (LY6C1)': 'LY6C1'
               }

    vae_val_col = {'LY6C1': 'LY6C1_log2enr',
                    'Brain Transduction (LY6A)': 'trans_brain_log2enr',
                    'Brain Biodistribution': 'biod_brain_log2enr',
                    'LY6A': 'LY6A_log2enr',
                    'Brain Transduction (LY6C1)': 'trans_brain_log2enr'
                   }

    qhr2_val_col = {'LY6C1': 'LY6C1_log2enr',
                    'Brain Transduction (LY6A)': 'C57BL/6_log2enr',
                    'Brain Biodistribution': 'C57BL/6_log2enr',
                    'LY6A': 'LY6A_log2enr',
                    'Brain Transduction (LY6C1)': 'C57BL/6_log2enr'
                   }
    
    vae_shuffle = ss_df.copy()
    qhr2_shuffle = r2_df.copy() 
    
    dataset_idents = [
        ('SM', None),
        ('SVAE', '9K'),
        ('Control', None),
        ('Positive_Control', 'SVAE'),
        ('Reference', None),
        ('StopCodon', None)
    ]

    dataset_labels = [
        'SM', 
        'SVAE',
        'Control',
        'Positive Control',
        'Reference Sequences',
        'Stop Codon Controls'
    ]
    
    print('Loading UMAP models...')
    
    with open('UMAPs/LY6A_umap_l1_l2.pickle', 'rb') as fp:
        trans_6a = pickle.load(fp)
        
    with open('UMAPs/LY6A_gm_l1_l2.pickle', 'rb') as fp:
        gm_ly6a = pickle.load(fp)
    
    with open('UMAPs/LY6C1_umap_l1_l2.pickle', 'rb') as fp:
        trans_6c = pickle.load(fp)

    with open('UMAPs/LY6C1_gm_l1_l2.pickle', 'rb') as fp:
        gm_ly6c = pickle.load(fp)
        
    # First load UMAP models and transform

    
    for i in range(0, 5):
        if i == 2:
            continue

        if i == 0:
            # Ly6C
            [is_ly6a, is_ly6c, is_brain_trans, is_brain_biod, dataname] = [False, True, False, False, 'LY6C1']

        elif i == 1:
            # Brain transduciton ly6a
            [is_ly6a, is_ly6c, is_brain_trans, is_brain_biod, dataname] = [True, False, True, False, 'Brain Transduction (LY6A)']

        elif i == 2:
            # Brain Biodistribution
            [is_ly6a, is_ly6c, is_brain_trans, is_brain_biod, dataname] = [False, False, True, False, 'Brain Biodistribution']

        elif i == 3:
            # Ly6A
            [is_ly6a, is_ly6c, is_brain_trans, is_brain_biod, dataname] = [True, False, False, False, 'LY6A']

        elif i == 4:
            # brain transduction ly6c
            [is_ly6a, is_ly6c, is_brain_trans, is_brain_biod, dataname] = [False, True, True, False, 'Brain Transduction (LY6C1)']

        else:
            raise Exception('Unknown i')

        if i == 2:
            continue

        k = 3
        
        print('Computing for '  + dataname)

        if is_brain_biod:
            vaeIdx = (vae_shuffle['Model'] == 'SVAE') & (vae_shuffle['Source'] == '9K')
            naiveIdx = (vae_shuffle['Model'] == 'SM')
        else:
            vaeIdx = (vae_shuffle['Model'] == 'SVAE') & (vae_shuffle['Receptor'] == receptor[dataname]) & (vae_shuffle['Source'] == '9K')
            naiveIdx = (vae_shuffle['Model'] == 'SM') & (vae_shuffle['Receptor'] == receptor[dataname])

        if is_brain_trans:
            if is_ly6a:
                # print('r2_idx:')
                qhr2_idx = qhr2_shuffle['SOURCE'] == 'asdfasdfas'
                for c in list(set(qhr2_shuffle['SOURCE'])):
                    if 'LY6A' in c:
                        # Add to list of indexes
                        idx2 = qhr2_shuffle['SOURCE'] == c
                        qhr2_idx |= idx2
                        # print('Added ' + c + ' to list, giving us ' + str(np.sum(qhr2_idx)) + ' rows.')
            elif is_ly6c:
                # print('r2_idx:')
                qhr2_idx = qhr2_shuffle['SOURCE'] == 'asdfasdfas'
                for c in list(set(qhr2_shuffle['SOURCE'])):
                    if 'LY6C1' in c:
                        # Add to list of indexes
                        idx2 = qhr2_shuffle['SOURCE'] == c
                        qhr2_idx |= idx2
                        # print('Added ' + c + ' to list, giving us ' + str(np.sum(qhr2_idx)) + ' rows.')
            else:
                raise Exception('unknown ly6')
        elif is_ly6c:
            # print()
            # print('r2_idx:')
            qhr2_idx = qhr2_shuffle['SOURCE'] == 'asdfasdfas'
            for c in list(set(qhr2_shuffle['SOURCE'])):
                if 'LY6C1' in c:
                    # Add to list of indexes
                    idx2 = qhr2_shuffle['SOURCE'] == c
                    qhr2_idx |= idx2
                    # print('Added ' + c + ' to list, giving us ' + str(np.sum(qhr2_idx)) + ' rows.')
        elif is_ly6a:
            # print()
            # print('r2_idx:')
            qhr2_idx = qhr2_shuffle['SOURCE'] == 'asdfasdfas'
            for c in list(set(qhr2_shuffle['SOURCE'])):
                if 'LY6A' in c:
                    # Add to list of indexes
                    idx2 = qhr2_shuffle['SOURCE'] == c
                    qhr2_idx |= idx2
                    # print('Added ' + c + ' to list, giving us ' + str(np.sum(qhr2_idx)) + ' rows.')
        elif is_brain_biod:
            pass
        else:
            raise Exception('unknown assay')

        if not is_brain_biod:
            qhr2_idx = qhr2_idx

        umap_inputs = {
            'SVAE': {
                'data': vae_shuffle[vaeIdx],
                'AA_col': 'AA_sequence',
                'val_col': vae_val_col[dataname]
            },
            'SM': {
                'data': vae_shuffle[naiveIdx],
                'AA_col': 'AA_sequence',
                'val_col': vae_val_col[dataname],
            },
            'R2': {
                'data': qhr2_shuffle[qhr2_idx],
                'AA_col': 'AA_sequence',
                'val_col': qhr2_val_col[dataname],
            }
        }
        
        if 'LY6A' in dataname:
            # ly6 = 'ly6a'
            trans_6 = trans_6a
            gm_ly6 = gm_ly6a

        elif 'LY6C1' in dataname:
            # ly6 = 'ly6c'
            trans_6 = trans_6c
            gm_ly6 = gm_ly6c

        else:
            raise Exception('unknown ly6')

        # Get the one-hot for this plot.
        outdata[dataname] = {}
        cluster_data = {}
        outdata[dataname]['data'] = {}
        for key in umap_inputs:
            # Because we are no longer dropping values for fitness or for assay value, we can reuse VAE/sat mut clustering results.
            if dataname == 'LY6A' or dataname == 'Brain Transduction (LY6C1)':
                if key == 'SVAE' or key == 'SM':
                    continue

            print('Computing umap/clusters for ' + key + '...')

            data = umap_inputs[key]['data'].copy()
            AA_col = umap_inputs[key]['AA_col']
            val_col = umap_inputs[key]['val_col']

            aa_seqs = pd.Series(data[AA_col])
            vals = pd.Series(data[val_col])
            seq_one_hot = np.vstack(aa_seqs.apply(one_hot).values)

            # print('UMAP on ' + engFormat(len(seq_one_hot)))

            res_6 = trans_6.transform(seq_one_hot)
            data['x1'] = res_6[:, 0]
            data['x2'] = res_6[:, 1]

            # print('Cluster on ' + engFormat(len(data)))

            # Cluster assignments
            data['cluster'] = gm_ly6.predict(data[['x1', 'x2']].values)
            outdata[dataname]['data'][key] = data

            # For each cluster, get the maximum values and the cluster size

            df = data

            cluster_data[key] = {'lengths': [], 'max_vals': [], 'ids': [], 'mean_top_5_cluster_val': [], 'mean_top_10_cluster_val': []}

            for cluster_id in list(set(df['cluster'])):

                max_cluster_val = df[df['cluster'] == cluster_id][val_col].max()

                sorted = df[df['cluster'] == cluster_id][val_col].sort_values()
                top_5_idx = round(len(sorted)*0.95)

                if top_5_idx >= len(sorted):
                    top_5_idx = len(sorted)-1

                top_10_idx = round(len(sorted)*0.90)
                if top_10_idx >= len(sorted):
                    top_10_idx = len(sorted)-1

                mean_top_5_cluster_val = sorted.iloc[top_5_idx:].mean()

                mean_top_10_cluster_val = sorted.iloc[top_10_idx:].mean()

                cluster_data[key]['lengths'].append(np.sum(df['cluster'] == cluster_id))
                cluster_data[key]['max_vals'].append(max_cluster_val)
                cluster_data[key]['mean_top_5_cluster_val'].append(mean_top_5_cluster_val)
                cluster_data[key]['mean_top_10_cluster_val'].append(mean_top_10_cluster_val)
                cluster_data[key]['ids'].append(cluster_id)

        outdata[dataname]['cluster_data'] = copy.deepcopy(cluster_data)
        outdata[dataname]['umap_inputs'] = copy.deepcopy(umap_inputs)
    
    with open('UMAPs/LY6A_cluster_colors_l1_l2.pickle', 'rb') as fp:
        cluster_colors_6a = pickle.load(fp)

    with open('UMAPs/LY6C1_cluster_colors_l1_l2.pickle', 'rb') as fp:
        cluster_colors_6c = pickle.load(fp)

    # Load Fig1 data for gray background.
    ly6a_fig1_df = pd.read_csv('UMAPs/LY6A_joint_umap_l1_l2.csv')
    ly6c_fig1_df = pd.read_csv('UMAPs/LY6C1_joint_umap_l1_l2.csv')

    # Copy outdata that is identical.

    outdata['LY6A']['data']['SM'] = outdata['Brain Transduction (LY6A)']['data']['SM']
    outdata['LY6A']['data']['SVAE'] = outdata['Brain Transduction (LY6A)']['data']['SVAE']

    outdata['Brain Transduction (LY6C1)']['data']['SM'] = outdata['LY6C1']['data']['SM'] 
    outdata['Brain Transduction (LY6C1)']['data']['SVAE'] = outdata['LY6C1']['data']['SVAE'] 

    outdata['LY6A']['cluster_data']['SM'] = None 
    outdata['LY6A']['cluster_data']['SVAE'] = None 

    outdata['Brain Transduction (LY6C1)']['cluster_data']['SM'] = None 
    outdata['Brain Transduction (LY6C1)']['cluster_data']['SVAE'] = None 
    
    print('Finished preparing UMAP clusters.')
    
    prep_output = [outdata, cluster_colors_6a, cluster_colors_6c, ly6a_fig1_df, ly6c_fig1_df]
    
    print('Pickling UMAP prep.')
    with open('UMAPs/fig4E_prep_output.pickle', 'wb') as f:
        pickle.dump(prep_output, f)
    
    return prep_output

####### Fig 4E #######

def plot_sequence_exploration_UMAPs(r2_df, ss_df, fig_outdir='figures', figname='fig4E'):
    
    with open('UMAPs/fig4E_prep_output.pickle', 'rb') as f:
        prep_output = pickle.load(f)

    [outdata, cluster_colors_6a, cluster_colors_6c, ly6a_fig1_df, ly6c_fig1_df] = prep_output
    
    vae_fitness_col = 'log2fitness'
    qhr2_fitness_col = 'log2fitness'
    
    log2enr_hit_cutoff = 3.0
    fitness_cutoff = -1.0

    color_invivo_red = True
    
    fig = plt.figure(figsize=(7.7, 5.1), dpi=200)
    big_gs = fig.add_gridspec(2, 1, hspace=0.04, left=0.1, right=0.9, bottom=0.2, top=0.8)
    
    hit_cutoff = log2enr_hit_cutoff

    for fi, plot_title in enumerate(['LY6A', 'LY6C1']): 

        #fig = big_fig.add_subfigure(big_gs[fi]) #plt.figure(figsize=(8, 2.5), dpi=200)
        gs = mpl.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=big_gs[fi], wspace=0.03)
        # gs = fig.add_gridspec(1, 3, wspace=.05, left=, right=1, bottom=0, top=1)

        # Set up cluster colors
        np.random.seed(17)
        if plot_title == 'Brain Transduction':
            np.random.seed(5)
        viridis = mpl.cm.get_cmap('nipy_spectral', 200)
        newcolors = viridis(np.random.uniform(0.1, 1, 200)) # start at 0.1 to avoid the black region of nipy_spectral since we use black for something else
        newcmp = mpl.colors.ListedColormap(newcolors)

        if plot_title == 'LY6A':
            cluster_colors = cluster_colors_6a
        elif plot_title == 'LY6C1':
            cluster_colors = cluster_colors_6c
        else:
            raise Exception('unknown plot title ly6')

        
        for i, key in enumerate(['R2', 'SM', 'SVAE']):
            df1 = outdata[plot_title]['data'][key]
            df2 = outdata['Brain Transduction (' + plot_title + ')']['data'][key]

            umap_inputs = outdata[plot_title]['umap_inputs']
            umap_inputs_braintrans = outdata['Brain Transduction (' + plot_title + ')']['umap_inputs']

            cols = list(df1.columns)

            # Combine the two dfs

            # (barrya): UMAP transforms are non-deterministic, uggg:
            # https://github.com/lmcinnes/umap/issues/158
            #
            # We will arbitrarily take the first result (x1_x / x2_x).  They are pretty close at least,
            # as the author says in issue 158:
            #    "Ideally the difference should be small, but I can't give any explicit guarantees."
            cols.remove('x1')
            cols.remove('x2')
            umap_coord_df = df1.merge(df2, on=cols, suffixes=(None, '_y'), how='outer')
            plot_array = [umap_coord_df]

            if key == 'R2':
                fitness_col = qhr2_fitness_col
            else:
                fitness_col = vae_fitness_col

            if hit_cutoff is not None:
                if color_invivo_red:
                    #print('hit on ', umap_inputs_braintrans[key]['val_col'])
                    hit_idx = umap_coord_df[umap_inputs_braintrans[key]['val_col']] > hit_cutoff
                    #print(np.sum(hit_idx)/len(umap_coord_df))
                else:
                    #print('hit on ', umap_inputs[key]['val_col'])
                    hit_idx = umap_coord_df[umap_inputs[key]['val_col']] > hit_cutoff
                    #print(np.sum(hit_idx)/len(umap_coord_df))
            else:
                hit_idx = np.ones(len(umap_coord_df), dtype=bool)

            
            ax = fig.add_subplot(gs[i])
            if i == 0:
                ax0 = ax

            if plot_title == 'LY6A':
                ax.scatter(ly6a_fig1_df['x1'], ly6a_fig1_df['x2'], color='#EDEDED', s=.05, rasterized=True) ##
            elif plot_title == 'LY6C1':
                ax.scatter(ly6c_fig1_df['x1'], ly6c_fig1_df['x2'], color='#EDEDED', s=.05, rasterized=True)
            else:
                raise Exception('unknown plot_title')

            fit_idx = umap_coord_df[fitness_col] > fitness_cutoff

            hit_idx = hit_idx & fit_idx

            ax.scatter(umap_coord_df[fit_idx]['x1'],umap_coord_df[fit_idx]['x2'],
                    c='#7171FF55', s=.5, edgecolor='none', rasterized=True)

            if hit_cutoff is not None:
                ax.scatter(umap_coord_df[hit_idx]['x1'],umap_coord_df[hit_idx]['x2'],
                        c='r', s=.5, edgecolor='none', rasterized=True)

            if fi == 1:
                ax.set_xlabel('UMAP 1', fontsize=8)
            if i == 0:
                ax.set_ylabel('UMAP 2', fontsize=8)
            title = ''
            if key == 'SVAE':
                title = 'SVAE'
            elif key == 'SM':
                title = 'Saturation Mutagenesis'
            elif key == 'R2':
                title = 'Round 2 Library'
            num_clusters = len(list(set(umap_coord_df['cluster'])))
            clust_str = 'clusters'
            if num_clusters == 1:
                clust_str = 'cluster'
            
            if fi == 0:
                ax.set_title(title, fontsize=11)

            xmin, xmax = ax.get_xlim()
            center_x = (xmax + xmin)/2
            ymin, ymax = ax.get_ylim()
            center_y = (ymax + ymin)/2
            delta = 7

            if plot_title == 'LY6A':
                ax.set_xlim([10.8, 22])
                ax.set_ylim([1.8, 11.5])
            elif plot_title == 'LY6C1':
                ax.set_xlim([4, 15])
                ax.set_ylim([-6, 2])
            else:
                raise Exception('unknown plot_title')

            ax.set_xticks([])
            ax.set_yticks([])


#         hitname_str = ''
#         if hit_cutoff is not None:
#             hitname_str = '_hits'

#         hit_text = ''
#         if hit_cutoff is not None:
#             hit_text = '        Log$_2$ Enrichment > ' + str(hit_cutoff)
        fig.text(-0.35, 0.5, plot_title, ha="left", transform=ax0.transAxes, fontsize=10)

            
    png_path = save_fig_formats(fig, figname, fig_outdir, dpi=250)
    plt.close()
    
    return png_path


####### ---- Fig 4F + support utils ---- #######

def make_cluster_data(outdata):
    # Now that we are not filtering for fitness or hits above, we need to implement those filters.
    # We will generate cluster_data here from the dfs.
    
    vae_fitness_col = 'log2fitness'
    qhr2_fitness_col = 'log2fitness'
    
    log2enr_hit_cutoff = 3.0
    fitness_cutoff = -1.0

    color_invivo_red = True

    # print('Fitness cutoff is: ' + str(fitness_cutoff))

    # html = '<table><tr><th>Assay</th><th>Key</th><th>Val col</th><th>Fitness col</th><th>isfinite_removed</th><th>Fitness removed</th><th>All removed</th><th>% removed</th><th>Total remaining</th></tr>'
    for assay in outdata:
        cluster_data = {}
        for key in outdata[assay]['data']:
            val_col = outdata[assay]['umap_inputs'][key]['val_col']
            #print(assay, '|', key, '|', val_col)

            if key == 'R2':
                fitness_col = qhr2_fitness_col
            else:
                fitness_col = vae_fitness_col

            # html += '<tr><td>' + assay + '</td><td>'+key+'</td><td>'+val_col+'</td><td>'+fitness_col+'</td>'

            df = outdata[assay]['data'][key]

            cluster_data[key] = {'lengths': [], 'max_vals': [], 'ids': [], 'mean_top_5_cluster_val': [], 'mean_top_10_cluster_val': []}
            # Filter out low fitness

            isfinite_idx = (np.isfinite(df[val_col])) & (np.isfinite(df[fitness_col]))
            # html += '<td>' + str(len(df) - np.sum(isfinite_idx)) + '</td>'

            fitness_idx = df[fitness_col] > fitness_cutoff

            # html += '<td>' + str(len(df) - np.sum(fitness_idx)) + '</td>'

            include_idx = (fitness_idx) & (isfinite_idx)
            # html += '<td>' + str(len(df) - np.sum(include_idx)) + '</td>'
            # html += '<td>{:.2%}</td>'.format((len(df)-len(df[include_idx]))/len(df))
            # html += '<td>'+str(np.sum(include_idx))+'</td>'
            # html += '</tr>'

            df['included_in_fig4'] = None
            df['isfinite_all'] = False
            cluster_data[key]['rows'] = df[include_idx]

            for cluster_id in list(set(df[include_idx]['cluster'])):

                max_cluster_val = df[include_idx][df[include_idx]['cluster'] == cluster_id][val_col].max()

                sorted = df[include_idx][df[include_idx]['cluster'] == cluster_id][val_col].sort_values()
                top_5_idx = round(len(sorted)*0.95)

                if top_5_idx >= len(sorted):
                    top_5_idx = len(sorted)-1

                top_10_idx = round(len(sorted)*0.90)
                if top_10_idx >= len(sorted):
                    top_10_idx = len(sorted)-1

                mean_top_5_cluster_val = sorted.iloc[top_5_idx:].mean()
                mean_top_10_cluster_val = sorted.iloc[top_10_idx:].mean()

                cluster_data[key]['lengths'].append(np.sum(df[include_idx]['cluster'] == cluster_id))
                cluster_data[key]['max_vals'].append(max_cluster_val)
                cluster_data[key]['mean_top_5_cluster_val'].append(mean_top_5_cluster_val)
                cluster_data[key]['mean_top_10_cluster_val'].append(mean_top_10_cluster_val)
                cluster_data[key]['ids'].append(cluster_id)
                df.loc[include_idx, 'isfinite_all'] = True
        outdata[assay]['cluster_data'] = cluster_data
    
    return outdata
    
####### Fig 4F #######

def plot_cluster_scatter(r2_df, ss_df, fig_outdir='figures', figname='fig4F'):
    
    with open('UMAPs/fig4E_prep_output.pickle', 'rb') as f:
        prep_output = pickle.load(f)
        
    [outdata, cluster_colors_6a, cluster_colors_6c, ly6a_fig1_df, ly6c_fig1_df] = prep_output
    
    overlap = get_overlap(ss_df, r2_df)
    
    get_calibrations = {}
    for assay in ['LY6A', 'LY6C1', 'brain_trans']:
        calib_output = prep_calibration(overlap, assay, hist=True)
        get_calibrations[assay] = calib_output[7]
        
    calibration = {}
    calibration['LY6C1'] = get_calibrations['LY6C1'] 
    calibration['Brain Transduction (LY6A)'] = get_calibrations['brain_trans']
    calibration['Brain Biodistribution'] = np.nan
    calibration['LY6A'] = get_calibrations['LY6A']
    calibration['Brain Transduction (LY6C1)'] = get_calibrations['brain_trans']
    
    receptor = {'LY6C1': 'LY6C1',
                'Brain Transduction (LY6A)': 'LY6A',
                'Brain Biodistribution': None,
                'LY6A': 'LY6A',
                'Brain Transduction (LY6C1)': 'LY6C1'
               }

    vae_val_col = {'LY6C1': 'LY6C1_log2enr',
                    'Brain Transduction (LY6A)': 'trans_brain_log2enr',
                    'Brain Biodistribution': 'biod_brain_log2enr',
                    'LY6A': 'LY6A_log2enr',
                    'Brain Transduction (LY6C1)': 'trans_brain_log2enr'
                   }

    qhr2_val_col = {'LY6C1': 'LY6C1_log2enr',
                    'Brain Transduction (LY6A)': 'C57BL/6_log2enr',
                    'Brain Biodistribution': 'C57BL/6_log2enr',
                    'LY6A': 'LY6A_log2enr',
                    'Brain Transduction (LY6C1)': 'C57BL/6_log2enr'
                   }
    
    
    fig = plt.figure(figsize=(3.5, 3.75), dpi=200)
    gs = fig.add_gridspec(2, 2, left=0.15, right=0.95, bottom=0.10, top=0.82, hspace=0.75)
    marker_size = 2

    Fig4Theme()
    outdata = make_cluster_data(outdata)

    titles = {'LY6A': 'LY6A-Fc',
              'LY6C1': 'LY6C1-Fc',
              'Brain Transduction (LY6A)': 'LY6A-Binders',
              'Brain Transduction (LY6C1)': 'LY6C1-Binders'
             }

    max_or_5 = 'max_vals'
    xlabel_percent = {'mean_top_10_cluster_val': ' (Mean of Top 10%)', 'mean_top_5_cluster_val': ' (Mean of Top 5%)', 'max_vals': ''}

    for i, dataname in enumerate(titles):
        cluster_data = outdata[dataname]['cluster_data']
        is_brain_biod = False

        ax = plt.subplot(gs[i])

        if not is_brain_biod:
            ax.plot(cluster_data['R2'][max_or_5] + calibration[dataname], cluster_data['R2']['lengths'], 'g.', markersize=marker_size)

        ax.plot(cluster_data['SM'][max_or_5], cluster_data['SM']['lengths'], 'o', markersize=marker_size, markeredgecolor='none')
        ax.plot(cluster_data['SVAE'][max_or_5], cluster_data['SVAE']['lengths'], 'r.', markersize=marker_size)
        ax.set_xticks([-8, -4, 0, 4, 8, 12])
        ax.set_title(titles[dataname], y=0.85, x=0.05, ha="left")

        if i == 1:
            ax.legend(['Round 2', 'Saturation Mutagenesis', 'SVAE'], frameon = False, borderaxespad=0, handletextpad=0, markerscale=2, ncol=3, bbox_to_anchor=(1.1, 1.6), loc='upper right', columnspacing=0)

        ax.set_yscale('log')

        xlim = [-11, 13]#11.5]
        ylim = [.4, 2e5]
        ax.grid(visible=True, which='major', color='gray', linewidth=0.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', length=1, pad=4, color='black', which='both', labelsize=7)

        ax.set_yticks([1, 1e1, 1e2, 1e3, 1e4, 1e5])
        # ax.set_yticks([0, 1, 2, 3, 4, 5])
        if i == 1 or i == 3:
            ax.set_yticklabels([])
        if i == 0 or i == 2:
            ax.set_yticklabels([0, 1, 2, 3, 4, 5])

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        ax.grid(False)


    ax.text(-1.5, 1.05, 'Cluster Log$_2$ Size', ha="right", transform=ax.transAxes, rotation=90)
    ax.text(-0.125, 2.95, 'Pull-down Assay Binding', ha="center", transform=ax.transAxes)
    ax.text(-0.175, 1.21, 'Brain Transduction (C57BL/6)', ha="center", transform=ax.transAxes)

    ax.text(-0.175, -0.3, 'Cluster Max Log$_2$ Enrichment' + xlabel_percent[max_or_5], ha="center", transform=ax.transAxes)
    
    with open('UMAPs/fig4_clustering_outdata.pickle', 'wb') as f:
            pickle.dump([outdata, calibration], f)
    
    png_path = save_fig_formats(fig, figname, fig_outdir, dpi=200)
    plt.close()
    
    return png_path


####### ---- Fig 4S1 + support utils ---- #######

def prep_binding_fitness_scatter(df, receptor):
    
    print('Computing density kernel for {}...'.format(receptor))
    top_9K = df[df['9K']]
    
    x = df['pred_log2fitness']
    y = df['{}_pred_log2enr'.format(receptor)]

    kernel = gaussian_kde(np.vstack([x, y]))
    c = kernel(np.vstack([x, y]))

    nontop_indices = (~df['AA_sequence'].isin(top_9K['AA_sequence']))
    x_nontop = x[nontop_indices]
    y_nontop = y[nontop_indices]
    c_nontop = c[nontop_indices]

    x_top = x[~nontop_indices]
    y_top = y[~nontop_indices]
    
    return [x, y, kernel, c, nontop_indices, x_nontop, y_nontop, c_nontop, x_top, y_top]

####### Fig 4S1B #######

def plot_binding_fitness_scatter(gen_df_a, gen_df_c, fig_outdir='figures', figname='fig4S1B'):
    
    pickle_file = 'UMAPs/fig4S1B_prep_output.pickle'
    
    loaded_pickle = False
    if Path(pickle_file).is_file():
        try: 
            with open(pickle_file, 'rb') as f:
                prep_output = pickle.load(f)
                [prep_output_ly6a, prep_output_ly6c] = prep_output
                loaded_pickle = True
        except:
            pass
            
    if not loaded_pickle:
        prep_output_ly6a = prep_binding_fitness_scatter(gen_df_a, receptor='LY6A')
        prep_output_ly6c = prep_binding_fitness_scatter(gen_df_c, receptor='LY6C1')
        with open(pickle_file, 'wb') as f:
            pickle.dump([prep_output_ly6a, prep_output_ly6c],f)
    
    [x_a, y_a, kernel_a, c_a, nontop_indices_a, 
     x_a_nontop, y_a_nontop, c_a_nontop, x_a_top, y_a_top] = prep_output_ly6a
    [x_c, y_c, kernel_c, c_c, nontop_indices_c, 
     x_c_nontop, y_c_nontop, c_c_nontop, x_c_top, y_c_top] = prep_output_ly6c
    
    top_9K_ly6a = gen_df_a[gen_df_a['9K']]
    top_4K_ly6a = gen_df_a[gen_df_a['4K']]
    
    top_9K_ly6c = gen_df_c[gen_df_c['9K']]
    top_4K_ly6c = gen_df_c[gen_df_c['4K']]
    
    fig = plt.figure(figsize=(3.5, 2.75), dpi=150)
    gs = fig.add_gridspec(2, 2, wspace=0.35, right=0.7, bottom=0.2)

    cmap_density = mpl.cm.Blues_r

    ax = fig.add_subplot(gs[0,0])
    sca = ax.scatter(x_a, y_a, c=c_a, s=0.5, edgecolor='none', cmap=cmap_density, rasterized=True)
    sca_4k = ax.scatter(top_4K_ly6a['pred_log2fitness'], top_4K_ly6a['LY6A_pred_log2enr'], 
                        c='r', s=0.5, alpha=0.5, edgecolor='none', rasterized=True)

    ax.set_xticks([-8, -4, 0, 4])

    ax.set_ylabel('Predicted Log2 Enrichment', y=-0.2, ha='center')
    ax.set_title('LY6A-Fc')

    ax = fig.add_subplot(gs[0,1])
    scb = ax.scatter(x_c, y_c, c=c_c, s=0.5, edgecolor='none', cmap=cmap_density, rasterized=True)
    scb_4k = ax.scatter(top_4K_ly6c['pred_log2fitness'], top_4K_ly6c['LY6C1_pred_log2enr'], 
                        c='r', s=0.5, alpha=0.5, edgecolor='none', rasterized=True)

    ax.set_xticks([-8, -4, 0, 4])

    ax.set_title('LY6C1-Fc')

    ax = fig.add_subplot(gs[1,0])
    sca = ax.scatter(x_a_nontop, y_a_nontop, c=c_a_nontop, s=0.5, edgecolor='none', cmap=cmap_density, rasterized=True)
    scb_top = ax.scatter(x_c_top, y_c_top, c='orange', s=0.5, edgecolor='none', rasterized=True)

    ax.set_xticks([-8, -4, 0, 4])

    ax = fig.add_subplot(gs[1,1])
    scb = ax.scatter(x_c_nontop, y_c_nontop, c=c_c_nontop, s=0.5, edgecolor='none', cmap=cmap_density, rasterized=True)
    scb_top = ax.scatter(x_c_top, y_c_top, c='orange', s=0.5, edgecolor='none', rasterized=True)

    ax.set_xticks([-8, -4, 0, 4])
    ax.set_xlabel('Predicted Production Fitness', x=-0.2, ha='center')

    gs2 = fig.add_gridspec(1, 1, left=0.72, right=0.74, bottom=0.4, top=0.7)
    ax = fig.add_subplot(gs2[0, 0])
    fig.colorbar(sca, cax=ax, ticks=[])
    ax.set_ylabel('Density', rotation=270, va='bottom')

    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path
    
####### Fig 4S1C #######

def plot_fitness_ridgeplots(ss_df, fig_outdir='figures', figname='fig4S1C'):
    
    fig_theme()
    
    df = ss_df
    fig = plt.figure(figsize=(6, 2.), dpi=150)
    gs = fig.add_gridspec(1, 2, wspace=0.4, left=0.5, bottom=0.3, right=0.8)

    xlim = [-9, 4]
    xticks = np.arange(
        np.round((xlim[0] // 2)*2),
        np.round((xlim[1] // 2)*2) + 1,
        2
    )
    xticks_ecdf = np.arange(
        np.round((xlim[0] // 4)*4),
        np.round((xlim[1] // 4)*4) + 1,
        4
    )
    # bins = np.arange(xlim[0], xlim[1], 0.2)
    bins = np.linspace(xlim[0], xlim[1], 120)
    colors = ['purple', 'r', 'orange']
    left_titles = ['Saturation mutagenesis', '4K Top enrichment', '9K Joint enrichment\n+production']

    for k, receptor in enumerate(['LY6A', 'LY6C1']):
        gsa = mpl.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[k], hspace=-0.5)

        for i, (model, source) in enumerate([('SM', None), ('SVAE', '4K'), ('SVAE', '9K')]):
            ax = fig.add_subplot(gsa[i, 0])

            x = df.loc[
                (df['Model'] == model) &
                (df['Receptor'] == receptor),
            ]
            if source:
                x = x.loc[x['Source'] == source]

            nd = x['starter_virus_mean_RPM'] == 0
            print(receptor, source, nd.sum(), (nd.sum() / len(x)) * 100)

            # x = x.loc[~nd]
            x.loc[nd, 'starter_virus_mean_RPM'] = 1e-100
            x = np.log2((x['starter_virus_mean_RPM']) / (x['DNA_mean_RPM']))

            ax.hist(x, bins=bins, color=colors[i], edgecolor='none', alpha=0.8)

            # Transparent background
            ax.patch.set_alpha(0)

            ax.tick_params(axis='x', labelsize=7)
            ax.set_xticks(xticks)
            ax.set_yticks([])
            ax.set_xlim(xlim)

            for spine in ax.spines.keys():
                ax.spines[spine].set_visible(False)

            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['bottom'].set_color('#AAA')

            if i == 2:
                ax.spines['bottom'].set_color('#444')
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            if k == 1 and i == 2:
                ax.set_xlabel('Measured Production Fitness', x=-0.4, ha='center', va='top', fontsize=8, labelpad=5)
            if k == 1 and i == 1:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel('Density', rotation=90, labelpad=7, y=0, ha='left')
            if k == 0 and i == 2:
                ax.set_xlabel('% Not Detected', x=-0.2, ha='right', va='top', fontsize=8, labelpad=5)

            rect_width = 0.15
            rect_height = 0.5
            ax.add_patch(mpl.patches.Rectangle(
                ((-1 * rect_width) - 0.02, 0), rect_width, rect_height,
                transform=ax.transAxes, clip_on=False, facecolor=colors[i], alpha=0.5
            ))
            ax.text((-1 * (rect_width / 2))-0.01, 0.27, '{:.1f}'.format((nd.sum() / len(x)) * 100), 
                    fontsize=7, transform=ax.transAxes, rotation=90, ha='center', va='center')
            if i == 2:
                ax.text((-1 * (rect_width / 2))-0.02, -0.1, 'ND', transform=ax.transAxes, ha='center', va='top',
                       fontsize=7)
            if k == 0:
                ax.text(-0.3, 0.25, left_titles[i], transform=ax.transAxes, ha='right', va='center', fontsize=8)
                
            if i == 0:
                ax.set_title('{}-Fc'.format(receptor))
            

    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path
    
    
####### ---- Fig 4S2 + support utils ---- #######

def get_overlap(dd_df, r2_df):
    
    dd_dff = dd_df[~dd_df['DNA_mean_RPM'].isna()]
    
    # Calibration seems reasonable and possible.  Determine the best shift.
    overlap = dd_dff.merge(r2_df, on='AA_sequence', suffixes=('_dd', '_r2'))
    return overlap

def prep_calibration(overlap, assay, hist=False):
    
    if assay == 'brain_trans':
        if hist:
            idxAll = overlap['Source'] != 'asdfasdffasfd'
            idxYes, idxNo = ExtractRegionIndices(overlap, idxAll, 'C57BL/6_log2enr', 'trans_brain_log2enr', (-12, -12), (10, -5))
            # For brain transduction calibration, drop the outlier points in the green box
            x = overlap[idxNo]['C57BL/6_log2enr']
            y = overlap[idxNo]['trans_brain_log2enr'] 
        else:
            x = overlap['C57BL/6_log2enr']
            y = overlap['trans_brain_log2enr'] 
        xlabel = 'Brain Trans. LY6A/LY6C1 Round 2 (Log$_2$ Enrichment)'
        ylabel = 'Brain Trans. LY6A/LY6C1 SVAE (Log$_2$ Enrichment)'
    
    else:
        x = overlap['{}_log2enr_r2'.format(assay)]
        y = overlap['{}_log2enr_dd'.format(assay)]
        xlabel = '{} Round 2 (Log$_2$ Enrichment)'.format(assay)
        ylabel = '{} SVAE (Log$_2$ Enrichment)'.format(assay)
    
    non_null_both = np.isfinite(x) & np.isfinite(y)

    x_nonnull = x[non_null_both]
    y_nonnull = y[non_null_both]

    meanx = np.mean(x_nonnull)
    meany = np.mean(y_nonnull)

    # y = mx + b
    #meany_all = 1 * meanx_all + b
    #b = meany_all - meanx_all
    calibration = meany - meanx

    return x, y, x_nonnull, y_nonnull, non_null_both, meanx, meany, calibration, xlabel, ylabel

####### Fig 4S2ABC #######

def plot_calibration(dd_df, r2_df, assay='LY6A', fig_outdir='figures', figname='fig4S2A'):
    
    fig_theme()
    # plt.rcParams.update({'font.size': 16})

    overlap = get_overlap(dd_df, r2_df)
    x, y, x_nonnull, y_nonnull, non_null_both, meanx, meany, calibration, xlabel, ylabel = prep_calibration(overlap, assay)

    fig, main_fig = ScatterDensityMissing(x + 0*calibration, y, xlabel=xlabel, ylabel=ylabel, centerline='on', legend=(0.35, -0.1), npoints=np.inf, grid=False, show_r=False)
    
    if assay == 'brain_trans':
        idxAll = overlap['Source'] != 'asdfasdffasfd'
        idxYes, idxNo = ExtractRegion(overlap, idxAll, 'C57BL/6_log2enr', 'trans_brain_log2enr', (-12, -12), (10, -5), fig=main_fig)

        main_fig.set_xticks([-10, -5, 0, 5])
        main_fig.set_yticks([-10, -5, 0, 5])

    png_path = save_fig_formats(fig, figname, fig_outdir, dpi=200)
    plt.close()
    
    return png_path

####### Fig 4S2D #######

def calibration_drops_hist(dd_df, r2_df, fig_outdir='figures', figname='fig4S2D'):

    val = 'starter_virus_mean_RPM_dd'
    
    overlap = get_overlap(dd_df, r2_df)
    
    idxAll = overlap['Source'] != 'asdfasdffasfd'
    # idxYes, idxNo = ExtractRegionIndices(overlap, idxAll, 'C57BL/6_log2enr', 'trans_brain_log2enr', (-12, -12), (10, -5))
    idxYes, idxNo = ExtractRegion(overlap, idxAll, 'C57BL/6_log2enr', 'trans_brain_log2enr', (-12, -12), (10, -5))
    
    fig, axes = MultiHist([overlap[idxNo][val], overlap[idxYes][val]], stat='proportion', nbins=50, binrange=(0, 20), colors=['r', 'g'])
    plt.xlim([0, 20])
    plt.legend(['Included in Calibration', 'Dropped'])
    plt.xlabel('Starter Virus RPM')
    plt.grid(False)
    
    png_path = save_fig_formats(fig, figname, fig_outdir, dpi=100)
    plt.close()
    
    return png_path

####### Fig 4S2E #######

def plot_calibration_hist(dd_df, r2_df, fig_outdir='figures', figname='fig4S2E'):
    params = {'legend.fontsize': 14}
    plt.rcParams.update(params)
    
    # Fig4Theme()
    
    assays = ['LY6A', 'LY6C1', 'brain_trans']
    
    fig = plt.figure(figsize=(6.4*3, 4.8))
    subfigs = fig.subfigures(1, 3, wspace=0.07)
    
    overlap = get_overlap(dd_df, r2_df)
    
    # ly6a_calibration = None
        
    for i, assay in enumerate(assays):
    
        x, y, x_nonnull, y_nonnull, non_null_both, meanx, meany, calibration, xlabel, ylabel = prep_calibration(overlap, assay, hist=True)
        
        # if assay == 'LY6A':
        #     ly6a_calibration = calibration
        # if assay == 'brain_trans':
        #     calibration = ly6a_calibration
        
        errors_no_cal = x_nonnull - y_nonnull
        errors_cal = x_nonnull - y_nonnull + calibration

        data = {
            'Calibrated': errors_cal,
            'Uncalibrated': errors_no_cal,
        }
        if assay == 'LY6A':
            show_legend = True
        else:
            show_legend = False
            
        if assay == 'brain_trans':
            title = 'Brain Transduction'
        else:
            title = assay


        subfigs[i], axes = MultiHist2(data, nbins=100, show_legend=show_legend, fig=subfigs[i])
        axes[0].set_title(title, fontsize=20)
        axes[0].set_xlabel('Log$_2$ Enrichment', fontsize=16)
        axes[0].set_ylabel('Count', fontsize=16)
        axes[0].tick_params(axis='both', size=3, labelsize=14, pad=5)
        plt.grid(False)

    png_path = save_fig_formats(fig, figname, fig_outdir, dpi=80)
    plt.close()
    
    return png_path


####### ---- Fig 4S3 + support utils ---- #######

def prep_latent_spaces(preds_df, receptor):

    preds_df, cluster_dict, cluster_centers, cluster_assay_means = kmeans_clustering(preds_df, receptor, n_clusters=5, y_scale_adjustment=0.5)
    
    top_cluster = np.argmax(cluster_assay_means)
    top_cluster_df = preds_df[preds_df['cluster_label'] == top_cluster].copy()

    # Z_sub = top_cluster_df[['z0', 'z1']].copy()
    # Y_sub = top_cluster_df['y_true'].copy()

    top_cluster_df, subcluster_dict, subcluster_centers, subcluster_assay_means = kmeans_clustering(top_cluster_df, receptor, n_clusters=10, cluster_with_assay_vals=False, y_scale_adjustment=0.15)
    
    return preds_df, top_cluster_df

####### Fig 4S3CD ####

def plot_latent_spaces(preds_df, receptor='LY6A', fig_outdir='figures', figname='fig4S3C'):
    
    pickle_file = 'UMAPs/fig4S3_{}_prep_output.pickle'.format(receptor)
    
    loaded_pickle = False
    try:
        if Path(pickle_file).is_file():
            with open(pickle_file, 'rb') as f:
                [preds_df, top_cluster_df] = pickle.load(f)
                loaded_pickle = True
    except:
        pass
    
    if not loaded_pickle:
        preds_df, top_cluster_df = prep_latent_spaces(preds_df, receptor)
        with open(pickle_file, 'wb') as f:
            pickle.dump([preds_df, top_cluster_df], f)
    
    vmin = preds_df[['{}_pred_log2enr'.format(receptor), '{}_log2enr'.format(receptor)]].min().min()
    vmax = preds_df[['{}_pred_log2enr'.format(receptor), '{}_log2enr'.format(receptor)]].max().max()
    
    vmin_sub = top_cluster_df['{}_pred_log2enr'.format(receptor)].min()
    vmax_sub = top_cluster_df['{}_pred_log2enr'.format(receptor)].max()
    
    pointsize = 0.1
    cmap = 'coolwarm'
    cmap_sub = 'plasma'
    
    top=0.87
    bottom=0.2
    left=0.1
    image_right = 0.9
    cbar_padding=0.01
    cbar_width = 0.01
    
    fig = plt.figure(figsize=(9,2.5), dpi=200) 
    gs = fig.add_gridspec(1, 2, top=top, bottom=bottom, left=left, right=image_right, 
                           width_ratios=[3, 1], wspace=0.3)


    #ax = fig.add_subplot(gs[0])
    gs0 = mpl.gridspec.GridSpecFromSubplotSpec(
        1, 4, width_ratios=[12,12,12,1], subplot_spec=gs[0], wspace=0.05)
    
    
    
    
    ax0 = fig.add_subplot(gs0[0])
    ax1 = fig.add_subplot(gs0[1])
    ax2 = fig.add_subplot(gs0[2])

    y_true = ax0.scatter(preds_df['z0'], preds_df['z1'], vmin=vmin, vmax=vmax, c=preds_df['{}_log2enr'.format(receptor)], s=pointsize, cmap=cmap)
    
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()
    
    y_clustered = ax1.scatter(preds_df['z0'], preds_df['z1'], vmin=vmin, vmax=vmax, c=preds_df['cluster_assay_mean'], s=pointsize, cmap=cmap) 
    
    top_cluster = ax2.scatter(top_cluster_df['z0'], top_cluster_df['z1'], vmin=vmin, vmax=vmax, c=top_cluster_df['{}_log2enr'.format(receptor)], s=pointsize, cmap=cmap)
    
    for axx in (ax1, ax2):
        axx.set_xlim(xlim)
        axx.set_ylim(ylim)
        axx.tick_params(left=False, labelleft=False)
        
    for axx in (ax0, ax1, ax2):
        axx.set_xlabel('z0', fontsize=8)
        axx.tick_params(labelsize=7)
        #ax.set_xticks([-1, -0.5, 0, 0.5])
    
    ax0.set_ylabel('z1', fontsize=8)
    ax0.tick_params(labelsize=7)
    
    ax0.text(-0.4, 0.5, '{}-Fc'.format(receptor), fontsize=10, transform=ax0.transAxes, rotation=90, ha='center', va='center')

    ax = fig.add_subplot(gs0[3])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), 
            cmap=cmap
        ), 
        cax=ax
    )
    #cbar.set_ticks([-10, -5, 0, 5, 10])
    cbar.set_label('Log2 Enrichment', rotation=90, labelpad=5, y=0.45)
    ax.tick_params(axis='y', labelsize=7) #, pad=1)
    
    
    #ax = fig.add_subplot(gs[1])
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(
        1, 2, width_ratios=[12,1], subplot_spec=gs[1], wspace=0.05)

    ax3 = fig.add_subplot(gs1[0])
    
    subclusters_recolored = ax3.scatter(top_cluster_df['z0'], top_cluster_df['z1'], #vmin=vmin_sub, vmax=vmax_sub, 
                                    c=top_cluster_df['cluster_assay_mean'], s=pointsize, cmap=cmap_sub) 
    
    ax3.set_ylabel('z1', fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.set_xlabel('z0', fontsize=8)
    
    ax = fig.add_subplot(gs1[1])
    cbar = fig.colorbar(
        subclusters_recolored, 
        cax=ax
    )
    #cbar.set_ticks([-10, -5, 0, 5, 10])
    cbar.set_label('Log2 Enrichment', rotation=90, labelpad=5, y=0.45)
    ax.tick_params(axis='y', labelsize=7) #, pad=1)
        
    png_path = save_fig_formats(fig, figname, fig_outdir)
    plt.close()
    
    return png_path

####### ---- Fig 4S4 + support utils ---- #######

def plot_assay_ridgeplots(outdata=None, fig_outdir='figures', figname='fig4S4'):
    
    if outdata is None:
        pickle_file = 'UMAPs/fig4_clustering_outdata.pickle'
        if Path(pickle_file).is_file():
            with open(pickle_file, 'rb') as f:
                [outdata, calibration] = pickle.load(f)
        else:
            raise Exception('Please go plot Fig 4F first.')
    
    
    outerfig = plt.figure(figsize=(3, 3), dpi=200)
    subfigs = outerfig.subfigures(2, 2)#, hspace=0.1)
    #gs = outerfig.add_gridspec(2, 2, left=0.15, right=0.95, bottom=0.10, top=0.82, hspace=0.55)
    marker_size = 2

    sns.set_context('paper', font_scale=.6)

    titles = {'LY6A': 'LY6A-Fc',
              'LY6C1': 'LY6C1-Fc',
              'Brain Transduction (LY6A)': '',
              'Brain Transduction (LY6C1)': ''
             }

    for i, dataname in enumerate(titles):
        data = outdata[dataname]['data']

        fig = flattenArray(subfigs)[i]

        colors = ['g', 'b', 'r']


        qhr2_valcol = outdata[dataname]['umap_inputs']['R2']['val_col']
        satmut_valcol = outdata[dataname]['umap_inputs']['SM']['val_col']
        vae_valcol = outdata[dataname]['umap_inputs']['SVAE']['val_col']

        if i == 0 or i == 2:
            ridgedata = {
                'Round 2': data['R2'][qhr2_valcol] + calibration[dataname],
                'Saturation\nMutagenesis': data['SM'][satmut_valcol],
                'SVAE': data['SVAE'][vae_valcol],
            }
        else:
            ridgedata = {
                '': data['R2'][qhr2_valcol] + calibration[dataname],
                ' ': data['SM'][satmut_valcol],
                '  ': data['SVAE'][vae_valcol],
            }
        xlabel=''

        RidgePlot(ridgedata, xlabel=xlabel, hspace=-0.7, nbins=100, title=titles[dataname], grid=False, colors=colors, binrange=[-14, 9], stat='proportion', fig=fig)

    # Add some text labels
    delta = -.15
    outerfig.gca().text(-.1, 2.45, '$\it{In}$ $\it{vitro}$ Binding', ha="center", transform=fig.gca().transAxes)
    outerfig.gca().text(-.1, 0.95, 'Brain Transduction', ha="center", transform=fig.gca().transAxes)
    outerfig.gca().text(-.75, 0.9, 'LY6A Variants', ha="center", transform=fig.gca().transAxes)
    outerfig.gca().text(.6, 0.9, 'LY6C1 Variants', ha="center", transform=fig.gca().transAxes)
    outerfig.gca().text(-.2, -.25, 'Log$_2$ Enrichment', ha="center", transform=fig.gca().transAxes)

    png_path = save_fig_formats(outerfig, figname, fig_outdir, bbox_inches='tight')
    plt.close()
    
    return png_path

def plot_assay_ridgeplots2(outdata=None, fig_outdir='figures', figname='fig4S4'):
    
    if outdata is None:
        pickle_file = 'UMAPs/fig4_clustering_outdata.pickle'
        if Path(pickle_file).is_file():
            with open(pickle_file, 'rb') as f:
                [outdata, calibration] = pickle.load(f)
        else:
            raise Exception('Please go plot Fig 4F first.')
    
    #outermost_fig = plt.figure(figsize=(3.2,3.2), dpi=200)
    #gs = outermost_fig.add_gridspec(1, 1, left=0.3, right=0.9, bottom=0.2, top=0.7)
    #outerfig = outermost_fig.add_subfigure(gs[0,0]) #plt.figure(figsize=(3, 3), dpi=200)
    outerfig = plt.figure(figsize=(3, 3), dpi=200)
    #subfigs = outerfig.subfigures(2, 2)#, hspace=0.1)
    subfigs = outerfig.add_gridspec(2, 2, left=0.2, right=0.95, bottom=0.1, top=0.8) #, hspace=0.55)
    marker_size = 2

    sns.set_context('paper', font_scale=.6)

    titles = {'LY6A': 'LY6A-Fc',
              'LY6C1': 'LY6C1-Fc',
              'Brain Transduction (LY6A)': '',
              'Brain Transduction (LY6C1)': ''
             }

    for i, dataname in enumerate(titles):
        data = outdata[dataname]['data']

        # fig = flattenArray(subfigs)[i]
        fig = outerfig.add_subplot(subfigs[i//2,i%2]) #(subfigs.flat[i])

        colors = ['g', 'b', 'r']


        qhr2_valcol = outdata[dataname]['umap_inputs']['R2']['val_col']
        satmut_valcol = outdata[dataname]['umap_inputs']['SM']['val_col']
        vae_valcol = outdata[dataname]['umap_inputs']['SVAE']['val_col']

        if i == 0 or i == 2:
            ridgedata = {
                'Round 2': data['R2'][qhr2_valcol] + calibration[dataname],
                'Saturation\nMutagenesis': data['SM'][satmut_valcol],
                'SVAE': data['SVAE'][vae_valcol],
            }
        else:
            ridgedata = {
                '': data['R2'][qhr2_valcol] + calibration[dataname],
                ' ': data['SM'][satmut_valcol],
                '  ': data['SVAE'][vae_valcol],
            }
        xlabel=''

        RidgePlot2(ridgedata, xlabel=xlabel, hspace=-0.7, nbins=100, title=titles[dataname], grid=False, colors=colors, binrange=[-14, 7], stat='proportion', fig=outerfig, gs_fig=subfigs[i//2,i%2])

    # Add some text labels
    delta = -.15
    outerfig.gca().text(-.1, 2.45, '$\it{In}$ $\it{vitro}$ Binding', ha="center", transform=fig.transAxes)
    outerfig.gca().text(-.1, 0.95, 'Brain Transduction', ha="center", transform=fig.transAxes)
    outerfig.gca().text(-.75, 0.9, 'LY6A Variants', ha="center", transform=fig.transAxes)
    outerfig.gca().text(.6, 0.9, 'LY6C1 Variants', ha="center", transform=fig.transAxes)
    outerfig.gca().text(-.2, -.25, 'Log$_2$ Enrichment', ha="center", transform=fig.transAxes)

    png_path = save_fig_formats(outerfig, figname, fig_outdir)
    plt.close()
    
    return png_path

def swarm_plot_4f(fig_outdir, figname, outdata=None):
    mpl.rcParams['axes.linewidth'] = 0.15
    if outdata is None:
        pickle_file = 'UMAPs/fig4_clustering_outdata.pickle'
        if Path(pickle_file).is_file():
            with open(pickle_file, 'rb') as f:
                [outdata, calibration] = pickle.load(f)
        else:
            raise Exception('Please go plot Fig 4F first.')

    sns.set_context('paper', font_scale=.6)
    titles = {
              'LY6A': 'LY6A-Fc pull-down',
              'LY6C1': 'LY6C1-Fc pull-down',
              'Brain Transduction (LY6A)': 'C57BL/6J mouse brain transduction\nby LY6A-binding variants',
              'Brain Transduction (LY6C1)': 'C57BL/6J mouse brain transduction\nby LY6C1-binding variants'
             }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4.5,3.5), dpi=300)
    fig.subplots_adjust(hspace=0.25)
    # Flatten the array of axes objects for easier iteration
    axes = axes.flatten()
    top_num = None
    markersize = 0.15
    fitness_cutoff = -1.0

    fitness_col = 'log2fitness'

    for i, dataname in enumerate(titles):
        print(dataname)
        data = outdata[dataname]['data']

        # Plot brain transduction for top inidividual sequences for SVAE and sat mut libraries.
        qhr2_valcol = outdata[dataname]['umap_inputs']['R2']['val_col']
        satmut_valcol = outdata[dataname]['umap_inputs']['SM']['val_col']
        vae_valcol = outdata[dataname]['umap_inputs']['SVAE']['val_col']

        # Filter out low fitness
        isfinite_idx = (np.isfinite(data['R2'][qhr2_valcol])) & (np.isfinite(data['R2'][fitness_col]))
        fitness_idx = data['R2'][fitness_col] > fitness_cutoff
        r2_include_idx = (fitness_idx) & (isfinite_idx)

        isfinite_idx = (np.isfinite(data['SM'][satmut_valcol])) & (np.isfinite(data['SM'][fitness_col]))
        fitness_idx = data['SM'][fitness_col] > fitness_cutoff
        sm_include_idx = (fitness_idx) & (isfinite_idx)

        isfinite_idx = (np.isfinite(data['SVAE'][vae_valcol])) & (np.isfinite(data['SVAE'][fitness_col]))
        fitness_idx = data['SVAE'][fitness_col] > fitness_cutoff
        svae_include_idx = (fitness_idx) & (isfinite_idx)

        if top_num is not None:
            swarmdf = pd.DataFrame({
                'Round 2': data['R2'][r2_include_idx][qhr2_valcol].nlargest(top_num),
                'Saturation\nMutagenesis': data['SM'][sm_include_idx][satmut_valcol].nlargest(top_num),
                'SVAE': data['SVAE'][svae_include_idx][vae_valcol].nlargest(top_num),
            })
        else:
            swarmdf = pd.DataFrame({
                'Round 2': data['R2'][r2_include_idx][qhr2_valcol],
                'Saturation\nMutagenesis': data['SM'][sm_include_idx][satmut_valcol],
                'SVAE': data['SVAE'][svae_include_idx][vae_valcol],
            })

        sns.swarmplot(data=swarmdf, s=markersize, ax=axes[i], palette = ['g', 'b', 'r'])

        

        axes[i].set_title(titles[dataname])# fontsize=20)
        for axis in ['top','bottom','left','right']:
            axes[i].spines[axis].set_linewidth(.5)

        plt.grid(False)
        axes[i].set_ylim([-14, 10])
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[0].text(-1.25, -21, 'Log$_2$ Enrichment', ha="center", rotation=90)
    sns.despine()

    png_path = save_fig_formats(fig, figname, fig_outdir, bbox_inches='tight')
    plt.close()
    
    return png_path
