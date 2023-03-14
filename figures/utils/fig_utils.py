import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import scipy
import functools
import operator
from sklearn.cluster import KMeans
from collections import defaultdict

from pathlib import Path

# Data download URLs.
urls = {
    'library1.csv': 'https://zenodo.org/record/7689795/files/library1.csv?download=1',
    'library2_pulldown.csv': 'https://zenodo.org/record/7689795/files/library2_pulldown.csv?download=1',
    'library2_invivo.csv': 'https://zenodo.org/record/7689795/files/library2_invivo.csv?download=1',
    'round2_codons_merged.csv': 'https://zenodo.org/record/7689795/files/round2_codons_merged.csv?download=1',
    'round2_codons_separate.csv': 'https://zenodo.org/record/7689795/files/round2_codons_separate.csv?download=1',
    'SVAE_SM_library_references_only.csv': 'https://zenodo.org/record/7689795/files/SVAE_SM_library_references_only.csv?download=1',
    'SVAE_SM_library_codons_merged.csv': 'https://zenodo.org/record/7689795/files/SVAE_SM_library_codons_merged.csv?download=1',
    'SVAE_SM_library_codons_separate.csv': 'https://zenodo.org/record/7689795/files/SVAE_SM_library_codons_separate.csv?download=1',
    'LY6A_SVAE_generated_sequences.csv': 'https://zenodo.org/record/7689795/files/LY6A_SVAE_generated_sequences.csv?download=1',
    'LY6C1_SVAE_generated_sequences.csv': 'https://zenodo.org/record/7689795/files/LY6C1_SVAE_generated_sequences.csv?download=1',
    'LY6A_joint_umap_l1_l2.csv': 'https://zenodo.org/record/7689795/files/LY6A_joint_umap_l1_l2.csv?download=1',
    'LY6C1_joint_umap_l1_l2.csv': 'https://zenodo.org/record/7689795/files/LY6C1_joint_umap_l1_l2.csv?download=1',
    'LY6A_SVAE_training_predictions.csv': 'https://zenodo.org/record/7689795/files/LY6A_SVAE_training_predictions.csv?download=1',
    'LY6C1_SVAE_training_predictions.csv': 'https://zenodo.org/record/7689795/files/LY6C1_SVAE_training_predictions.csv?download=1',
}

def download_data(filename, url):
    from tqdm.notebook import tqdm_notebook
    import requests
    import os

    # Determine if we have already downloaded the data:
    if not os.path.exists(filename):
        print(filename + ' is not found.  You can download it here: ' + url)
        print('Attempting automatic download of ' + filename)
        print('Connecting, this may take a moment...')
        response = requests.get(url, stream=True)
        print(f'Zenodo rate limit remaining: {response.headers.get("X-RateLimit-Remaining", 0)}')
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm_notebook(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filename, 'wb') as file:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("Error: failed to download file.")
    else:
        print('Already have ' + filename + ' skipping download.')

####### ---- General settings ---- #######

def fig_theme():
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
    '*': '#888888'
}

####### ---- General functions ---- #######

aa_alphabet = np.array([
    'A', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'K', 'L', 'M', 'N', 'P', 'Q',
    'R', 'S', 'T', 'V', 'W', 'Y'
])

def one_hot(seq):
    encoding = []
    for char in seq:
        encoding.append(aa_alphabet == char)
    return np.concatenate(encoding).astype(int)

def save_fig_formats(fig, figname, fig_outdir, formats=['.png', '.svg', '.pdf'], dpi=300, bbox_inches=None):
    png_path = str(Path(fig_outdir) / 'PNGs' / (figname+'.png'))
    svg_path = str(Path(fig_outdir) / 'SVGs' / (figname+'.svg'))
    pdf_path = str(Path(fig_outdir) / 'PDFs' / (figname+'.pdf'))
    
    all_paths = [png_path, svg_path, pdf_path]
    
    for p in all_paths:
        if any([form in p for form in formats]):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches=bbox_inches)
            
    return png_path

def human_format(num):
    """Convert a number to a human-readable format
    Thanks to user "rtaft" from https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
    """

    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )

def hamming_xor(a, b):
    # Convert sequences a and b to bytes, then do an xor
    return sum([(a ^ b) > 0 for a, b in zip(a.encode(), b.encode())])

def aa_to_matrix(aa_seqs, normalize=True, n_pos=7):
    seq_mat = pd.DataFrame.from_records(aa_seqs.apply(list).values)
    mat = pd.DataFrame(np.zeros((len(aa_alphabet), n_pos)), index=aa_alphabet)
    
    for i in range(n_pos):
        mat[i] = seq_mat.iloc[:, i].value_counts().sort_index()
    
    mat = mat.fillna(0)
    
    if normalize:
        mat = mat / len(aa_seqs)
    
    return mat.values - (1/len(aa_alphabet))

def kmeans_clustering(preds_df, receptor, y_scale_adjustment=0.5, cluster_with_assay_vals=True, n_clusters=5):
    
    Z = preds_df[['z0', 'z1']]
    Y = preds_df['{}_log2enr'.format(receptor)]
    
    z_df = Z.copy()
    z = Z.to_numpy()
    
    scaled_labels = ((z.max()-z.min()) / 
                     (Y.max()- Y.min()) * 
                     (Y - Y.max()) + 
                     z.max()) * y_scale_adjustment
    z_df['y_true_scaled'] = scaled_labels 
    
    if cluster_with_assay_vals:
        data = z_df
    else:
        data = z
        
        
    ######## KMEANS ########
    
    print('Fitting kmeans...')
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    print('Done fitting kmeans.')
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
        
    cluster_assay_means = np.array([np.mean(Y[labels==i]) for i in range(n_clusters)])
    
    cluster_dict = defaultdict(dict)
    for i in range(n_clusters):
        cluster_dict[i]['cluster_center'] = cluster_centers[i]
        cluster_dict[i]['cluster_assay_mean'] = cluster_assay_means[i]
    
    preds_df['cluster_label'] = labels #.astype('int32') 
    #preds_df['cluster_assay_mean'] = cluster_assay_means
    preds_df['cluster_assay_mean'] = cluster_assay_means[preds_df['cluster_label'].values]

    return preds_df, cluster_dict, cluster_centers, cluster_assay_means

# From Andy's Fig 4 utils

def engFormat(num):
    from matplotlib.ticker import EngFormatter
    engFormat = EngFormatter(unit='',places=0,sep='')
    return engFormat(num)

def flattenArray(array):
    return functools.reduce(operator.iconcat, array, [])

def ScatterDensity(x, y, ax=None, square=True, centerline='on', color='flare', quiet=False, npoints=None, vertical_line_size = None, marker_size=1, show_r=True):
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score
    r2 = r2_score(x, y)
    r = scipy.stats.pearsonr(x, y)[0]

    non_null_both = np.isfinite(x) & np.isfinite(y)
    x_nonnull = x[non_null_both]
    y_nonnull = y[non_null_both]
    removed = len(x) - len(x_nonnull)
    if removed > 0 and not quiet:
        print('Removed ' + str(removed) + ' NaNs / Infs leaving ' + str(len(x_nonnull)) + ' points')

    cmap = sns.color_palette(color, as_cmap=True)

    # Initialize a downsampled kernel for speed.
    # When making the final plot, we can take this out

    if npoints is None:
        n = min(1000, len(x_nonnull))
    else:
        n = min(npoints, len(x_nonnull))

    if n < 500:
        c = None
    else:
        kernel = gaussian_kde(np.vstack([
            x_nonnull.sample(n=n, random_state=1),
            y_nonnull.sample(n=n, random_state=1)
        ]))
        c = kernel(np.vstack([x_nonnull, y_nonnull]))

    pt_size = marker_size
    marker = 'o'
    if vertical_line_size is not None:
        pt_size = vertical_line_size
        marker = '|'

    # Rasterized so that it doesn't crash Illustrator when exported
    # to SVG or PDF formats

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    ax.grid(b=True, which='major', color='gray', linewidth=0.25)
    ax.scatter(
        x_nonnull, y_nonnull, c=c, s=pt_size, cmap=cmap,
        rasterized=True,
        linewidth=1, edgecolors=None, marker=marker
    )

    # Draw the correlation line
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    amin = min(xmin, ymin)
    amax = max(xmax, ymax)

    if centerline == 'on':
        ax.plot([amin, amax], [amin, amax], 'k', linewidth=0.35)
    elif centerline == 'off':
        pass
    elif centerline == 'mean':
        meanx = np.mean(x_nonnull)
        meany = np.mean(y_nonnull)

        deltax = max(x_nonnull) - min(x_nonnull)
        deltay = max(y_nonnull) - min(y_nonnull)
        delta = max(deltax, deltay)

        # Draw a line on the mean, going through the whole plot.
        ax.plot([meanx-delta, meanx+delta], [meany-delta, meany+delta], 'k', linewidth=0.35)
    else:
        raise Exception("Unknown centerline command: " + centerline)

    if square:
        ax.set_xlim([amin, amax])
        ax.set_ylim([amin, amax])
    else:
        diffx = xmax - xmin
        diffy = ymax - ymin
        diff = max(diffx, diffy)

        if diff == diffx:
            ax.set_xlim([xmin, xmax])
            delta_diff = diff - (ymax - ymin)
            ax.set_ylim([ymin - delta_diff/2, ymax + delta_diff/2])
        else:
            ax.set_ylim([ymin, ymax])
            delta_diff = diff - (xmax - xmin)
            ax.set_xlim([xmin - delta_diff/2, xmax + delta_diff/2])

    if show_r:
        ax.text(0.1, 0.98, 'R: {:.2}'.format(r), ha="right", transform=ax.transAxes)

    return ax

def ScatterDensityMissing(x, y, title='', xlabel='', ylabel='', fig=None, gs=None, square=True, centerline='on', color='flare', npoints=None, legend=(0.25, 1.4), show_r=True, grid=True):
    # Centerline can be:
    #   'on': enables the centerline
    #   'off': disables the centerline
    #   'mean': draws a line at the mean of the data
    #
    # legend:
    #   'none' or location, like (0.25, 1.4)
    from matplotlib import gridspec
    import seaborn as sns

    sns.set_style("white")
    sns.set_color_codes()

    if (fig is None) and (gs is None):
        fig = plt.figure(figsize=(4, 4))
    widths = [7, 1]
    heights = [1, 7]
    if gs is not None:
        spec2 = mpl.gridspec.GridSpecFromSubplotSpec(
                ncols=2, nrows=2, width_ratios=widths, height_ratios=heights, subplot_spec=gs)
    else:
        spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=widths, height_ratios=heights)

    top_bar = fig.add_subplot(spec2[0, 0])
    main_fig = fig.add_subplot(spec2[1, 0])
    right_bar = fig.add_subplot(spec2[1, 1])
    #tiny_bottom_left = fig.add_subplot(spec2[2, 0])

    # Find rows that are null in X but valid in Y (and vice versa):
    null_x = ~np.isfinite(x) & np.isfinite(y)
    null_y = np.isfinite(x) & ~np.isfinite(y)
    null_both = ~np.isfinite(x) & ~np.isfinite(y)
    non_null_x = np.isfinite(x)
    non_null_y = np.isfinite(y)
    non_null_both = np.isfinite(x) & np.isfinite(y)

    # Plot the main data
    scatter_ax = ScatterDensity(x[non_null_both], y[non_null_both], ax=main_fig, square=square, centerline=centerline, color=color, npoints=npoints, show_r=show_r)
    scatter_ax.grid(grid)

    # Do auto-scaling
    nbins, valmax = AutoScaleHistogram(x, y, null_x, null_y, non_null_x, non_null_y)
    valmax *= 1.1

    assert len(x) == len(y)

    histax = MakeHistogram(x, null_y, nbins, top_bar, 'top', main_fig, valmax, 'r')
    histax.grid(grid)
    histax = MakeHistogram(y, null_x, nbins, right_bar, 'right', main_fig, valmax, 'r')
    histax.grid(grid)
    histax = MakeHistogram(x, non_null_x, nbins, top_bar, 'top', main_fig, valmax, 'b', stepgraph=True)
    histax.grid(grid)
    histax = MakeHistogram(y, non_null_y, nbins, right_bar, 'right', main_fig, valmax, 'b', stepgraph=True)
    histax.grid(grid)


    if legend != 'none':
        top_bar.legend(['Total', 'Missing'], bbox_to_anchor=legend, loc='upper right')
        #, bbox_to_anchor=(-.09, 1.1), loc='upper right')

    top_bar.set_title(title)

    main_fig.set(frame_on=False)
    main_fig.set_xlabel(xlabel, fontsize=8)
    main_fig.set_ylabel(ylabel, fontsize=8)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)


    all_nan = sum(null_both) / len(x) * 100.0
    nanstr = 'Both\nMissing:\n{:.1f}%'.format(all_nan)

    from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
    Texts = []
    Texts.append(TextArea(nanstr, textprops=dict(color='r', ha='center')))
    texts_vbox = VPacker(children=Texts,pad=0,sep=0)
    ann = AnnotationBbox(texts_vbox,(1.05, .5),xycoords=top_bar.transAxes,
                            box_alignment=(0,.5),bboxprops =
                            dict(boxstyle='round',color='black', facecolor='white'))
    ann.set_figure(fig)
    fig.artists.append(ann)

    return fig, main_fig

def ExtractRegion(features, idxs, xstr, ystr, min_point, max_point, fig=None, confirm_fig=False, color='g', show_r=True):
    from matplotlib import patches

    # Draw the plot with the bounding box on it:
    width = max_point[0] - min_point[0]
    height = max_point[1] - min_point[1]

    if fig is not None:
        fig.add_patch(patches.Rectangle(xy=min_point, width=width, height=height, color=color, fill=False, linewidth=2))

    idxNo =  idxs & ((features[idxs][xstr] <= min_point[0]) | (features[idxs][xstr] >= max_point[0]) | (features[idxs][ystr] <= min_point[1]) | (features[idxs][ystr] >= max_point[1]))

    idxYes = idxs & ((features[idxs][xstr] > min_point[0]) & (features[idxs][xstr] < max_point[0]) & (features[idxs][ystr] > min_point[1]) & (features[idxs][ystr] < max_point[1]))

    if confirm_fig:
        plt.figure()
        fig = ScatterDensity(features[idxNo][xstr], features[idxNo][ystr], centerline='mean', square=False)
        fig = ScatterDensity(features[idxYes][xstr], features[idxYes][ystr], centerline='off', square=False, ax=fig, color='mako')

    return idxYes, idxNo

def ExtractRegionIndices(features, idxs, xstr, ystr, min_point, max_point):

    idxNo =  idxs & ((features[idxs][xstr] <= min_point[0]) | (features[idxs][xstr] >= max_point[0]) | (features[idxs][ystr] <= min_point[1]) | (features[idxs][ystr] >= max_point[1]))

    idxYes = idxs & ((features[idxs][xstr] > min_point[0]) & (features[idxs][xstr] < max_point[0]) & (features[idxs][ystr] > min_point[1]) & (features[idxs][ystr] < max_point[1]))

    return idxYes, idxNo

def DropNaInf(x):
    return x.replace([np.inf, -np.inf], np.nan).dropna()

def GetBins(x):
    hist = np.histogram(DropNaInf(x), bins='auto')
    return hist[1]

def MultiHist(data, colors=None, total_count=None, filled=None, stat=None, alpha=0.75, nbins=None, binrange=None):
    import matplotlib as mpl

    if stat is not None and total_count is not None:
        raise Exception('Cannot specify total_count and stat.')

    if stat is None:
        stat = 'count'

    if nbins is None:
        assert binrange is None
        nbins, binrange = AutoScaleHistogram2(data)

    if nbins is not None:
        assert binrange is not None

    axes = []
    for i, d in enumerate(data):
        if colors is None:
            color = sns.color_palette()[i]
        else:
            color = colors[i]
        if i == 0:
            fig4 = plt.figure()
        hist = np.histogram(d, bins=nbins, range=binrange)
        if total_count is None:
            if filled is not None and stat != 'count':
                raise Exception('Filled with stat that isn\'t "count" is not implemented.')
            elif filled is None:
                ax = sns.histplot(d, linewidth=0, color=color, bins=nbins, binrange=binrange, stat=stat, alpha=alpha)
                axes.append(ax)
            elif filled[i]:
                width = hist[1][1] - hist[1][0]
                ax = plt.bar(hist[1][:-1], hist[0], align='edge', width=width, color=color, edgecolor='none', alpha=alpha)
            else:
                ax = plt.step(hist[1][:-1], hist[0], where='post', color=color)
                # ax = plt.stairs(hist[0], hist[1], color=color, alpha=alpha, linewidth=width)
            axes.append(ax)
        else:
            # Compute percentages instead of counts.
            xbins2 = np.divide(hist[0], total_count)*100.0
            #plt.plot(xbins2, hist[1][:-1], color=color)
            if filled is not None:
                if filled[i]:
                    width = hist[1][1] - hist[1][0]
                    plt.bar(hist[1][:-1], xbins2, align='edge', width=width, color=color, edgecolor='none', alpha=alpha)
                else:
                    ax = plt.step(hist[1][:-1], xbins2, where='post', color=color)

    if total_count is not None:
        fig4.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))

    fig4.gca().grid(b=True, which='major', color='gray', linewidth=0.075)

    return fig4, axes

def MultiHist2(data, colors=None, total_count=None, filled=None, stat=None, alpha=0.75, nbins=None, binrange=None, fig=None, show_legend=True):
    import matplotlib as mpl
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import tol_colors
    import copy

    if stat is not None and total_count is not None:
        raise Exception('Cannot specify total_count and stat.')

    if stat is None:
        stat = 'count'

    if nbins is None:
        assert binrange is None
        nbins, binrange = AutoScaleHistogram2(data.values())

    if nbins is not None and binrange is None:
        nbins, binrange = AutoScaleHistogram2(data.values(), nbins)

    axes = []
    keys = data.keys()
    legend_elements = []
    for i, key in enumerate(keys):
        d = data[key]
        if colors is None:
            #color = sns.color_palette()[i % len(sns.color_palette())]
            color = tol_colors.tol_cset('light')[i % 10]
        else:
            color = colors[i]
        if i == 0:
            if fig is None:
                fig4 = plt.figure()
            else:
                fig4 = fig
            axis = fig4.add_subplot(1,1,1)
        density = False
        if stat == 'density':
            density = True
        hist = np.histogram(d, bins=nbins, range=binrange, density=density)
        width = hist[1][1] - hist[1][0]
        if total_count is None:
            # if filled is not None and stat != 'count':
            #     raise Exception('Filled with stat that isn\'t "count" is not implemented.')
            if filled is None:
                ax = sns.histplot(d, linewidth=0, color=color, bins=nbins, binrange=binrange, stat=stat, alpha=alpha)
                axes.append(ax)
            elif stat == 'count':
                if filled[key]:
                    ax = axis.bar(hist[1][:-1], hist[0], align='edge', width=width, color=color, edgecolor='none', alpha=alpha)

                    legend_elements.append(Patch(facecolor=color, edgecolor='none',
                                            label=key))
                else:
                    # For the step graph, we want to draw the end caps and
                    # the final step.  To do this, we add some points:
                    final_step = copy.deepcopy(list(hist[0]))
                    final_step = [0] + final_step
                    final_step.append(hist[0][-1])
                    final_step.append(0)

                    xbins1 = copy.deepcopy(list(hist[1]))
                    xbins1 = [xbins1[0]-width] + xbins1 + [xbins1[-1]]
                    ax = axis.step(xbins1, final_step, where='post', color=color)
                    #ax = plt.step(hist[1][:-1], hist[0], where='post', color=color)
                    # ax = plt.stairs(hist[0], hist[1], color=color, alpha=alpha, linewidth=width)
                    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=key))
                axes.append(ax)
            elif stat == 'density':
                if filled[key]:
                    # density, filled, just use normal stuff.
                    ax = sns.histplot(d, linewidth=0, color=color, bins=nbins, binrange=binrange, stat=stat, alpha=alpha)
                    legend_elements.append(Patch(facecolor=color, edgecolor='none',
                                            label=key))
                else:
                    # density, not filled.
                    ax = plt.step(hist[1][:-1], hist[0], where='post', color=color)
                    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=key))
                axes.append(ax)
            elif stat == 'frequency':
                if filled[key]:
                    # filled, just use normal stuff.
                    ax = sns.histplot(d, linewidth=0, color=color, bins=nbins, binrange=binrange, stat=stat, alpha=alpha)
                    legend_elements.append(Patch(facecolor=color, edgecolor='none',
                                            label=key))
                else:
                    # frequency, not filled.
                    binwidth = hist[1][1] - hist[1][0]
                    ax = plt.step(hist[1][:-1], hist[0]/binwidth, where='post', color=color)
                    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=key))
                axes.append(ax)
            elif stat == 'probability' or stat == 'proportion' or stat == 'percent':
                if filled[key]:
                    # filled, just use normal stuff.
                    ax = sns.histplot(d, linewidth=0, color=color, bins=nbins, binrange=binrange, stat=stat, alpha=alpha)
                    legend_elements.append(Patch(facecolor=color, edgecolor='none',
                                            label=key))
                else:
                    # normalize such that bar heights sum to 1.
                    total = np.sum(hist[0])
                    factor = 1/total
                    if stat == 'percent':
                        factor *= 100
                    ax = plt.step(hist[1][:-1], hist[0]*factor, where='post', color=color)
                    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=key))
                axes.append(ax)


        else:
            # Compute percentages instead of counts.
            xbins2 = np.divide(hist[0], total_count)*100.0
            if filled is not None:
                if filled[i]:
                    width = hist[1][1] - hist[1][0]
                    plt.bar(hist[1][:-1], xbins2, align='edge', width=width, color=color, edgecolor='none', alpha=alpha)
                else:
                    ax = plt.step(hist[1][:-1], xbins2, where='post', color=color)
    if show_legend:
        if len(legend_elements) > 0:
            plt.legend(handles=legend_elements)
        else:
            plt.legend(keys)

    if total_count is not None:
        fig4.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))

    fig4.gca().grid(b=True, which='major', color='gray', linewidth=0.075)

    return fig4, axes

def AutoScaleHistogram(x, y, null_x, null_y, non_null_x, non_null_y):
    # Compute good values for the 4 plots and then go from there.

    datax = []
    datax.append(x[null_y].replace([np.inf, -np.inf], np.nan).dropna())
    datax.append(x[non_null_x].replace([np.inf, -np.inf], np.nan).dropna())
    xmin = np.min(datax[-1])
    xmax = np.max(datax[-1])

    datay = []
    datay.append(y[null_x].replace([np.inf, -np.inf], np.nan).dropna())
    datay.append(y[non_null_y].replace([np.inf, -np.inf], np.nan).dropna())
    ymin = np.min(datay[-1])
    ymax = np.max(datay[-1])


    n = len(x)
    nbins = []
    for d in datax:
        data_min = np.min(d)
        data_max = np.max(d)
        hist = np.histogram(d, bins='auto', range=(xmin, xmax))
        nbins.append(len(hist[1]) - 1)
    for d in datay:
        hist = np.histogram(d, bins='auto', range=(ymin, ymax))
        nbins.append(len(hist[1]) - 1)

    outbins = int(round(np.mean(nbins)))

    # Now that we have bin count, compute max value.
    maxval = -np.inf
    for d in datax:
        hist = np.histogram(d, bins=outbins, range=(xmin, xmax))
        val = max(np.divide(hist[0], n)*100.0)
        maxval = max(maxval, val)

    for d in datay:
        hist = np.histogram(d, bins=outbins, range=(ymin, ymax))
        val = max(np.divide(hist[0], n)*100.0)
        maxval = max(maxval, val)

    return (outbins, maxval)

def AutoScaleHistogram2(data, nbins=None):
    # Compute min and max range
    datana = []
    dmin = np.inf
    dmax = -np.inf
    for d in data:
        if isinstance(d, np.ndarray):
            d_series = pd.Series(d)
        else:
            d_series = d
        d2 = d_series.replace([np.inf, -np.inf], np.nan).dropna()
        datana.append(d2)
        dmin = min(min(d2), dmin)
        dmax = max(max(d2), dmax)

    if nbins is None:
        nbins = []
        for d in datana:
            hist = np.histogram(d, bins='auto', range=(dmin, dmax))
            nbins.append(len(hist[1]) - 1)

        outbins = int(round(np.mean(nbins)))
    else:
        outbins = nbins

    return (outbins, (dmin, dmax))

def MakeHistogram(data, indexes, bins, ax, position, main_plot, valmax, color, stepgraph=False):
    import copy
    import matplotlib as mpl
    valmin = 0

    xmin, xmax = main_plot.get_xlim()
    ymin, ymax = main_plot.get_ylim()

    non_null_both = (np.isfinite(data)) & (np.isfinite(data))
    data_min = np.min(data[non_null_both])
    data_max = np.max(data[non_null_both])

    # assert xmin == ymin
    # assert xmax == ymax

    if position == 'top' or position == 'bottom':
        vmin = xmin
        vmax = xmax
    else:
        vmin = ymin
        vmax = ymax

    # Bin those values.
    xbins = np.histogram(data[indexes], bins=bins, range=(data_min, data_max))

    # Divide the counts by the total number of rows to get the fraction of NaNs of the total dataset that are in the bin.
    xbins2 = np.divide(xbins[0], len(data))*100.0
    # Make a bar chart with counts as xbins2 and intervals as xbins[1]

    # Compute the interval width
    width = xbins[1][1] - xbins[1][0]

    # Set the limits to match the main plot.

    # print('heights', xbins[0])
    # print('edges', xbins[1])
    if position == 'bottom' or position == 'top':
        if stepgraph:
            ax.stairs(xbins2, xbins[1], linewidth=1.5)
        else:
            # ax.bar( x position, heights)
            ax.bar(xbins[1][:-1], xbins2, align='edge', width=width, color=color, edgecolor='none')

        ax.set_xlim([vmin, vmax])
        if position == 'bottom':
            ax.set_ylim([valmax, valmin])
        else:
            ax.set_ylim([valmin, valmax])

    elif position == 'left' or position == 'right':
        if stepgraph:
            # For the step graph, we want to draw the end caps and
            # the final step.  To do this, we add some points:
            final_step = copy.deepcopy(list(xbins2))
            final_step = [0] + final_step
            final_step.append(xbins2[-1])
            final_step.append(0)

            xbins1 = copy.deepcopy(list(xbins[1]))
            xbins1 = [xbins1[0]] + xbins1 + [xbins1[-1]]

            ax.plot(final_step, xbins1, drawstyle='steps-pre')
            #ax.step(xbins[1][:-1], xbins2)
            #ax.barh(xbins[1][:-1], xbins2, align='edge', height=width, color=color, edgecolor='none')
        else:
            ax.barh(xbins[1][:-1], xbins2, align='edge', height=width, color=color, edgecolor='none')
        ax.set_ylim([ymin, ymax])

        if position == 'left':
            ax.set_xlim([valmax, valmin])
        else:
            ax.set_xlim([valmin, valmax])
    else:
        raise Exception('invalid position: ' + position)

    [ax.spines[i].set_visible(False) for i in ["top", "left", "right", "bottom"]]

    if position == 'top':
        ax.set_xticklabels([])
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
        ax.spines["bottom"].set_visible(True)
    elif position == 'right':
        ax.set_yticklabels([])
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
        ax.spines["left"].set_visible(True)
    elif position == 'bottom':
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
        ax.yaxis.get_major_ticks()[0].draw = lambda *args: None
        ax.spines["top"].set_visible(True)
    elif position == 'left':
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
        ax.xaxis.get_major_ticks()[0].draw = lambda *args: None
        ax.spines["right"].set_visible(True)


    ax.grid(b=True, which='major', color='gray', linewidth=0.25)
    return ax

def RidgePlot(data, xlabel, colors=None, filled=None, stat=None, alpha=0.75, hspace=-0.75, nbins=None, binrange=None, title='', figsize=None, grid=True, fig=None):
    import matplotlib as mpl
    import tol_colors
    
    assert fig is None or figsize is None

    if stat is None:
        stat = 'count'
        
    if binrange is None:
        nbins, binrange = AutoScaleHistogram2(data.values(), nbins)
        
    ymin_all = None
    ymax_all = None

    gs = (mpl.gridspec.GridSpec(len(data), 1))
    if fig is None:
        fig = plt.figure(figsize=figsize)

    ax_objs = []
    for i, d in enumerate(data.values()):
        if colors is None:
            #color = sns.color_palette()[i]
            color = tol_colors.tol_cset('light')[i % 10]
        else:
            color = colors[i]

        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))#, sharex=sharex))

        density = False
        if stat == 'density':
            density = True
        hist = np.histogram(d, bins=nbins, range=binrange, density=density)
        width = hist[1][1] - hist[1][0]

        if stat == 'count' or stat == 'density': # density handled by histogram computation above.
            Bars = plt.bar(hist[1][:-1], hist[0], align='edge', width=width, color=color, edgecolor='none', alpha=alpha)
        elif stat == 'frequency':
            Bars = plt.bar(hist[1][:-1], hist[0]/width, align='edge', width=width, color=color, edgecolor='none', alpha=alpha)
        elif stat == 'probability' or stat == 'proportion' or stat == 'percent':
            # normalize such that bar heights sum to 1.
            total = np.sum(hist[0])
            stat_factor = 1/total
            if stat == 'percent':
                stat_factor *= 100
            Bars = plt.bar(hist[1][:-1], hist[0]*stat_factor, align='edge', width=width, color=color, edgecolor='none', alpha=alpha)
        else:
            raise Exception("Unknown 'stat': " + stat)
            
        #if you want your hlines to align with the bars.
        #i.e. start and end at the same x coordinates:

        x_start = np.array([plt.getp(item, 'x') for item in Bars])
        x_end   = x_start+[plt.getp(item, 'width') for item in Bars]
        
        # Draw white lines that outline the ridge plots, making them look nice.
        
        if stat == 'count' or stat == 'density': # density handled by histogram computation above.
            plt.hlines(hist[0], x_start, x_end, color='w', linewidth=0.5)
            plt.vlines(x_start[1:], hist[0][0:-1], hist[0][1:], color='w', linewidth=0.5)
        elif stat == 'frequency':
            plt.hlines(hist[0]/width, x_start, x_end, color='w', linewidth=0.5)
            plt.vlines(x_start[1:], hist[0][0:-1]/width, hist[0][1:]/width, color='w', linewidth=0.5)
        elif stat == 'probability' or stat == 'proportion' or stat == 'percent':
            plt.hlines(hist[0]*stat_factor, x_start, x_end, color='w', linewidth=0.5)
            plt.vlines(x_start[1:], hist[0][0:-1]*stat_factor, hist[0][1:]*stat_factor, color='w', linewidth=0.5)
        else:
            raise Exception("Unknown 'stat': " + stat)
        
        # Capture ylims
        ymin, ymax = plt.gca().get_ylim()
        
        if ymin_all is None:
            ymin_all = ymin
        else:
            ymin_all = min(ymin, ymin_all)
            
        if ymax_all is None:
            ymax_all = ymax
        else:
            ymax_all = max(ymax, ymax_all)
            
        
        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_ylabel('')

        ax_objs[-1].set_xlabel('')
        
        if i == len(data)-1:
            ax_objs[-1].set_xlabel(xlabel)
            ax_objs[-1].tick_params(labelleft=False, labelbottom=True, left=False, right=False, bottom=False )
        else:
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False )

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        ax_objs[-1].text(0.025, 0, list(data.keys())[i], ha="right", transform=ax_objs[-1].transAxes)

    gs.update(hspace = hspace)

    # Add an axis behind all the other axes so we can draw the grid.
    grid_ax = fig.add_subplot(1, 1, 1, zorder=-1, sharex = ax_objs[-1])

    # Disable outline for the background axis.
    for _, spine in grid_ax.spines.items():
        spine.set_visible(False)
        grid_ax.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False)

    # Draw the grid.
    if grid:
        grid_ax.grid(b=True, which='major', color='gray', linewidth=0.5, axis='x')
        
    # Set ylimits to be the same on all axes
    for ax in ax_objs:
        ax.set_ylim([ymin_all, ymax_all])
    grid_ax.set_ylim([ymin_all, ymax_all])
        
    plt.title(title)
    return fig

def RidgePlot2(data, xlabel, colors=None, filled=None, stat=None, alpha=0.75, hspace=-0.75, nbins=None, binrange=None, title='', figsize=None, grid=True, fig=None, gs_fig=None):
    import matplotlib as mpl
    import tol_colors
    
    assert fig is None or figsize is None

    if stat is None:
        stat = 'count'
        
    if binrange is None:
        nbins, binrange = AutoScaleHistogram2(data.values(), nbins)
        
    ymin_all = None
    ymax_all = None

    gs = mpl.gridspec.GridSpecFromSubplotSpec(
            len(data), 1, subplot_spec=gs_fig) #(mpl.gridspec.GridSpec(len(data), 1))
    if fig is None:
        fig = plt.figure(figsize=figsize)

    ax_objs = []
    for i, d in enumerate(data.values()):
        if colors is None:
            #color = sns.color_palette()[i]
            color = tol_colors.tol_cset('light')[i % 10]
        else:
            color = colors[i]

        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))#, sharex=sharex))

        density = False
        if stat == 'density':
            density = True
        hist = np.histogram(d, bins=nbins, range=binrange, density=density)
        width = hist[1][1] - hist[1][0]

        if stat == 'count' or stat == 'density': # density handled by histogram computation above.
            Bars = plt.bar(hist[1][:-1], hist[0], align='edge', width=width, color=color, edgecolor='none', alpha=alpha)
        elif stat == 'frequency':
            Bars = plt.bar(hist[1][:-1], hist[0]/width, align='edge', width=width, color=color, edgecolor='none', alpha=alpha)
        elif stat == 'probability' or stat == 'proportion' or stat == 'percent':
            # normalize such that bar heights sum to 1.
            total = np.sum(hist[0])
            stat_factor = 1/total
            if stat == 'percent':
                stat_factor *= 100
            Bars = plt.bar(hist[1][:-1], hist[0]*stat_factor, align='edge', width=width, color=color, edgecolor='none', alpha=alpha)
        else:
            raise Exception("Unknown 'stat': " + stat)
            
        #if you want your hlines to align with the bars.
        #i.e. start and end at the same x coordinates:

        x_start = np.array([plt.getp(item, 'x') for item in Bars])
        x_end   = x_start+[plt.getp(item, 'width') for item in Bars]
        
        # Draw white lines that outline the ridge plots, making them look nice.
        
        if stat == 'count' or stat == 'density': # density handled by histogram computation above.
            plt.hlines(hist[0], x_start, x_end, color='w', linewidth=0.5)
            plt.vlines(x_start[1:], hist[0][0:-1], hist[0][1:], color='w', linewidth=0.5)
        elif stat == 'frequency':
            plt.hlines(hist[0]/width, x_start, x_end, color='w', linewidth=0.5)
            plt.vlines(x_start[1:], hist[0][0:-1]/width, hist[0][1:]/width, color='w', linewidth=0.5)
        elif stat == 'probability' or stat == 'proportion' or stat == 'percent':
            plt.hlines(hist[0]*stat_factor, x_start, x_end, color='w', linewidth=0.5)
            plt.vlines(x_start[1:], hist[0][0:-1]*stat_factor, hist[0][1:]*stat_factor, color='w', linewidth=0.5)
        else:
            raise Exception("Unknown 'stat': " + stat)
        
        # Capture ylims
        ymin, ymax = plt.gca().get_ylim()
        
        if ymin_all is None:
            ymin_all = ymin
        else:
            ymin_all = min(ymin, ymin_all)
            
        if ymax_all is None:
            ymax_all = ymax
        else:
            ymax_all = max(ymax, ymax_all)
            
        
        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_ylabel('')

        ax_objs[-1].set_xlabel('')
        
        if i == len(data)-1:
            ax_objs[-1].set_xlabel(xlabel)
            ax_objs[-1].tick_params(labelleft=False, labelbottom=True, left=False, right=False, bottom=False )
        else:
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False )

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        ax_objs[-1].text(0.025, 0, list(data.keys())[i], ha="right", transform=ax_objs[-1].transAxes)

    #gs.update(hspace = hspace)

    # Add an axis behind all the other axes so we can draw the grid.
    grid_ax = fig.add_subplot(1, 1, 1, zorder=-1, sharex = ax_objs[-1])

    # Disable outline for the background axis.
    for _, spine in grid_ax.spines.items():
        spine.set_visible(False)
        grid_ax.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False)

    # Draw the grid.
    if grid:
        grid_ax.grid(b=True, which='major', color='gray', linewidth=0.5, axis='x')
        
    # Set ylimits to be the same on all axes
    for ax in ax_objs:
        ax.set_ylim([ymin_all, ymax_all])
    grid_ax.set_ylim([ymin_all, ymax_all])
        
    plt.title(title)
    return fig, gs_fig