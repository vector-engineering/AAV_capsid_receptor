import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os, re

import tensorflow as tf

import logomaker
from joblib import dump, load
from sklearn.cluster import KMeans, DBSCAN

# from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor

from itertools import product
from pathlib import Path

import logomaker

#  ClustalX AA Colors
#  (by properties + conservation)
#  http://www.jalview.org/help/html/colourSchemes/clustal.html
clustalXAAColors = {
    #    Hydrophobic (Blue)
    "A": "#809df0",
    "I": "#809df0",
    "L": "#809df0",
    "M": "#809df0",
    "F": "#809df0",
    "W": "#809df0",
    "V": "#809df0",
    #    Positive charge (Red)
    "K": "#ed000a",
    "R": "#ed000a",
    #    Negative charge (Magenta)
    "D": "#be38bf",
    "E": "#be38bf",
    #    Polar (Green)
    "N": "#29c417",
    "Q": "#29c417",
    "S": "#29c417",
    "T": "#29c417",
    #    Cysteins (Pink)
    "C": "#ee7d80",
    #    Glycines (Orange)
    "G": "#ef8f48",
    #    Prolines (Yellow)
    "P": "#c1c204",
    #    Aromatics (Cyan)
    "H": "#23a6a4",
    "Y": "#23a6a4",
    #    STOP
    "_": "#FF0000",
    "*": "#AAAAAA",
}


CEL_logits = tf.nn.softmax_cross_entropy_with_logits
mse = tf.keras.losses.MeanSquaredError()
ELU = tf.nn.elu

AAs = np.array(
    [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]
)


def cluster_latent_space(
    preds_df,
    assay="Pulldown Assay",
    clustering_method="kmeans",
    cluster_model_outdir=None,
    cluster_model_name="kmeans",
    cluster_with_assay_vals=True,
    kmeans_n_clusters=5,
    dbscan_eps=0.06,
    dbscan_min_samples=5,
    y_scale_adjustment=0.5,
    feat_imps=False,
    plots_outdir=None,
    plot_name=None,
    show_plots=True,
):

    # X = train_df[[col for col in train_df.columns if re.match(r"x\d+", col)]].values
    Y = preds_df["y_true"].values
    Z = preds_df[[col for col in preds_df.columns if re.match(r"z\d+", col)]].values

    if feat_imps:
        rf = RandomForestRegressor(random_state=0)
        rf.fit(Z, Y)
        feat_imps = rf.feature_importances_
        imp_scaled_z = Z * feat_imps
    else:
        imp_scaled_z = Z

    scaled_labels = (
        (imp_scaled_z.max() - imp_scaled_z.min()) / (Y.max() - Y.min()) * (Y - Y.max())
        + imp_scaled_z.max()
    ) * y_scale_adjustment
    preds_df["y_true_scaled"] = scaled_labels

    z_with_scaled_labels = np.concatenate(
        (imp_scaled_z, scaled_labels.reshape((len(Y), 1))), axis=1
    )

    if cluster_with_assay_vals:
        data = z_with_scaled_labels
    else:
        data = imp_scaled_z

    ######## KMEANS ########

    if clustering_method == "kmeans":
        n_clusters = kmeans_n_clusters
        print("Fitting kmeans...")
        kmeans = KMeans(n_clusters=n_clusters).fit(data)
        print("Done fitting kmeans.")
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        if cluster_model_outdir is not None:
            Path(cluster_model_outdir).mkdir(parents=True, exist_ok=True)
            cluster_model_name = cluster_model_outdir / cluster_model_name
        dump(kmeans, "{}.joblib".format(cluster_model_name))

    ######## DBSCAN ########

    elif clustering_method == "dbscan":
        print("Fitting DBSCAN...")
        db_clust_latent = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(
            data
        )
        print("Done fitting DBSCAN.")
        labels = db_clust_latent.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_dbscan = list(labels).count(-1)
        print("Estimated number of clusters: {}".format(n_clusters))
        print("Estimated number of noise points: {}".format(n_noise_dbscan))
        cluster_centers = np.array(
            [np.mean(data[labels == i], axis=0) for i in range(n_clusters)]
        )
        if cluster_model_outdir is not None:
            Path(cluster_model_outdir).mkdir(parents=True, exist_ok=True)
            cluster_model_name = cluster_model_outdir / cluster_model_name
        dump(db_clust_latent, "{}.joblib".format(cluster_model_name))

    print("Saved clustering model to {}.".format(cluster_model_name))
    preds_df["cluster_label"] = labels

    # Compute the mean assay log2enr per cluster, add to preds_df
    cluster_assay_means = np.array([np.mean(Y[labels == i]) for i in range(n_clusters)])

    preds_df["cluster_assay_mean"] = np.nan
    preds_df.loc[
        preds_df["cluster_label"] >= 0, "cluster_assay_mean"
    ] = cluster_assay_means[
        preds_df[preds_df["cluster_label"] >= 0]["cluster_label"].values
    ]

    ### CLUSTERING PLOTS ###

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9, 2.5), dpi=200, sharey=True)
    fig.subplots_adjust(bottom=0.2, wspace=0.02, top=0.98)

    pointsize = 0.1
    cmap = "coolwarm"
    vmin = preds_df[["y_pred", "y_true"]].min().min()
    vmax = preds_df[["y_pred", "y_true"]].max().max()

    y_true = ax0.scatter(
        preds_df["z0"],
        preds_df["z1"],
        vmin=vmin,
        vmax=vmax,
        c=preds_df["y_true"],
        s=pointsize,
        cmap=cmap,
        rasterized=True,
    )
    ax1.scatter(
        preds_df["z0"],
        preds_df["z1"],
        vmin=vmin,
        vmax=vmax,
        c=preds_df["y_pred"],
        s=pointsize,
        cmap=cmap,
        rasterized=True,
    )
    ax2.scatter(
        preds_df["z0"],
        preds_df["z1"],
        vmin=vmin,
        vmax=vmax,
        c=preds_df["cluster_assay_mean"],
        s=pointsize,
        cmap=cmap,
        rasterized=True,
    )

    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes([0.95, 0.2, 0.01, 0.78])
    fig.colorbar(y_true, cax=cbar_ax, orientation="vertical")
    cbar_ax.tick_params(labelsize=7)

    for ax in (ax0, ax1, ax2):
        ax.tick_params(left=False, labelsize=7)
        ax.set_xlabel("z0", fontsize=8)
    ax0.set_ylabel("z1", fontsize=8)
    ax0.tick_params(left=True, labelsize=7)

    ax0.set_title("{} Latent Space | y_true".format(assay), fontsize=9)
    ax1.set_title("{} Latent Space | y_pred".format(assay), fontsize=9)
    ax2.set_title("{} Latent Space | cluster_assay_mean".format(assay), fontsize=9)

    if plot_name is None:
        assay_name = assay.replace(" ", "_").lower()
        plot_name = "{}_latent_space_clusters".format(assay_name)
    if plots_outdir is not None:
        Path(plots_outdir).mkdir(parents=True, exist_ok=True)
        plot_name = str(Path(plots_outdir) / plot_name)

    fig.savefig("{}.svg".format(plot_name), transparent=True, format="svg")
    plt.savefig("{}.png".format(plot_name), transparent=True)

    print("Plot saved to {}.png".format(plot_name))
    if show_plots:
        plt.show()

    # print("\nGenerating Gaussian mixture...\n")

    # if len(fig_label) > 0:
    #     fig_label = "{}_{}".format(fig_label, clustering_method)
    # else:
    #     fig_label = clustering_method

    # mean_y_preds, gen_cov = GMM_on_clusters(
    #     model, latent_df, cluster_centers, cluster_assay_means, fig_label=fig_label
    # )

    return preds_df, {
        "cluster_centers": cluster_centers,
        "cluster_assay_means": cluster_assay_means,
    }


def GMM_on_clusters(
    model,
    latent_df,
    cluster_centers,
    cluster_assay_means,
    fig_label="",
    test_samples_per_cluster=50,
    var_scale=1,
    alphabet=AAs,
    mer=7,
    results_outpath=None,
):

    if results_outpath == None:
        results_outpath = os.getcwd() + "/"

    regressor = model.get_layer("regressor")
    decoder = model.get_layer("decoder")

    data = latent_df[["z0", "z1"]]

    n_clusters = latent_df["cluster_label"].max() + 1

    gen_cov = [
        np.cov(data[latent_df["cluster_label"] == k].T) for k in range(n_clusters)
    ]

    mean_y_preds = []

    # construct each cluster
    for k in range(n_clusters):

        # take only first two coordinates of cluster center to use in gaussian distribution
        #    (third corresponds to assay values used in clustering)
        latent_cluster_mean = cluster_centers[k][:2]

        # skip over single-point clusters, set mean y to be nan (so not to select in top when sampling)
        if np.isnan(gen_cov[k]).any():
            mean_y_preds.append(np.nan)
        else:

            reg_inputs = []

            # sample from cluster to get mean assay predictions for cluster ranking
            cluster_samples = np.random.multivariate_normal(
                latent_cluster_mean, var_scale * gen_cov[k], test_samples_per_cluster
            )
            x_hats = decoder.predict(cluster_samples)

            for i, x_hat in enumerate(x_hats):
                # turn decoded samples into one-hot vectors
                one_hot = np.zeros((mer, len(alphabet)))
                reshaped = x_hat.reshape((mer, len(alphabet)))
                hot_indices = reshaped.argmax(axis=1)
                one_hot[np.arange(len(hot_indices)), hot_indices] = 1

                # make sample compatible with model regressor
                reg_input = np.append(cluster_samples[i], one_hot.flatten())
                reg_inputs.append(reg_input)

            # predict regression score of samples
            y_preds = regressor.predict(np.array(reg_inputs))

            mean_y_preds.append(np.mean(y_preds))

    print("Mean y preds: ", mean_y_preds)
    print("Mean y assay: ", cluster_assay_means)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    scatter = ax.scatter(
        latent_df["z0"][latent_df["cluster_label"] >= 0],
        latent_df["z1"][latent_df["cluster_label"] >= 0],
        vmin=np.min(latent_df["y_true"]),
        vmax=np.max(latent_df["y_true"]),
        c=latent_df["cluster_assay_mean"][latent_df["cluster_label"] >= 0],
        cmap="RdYlGn",
        s=2,
        alpha=0.5,
    )
    ax.scatter(
        latent_df["z0"][latent_df["cluster_label"] < 0],
        latent_df["z1"][latent_df["cluster_label"] < 0],
        c="black",
        s=2,
    )

    norm = mpl.colors.Normalize(
        vmin=np.min(latent_df["y_true"]), vmax=np.max(latent_df["y_true"])
    )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdYlGn")
    colors = sm.to_rgba(cluster_assay_means)

    # Plot sampling distributions
    for i, (center, covar) in enumerate(zip(cluster_centers, gen_cov)):
        if np.isnan(covar).any():
            continue

        mean = center[:2]
        covar *= var_scale
        covar = covar.astype("float64")

        v, w = LA.eigh(covar)

        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / LA.norm(w[0])

        # Plot an ellipse to show the Gaussian distribution
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees

        ell = mpl.patches.Ellipse(
            mean,
            v[0],
            v[1],
            180.0 + angle,
            edgecolor="black",
            ls="dotted",
            lw=1,
            fill=False,
            alpha=0.6,
        )
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)

    fig.colorbar(scatter, ax=ax)
    ax.set_title("{} latent clusters".format(fig_label))
    figpath = "{}_latent_clusters".format(fig_label)
    plt.savefig(results_outpath + "{}.png".format(figpath))
    plt.show()

    return mean_y_preds, gen_cov


def generate_without_sampling(
    preds_df,
    cluster_outputs,
    assay="Pulldown Assay",
    use_top_clusters=True,
    num_top_clusters=1,
    subcluster_model_outdir=None,
    subcluster_model_name="kmeans_sub",
    cluster_mean_threshold=None,
    kmeans_n_subclusters=10,
    cluster_with_assay_vals=True,
    y_scale_adjustment=0.5,
    AA_percentile_threshold=80,
    AA_colname="AA_sequence",
    show_logos=False,
    show_subcluster_plots=False,
    show_sublogos=False,
    plots_outdir=None,
    sequences_outdir=None,
):

    cluster_centers = cluster_outputs["cluster_centers"]
    cluster_assay_means = cluster_outputs["cluster_assay_means"]

    if use_top_clusters:
        if num_top_clusters is not None:
            top_clusters = list(np.argsort(-cluster_assay_means)[:num_top_clusters])
        else:
            if cluster_mean_threshold == None:
                cluster_mean_threshold = np.nanmean(cluster_assay_means)
            top_clusters = [
                c
                for c in range(len(cluster_centers))
                if cluster_assay_means[c] > cluster_mean_threshold
            ]
    else:
        top_clusters = range(len(cluster_centers))

    cluster_ranking = np.argsort(np.argsort(-cluster_assay_means))

    # Make cluster logo plots and save to disk
    if show_logos:
        print("\n----- Top cluster logos -----")

    for cluster in top_clusters:
        print(
            "\nCluster {}: assay mean = {}, ranking = {}".format(
                cluster, cluster_assay_means[cluster], cluster_ranking[cluster]
            )
        )
        cluster_seqs = preds_df[preds_df["cluster_label"] == cluster]

        # Make logo plots
        fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
        logomaker.Logo(
            logomaker.alignment_to_matrix(
                cluster_seqs[AA_colname].values, counts=cluster_seqs["y_true"].values
            ),
            color_scheme=clustalXAAColors,
            ax=ax,
        )
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels(np.arange(7))
        ax.set_yticks([])
        ax.set_title(
            "Cluster {}: assay mean = {}, ranking = {}".format(
                cluster, cluster_assay_means[cluster], cluster_ranking[cluster]
            )
        )
        plot_name = "cluster{}_mean{:.2f}".format(
            cluster_ranking[cluster], cluster_assay_means[cluster]
        )
        if plots_outdir is not None:
            Path(plots_outdir / "cluster_logos").mkdir(parents=True, exist_ok=True)
            plot_name = str(Path(plots_outdir) / "cluster_logos" / plot_name)

        fig.savefig("{}.svg".format(plot_name), transparent=True, format="svg")
        plt.savefig("{}.png".format(plot_name), transparent=True)

        print("Logo plot saved to {}.png".format(plot_name))

        if show_logos:
            plt.show()

    #### SEQUENCE GENERATION ####

    generated_sequences_list = np.array([])
    generated_sequences_dict = {}

    # For each of the top clusters, do latent clustering again to form subclusters
    for cluster in top_clusters:
        generated_sequences_dict["cluster{}".format(cluster)] = {}

        preds_df_sub = preds_df[preds_df["cluster_label"] == cluster].copy()
        preds_df_sub, subcluster_outputs = cluster_latent_space(
            preds_df_sub,
            assay=assay + " Subclusters",
            cluster_model_outdir=subcluster_model_outdir,
            cluster_model_name=subcluster_model_name,
            cluster_with_assay_vals=cluster_with_assay_vals,
            kmeans_n_clusters=kmeans_n_subclusters,
            y_scale_adjustment=y_scale_adjustment,
            plots_outdir=plots_outdir / "subclusters",
            plot_name="cluster{}_mean{:.2f}_subclusters".format(
                cluster_ranking[cluster], cluster_assay_means[cluster]
            ),
            show_plots=show_subcluster_plots,
        )

        preds_df["subcluster_label"] = np.nan
        preds_df.loc[
            preds_df["cluster_label"] == cluster, "subcluster_label"
        ] = preds_df_sub["cluster_label"]

        subcluster_centers = subcluster_outputs["cluster_centers"]
        subcluster_assay_means = subcluster_outputs["cluster_assay_means"]
        subcluster_ranking = np.argsort(np.argsort(-subcluster_assay_means))

        if show_sublogos:
            print("\n----- Subcluster logos by assay mean (descending) -----")

        # Iterating through subclusters, in descending order of subcluster assay mean
        for subcluster in np.argsort(-subcluster_assay_means):
            generated_sequences_dict["cluster{}".format(cluster)][
                "subcluster{}".format(subcluster)
            ] = []

            preds_df["subcluster_assay_mean"] = np.nan
            preds_df.loc[
                preds_df["subcluster_label"] == subcluster, "subcluster_assay_mean"
            ] = subcluster_assay_means[subcluster]

            print(
                "\nSubcluster {}: assay mean = {}, ranking = {}".format(
                    subcluster,
                    subcluster_assay_means[subcluster],
                    subcluster_ranking[subcluster],
                )
            )

            subcluster_seqs = preds_df_sub[preds_df_sub["cluster_label"] == subcluster]

            # Make logo plots
            fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
            subcluster_matrix = logomaker.alignment_to_matrix(
                subcluster_seqs[AA_colname].values,
                counts=subcluster_seqs["y_true"].values,
            )
            logomaker.Logo(
                subcluster_matrix,
                color_scheme=clustalXAAColors,
                ax=ax,
            )
            ax.set_xticks(np.arange(7))
            ax.set_xticklabels(np.arange(7))
            ax.set_yticks([])
            ax.set_title(
                "Subcluster {}: assay mean = {}, ranking = {}".format(
                    subcluster,
                    subcluster_assay_means[subcluster],
                    subcluster_ranking[subcluster],
                )
            )
            plot_name = "subcluster{}_mean{:.2f}".format(
                subcluster_ranking[subcluster], subcluster_assay_means[subcluster]
            )
            if plots_outdir is not None:
                Path(plots_outdir / "subcluster_logos").mkdir(
                    parents=True, exist_ok=True
                )
                plot_name = str(Path(plots_outdir) / "subcluster_logos" / plot_name)

            fig.savefig("{}.svg".format(plot_name), transparent=True, format="svg")
            plt.savefig("{}.png".format(plot_name), transparent=True)

            print("Logo plot saved to {}.png".format(plot_name))
            if show_sublogos:
                plt.show()

            print(
                "Generating sequences for cluster {}, subcluster {}.".format(
                    cluster, subcluster
                )
            )

            # Get AAs per position with frequency above threshold
            subcluster_matrix = subcluster_matrix.T
            topAAs = [
                list(
                    subcluster_matrix[
                        subcluster_matrix[t]
                        > np.percentile(subcluster_matrix[t], AA_percentile_threshold)
                    ].index
                )
                for t in range(7)
            ]

            # Combinatorially generate novel sequences from list of AAs per position
            # above frequency threshold
            combos = product(*topAAs)
            sequences = []
            for combo in combos:
                sequence = "".join(np.array(combo))
                sequences.append(sequence)

            print("{} sequences generated.".format(len(sequences)))
            generated_sequences_dict["cluster{}".format(cluster)][
                "subcluster{}".format(subcluster)
            ].extend(sequences)

            # Write sequences per subcluster to file
            sequences_filename = "cluster{}_subcluster{}".format(cluster, subcluster)
            if sequences_outdir is not None:
                Path(sequences_outdir).mkdir(parents=True, exist_ok=True)
                sequences_filename = Path(sequences_outdir) / sequences_filename
            with open("{}.txt".format(sequences_filename), "w") as f:
                for seq in sequences:
                    f.write(f"{seq}\n")

            # For the full generated sequences list, keep only the unique ones
            generated_sequences_list = pd.unique(
                np.append(generated_sequences_list, sequences)
            )

            print(
                "{} unique novel sequences generated so far.".format(
                    len(generated_sequences_list)
                )
            )

    # Write full novel sequences list to file
    sequences_filename = "all_novel_sequences"
    if sequences_outdir is not None:
        Path(sequences_outdir).mkdir(parents=True, exist_ok=True)
        sequences_filename = Path(sequences_outdir) / sequences_filename
    with open("{}.txt".format(sequences_filename), "w") as f:
        for seq in generated_sequences_list:
            f.write(f"{seq}\n")

    return generated_sequences_dict, generated_sequences_list


def generate_sequences(
    preds_df,
    assay="Pulldown Assay",
    clustering_method="kmeans",
    cluster_model_outdir=None,
    cluster_model_name="kmeans",
    subcluster_model_outdir=None,
    subcluster_model_name="kmeans_sub",
    cluster_with_assay_vals=True,
    subcluster_with_assay_vals=True,
    kmeans_n_clusters=5,
    kmeans_n_subclusters=10,
    y_scale_adjustment=0.5,
    y_scale_adjustment_sub=0.5,
    AA_percentile_threshold=80,
    show_plots=True,
    show_logos=False,
    show_subcluster_plots=False,
    show_sublogos=False,
    plots_outdir=None,
    sequences_outdir=None,
):

    preds_df, cluster_outputs = cluster_latent_space(
        preds_df,
        assay=assay,
        cluster_model_outdir=cluster_model_outdir,
        cluster_model_name=cluster_model_name,
        clustering_method=clustering_method,
        cluster_with_assay_vals=cluster_with_assay_vals,
        kmeans_n_clusters=kmeans_n_clusters,
        y_scale_adjustment=y_scale_adjustment,
        plots_outdir=plots_outdir,
        show_plots=show_plots,
    )

    generated_sequences_dict, generated_sequences_list = generate_without_sampling(
        preds_df,
        cluster_outputs,
        assay=assay,
        subcluster_model_outdir=subcluster_model_outdir,
        subcluster_model_name=subcluster_model_name,
        kmeans_n_subclusters=kmeans_n_subclusters,
        cluster_with_assay_vals=subcluster_with_assay_vals,
        y_scale_adjustment=y_scale_adjustment_sub,
        AA_percentile_threshold=AA_percentile_threshold,
        AA_colname="AA_sequence",
        show_logos=show_logos,
        show_subcluster_plots=show_subcluster_plots,
        show_sublogos=show_sublogos,
        plots_outdir=plots_outdir,
        sequences_outdir=sequences_outdir,
    )

    return generated_sequences_dict, generated_sequences_list
