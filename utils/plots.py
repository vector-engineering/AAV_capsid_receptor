"""Plotting functions for SVAE model
Author: Hikari Sorensen - Vector Engineering Team (hsorense@broadinstitute.org)
Notes: 
"""

import matplotlib.pyplot as plt
from pathlib import Path

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


def plot_latent_space(
    preds_df,
    plots_outdir=None,
    plot_name=None,
    assay="Pulldown Assay",
    fig=None,
    cmap="coolwarm",
):

    vmin = preds_df[["y_pred", "y_true"]].min().min()
    vmax = preds_df[["y_pred", "y_true"]].max().max()

    if fig is None:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    else:
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

    true_vals = ax0.scatter(
        preds_df["z0"],
        preds_df["z1"],
        vmin=vmin,
        vmax=vmax,
        c=preds_df["y_true"],
        s=2,
        cmap=cmap,
        rasterized=True,
    )
    ax0.set_xlabel("z0")
    ax0.set_ylabel("z1")
    ax0.set_title("{} Latent Space | Training | y_true".format(assay))
    fig.colorbar(true_vals, ax=ax0, label="True Assay log2enr")

    pred_vals = ax1.scatter(
        preds_df["z0"],
        preds_df["z1"],
        vmin=vmin,
        vmax=vmax,
        c=preds_df["y_pred"],
        s=2,
        cmap=cmap,
        rasterized=True,
    )
    ax1.set_xlabel("z0")
    ax1.set_ylabel("z1")
    ax1.set_title("{} Latent Space | Training | y_pred".format(assay))
    fig.colorbar(pred_vals, ax=ax1, label="Predicted Assay log2enr")

    ax = (ax0, ax1)

    if plot_name is None:
        assay_name = assay.replace(" ", "_").lower()
        plot_name = "{}_train_latent_space".format(assay_name)
    if plots_outdir is not None:
        Path(plots_outdir).mkdir(parents=True, exist_ok=True)
        plot_name = str(Path(plots_outdir) / plot_name)

    fig.savefig("{}.svg".format(plot_name), transparent=True, format="svg")
    plt.savefig("{}.png".format(plot_name), transparent=True)

    print("Plot saved to {}.png".format(plot_name))

    plt.show()

    return fig, ax
