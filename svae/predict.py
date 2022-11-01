"""Predictions module for SVAE model
Author: Hikari Sorensen - Vector Engineering Team (hsorense@broadinstitute.org)
Notes: 
"""

import pandas as pd
import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()

""" 
# -- predict --
Description: given a trained (supervised) VAE model and (one-hot encoded) inputs X, encodes X
into latent representation (z0, z1, ...) and makes regressor label predictions y_pred.
If true labels Y (optional argument) are specified, predictions are checked against Y
and the mean squared error (MSE) of predictions against true label is printed out.
Outputs a dataframe containing latent encodings, predicted labels, and true label values
if specified. Can optionally specify list of variants AA, in amino acid string form, as
an index for dataframe. Optionally writes this dataframe to file if a full output path
is specified. 
"""


def predict(model, X, Y=None, AA=None, outpath=None):

    encoder = model.get_layer("encoder")
    regressor = model.get_layer("regressor")

    # Get latent encoding for regressor input
    z_encoded_dataset = encoder.predict(X)[0]
    regressor_input = tf.concat([z_encoded_dataset, X], 1)

    # Make regressor predictions
    preds = regressor.predict(regressor_input)

    # Dataframe to store latent encoding, regressor predictions, and true labels (if
    # specified), indexed by AA strings (if specified)
    preds_df = pd.DataFrame(preds.flatten(), columns=["y_pred"])

    # Add columns in preds_df for latent space coordinates
    for i in range(z_encoded_dataset.shape[1]):
        preds_df["z{}".format(i)] = z_encoded_dataset[:, i]

    # Add optional AA index
    if AA is not None:
        preds_df.insert(0, "AA_sequence", AA)

    # If Y specified, compute MSE of predictions vs true labels and add Y column to df
    if Y is not None:
        mse_y = mse(Y, preds.flatten()).numpy()
        print("\nMSE of predictions vs true labels: {}".format(mse_y))
        preds_df["y_true"] = Y

    # Write dataframe to file, if full path is specified
    if outpath is not None:
        preds_df.to_csv(outpath)

    return preds_df
