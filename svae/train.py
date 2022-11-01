"""Training module for SVAE model
Author: Hikari Sorensen - Vector Engineering Team (hsorense@broadinstitute.org)
Notes: 
"""

import os, re
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from pathlib import Path

from svae.predict import predict
from utils.loss import loss, grad
from tensorflow.keras.utils import Progbar


# -- train -- (incorporates CV)
# trains model, returns (model, train_losses, val_losses, preds_df)


def train(
    model,
    train_batches,
    val_batches,
    train_df,
    optimizer=None,
    model_outdir=None,  # Directory for saving model and logs
    loss_weights=[1.0, 0.5, 0.1],
    patience=10,  # Number of epochs to continue training for after convergence
    min_epochs=50,
    max_epochs=100,
    convergence_threshold=0.005,
    progbar_verbosity=1,  # Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(1e-3)

    # Get some info about the batches - deducing these values from the batches is pretty
    # inefficient given that they're parameters passed in in the first place, but it's
    # also sort of nicer not to have to re-input parameters repeatedly...
    batches_list = list(train_batches.as_numpy_iterator())
    batch_size = batches_list[0][0].shape[0]
    last_batch_size = batches_list[-1][0].shape[0]
    num_batches = len(batches_list)
    num_training_samples = (num_batches - 1) * batch_size + last_batch_size

    Y_colname = [col for col in train_df.columns if "Y--" in col][0]
    assay = Y_colname.split("Y--")[-1].split("-")[0]

    ## -- Metrics -- ##
    # model loss: aggregate loss that incorporates reconstruction, kl and regression losses
    # reconstruction loss: (strictly vae loss) binary crossentropy between input one-hot
    # sequences and decoded (output) one-hot
    # kl loss: (statistical) difference of distributions - this is a form of
    # regularization, essentially
    # regression loss: mean squared error loss between true log2enr values and
    # regressor's prediction

    train_metrics = [
        "train_model_loss",
        "train_reconstruction_loss",
        "train_kl_loss",
        "train_regression_loss",
    ]
    val_metrics = [
        "val_model_loss",
        "val_reconstruction_loss",
        "val_kl_loss",
        "val_regression_loss",
    ]

    metrics_names = train_metrics + val_metrics

    print("Logging following metrics: {}".format(metrics_names))
    print("\n\n----- Beginning training. -----")

    # Losses logged to file
    train_losses = []
    val_losses = []

    # Keeps track of max difference in loss values between consecutive epochs, across all training losses (model total, reconstruction, kl, regression)
    convergence_history = []

    # After convergence, model continues to train (stalls) for number of epochs specified by 'patience' parameter
    stall = 0
    converged = False
    epochs_run = 0

    #### ---- TRAINING ---- ####
    while not converged or stall < patience:
        epoch_train_losses = []
        epoch_val_losses = []
        for i in range(len(train_metrics)):
            # Adding a tf.keras.metrics.Mean() object for each tracked loss that serves
            # as a container for loss values per iteration; after each epoch, each loss
            # container's state gets updated to contain the most recent loss values, and
            # the mean loss over the epoch is computed
            epoch_train_losses.append(tf.keras.metrics.Mean())
            epoch_val_losses.append(tf.keras.metrics.Mean())

        print("\nepoch {}".format(epochs_run + 1))
        if converged:
            print("Converged. Stall: {}/{}".format(stall, patience))

        # Nice printout of training progress
        progBar = Progbar(
            num_training_samples,
            stateful_metrics=metrics_names,
            verbose=progbar_verbosity,
            interval=0.1,
        )

        ### --- BEGIN EPOCH --- ###
        for i, examples in enumerate(train_batches):
            x = examples[0]
            y = examples[1]

            # Data optionally contains CV values, in addition to x and y values
            if len(examples) > 2:
                cv = examples[2]
            else:
                cv = 0

            # Compute training losses
            train_loss_values = loss(model, x, y, cv, loss_weights=loss_weights)

            # Backprop step
            grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Update training losses for printing and progbar
            for l, loss_val in enumerate(train_loss_values):
                epoch_train_losses[l].update_state(loss_val)
            values = zip(train_metrics, train_loss_values)

            # Update progress bar for every training example throughout the epoch
            progBar.update(i * batch_size, values=values)

        ### --- END EPOCH --- ###
        train_losses.append([loss_avg.result() for loss_avg in epoch_train_losses])

        #### ---- VALIDATION ---- ####
        for i, examples in enumerate(val_batches):
            batch_x = examples[0]
            batch_y = examples[1]

            if len(examples) > 2:
                batch_cv = examples[2]
            else:
                batch_cv = 0

            # Compute validation losses
            val_loss_values = loss(
                model,
                batch_x,
                batch_y,
                batch_cv,
                loss_weights=loss_weights,
            )
            # Update validation losses for printing and progbar
            for l, loss_val in enumerate(val_loss_values):
                epoch_val_losses[l].update_state(loss_val)

        val_losses.append([loss_avg.result() for loss_avg in epoch_val_losses])

        # Report both training and validation losses together at epoch end
        all_loss_avgs = [loss_avg.result() for loss_avg in epoch_train_losses] + [
            loss_avg.result() for loss_avg in epoch_val_losses
        ]
        values = zip(metrics_names, all_loss_avgs)

        # Update progress bar with end-of-epoch training and validation losses
        progBar.update(num_training_samples, values=values, finalize=True)

        epochs_run += 1

        if epochs_run >= max_epochs:
            print("Hit maximum epochs.")
            converged = True
            break

        if epochs_run < min_epochs:
            continue

        # Compute training loss improvement from previous epoch
        if not converged:
            improvement = np.array(train_losses[-1]) - np.array(train_losses[-2])

            if np.max(improvement) < convergence_threshold:
                convergence_history.append(1)
            else:
                convergence_history.append(0)

            print("Convergence history: {}".format(convergence_history))

        # Training has converged if the max loss difference between consecutive epochs
        # is less than convergence threshold for at least 3 of the last 5 runs
        if len(convergence_history) >= 5:
            recent_history = convergence_history[-5:]
            if not converged and np.sum(recent_history) / (len(recent_history)) >= 0.7:
                converged = True
                stall += 1
                continue

        if stall > 0:
            stall += 1

    final_train_model_loss = train_losses[-1][0]
    final_val_model_loss = val_losses[-1][0]
    print(
        "Finished training model. Final overall losses:\ntrain: {:.3f}    val:{:.3f}".format(
            final_train_model_loss, final_val_model_loss
        )
    )
    #### --- END OF TRAINING --- ####

    # Make dataframes of train and val loss histories and combine into in a single loss dataframe
    train_loss_df = pd.DataFrame(
        train_losses,
        columns=[
            "train_model_loss",
            "train_reconstruction_loss",
            "train_kl_loss",
            "train_regression_loss",
        ],
    )
    val_loss_df = pd.DataFrame(
        val_losses,
        columns=[
            "val_model_loss",
            "val_reconstruction_loss",
            "val_kl_loss",
            "val_regression_loss",
        ],
    )
    loss_df = pd.concat([train_loss_df, val_loss_df], axis=1)

    # Make train regressor predictions on train and val data combined
    AA = train_df["AA_sequence"].values
    X = train_df[[col for col in train_df.columns if re.match(r"x\d+", col)]].values
    Y = train_df[Y_colname].values
    preds_df = predict(model, X, Y=Y, AA=AA)
    train_df[Y_colname.replace("Y", "y_pred")] = preds_df["y_pred"]
    Z_cols = [col for col in preds_df.columns if re.match(r"z\d+", col)]
    for col in Z_cols:
        train_df[col] = preds_df[col]

    # Save everything to file

    if model_outdir is None:
        cwd = Path(os.getcwd())
        top_dir = cwd.parent
        now = datetime.now()
        latent_dim = model.get_layer("encoder").weights[-1].numpy().shape[0]
        model_outdir = Path(
            "{}/trained_models/{}{}{}_{}_{}D_{}epochs_{:.2f}T_{:.2f}V/".format(
                top_dir,
                now.year,
                now.month,
                now.day,
                assay,
                latent_dim,
                epochs_run,
                final_train_model_loss,
                final_val_model_loss,
            )
        )
    model_outdir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_outpath = model_outdir / "model"
    tf.keras.models.save_model(model, model_outpath)
    print("Model saved to {}.".format(str(model_outpath)))

    # Save loss df
    loss_outpath = model_outdir / "loss_log.csv"
    loss_df.to_csv(loss_outpath, index=False)
    print("Losses saved to {}.".format(str(loss_outpath)))

    # Save train predictions df
    preds_outpath = model_outdir / "preds.csv"
    preds_df.to_csv(preds_outpath, index=False)
    print("Train predictions saved to {}.".format(str(preds_outpath)))

    return model, preds_df, model_outdir
