"""Data processing functions module for pulldown assay data and SVAE model
Author: Hikari Sorensen - Vector Engineering Team (hsorense@broadinstitute.org)
Notes: 
"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf

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
#### ---- pulldown assay processing ---- ####

def RPM(dataframe):
    """Compute reads per million (RPM) of input dataframe columns.
    """
    total_read_count = dataframe.sum(axis=0)
    rpm = dataframe / (total_read_count / 1_000_000)

    return rpm

def log2enr(RPM_dataframe, starter_sample):
    """Compute log2 enrichment (log2enr) of dataframe of RPM values, relative to starter
    sample.
    """
    enr = RPM_dataframe/starter_sample

#### ---- SVAE model ---- ####

# -- onehot_flatten --
# Input: integer vector, total number of integers to expand into one-hot encoding
# Returns: one-hot encoding of integer vector input, flattened into long 1D vector


def onehot_flatten(integer_vec, alphabet=AAs):
    num_labels = len(alphabet)
    one_hot = np.zeros((len(integer_vec), num_labels))
    one_hot[np.arange(len(integer_vec)), integer_vec] = 1
    return one_hot.flatten()


# -- seq_to_onehot --
# Input: list of character-encoded amino acid sequences
# Returns: dataframe of flattened one-hot encodings of AA sequences, split into columns bitwise


def seq_to_onehot(character_sequences, alphabet=AAs):

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    seq_char_array = [list(sequence) for sequence in character_sequences]
    integer_encoded = pd.DataFrame(
        [[char_to_int[char] for char in seq] for seq in seq_char_array]
    )

    one_hot_dict = dict()
    for index, row in integer_encoded.iterrows():
        one_hot_dict[index] = list(onehot_flatten(row, alphabet))

    one_hot_df = pd.DataFrame.from_dict(one_hot_dict, orient="index").astype('int')

    return one_hot_df, integer_encoded


# -- onehot_to_seq --
# Input: one-hot encoded vector of shape (1, mer * len(alphabet))
# Returns: character string of amino acid sequence


def onehot_to_seq(one_hot_sequence, alphabet=AAs):
    mer = int(len(one_hot_sequence) / len(alphabet))
    letters = alphabet[
        np.argmax(one_hot_sequence.reshape((int(mer), len(alphabet))), axis=1)
    ]
    sequence = "".join(letters)
    return sequence


# -- make_tf_batches --
# Description: given one-hot-encoded inputs X and labels Y (and optionally cv values
# CV), splits data into training and validation batch sets compatible with TensorFlow models.
# Returns training and validation batches, as well as the unbatched data in a tuple.


def make_tf_batches(X, Y, CV=None, val_size=0.2, batch_size=32, shuffle=True):

    if CV is not None:
        unbatched_data = (X.values, Y.values, CV.values)
    else:
        unbatched_data = (X.values, Y.values)

    if shuffle:
        batches = tf.data.Dataset.from_tensor_slices(unbatched_data).shuffle(
            len(Y.values)
        )
    else:
        batches = tf.data.Dataset.from_tensor_slices(unbatched_data)

    num_train_samples = int((1.0 - val_size) * len(Y.values))

    train_batches = batches.take(num_train_samples).batch(batch_size)
    val_batches = batches.skip(num_train_samples).batch(batch_size)

    return train_batches, val_batches, unbatched_data


# -- prep_data --
# Description: given a dataframe df with AA sequences, target assay, CV and mean RPM,
# one-hot encodes the AA sequences and splits the data into training and test sets. The
# training split will be split further into training and validation batches with
# make_tf_batches().


def prep_data(
    df,
    target_assay_col,
    cv_col=None,
    RPM_col=None,
    spec_col=None,
    target_threshold=-np.inf,
    RPM_threshold=-np.inf,
    spec_threshold=-np.inf,
    test_size=0.1,
    alphabet=AAs,
    AA_colname="AA_sequence",
    random_seed=None
):
    if random_seed is not None:
        random.seed(random_seed)

    if RPM_col is not None:
        df = df[df[RPM_col] > RPM_threshold]
    if spec_col is not None:
        df = df[df[spec_col] > spec_threshold]

    df = df[df[target_assay_col] > target_threshold]

    X, integer_encoded = seq_to_onehot(list(df[AA_colname].values), alphabet=alphabet)
    Y = df[target_assay_col]

    x_cols = ["x{}".format(i) for i in range(X.shape[1])]
    new_df = X.rename(columns=dict(zip(X.columns, x_cols)))
    new_df.insert(0, "AA_sequence", df[AA_colname].values)
    new_df["Y--{}".format(target_assay_col)] = Y
    if cv_col is not None:
        CV = df[cv_col]
        new_df["CV--{}".format(cv_col)] = CV
        data_list = [X, Y, CV]
    else:
        data_list = [X, Y]

    new_df["test"] = False
    num_test_examples = int(test_size * len(Y))
    test_set_index = np.array(random.sample(list(np.arange(len(Y))), num_test_examples))

    new_df.loc[test_set_index,"test"] = True
    train_df = new_df[~new_df['test']].copy().reset_index(drop=True).drop(columns='test')
    test_df = new_df[new_df["test"]].copy().reset_index(drop=True).drop(columns='test')
    

    return data_list, train_df, test_df, new_df
