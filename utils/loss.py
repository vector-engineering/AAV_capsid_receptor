
import numpy as np
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow.keras import backend as K

CEL_logits = tf.nn.softmax_cross_entropy_with_logits
mse = tf.keras.losses.MeanSquaredError()

AAs = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])


# -- KLD_Gaussian -- 
# Description: KL divergence of two (multi-dimensional) Gaussian distributions

def KLD_Gaussian(q_mu, q_sigma, p_mu, p_sigma):
    # 1/2 [log|Σ2|/|Σ1| −d + tr{Σ2^-1 Σ1} + (μ2−μ1)^T Σ2^-1 (μ2−μ1)]
    KLD = 1 / 2 * (2 * (p_sigma - q_sigma)
                   - 1
                   + tf.math.pow(((tf.math.exp(q_sigma)) / (tf.math.exp(p_sigma) + 1e-6)), 2)
                   + tf.math.pow(((p_mu - q_mu) / (tf.math.exp(p_sigma) + 1e-6)), 2))
    return K.sum(KLD, axis=-1)


# -- KLD_Categorical -- 
# Description: KL divergence of two categorical distributions

def KLD_Categorical(q, p):
    # sum (q log (q/p) )
    KLD = q * tf.math.log((q + 1e-4) / (p + 1e-4))
    return K.sum(KLD, axis=-1)



# -- loss -- 
# Description: computes VAE loss of given vae model on (input, label) pair 
# (X,Y). Optionally takes CV (coefficient of variation) of Y values into account
# dividing regression loss by CV+1 (so points with higher uncertainty incur
# lower loss penalty).


def loss(vae, X, Y, 
           CV=0, 
           alphabet=AAs, 
           loss_weights=[0.2, 1.0, 0.2], 
           kind='default'): 
    
    input_dim = tf.shape(X)[1].numpy()
    
    mer = int(input_dim / len(alphabet))
    
    encoder = vae.get_layer('encoder')
    decoder = vae.get_layer('decoder')
    regressor = vae.get_layer('regressor')
    z_mean, z_log_var, z = encoder(X)
    reconstruction, y_preds = vae(X)

    # reconstruction loss
    reconstruction_loss = 0
    for i in range(mer):
        logits = reconstruction[:, i*len(alphabet):(i+1)*len(alphabet)]
        labels = X[:, i*len(alphabet):(i+1)*len(alphabet)]
        reconstruction_loss += CEL_logits(labels, logits)
    
    # kl loss    
    if kind == 'default':
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    elif kind == 'linear':
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean - tf.reduce_mean(z_mean)) - tf.exp(z_log_var) + 1)
    
    # regression loss
    regression_loss = mse(y_true=Y, y_pred=y_preds)
    
    vae_loss = (loss_weights[0]*tf.cast(reconstruction_loss, tf.float64) + 
                loss_weights[1]*tf.cast(kl_loss, tf.float64) + 
                loss_weights[2]*tf.cast(regression_loss, tf.float64)/(CV+1)
               )
    
    return (vae_loss, reconstruction_loss, kl_loss, regression_loss)


# -- grad -- 
# Description: computes VAE gradient of given vae model on (input, label) pair (X,Y) (w.r.t. defined loss)

def grad(model, X, Y):
    with tf.GradientTape() as tape:
        vae_loss, _, _, _ = loss(model, X, Y)
    return tape.gradient(vae_loss, model.trainable_variables)
