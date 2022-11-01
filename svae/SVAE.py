"""SVAE model
Author: Hikari Sorensen - Vector Engineering Team (hsorense@broadinstitute.org)
Notes: 
- this model uses TensorFlow's functional API
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# activation function
ELU = tf.nn.elu


# -- reparam_trick -- 
# see Kingma DP, Salimans T and Welling M. 2015. Variational Dropout and the Local Reparameterization Trick. Advances in Neural Information Processing Systems. https://proceedings.neurips.cc/paper/2015/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf

def reparam_trick(z_mean, z_log_var):
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# -------- ENCODER --------
def Encoder(enc_input_dim, enc_output_dim, enc_hidden_dims):
    
    latent_dim = enc_output_dim

    enc_inputs = Input(shape=(enc_input_dim,), name='encoder_input')
    enc_hidden = [Dense(dim, activation='linear', 
            name='enc_hidden_{}'.format(i)) for i, dim in enumerate(enc_hidden_dims)]
    
    z_mean_layer = Dense(latent_dim, activation='linear', name='z_mean')
    z_log_var_layer = Dense(latent_dim, activation='linear', name='z_log_var')
    
    # encoder evaluation
    x = enc_inputs
    for layer in enc_hidden:
        x = ELU(layer(x))

    z_mean = z_mean_layer(x)
    z_log_var = z_log_var_layer(x)
    
    #----- REPARAM TRICK ---------
    z = reparam_trick(z_mean, z_log_var)

    
    # instantiate encoder model
    encoder = Model(enc_inputs, [z_mean, z_log_var, z], name='encoder')
    
    return encoder, enc_inputs


# --------- DECODER -------
def Decoder(dec_input_dim, dec_output_dim, dec_hidden_dims):
    
    latent_dim = dec_input_dim
    
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    dec_hidden = [Dense(dim, activation='linear', 
            name='dec_hidden_{}'.format(i)) for i, dim in enumerate(dec_hidden_dims)]
    
    dec_output_layer = Dense(dec_output_dim, activation='linear')
    
    # decoder evaluation
    x = latent_inputs
    for layer in dec_hidden:
        x = ELU(layer(x))

    dec_outputs = dec_output_layer(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, dec_outputs, name='decoder')
    
    return decoder


# -------- REGRESSOR -----------
def Regressor(latent_dim, enc_input_dim, reg_hidden_dims):
    
    reg_latent_inputs = Input(shape=(latent_dim+enc_input_dim,), name='regressor_input')
    reg_layerz = Dense(reg_hidden_dims[0], activation='linear', name='reg_z')
    reg_layerx = Dense(reg_hidden_dims[0], activation='linear', name='reg_x')
    reg_hidden = [Dense(dim, activation='linear', 
            name='reg_hidden_{}'.format(i)) for i, dim in enumerate(reg_hidden_dims[1:])]
    reg_output_layer = Dense(1, activation='linear', name='reg_output')
    
    
    # regressor takes both the original one-hot encoding and the transformed latent 
    #   encoding as input
    z_input = reg_latent_inputs[:, :latent_dim]
    x_input = reg_latent_inputs[:, latent_dim:]
    
    # regressor evaluation
    reg_z = ELU(reg_layerz(z_input))
    reg_x = ELU(reg_layerx(x_input))
    x = tf.concat([reg_z, reg_x], 1)
    for layer in reg_hidden:
        x = ELU(layer(x))
    reg_outputs = reg_output_layer(x)
    
    # instantiate regressor model
    regressor = Model(reg_latent_inputs, reg_outputs, name='regressor')
    
    return regressor



# -- SVAE -- 

def SVAE(input_dim=140, 
        latent_dim=2, 
        enc_hidden_dims=[100,40], 
        dec_hidden_dims=[40,100],
        reg_hidden_dims=[100,10],
        name='svae'
       ):
    
    encoder, enc_inputs = Encoder(input_dim, latent_dim, enc_hidden_dims)
    decoder = Decoder(latent_dim, input_dim, dec_hidden_dims)
    if reg_hidden_dims and reg_hidden_dims != None:
        regressor = Regressor(latent_dim, input_dim, reg_hidden_dims)
        
        # configure full SVAE outputs (with regressor)
        model_outputs = [decoder(encoder(enc_inputs)[2]), 
                       regressor(tf.concat([encoder(enc_inputs)[0], enc_inputs],1))
                      ]
    else:
        # configure full VAE outputs (no regressor)
        model_outputs = [decoder(encoder(enc_inputs)[2])]
    
    # instantiate full model with regressor
    model = Model(enc_inputs, model_outputs, name=name)

    return model
