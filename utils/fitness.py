import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from .tools.data_processing import *

def predict_fitness(AA_sequence, 
                    model=None,
                    model_file='/mnt/shared/notebooks/albert/FitPred/Final_model_24K_v1.h5'
                   ):
    
    if model == None:
        model = tf.keras.models.load_model(model_file)
        
    X = seq_to_onehot(AA_sequence)[0]
    
    fitness_predictions = pd.DataFrame(AA_sequence, columns=['AA_sequence'])
    fitness_predictions['predicted_fitness'] = model.predict(X)
    
    return fitness_predictions
