

import numpy as np
import pandas as pd


def normalize(data,
              new_min=0,
              new_max=1
             ):
    
    old_range = data.max() - data.min() 
    new_range = new_max - new_min  
    new_data = (((data - data.min()) * new_range) / old_range) + new_min
    return new_data
    
#     a = (vmax - vmin) / (data.max()-data.min())
#     b = data.max() - a * data.max()
    
#     return a * data + b