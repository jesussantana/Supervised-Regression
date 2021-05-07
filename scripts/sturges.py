# Sturges rule 

import numpy as np

def createBins(df, column):
    k = int(np.ceil(1+np.log2(len(df[f'{column}']))))
    

    return k