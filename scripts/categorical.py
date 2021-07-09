# Create Dummies
# ==============================================================================
""" import sys
    new_path = '../scripts/'
    if new_path not in sys.path:
    sys.path.append(new_path)"""
# ==============================================================================

import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

def transform(df, var_name):

    dummy = pd.get_dummies(df[var_name], prefix=var_name)
    df = df.drop(var_name, axis = 1)
    df = pd.concat([df, dummy], axis = 1)

    return df


