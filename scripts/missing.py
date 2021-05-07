# Missing Data

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Nan treatment
def transform(df):

    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0) 
    imputer = imputer.fit(df)
    df = imputer.transform(df)

    return df