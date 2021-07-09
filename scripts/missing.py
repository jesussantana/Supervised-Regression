# Missing Data
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# ==============================================================================

# Nan treatment
def transform(df, strategies):

    imputer = SimpleImputer(missing_values = np.nan, strategy = f'{strategies}', verbose=0) 
    imputer = imputer.fit(df)
    df = imputer.transform(df)

    return df