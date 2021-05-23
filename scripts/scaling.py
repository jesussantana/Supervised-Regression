# Variable Scaling Methods

import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer, MaxAbsScaler

# Standard Scaler
# ==============================================================================

def transform(X_train, X_test, column):

    # fit on training data column
    scale = StandardScaler().fit(X_train)
    
    # transform the training & Test data column
    X_train_stand = scale.transform(X_train)
    X_test_stand = scale.transform(X_test)

    X_train_stand = pd.DataFrame(X_train_stand, columns = column)
    X_test_stand = pd.DataFrame(X_test_stand, columns = column)

    return X_train_stand, X_test_stand


def transform_X(X_train, column):

    # fit on training data column
    scale = StandardScaler().fit(X_train)
    
    # transform the training & Test data column
    X_train_stand = scale.transform(X_train)

    X_train_stand = pd.DataFrame(X_train_stand, columns = column)
  

    return X_train_stand


# Min Max Scaler
# ==============================================================================

def transform_X_MinMax(X_train, column):

    # fit on training data column
    minmax = MinMaxScaler()
    
    # transform the training & Test data column
    X_train_MinMax = minmax.fit_transform(X_train)

    X_train_MinMax = pd.DataFrame(X_train_MinMax, columns = column)
  

    return X_train_MinMax


# Robust Scaler
# ==============================================================================

def transform_X_Robust(X_train, column):

    # fit on training data column
    Robust = RobustScaler(with_centering=False)
    
    # transform the training & Test data column
    X_train_Robust = Robust.fit_transform(X_train)

    X_train_Robust = pd.DataFrame( X_train_Robust, columns = column)
  

    return X_train_Robust

# Quantile transformer
# ==============================================================================

def transform_X_Quantile(X_train, column):

    # fit on training data column
    Quantile = QuantileTransformer(output_distribution='normal', n_quantiles=150)
    
    # transform the training & Test data column
    X_train_Quantile= Quantile.fit_transform(X_train)

    X_train_Quantile = pd.DataFrame( X_train_Quantile, columns = column)
  

    return X_train_Quantile

# Power transformer
# ==============================================================================

def transform_X_Power(X_train, column):

    # fit on training data column
    Power = PowerTransformer(method='yeo-johnson', standardize=True)
    
    # transform the training & Test data column
    X_train_Power = Power.fit_transform(X_train)

    X_train_Power = pd.DataFrame( X_train_Power, columns = column)
  

    return X_train_Power