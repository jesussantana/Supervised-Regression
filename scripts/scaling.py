# Variable Scaling

import pandas as pd 
from sklearn.preprocessing import StandardScaler

def transform(X_train, X_test, column):

    # fit on training data column
    scale = StandardScaler().fit(X_train)
    
    # transform the training & Test data column
    X_train_stand = scale.transform(X_train)
    
    X_test_stand = scale.transform(X_train)

    X_train_stand = pd.DataFrame(X_train_stand, columns = column)
    X_test_stand = pd.DataFrame(X_test_stand, columns = column)

    return X_train_stand, X_test_stand