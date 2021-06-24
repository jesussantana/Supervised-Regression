
# Logistic regression
optimization_grid.append({
    'preprocessor__numerical__scaler':[RobustScaler(),StandardScaler(),MinMaxScaler()],
    'preprocessor__numerical__cleaner__strategy':['mean','median'],
    'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
    'estimator':[LogisticRegression()]
})