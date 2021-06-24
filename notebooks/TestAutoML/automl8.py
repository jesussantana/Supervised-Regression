 # K-nearest neighbors
optimization_grid.append({
    'preprocessor__numerical__scaler':[RobustScaler(),StandardScaler(),MinMaxScaler()],
    'preprocessor__numerical__cleaner__strategy':['mean','median'],
    'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
    'estimator':[KNeighborsClassifier()],
    'estimator__weights':['uniform','distance'],
    'estimator__n_neighbors':np.arange(1,20,1)
})

# Random Forest
optimization_grid.append({
    'preprocessor__numerical__scaler':[None],
    'preprocessor__numerical__cleaner__strategy':['mean','median'],
    'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
    'estimator':[RandomForestClassifier(random_state=0)],
    'estimator__n_estimators':np.arange(5,500,10),
    'estimator__criterion':['gini','entropy']
})


# Gradient boosting
optimization_grid.append({
    'preprocessor__numerical__scaler':[None],
    'preprocessor__numerical__cleaner__strategy':['mean','median'],
    'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
    'estimator':[GradientBoostingClassifier(random_state=0)],
    'estimator__n_estimators':np.arange(5,500,10),
    'estimator__learning_rate':np.linspace(0.1,0.9,20),
})



# Decision tree
optimization_grid.append({
    'preprocessor__numerical__scaler':[None],
    'preprocessor__numerical__cleaner__strategy':['mean','median'],
    'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
    'estimator':[DecisionTreeClassifier(random_state=0)],
    'estimator__criterion':['gini','entropy']
})

# Linear SVM
optimization_grid.append({
    'preprocessor__numerical__scaler':[RobustScaler(),StandardScaler(),MinMaxScaler()],
    'preprocessor__numerical__cleaner__strategy':['mean','median'],
    'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
    'estimator':[LinearSVC(random_state = 0)],
    'estimator__C': np.arange(0.1,1,0.1),

})