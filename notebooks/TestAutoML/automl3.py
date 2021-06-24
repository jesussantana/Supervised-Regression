def fit(self,X,y):
    X_train = X
    y_train = y

    categorical_values = []

    cat_subset = X_train.select_dtypes(include = ['object','category','bool'])

    for i in range(cat_subset.shape[1]):
      categorical_values.append(list(cat_subset.iloc[:,i].dropna().unique()))