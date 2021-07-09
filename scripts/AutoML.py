# AutoML Classifier
# ==============================================================================
# Build an AutoML library in Python, which will perform actions automatically
# ==============================================================================

from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import RFECV, SelectKBest, f_regression, f_classif 

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor, LogisticRegression

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.datasets import make_blobs

from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ==============================================================================

class AutoMLClassifier:
  def __init__(self, scoring_function = 'balanced_accuracy', n_iter = 50):
    self.scoring_function = scoring_function
    self.n_iter = n_iter

  # Build the "fit" method 
  def fit(self,X,y):
    X_train = X
    y_train = y

    categorical_values = []

    cat_subset = X_train.select_dtypes(include = ['object','category','bool'])

    for i in range(cat_subset.shape[1]):
      categorical_values.append(list(cat_subset.iloc[:,i].dropna().unique()))

    # Numerical Pipeline
    num_pipeline = Pipeline([
                         ('cleaner',SimpleImputer()),
                         ('scaler',StandardScaler())
                         ])

    # Categorical Pipeline
    cat_pipeline = Pipeline([
                        ('cleaner',SimpleImputer(strategy = 'most_frequent')),
                        ('encoder',OneHotEncoder(sparse = False, categories=categorical_values))
    ])

    # Preprocessing
    preprocessor = ColumnTransformer([
      ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
      ('categorical', cat_pipeline, make_column_selector(dtype_include=['object','category','bool']))
    ])



    # Define the ML pipeline
    # We set the model to LogisticRegression at this point; will be changed later by random search
    model_pipeline_steps = []
    model_pipeline_steps.append(('preprocessor',preprocessor))
    model_pipeline_steps.append(('feature_selector',SelectKBest(f_classif,k='all')))
    model_pipeline_steps.append(('estimator',LogisticRegression()))
    model_pipeline = Pipeline(model_pipeline_steps)

    # We calculate the number of characteristics
    total_features = preprocessor.fit_transform(X_train).shape[1]

    # Add models to our optimization grid
    optimization_grid = []

    # Logistic regression
    # Change the scale between RobustScaler, StandardScaler and MinMaxscaler
    # Then you will change the cleaning strategy between mean and median and select the characteristics
    optimization_grid.append({
        'preprocessor__numerical__scaler':[RobustScaler(),StandardScaler(),MinMaxScaler()],
        'preprocessor__numerical__cleaner__strategy':['mean','median'],
        'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
        'estimator':[LogisticRegression()]
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

    # Random Forest
    optimization_grid.append({
        'preprocessor__numerical__scaler':[None],
        'preprocessor__numerical__cleaner__strategy':['mean','median'],
        'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
        'estimator':[RandomForestClassifier(random_state=0)],
        'estimator__n_estimators':np.arange(5,500,10),
        'estimator__criterion':['gini','entropy']
    })

    # Decision tree
    optimization_grid.append({
        'preprocessor__numerical__scaler':[None],
        'preprocessor__numerical__cleaner__strategy':['mean','median'],
        'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
        'estimator':[DecisionTreeClassifier(random_state=0)],
        'estimator__criterion':['gini','entropy']
    })

    # K-nearest neighbors
    optimization_grid.append({
        'preprocessor__numerical__scaler':[RobustScaler(),StandardScaler(),MinMaxScaler()],
        'preprocessor__numerical__cleaner__strategy':['mean','median'],
        'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
        'estimator':[KNeighborsClassifier()],
        'estimator__weights':['uniform','distance'],
        'estimator__n_neighbors':np.arange(1,20,1)
    })

    # Linear SVM
    optimization_grid.append({
        'preprocessor__numerical__scaler':[RobustScaler(),StandardScaler(),MinMaxScaler()],
        'preprocessor__numerical__cleaner__strategy':['mean','median'],
        'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
        'estimator':[LinearSVC(random_state = 0)],
        'estimator__C': np.arange(0.1,1,0.1),
        
    })

    # Random search applies
    search = RandomizedSearchCV(
      model_pipeline,
      optimization_grid,
      n_iter=self.n_iter,
      scoring = self.scoring_function, 
      n_jobs = -1, 
      random_state = 0, 
      verbose = 3,
      cv = 5
    )

    search.fit(X_train, y_train)
    self.best_estimator_ = search.best_estimator_
    self.best_pipeline = search.best_params_
    

  # Prediction methods
  def predict(self,X,y = None):
    return self.best_estimator_.predict(X)

  def predict_proba(self,X,y = None):
    return self.best_estimator_.predict_proba(X)