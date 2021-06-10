num_pipeline = Pipeline([
   ('cleaner',SimpleImputer()),
   ('scaler',StandardScaler())
])

cat_pipeline = Pipeline([
    ('cleaner',SimpleImputer(strategy = 'most_frequent')),
    ('encoder',OneHotEncoder(sparse = False, categories=categorical_values))
])


preprocessor = ColumnTransformer([
  ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
  ('categorical', cat_pipeline, make_column_selector(dtype_include=['object','category','bool']))
])