model_pipeline_steps = []
model_pipeline_steps.append(('preprocessor',preprocessor))
model_pipeline_steps.append(('feature_selector',SelectKBest(f_classif,k='all')))
model_pipeline_steps.append(('estimator',LogisticRegression()))
model_pipeline = Pipeline(model_pipeline_steps)