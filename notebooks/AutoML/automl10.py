def predict(self,X,y = None):
  return self.best_estimator_.predict(X)

def predict_proba(self,X,y = None):
  return self.best_estimator_.predict_proba(X)