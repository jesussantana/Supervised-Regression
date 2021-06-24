
class MyAutoMLClassifier:
  def __init__(self, 
    scoring_function = 'balanced_accuracy', 
    n_iter = 50):
    self.scoring_function = scoring_function
    self.n_iter = n_iter