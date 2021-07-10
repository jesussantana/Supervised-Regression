# Evalue Models
# ==============================================================================
# Evalue Model Score &  scatter plot & Histogram
# ==============================================================================

# Modify functions to add X & y Train test as parameters !!!!!!!!!!!!!!!!!

# Dependencies
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, euclidean_distances, accuracy_score



# Evaluation Function
# ================================

def rmse(model, y_test, y_pred, X_train, y_train):
    r_squared = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return str(r_squared), str(rmse)


# Create model line scatter plot
# ================================

def scatter_plot(y_test, y_pred, model_name):
    plt.figure(figsize=(10,6))
    sns.residplot(y_test, y_pred, lowess=True, color='#4682b4',
              line_kws={'lw': 2, 'color': 'r'})
    plt.title(str('ArrDelay vs Residuals for \n' + model_name), fontsize=16)
    plt.xlabel('ArrDelay',fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig("../reports/figures/%s.png" % model_name)
    plt.show()


# Create model Histogram
# ================================
measure = []

def histogram(model, best_score=""):
    hist = model
    hist.fit(X_train, y_train)
    y_pred = hist.predict(X_test)
    r2, r_mse = rmse(hist, y_test, y_pred, X_train, y_train)
    print(r'R-Squared: %s' % r2)
    print(r'Mean Squared Error: %s' % r_mse)
    print('')
    scatter_plot(y_test, y_pred, r'Histogram-based %s' % model)
    measure.append([model, r2, r_mse, best_score])