# Libraries for Data Science Project
# ==============================================================================

def main():

    

# Data treatment
# ==============================================================================
    import numpy as np
    import pandas as pd
    from tabulate import tabulate

# # Graphics
# ==============================================================================
    import matplotlib.pyplot as plt
    from matplotlib import style
    import matplotlib.ticker as ticker
    import seaborn as sns

# Preprocessing and modeling
# ==============================================================================
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RepeatedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_blobs
    from sklearn.metrics import euclidean_distances
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Ridge

    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt import gp_minimize
    from skopt.plots import plot_convergence

# Various
# ==============================================================================
    import multiprocessing
    import random
    from itertools import product
    from fitter import Fitter, get_common_distributions

# Matplotlib configuration
# ==============================================================================
    plt.rcParams['image.cmap'] = "bwr"
    #plt.rcParams['figure.dpi'] = "100"
    plt.rcParams['savefig.bbox'] = "tight"
    style.use('ggplot') or plt.style.use('ggplot')

# Warnings configuration
# ==============================================================================
    import warnings
    warnings.filterwarnings('ignore')


if __name__ == "__main__":
    main()