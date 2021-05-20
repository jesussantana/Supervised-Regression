def main():
    # Data treatment
# ==============================================================================
    import json
    import numpy as np
    import pandas as pd
    from pandas import json_normalize
    from datetime import datetime
    from tabulate import tabulate

# # Graphics
# ==============================================================================
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib import style
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import geopandas as gpd
    import cartopy.crs as ccrs
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    from PIL import Image
    from IPython.display import Image

# Preprocessing and modeling
# ==============================================================================
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.formula.api import ols

    from scipy import stats
    from scipy.stats import ttest_1samp,ttest_ind
    from scipy.stats import normaltest
    from scipy.stats import f_oneway
    from scipy.stats.mstats import gmean,hmean

    from imblearn.over_sampling import SMOTE

    from sklearn import metrics
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RepeatedKFold
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_blobs
    from sklearn.metrics import euclidean_distances
    from sklearn.ensemble import RandomForestClassifier

    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence

# Paralllel Processing
# ==============================================================================
    import multiprocessing
    from joblib import Parallel, delayed
    import numba
    from numba import jit

# Various
# ==============================================================================
    import time
    import random as rd
    from itertools import product
    from fitter import Fitter, get_common_distributions
    from device_detector import SoftwareDetector

# Pandas configuration
# ==============================================================================
    pd.set_option('display.max_columns', None)

# Matplotlib configuration
# ==============================================================================
    plt.rcParams['image.cmap'] = "bwr"
    plt.rcParams['figure.dpi'] = "100"
    plt.rcParams['savefig.bbox'] = "tight"
    style.use('ggplot') or plt.style.use('ggplot')
        #%matplotlib inline

# Seaborn configuration
# ==============================================================================
    sns.set_theme(style='darkgrid', palette='deep')
    dims = (20, 16)

# Warnings configuration
# ==============================================================================
    import warnings
    warnings.filterwarnings('ignore')

# Folder configuration
# ==============================================================================
    from os import path
    import sys
    new_path = '../scripts/'
    if new_path not in sys.path:
        sys.path.append(new_path)

if __name__ == "__main__":
    main()

