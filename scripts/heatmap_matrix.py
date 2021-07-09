# Heatmap matrix of correlations
# ==============================================================================

import matplotlib.pyplot as plt

import seaborn as sns

# ==============================================================================


def Correlations(df):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    corr_matrix = df.select_dtypes(include=['float64', 'int']).corr(method='pearson')
    sns.heatmap(
        corr_matrix,
        annot     = True,
        cbar      = False,
        annot_kws = {"size": 10},
        vmin      = -1,
        vmax      = 1,
        center    = 0,
        cmap      = sns.diverging_palette(20, 220, n=200),
        square    = True,
        ax        = ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 45,
        horizontalalignment = 'right',
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation = 0,
        horizontalalignment = 'right',
    )
    ax.tick_params(labelsize = 10)
    plt.savefig("../reports/figures/heatmap_Matrix_Correlations.png")