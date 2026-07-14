import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt

sns.set_style('ticks')   
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 600


def plot_2D(ad, key='X_umap',
            colour_metacells=True,
            title='Metacell Assignments',
            save_as=None,
            show=True,
            cmap='Set2',
            figsize=(5, 5),
            SEACell_size=20,
            cell_size=10
            ):

    umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(ad.obs['metacell'])
    umap['metacell'] = umap['metacell'].astype("category")
    mcs = umap.groupby('metacell').mean().reset_index()

    plt.figure(figsize=figsize)
    if colour_metacells:
        sns.scatterplot(x=0, y=1,
                        hue='metacell',
                        data=umap,
                        s=cell_size,
                        cmap=cmap,
                        legend=None)
        sns.scatterplot(x=0, y=1, s=SEACell_size,
                        hue='metacell',
                        data=mcs,
                        cmap=cmap,
                        edgecolor='black', linewidth=1.25,
                        legend=None)
    else:
        sns.scatterplot(x=0, y=1,
                        color='grey',
                        data=umap,
                        s=cell_size,
                        cmap=cmap,
                        legend=None)
        sns.scatterplot(x=0, y=1, s=SEACell_size,
                        color='red',
                        data=mcs,
                        cmap=cmap,
                        edgecolor='black', linewidth=1.25,
                        legend=None)

    plt.xlabel(f'{key}-0')
    plt.ylabel(f'{key}-1')
    plt.title(title)
    ax = plt.gca()
    ax.set_axis_off()

    if save_as is not None:
        plt.savefig(save_as, dpi=150, transparent=True)
    if show:
        plt.show()
    plt.close()