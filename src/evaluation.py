import numpy as np
import pandas as pd
import palantir
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 600


def compactness(ad, low_dim_embedding="X_pca", idmi_label='metacell'):
    """Compute compactness of each metacell.

    Compactness is defined is the average variance of diffusion components across cells that constitute a metcell.

    :param ad: (Anndata) Anndata object
    :param low_dim_embedding: (str) `ad.obsm` field for constructing diffusion components
    :param SEACell_label: (str) `ad.obs` field for computing diffusion component variances

    :return: `pd.DataFrame` with a dataframe of compactness per metacell

    """
    components = pd.DataFrame(ad.obsm[low_dim_embedding]).set_index(ad.obs_names)
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)

    return pd.DataFrame(
        dc.join(ad.obs[idmi_label]).groupby(idmi_label).var().mean(1)
    ).rename(columns={0: "compactness"})


def separation(ad, low_dim_embedding="X_pca", nth_nbr=1, cluster=None, idmi_label='metacell'):
    """Compute separation of each metacell.

    Separation is defined is the distance to the nearest neighboring metacell.

    :param ad: (Anndata) Anndata object
    :param low_dim_embedding: (str) `ad.obsm` field for constructing diffusion components
    :param nth_nbr: (int) Which neighbor to use for computing separation
    :param SEACell_label: (str) `ad.obs` field for computing diffusion component variances

    :return: `pd.DataFrame` with a separation of compactness per metacell

    """
    components = pd.DataFrame(ad.obsm[low_dim_embedding]).set_index(ad.obs_names)
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)

    # Compute DC per metacell
    metacells_dcs = (
        dc.join(ad.obs[idmi_label], how="inner").groupby(idmi_label).mean()
    )

    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(n_neighbors=nth_nbr)
    nbrs = neigh.fit(metacells_dcs)
    dists, nbrs = nbrs.kneighbors()
    dists = pd.DataFrame(dists).set_index(metacells_dcs.index)
    dists.columns += 1

    nbr_cells = np.array(metacells_dcs.index)[nbrs]

    metacells_nbrs = pd.DataFrame(nbr_cells)
    metacells_nbrs.index = metacells_dcs.index
    metacells_nbrs.columns += 1

    if cluster is not None:
        # Get cluster type of neighbors to ensure they match the metacell cluster
        clusters = ad.obs.groupby(idmi_label)[cluster].agg(
            lambda x: x.value_counts().index[0]
        )
        nbr_clusters = pd.DataFrame(clusters.values[nbrs]).set_index(clusters.index)
        nbr_clusters.columns = metacells_nbrs.columns
        nbr_clusters = nbr_clusters.join(pd.DataFrame(clusters))

        clusters_match = nbr_clusters.eq(nbr_clusters[cluster], axis=0)
        return pd.DataFrame(dists[nth_nbr][clusters_match[nth_nbr]]).rename(
            columns={1: "separation"}
        )
    else:
        return pd.DataFrame(dists[nth_nbr]).rename(columns={1: "separation"})
    

def celltype_frac(x, col_name):
    val_counts = x[col_name].value_counts()
    return val_counts.values[0] / val_counts.values.sum()


def metacell_purity(ad, col_name):
    """Compute the purity (prevalence of most abundant value) of the specified col_name from ad.obs within each metacell.

    @param: ad - AnnData object with SEACell assignment and col_name in ad.obs dataframe
    @param: col_name - (str) column name within ad.obs representing celltype groupings for each cell.
    """
    celltype_fraction = ad.obs.groupby("metacell").apply(
        lambda x: celltype_frac(x, col_name)
    )
    celltype = ad.obs.groupby("metacell").apply(
        lambda x: x[col_name].value_counts().index[0]
    )

    return pd.concat([celltype, celltype_fraction], axis=1).rename(
        columns={0: col_name, 1: f"{col_name}_purity"}
    )

def plot_metacell_sizes(ad,
                        save_as=None,
                        show = True,
                        title='Distribution of Metacell Sizes',
                        bins = None,
                        figsize=(5,5)):

    """
    Plot distribution of number of cells contained per metacell.
    :param ad: annData containing 'Metacells' label in .obs
    :param save_as: (str) path to which figure is saved. If None, figure is not saved.
    :param title: (str) title of figure.
    :param bins: (int) number of bins for histogram
    :param figsize: (int,int) tuple of integers representing figure size
    :return: None
    """

    
    assert 'metacell' in ad.obs, 'AnnData must contain "metacell" in obs DataFrame.'
    label_df = ad.obs[['metacell']].reset_index()
    plt.figure(figsize=figsize)
    sns.distplot(label_df.groupby('metacell').count().iloc[:, 0], bins=bins)
    sns.despine()
    plt.xlabel('Number of Cells per metacell')
    plt.title(title)
    
    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()
    return pd.DataFrame(label_df.groupby('metacell').count().iloc[:, 0]).rename(columns={'index':'size'})


def calculate_gene_coverage(ad):
    
    """
    Plot gene coverage per metacell.
    :param ad: annData 
    """
    data = ad.X.toarry()
    nonzero_ratios = np.sum(data > 0, axis=1) / data.shape[1]
    return nonzero_ratios

# def plot_gene_coverage_distribution(nonzero_ratios):
#     # 绘制盒图
#     fig, ax = plt.subplots()
#     ax.boxplot(nonzero_ratios, vert=True)
    
#     # # 绘制中位数的白点
#     # ax.scatter(np.median(nonzero_ratios), 1, color='white', marker='o', s=100, edgecolors='black', zorder=3)
    
#     ax.set_ylabel('Coverage')
#     ax.set_xticklabels(['Non-zero Gene Ratio'])
#     plt.show()

# # random generate matrix
# np.random.seed(42)
# # gene_matrix = np.random.randint(0, 2, size=(10, 1000))  # 随机生成0和1的矩阵
# gene_matrix = np.array([[0, 0, 0, 1, 2],
#                        [1, 2, 3, 3, 4]])
# # compute gene coverage
# nonzero_ratios = calculate_gene_coverage(gene_matrix)
# # plot 
# plot_gene_coverage_distribution(nonzero_ratios)


