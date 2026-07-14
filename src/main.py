import sys  
import scanpy as sc
import numpy as np
import pandas as pd
import utils 
import plot 
import core 
import result
import evaluation
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


if __name__=='__main__':
    
    
    # adata = sc.read('/baimoc/hongyan/work/IDMI/data/sample_data.h5ad')  
    adata = sc.read('/baimoc/hongyan/work/imetacell/idata/cd34_multiome_rna.h5ad') 
    sc.pl.scatter(adata, basis='umap', color='celltype', frameon=False)
    
    #preprocessing
    raw_data = sc.AnnData(adata.X)
    raw_data.obs_names, raw_data.var_names = adata.obs_names, adata.var_names
    adata.raw = raw_data

    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1500)

    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    
    #params setting
    adata.obs['metacell'] = 'unknown'
    param = 100   
    waypoint = utils.get_waypoint_centers(adata, param, dr='X_pca') 
    for i in waypoint:
        adata.obs['metacell'][i] = f'metacell_{i}'

    core.metacell_assignments(adata, waypoint)


    plot.plot_2D(adata, key='X_umap', colour_metacells=False, save_as='/baimoc/hongyan/work/IDMI/test/sample_assig.png')
    plot.plot_2D(adata, key='X_umap', colour_metacells=True, save_as='/baimoc/hongyan/work/IDMI/test/sample_assig1.png')

    metacell_ad = result.summarize_by_idmi(adata, idmi_label='metacell', summarize_layer='raw')
    metacell_ad.write('/baimoc/hongyan/work/IDMI/test/sample_metacell.h5ad')
    
#     evaluation.plot_metacell_sizes(adata, bins=5)
    
#     purity = evaluation.metacell_purity(adata, 'celltype')

#     plt.figure(figsize=(4,4))
#     sns.boxplot(data=purity, y='celltype_purity')
#     plt.title('Celltype Purity')
#     sns.despine()
#     plt.savefig('/baimoc/hongyan/work/IDMI/test/purity.png')
#     plt.show()
#     plt.close()
    
#     compactness = evaluation.compactness(adata, 'X_pca')

#     plt.figure(figsize=(4,4))
#     sns.boxplot(data=compactness, y='compactness')
#     plt.title('Compactness')
#     sns.despine()
#     plt.savefig('/baimoc/hongyan/work/IDMI/test/compactness.png')
#     plt.show()
#     plt.close()

#     separation = evaluation.separation(adata, 'X_pca',nth_nbr=1)

#     plt.figure(figsize=(4,4))
#     sns.boxplot(data=separation, y='separation')
#     plt.title('Separation')
#     sns.despine()
#     plt.savefig('/baimoc/hongyan/work/IDMI/test/separation.png')
#     plt.show()
#     plt.close()

#     evaluation.plot_metacell_sizes(adata, bins=5, save_as = '/baimoc/hongyan/work/IDMI/test/metacell_sizes.png')



#     coverage = evaluation.calculate_gene_coverage(adata)

#     fig, ax = plt.subplots(figsize=(4,4))
#     ax.boxplot(nonzero_ratios, vert=True)
#     ax.set_ylabel('Coverage')
#     ax.set_xticklabels(['Non-zero Gene Ratio'])
#     plt.show()
    

