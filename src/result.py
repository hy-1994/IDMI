import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scanpy as sc
from tqdm import tqdm



def summarize_by_idmi(ad, idmi_label='metacell', celltype_label=None, summarize_layer='raw'):

    # Set of metacells
    metacells = ad.obs[idmi_label].unique()

    # Summary matrix
    summ_matrix = pd.DataFrame(0.0, index=metacells, columns=ad.var_names)

    for m in tqdm(summ_matrix.index):
        cells = ad.obs_names[ad.obs[idmi_label] == m]
        if summarize_layer == 'X':
            summ_matrix.loc[m, :] = np.ravel(ad[cells, :].X.sum(axis=0))
        elif summarize_layer == 'raw' and ad.raw != None:
            summ_matrix.loc[m, :] = np.ravel(ad[cells, :].raw.X.sum(axis=0))
        else:
            summ_matrix.loc[m, :] = np.ravel(ad[cells, :].layers[summarize_layer].sum(axis=0))

    # Ann data

    # Counts
    meta_ad = sc.AnnData(csr_matrix(summ_matrix), dtype=csr_matrix(summ_matrix).dtype)
    meta_ad.obs_names, meta_ad.var_names = summ_matrix.index.astype(str), ad.var_names
    meta_ad.layers['raw'] = csr_matrix(summ_matrix)

    # Also compute cell type purity
    if celltype_label is not None:
        try:
            purity_df = evaluate.compute_celltype_purity(ad, celltype_label)
            meta_ad.obs = meta_ad.obs.join(purity_df)
        except Exception as e:
            print(f'Cell type purity failed with Exception {e}')

    return meta_ad