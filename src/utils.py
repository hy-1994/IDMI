import numpy as np
import pandas as pd
import palantir

def get_waypoint_centers(ad, k, dr):
                
    pca_components = pd.DataFrame(ad.obsm[dr]).set_index(ad.obs_names)

    dm_res = palantir.utils.run_diffusion_maps(pca_components, n_components=15)
    dc_components = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)

    # Initialize SEACells via waypoint sampling
    waypoint_init = palantir.core._max_min_sampling(data=dc_components, num_waypoints=k)    # waypoint_init是对应行标签
    dc_components['iix'] = np.arange(len(dc_components))
    waypoint_ix = dc_components.loc[waypoint_init]['iix'].values

    return waypoint_ix

# _max_min_sampling函数
# def _max_min_sampling(data, num_waypoints, seed=None):
#     """Function for max min sampling of waypoints

#     :param data: Data matrix along which to sample the waypoints,
#                  usually diffusion components
#     :param num_waypoints: Number of waypoints to sample
#     :param seed: Random number generator seed to find initial guess.
#     :return: pandas Series reprenting the sampled waypoints
#     """

#     waypoint_set = list()
#     no_iterations = int((num_waypoints) / data.shape[1])
#     if seed is not None:
#         np.random.seed(seed)

#     # Sample along each component
#     N = data.shape[0]
#     for ind in data.columns:
#         # Data vector
#         vec = np.ravel(data[ind])

#         # Random initialzlation
#         iter_set = [
#             np.random.randint(N),
#         ]

#         # Distances along the component
#         dists = np.zeros([N, no_iterations])
#         dists[:, 0] = abs(vec - data[ind].values[iter_set])
#         for k in range(1, no_iterations):
#             # Minimum distances across the current set
#             min_dists = dists[:, 0:k].min(axis=1)

#             # Point with the maximum of the minimum distances is the new waypoint
#             new_wp = np.where(min_dists == min_dists.max())[0][0]
#             iter_set.append(new_wp)

#             # Update distances
#             dists[:, k] = abs(vec - data[ind].values[new_wp])

#         # Update global set
#         waypoint_set = waypoint_set + iter_set

#     # Unique waypoints
#     waypoints = data.index[waypoint_set].unique()

#     return waypoints


# run_diffusion_maps函数
# def run_diffusion_maps(
#     data: Union[pd.DataFrame, sc.AnnData],
#     n_components: int = 10,
#     knn: int = 30,
#     alpha: float = 0,
#     seed: Union[int, None] = 0,
#     pca_key: str = "X_pca",
#     kernel_key: str = "DM_Kernel",
#     sim_key: str = "DM_Similarity",
#     eigval_key: str = "DM_EigenValues",
#     eigvec_key: str = "DM_EigenVectors",
# ):
#     """
#     Run Diffusion maps using the adaptive anisotropic kernel.

#     Parameters
#     ----------
#     data : Union[pd.DataFrame, sc.AnnData]
#         PCA projections of the data or adjacency matrix.
#         If sc.AnnData is passed, its obsm[pca_key] is used and the result is written to
#         its obsp[kernel_key], obsm[eigvec_key], and uns[eigval_key].
#     n_components : int, optional
#         Number of diffusion components. Default is 10.
#     knn : int, optional
#         Number of nearest neighbors for graph construction. Default is 30.
#     alpha : float, optional
#         Normalization parameter for the diffusion operator. Default is 0.
#     seed : Union[int, None], optional
#         Numpy random seed, randomized if None, set to an arbitrary integer for reproducibility.
#         Default is 0.
#     pca_key : str, optional
#         Key to retrieve PCA projections from data if it is a sc.AnnData object. Default is 'X_pca'.
#     kernel_key : str, optional
#         Key to store the kernel in obsp of data if it is a sc.AnnData object. Default is 'DM_Kernel'.
#     sim_key : str, optional
#         Key to store the similarity in obsp of data if it is a sc.AnnData object. Default is 'DM_Similarity'.
#     eigval_key : str, optional
#         Key to store the EigenValues in uns of data if it is a sc.AnnData object. Default is 'DM_EigenValues'.
#     eigvec_key : str, optional
#         Key to store the EigenVectors in obsm of data if it is a sc.AnnData object. Default is 'DM_EigenVectors'.

#     Returns
#     -------
#     dict
#         Diffusion components, corresponding eigen values and the diffusion operator.
#         If sc.AnnData is passed as data, these results are also written to the input object
#         and returned.
#     """

#     if isinstance(data, sc.AnnData):
#         data_df = pd.DataFrame(data.obsm[pca_key], index=data.obs_names)
#     else:
#         data_df = data

#     if not isinstance(data_df, pd.DataFrame) and not issparse(data_df):
#         raise ValueError("'data_df' should be a pd.DataFrame or sc.AnnData")

#     if not issparse(data_df):
#         kernel = compute_kernel(data_df, knn, alpha)
#     else:
#         warn(
#             "'data' is a sparse matrix and will be interpreted as kernel. "
#             "To avoid this warning compute diffusion maps from a precompued kernel using "
#             "palantir.utils.diffusion_maps_from_kernel()."
#         )
#         kernel = data_df

#     res = diffusion_maps_from_kernel(kernel, n_components, seed)

#     res["kernel"] = kernel
#     if not issparse(data_df):
#         res["EigenVectors"].index = data_df.index

#     if isinstance(data, sc.AnnData):
#         data.obsp[kernel_key] = res["kernel"]
#         data.obsp[sim_key] = res["T"]
#         data.obsm[eigvec_key] = res["EigenVectors"].values
#         data.uns[eigval_key] = res["EigenValues"].values

#     return res


# determine_multiscale_space函数
# def determine_multiscale_space(
#     dm_res: Union[dict, sc.AnnData],
#     n_eigs: Union[int, None] = None,
#     eigval_key: str = "DM_EigenValues",
#     eigvec_key: str = "DM_EigenVectors",
#     out_key: str = "DM_EigenVectors_multiscaled",
# ) -> Union[pd.DataFrame, None]:
#     """
#     Determine the multi-scale space of the data.

#     Parameters
#     ----------
#     dm_res : Union[dict, sc.AnnData]
#         Diffusion map results from run_diffusion_maps.
#         If sc.AnnData is passed, its uns[eigval_key] and obsm[eigvec_key] are used.
#     n_eigs : Union[int, None], optional
#         Number of eigen vectors to use. If None is specified, the number
#         of eigen vectors will be determined using the eigen gap. Default is None.
#     eigval_key : str, optional
#         Key to retrieve EigenValues from dm_res if it is a sc.AnnData object. Default is 'DM_EigenValues'.
#     eigvec_key : str, optional
#         Key to retrieve EigenVectors from dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors'.
#     out_key : str, optional
#         Key to store the result in obsm of dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors_multiscaled'.

#     Returns
#     -------
#     Union[pd.DataFrame, None]
#         Multi-scale data matrix. If sc.AnnData is passed as dm_res, the result
#         is written to its obsm[out_key] and None is returned.
#     """
#     if isinstance(dm_res, sc.AnnData):
#         eigenvectors = dm_res.obsm[eigvec_key]
#         if not isinstance(eigenvectors, pd.DataFrame):
#             eigenvectors = pd.DataFrame(eigenvectors, index=dm_res.obs_names)
#         dm_res_dict = {
#             "EigenValues": dm_res.uns[eigval_key],
#             "EigenVectors": eigenvectors,
#         }
#     else:
#         dm_res_dict = dm_res

#     if not isinstance(dm_res_dict, dict):
#         raise ValueError("'dm_res' should be a dict or a sc.AnnData instance")
#     if n_eigs is None:
#         vals = np.ravel(dm_res_dict["EigenValues"])
#         n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
#         if n_eigs < 3:
#             n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 1

#     # Scale the data
#     use_eigs = list(range(1, n_eigs))
#     eig_vals = np.ravel(dm_res_dict["EigenValues"][use_eigs])
#     data = dm_res_dict["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))
#     data = pd.DataFrame(data, index=dm_res_dict["EigenVectors"].index)

#     if isinstance(dm_res, sc.AnnData):
#         dm_res.obsm[out_key] = data.values

#     return data
