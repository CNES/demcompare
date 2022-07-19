"""
Denoising methods aiming at smoothing surfaces without losing genuine high-frequency information.
"""

import multiprocessing as mp
from typing import Callable

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from tqdm import tqdm

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tools import point_cloud_handling


def compute_pcd_normals_o3d(cloud, weights=None):
    if isinstance(cloud, pd.DataFrame):
        data = cloud[["x", "y", "z"]].to_numpy()

    elif isinstance(cloud, np.ndarray):
        data = cloud

        if len(data.shape) != 2:
            raise ValueError(f"Data dimension is incorrect. It should be 2 dimensional. "
                             f"Found {len(data.shape)} dimensions.")
        if data.shape[1] != 3:
            raise ValueError("Data should be expressed as points along the rows and coordinates along the columns.")

    else:
        raise TypeError(f"Cloud is of an unknown type {type(cloud)}. It should either be a pandas DataFrame or a numpy " 
                        f"ndarray.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    # Compute normals
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(100), )
    print(np.asarray(pcd.normals))

    df_pcd = pd.DataFrame(data=np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals)), axis=1),
                          columns=["x", "y", "z", "n_x", "n_y", "n_z"])

    return df_pcd


def compute_point_normal(pcd, weights=None):
    """
    Compute normal with the PCA approach
    The normal to a point on the surface of an object is approximated to the normal to the tangent plane
    defined by the point and its neighbours. It becomes a least square problem.
    See https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html

    The normal vector corresponds to the vector associated with the smallest eigen value of the neighborhood point
    covariance matrix.
    """
    if isinstance(pcd, pd.DataFrame):
        data = pcd.loc[["x", "y", "z"]]

        # Compute the centroid of the nearest neighbours
        centroid = data.mean(axis=0)

        data = data.to_numpy()

    elif isinstance(pcd, np.ndarray):
        data = pcd

        if len(data.shape) != 2:
            raise ValueError(f"Data dimension is incorrect. It should be 2 dimensional. "
                             f"Found {len(data.shape)} dimensions.")
        if data.shape[1] != 3:
            raise ValueError("Data should be expressed as points along the rows and coordinates along the columns.")

        # Compute the centroid of the nearest neighbours
        centroid = np.mean(data, axis=0)

    else:
        raise TypeError(f"Cloud is of an unknown type {type(pcd)}. It should either be a pandas DataFrame or a numpy " 
                        f"ndarray.")

    # Compute the covariance matrix
    cov_mat = np.cov(data - centroid, rowvar=False, aweights=weights)

    # Find eigen values and vectors
    # use the Singular Value Decomposition A = U * S * V^T
    u, s, vT = np.linalg.svd(cov_mat)

    # TODO: find the right orientation for the normal

    return u[:, -1]


def weight_exp(distance: np.ndarray, mean_distance: np.ndarray):
    return np.exp(- distance ** 2 / mean_distance ** 2)

#en entrée une liste (une valeur par voisin) et un chiffre, sortie liste
def weight_exp_2(d, sigma):
    out = [ np.exp(- val ** 2 / 2 * (sigma ** 2)) for val in d]
    return out


def compute_pcd_normals(df_pcd: pd.DataFrame,
                        knn: int = 30,
                        weights_distance: bool = False,
                        weights_color: bool = False,
                        workers: int = 1,
                        use_open3d=False):
    """
    Compute the normal for each point of the cloud
    """

    if use_open3d:
        df_pcd = compute_pcd_normals_o3d(df_pcd)

    else:
        # Init
        tree = KDTree(df_pcd[["x", "y", "z"]].to_numpy())
        weights = None
        results = []

    # Query the knn for each point cloud data
    _, ind = tree.query(df_pcd[["x", "y", "z"]].to_numpy(), k=knn, workers=workers)

    # Loop on each point of the data to compute its normal
    for k, row in tqdm(enumerate(ind)):

        if weights_distance:
            # Weighting of the variance according to the distance to the neighbours
            distance = tree.data[row, :] - tree.data[k, :]
            mean_distance = np.mean(distance)

            weights = weight_exp(distance, mean_distance)

        if weights_color:
            # Weighting of the variance according to the radiometric difference with the neighbours
            color_list = [color for color in ["red", "green", "blue", "nir"] if color in df_pcd]
            if not color_list:
                raise ValueError("Weights on radiometry has been asked but no color channel was found in data. "
                                 "Color channels considered are among ['red', 'green', 'blue', 'nir'].")

            color_data = df_pcd[color_list].to_numpy()
            distance = color_data[row, :] - color_data[0, :]
            mean_distance = np.mean(distance)

            weights = weight_exp(distance, mean_distance) if weights is None \
                else weights * weight_exp(distance, mean_distance)

        # Compute the normal
        results.append(compute_point_normal(tree.data[row, :], weights))

    results = np.asarray(results)

    # Add normals information to the dataframe
    df_pcd = df_pcd.assign(n_x=results[:, 0], n_y=results[:, 1], n_z=results[:, 2])

    return df_pcd


def bilateral_denoising(df: pd.DataFrame,
                        radius: int=5,
                        sigma_d: float=0.5,
                        sigma_n: float=0.5,
                        knn: int = 50,
                        weights_distance: bool = False,
                        weights_color: bool = False,
                        workers: int = 1):
    df_pcd = compute_pcd_normals(df, knn, weights_distance, weights_color, workers)
    # dans la méthode, les voisins sont calculés dans une sphere de rayon fixe, test ici avec knn idem que calul des normales
    normal_cloud = KDTree(df_pcd[["n_x", "n_y", "n_z"]].to_numpy())
    cloud = df_pcd.loc[:, ["x", "y", "z"]].values
    cloud_tree = KDTree(cloud)
    _, ind = cloud_tree.query(df_pcd[["x", "y", "z"]].to_numpy(), k=knn, workers=workers)
    for k, row in tqdm(enumerate(ind)):
        sum_w=0
        delta_p=0
        distance = cloud_tree.data[row, :] - cloud_tree.data[k, :]
        d_d = [np.linalg.norm(i) for i in distance]
        d_n = np.dot(distance, normal_cloud.data[k,:])
        w = np.multiply(weight_exp_2(d_d,sigma_d),weight_exp_2(d_n,sigma_n))
        delta_p = sum(w*d_n)
        sum_w = sum(w)
        p_new=  cloud_tree.data[k,:]+(delta_p/sum_w)*normal_cloud.data[k,:]
        df_pcd.loc[k,'x':'z']=p_new

    return(df_pcd)


def bilateral_denoising_2(df: pd.DataFrame,
                        radius: int=3,
                        sigma_d: float=0.5,
                        sigma_n: float=0.5,
                        knn: int = 10,
                        weights_distance: bool = False,
                        weights_color: bool = False,
                        workers: int = 1):
    normals = compute_pcd_normals_o3d(df)
    normal_cloud = KDTree(normals[["n_x", "n_y", "n_z"]].to_numpy())
    cloud = df.loc[:, ["x", "y", "z"]].values
    cloud_tree = KDTree(cloud)
    for idx, _ in tqdm(enumerate(cloud)):
        neighbors_list = cloud_tree.query_ball_point(cloud[idx], radius)
        distance = cloud_tree.data[neighbors_list, :] - cloud_tree.data[idx, :]
        d_d = [np.linalg.norm(i) for i in distance]
        d_n = np.dot(distance, normal_cloud.data[idx,:])
        w = np.multiply(weight_exp_2(d_d,sigma_d),weight_exp_2(d_n,sigma_n))
        delta_p = sum(w*d_n)
        sum_w = sum(w)
        p_new=  cloud_tree.data[idx,:]+(delta_p/sum_w)*normal_cloud.data[idx,:]
        df.loc[idx,'x':'z']=p_new
        # ~ for neigh_idx in neighbors_list:
    return(df)
            

    
    
# ~ def bilateral_denoising(
    # ~ df_cloud: pandas.DataFrame,
    # ~ iteration: int = 10,
    # ~ k: int = 10,
    # ~ sigma_c: float = 40.0,
    # ~ sigma_d: float = 2.0,
    # ~ sigma_ps: float = 0.5,
# ~ ) -> pandas.DataFrame:
    # ~ """
    # ~ todo
    # ~ """
    # ~ bilateral_logger = logging
    # ~ start_bilat = time.time()
    # ~ print("bilateral_denoising start")
    # ~ # projection.points_cloud_conversion_dataframe(df_cloud,epsg_in,epsg_out)

    # ~ if len(df_cloud) == 0:
        # ~ print("len(df_cloud)==0")
        # ~ print("bilateral_denoising finish")
        # ~ return df_cloud

    # ~ clr_indexes = [
        # ~ idx for idx in df_cloud.columns.values.tolist() if idx.startswith("clr")
    # ~ ]

    # ~ df_xyz = df_cloud[["x", "y", "z"]]
    # ~ df_colors = df_cloud[clr_indexes]

    # ~ # calcul du tree
    # ~ start_tree = time.time()
    # ~ print("bilat: start cKDtree")
    # ~ tree = cKDTree(df_xyz.values)
    # ~ stop_tree = time.time()
    # ~ print(
        # ~ "bilat: stop cKDtree, duration {}s".format(stop_tree - start_tree)
    # ~ )

    # ~ # calcul des normales
    # ~ start_normal = time.time()
    # ~ print("bilat: start compute normal")
    # ~ np_normals = normal_selective(
        # ~ df_xyz, df_colors, sigma_d=sigma_d, sigma_c=sigma_c, k=k, tree=tree
    # ~ )
    # ~ stop_normal = time.time()
    # ~ print(
        # ~ "bilat: stop compute normal, duration {}s".format(
            # ~ stop_normal - start_normal
        # ~ )
    # ~ )

    # ~ nb_group = 20000
    # ~ start_filter = time.time()
    # ~ print("bilat: start for loop")
    # ~ for _ in range(iteration):
        # ~ start_iter = time.time()
        # ~ print(
            # ~ "bilat: start iteration,nb-group {}".format(nb_group)
        # ~ )

        # ~ tmp_z = df_xyz.copy()
        # ~ tmp_normal = np_normals.copy()
        # ~ for i in range(nb_group, len(df_cloud) + nb_group, nb_group):
            # ~ ind = tree.data[i - nb_group : i]
            # ~ _, nn_ind = tree.query(ind, k=(k ** 2))

            # ~ neighbours_xyz = df_xyz.values[nn_ind]
            # ~ neighbours_colors = df_colors.values[nn_ind]
            # ~ neighbours_normals = np_normals[nn_ind]

            # ~ points_xyz = neighbours_xyz[:, 0, :].copy()
            # ~ points_colors = neighbours_colors[:, 0, :].copy()

            # ~ delta_xyz = neighbours_xyz - points_xyz[..., None, :]
            # ~ delta_colors = neighbours_colors - points_colors[..., None, :]

            # ~ # calcul des poids
            # ~ w_d = np.exp(-(delta_xyz ** 2).sum(axis=-1) / (2 * sigma_d ** 2))
            # ~ w_c = np.exp(-(delta_colors ** 2).sum(axis=-1) / (2 * sigma_c ** 2))
            # ~ w_total = w_c * w_d

            # ~ # filtrage des normales
            # ~ points_normals = neighbours_normals.copy()
            # ~ points_normals = (points_normals * w_total[..., None]).sum(axis=-2)
            # ~ points_normals /= w_total[..., None].sum(axis=-2)

            # ~ # Normalisation
            # ~ points_normals /= np.sqrt((points_normals ** 2).sum(axis=1))[
                # ~ :, None
            # ~ ]

            # ~ # calcul des distances par rapport à la normal
            # ~ mean_pos = (delta_xyz * w_total[..., None]).sum(axis=-2)
            # ~ mean_pos /= w_total[..., None].sum(axis=-2)

            # ~ distance_ortho = (
                # ~ (delta_xyz - mean_pos[:, None, :]) * points_normals[:, None, :]
            # ~ ).sum(axis=-1)
            # ~ w_o = np.exp(-np.abs(distance_ortho) ** 2 / (sigma_ps ** 2))
            # ~ w_total *= w_o

            # ~ # calcul du barycentre
            # ~ new_mean_pos = (neighbours_xyz * w_total[..., None]).sum(axis=-2)
            # ~ new_mean_pos /= w_total[..., None].sum(axis=-2)

            # ~ # calcul de la nouvelle position
            # ~ new_pos_z = (
                # ~ points_xyz
                # ~ - (((points_xyz - new_mean_pos) * points_normals).sum(axis=1))[
                    # ~ :, None
                # ~ ]
                # ~ * points_normals
            # ~ )

            # ~ tmp_z.iloc[i - nb_group : i, :] = new_pos_z
            # ~ tmp_normal[i - nb_group : i] = points_normals

            # ~ # break
        # ~ df_xyz = tmp_z[["x", "y", "z"]]
        # ~ np_normals = tmp_normal

        # ~ stop_iter = time.time()
        # ~ print(
            # ~ "bilat: stop iteration,duration {}".format(stop_iter - start_iter)
        # ~ )
    # ~ stop_filter = time.time()
    # ~ print(
        # ~ "bilat: stop for loop, duration {}s".format(stop_filter - start_filter)
    # ~ )

    # ~ df_cloud.loc[:, ["x", "y", "z"]] = df_xyz.values
    # ~ print("bilateral_denoising finish")

    # ~ stop_bilat = time.time()
    # ~ print(
        # ~ "bilateral_denoising stop, duration {}s nb-points {}".format(
            # ~ stop_bilat - start_bilat, len(df_cloud)
        # ~ )
    # ~ )
    # ~ return df_cloud

#
# def normal_selective(
#     df_xyz: pd.DataFrame,
#     df_colors: pd.DataFrame,
#     sigma_c: float = 40.0,
#     sigma_d: float = 2.0,
#     k: int = 10,
#     tree=None,
# ):
#     """
#     Compute normal vectors of cloud dataframe
#     """
#     sigma_c_2 = sigma_c ** 2
#     normals = np.zeros((len(df_xyz), 3))
#
#     np_xyz = df_xyz.values
#     np_colors = df_colors.values
#
#     if tree is None:
#         tree = cKDTree(np_xyz)
#
#     nb_group = 20000
#     for i in range(0, len(df_xyz), nb_group):
#         ind = tree.data[i : i + nb_group, :]
#         _, nn_ind = tree.query(ind, k=(k ** 2))
#
#         neighbours_xyz = np_xyz[nn_ind]
#         neighbours_colors = np_colors[nn_ind]
#
#         points_xyz = neighbours_xyz[:, 0, :]
#         points_colors = neighbours_colors[:, 0, :]
#
#         delta_xyz = neighbours_xyz - points_xyz[..., None, :]
#         delta_colors = neighbours_colors - points_colors[..., None, :]
#
#         # calcul de la ponderation spatiale
#         w_total = np.exp(-(delta_xyz ** 2).sum(axis=-1) / (2 * sigma_d ** 2))
#         # calcul de la ponderation couleurs
#         w_total *= np.exp(-(delta_colors ** 2).sum(axis=-1) / (2 * sigma_c_2))
#
#         eigenvectors, _, _ = sdv_from_neighbor_array(
#             np_xyz, nn_ind, coef=w_total[..., None]
#         )
#         normals[i : i + nb_group, :] = eigenvectors[..., 2]
#     normals *= np.sign(normals[:, 2, None])
#     return normals


def main(df):
    # ~ compute_normal_o3d(df)
    # ~ compute_pcd_normals(df)
    # ~ df_f = bilateral_denoising(df)
    df_f = bilateral_denoising_2(df)
    point_cloud_handling.serialize_point_cloud("/home/data/bil_tlse3.las", df_f)


if __name__ == "__main__":
    fileName ='/home/code/stage/toulouse-points_color.pkl'
    df = pd.read_pickle(fileName)
    fileName2 ='/home/data/radiuso3dpyramidedekmin_04.las'
    # ~ df = pd.read_pickle(fileName)
    df2,_ = point_cloud_handling.las2df(fileName2)
    main(df)
