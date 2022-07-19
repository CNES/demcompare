#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Chloe Thenoz (Magellium), Lisa Vo Thanh (Magellium).
#
# This file is part of mesh_3d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Denoising methods aiming at smoothing surfaces without losing genuine high-frequency information.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import multiprocessing as mp
from typing import Callable

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from tqdm import tqdm

from ..tools import point_cloud_io
from ..tools.handlers import PointCloud


def compute_pcd_normals_o3d(pcd: PointCloud,
                            neighbour_search_method: str = "ball",
                            knn: int = 100,
                            radius: float = 5.) -> PointCloud:
    """
    Compute point cloud normals with open3d library

    Parameters
    ----------
    pcd: PointCloud
        Point cloud instance
    neighbour_search_method: str (default="ball")
        Neighbour search method
    knn: int (default=30)
        If "neighbour_search_method" is "knn", number of neighbours to consider
    radius: float (default=5.)
        If "neighbour_search_method" is "ball", ball radius in which to find the neighbours
    """

    if neighbour_search_method not in ["knn", "ball"]:
        raise ValueError(f"Neighbour search method should either be 'knn' or 'ball'. Here found "
                         f"'{neighbour_search_method}'.")

    # Init
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.df[["x", "y", "z"]].to_numpy())

    # Compute normals
    if neighbour_search_method == "knn":
        o3d_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn), )
    elif neighbour_search_method == "ball":
        # Compute normals
        o3d_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius), )
    else:
        raise NotImplementedError

    # Assign it to the df
    pcd.df = pd.concat([pcd.df, pd.DataFrame(data=np.asarray(o3d_pcd.normals), columns=["n_x", "n_y", "n_z"])])

    return pcd


def compute_point_normal(point_coordinates: np.array, weights: float = None) -> np.array:
    """
    Compute unitary normal with the PCA approach from a point and its neighbours
    The normal to a point on the surface of an object is approximated to the normal to the tangent plane
    defined by the point and its neighbours. It becomes a least square problem.
    See https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html

    The normal vector corresponds to the vector associated with the smallest eigen value of the neighborhood point
    covariance matrix.

    Parameters
    ----------
    point_coordinates: np.array
        Point coordinates
    weights: float (default=None)
        Absolute Weights for the covariance matrix (see numpy.cov documentation)

    Returns
    -------
    normal: np.array
        Local normal vector
    """
    print(f"Number of points: {point_coordinates.shape[0]}")

    if point_coordinates.shape[0] <= 1:
        raise ValueError(f"The cluster of points from which to compute the local normal is empty or with just "
                         f"one point. Increase the ball radius.")

    # Compute the centroid of the nearest neighbours
    centroid = np.mean(point_coordinates, axis=0)

    # Compute the covariance matrix
    cov_mat = np.cov(point_coordinates - centroid, rowvar=False, aweights=weights)

    # Find eigen values and vectors
    # use the Singular Value Decomposition A = U * S * V^T
    u, s, vT = np.linalg.svd(cov_mat)

    # TODO: find the right orientation for the normal

    # Extract local normal
    normal = u[:, -1]

    return normal


def weight_exp(distance: np.ndarray, mean_distance: np.ndarray) -> np.array:
    """Decreasing exponential function for weighting"""
    return np.exp(- distance ** 2 / mean_distance ** 2)


# en entrée une liste (une valeur par voisin) et un chiffre, sortie liste
def weight_exp_2(d, sigma):
    out = [ np.exp(- val ** 2 / 2 * (sigma ** 2)) for val in d]
    return out


def compute_pcd_normals(pcd: PointCloud,
                        neighbour_search_method: str = "ball",
                        knn: int = 30,
                        radius: float = 5.,
                        weights_distance: bool = False,
                        weights_color: bool = False,
                        workers: int = 1,
                        use_open3d: bool = False) -> PointCloud:
    """
    Compute the normal for each point of the cloud

    Parameters
    ----------
    pcd: PointCloud
        Point cloud instance
    neighbour_search_method: str (default="ball")
        Neighbour search method
    knn: int (default=30)
        If "neighbour_search_method" is "knn", number of neighbours to consider
    radius: float (default=5.)
        If "neighbour_search_method" is "ball", ball radius in which to find the neighbours
    weights_distance: bool (default=False)
        Whether to add a weighting to the neighbours on the distance information
    weights_color: bool (default=False)
        Whether to add a weighting to the neighbours on the color information
    workers: int (default=1)
        Number of workers to query the KDtree (neighbour search)
    use_open3d: bool (default=False)
        Whether to use open3d normal computation instead. No weighting is applied to neighbours in that case.

    Returns
    -------
    pcd: PointCloud
        Point cloud instance
    """

    if neighbour_search_method not in ["knn", "ball"]:
        raise ValueError(f"Neighbour search method should either be 'knn' or 'ball'. Here found "
                         f"'{neighbour_search_method}'.")

    if use_open3d:
        pcd = compute_pcd_normals_o3d(pcd, neighbour_search_method, knn=knn, radius=radius)

    else:
        # Init
        tree = KDTree(pcd.df[["x", "y", "z"]].to_numpy())
        weights = None
        results = np.zeros_like(pcd.df[["x", "y", "z"]].to_numpy())

        if neighbour_search_method == "knn":
            # Query the tree by knn for each point cloud data
            _, ind = tree.query(pcd.df[["x", "y", "z"]].to_numpy(), k=knn, workers=workers)
        elif neighbour_search_method == "ball":
            raise NotImplementedError("Due to memory consumption, scipy ball query is unusable: "
                                      "https://github.com/scipy/scipy/issues/12956.")
            # # Query the tree by radius for each point cloud data
            # ind = tree.query_ball_point(pcd.df[["x", "y", "z"]].to_numpy(), r=radius, workers=workers,
            #                             return_sorted=False, return_length=False)
        else:
            raise NotImplementedError

        if weights_color:
            # Weighting of the variance according to the radiometric difference with the neighbours
            color_data = pcd.get_colors()

        # Loop on each point of the data to compute its normal
        for k, row in tqdm(enumerate(ind)):

            if weights_distance:
                # Weighting of the variance according to the distance to the neighbours
                distance = tree.data[row, :] - tree.data[k, :]
                mean_distance = np.mean(distance)

                weights = weight_exp(distance, mean_distance)

            if weights_color:
                distance = color_data[row, :] - color_data[0, :]
                mean_distance = np.mean(distance)

                weights = weight_exp(distance, mean_distance) if weights is None \
                    else weights * weight_exp(distance, mean_distance)

            # Compute the normal
            results[k, :] = compute_point_normal(tree.data[row, :], weights)

        # results = np.asarray(results)

        # Add normals information to the dataframe
        pcd.df = pcd.df.assign(n_x=results[:, 0], n_y=results[:, 1], n_z=results[:, 2])

    return pcd


def bilateral_filtering(pcd: PointCloud,
                        neighbour_search_method: str = "ball",
                        knn: int = 30,
                        radius: float = 5.,
                        sigma_d: float = 0.5,
                        sigma_n: float = 0.5,
                        neighbour_search_method_normals: str = "ball",
                        knn_normals: int = 50,
                        radius_normals: float = 5.,
                        weights_distance: bool = False,
                        weights_color: bool = False,
                        workers: int = 1,
                        use_open3d: bool = False):
    """
    Bilateral denoising

    Parameters
    ----------
    pcd: PointCloud
        Point cloud instance
    neighbour_search_method: str (default="ball")
        Neighbour search method
    knn: int (default=30)
        If "neighbour_search_method" is "knn", number of neighbours to consider
    radius: float (default=5.)
        If "neighbour_search_method" is "ball", ball radius in which to find the neighbours
    sigma_d: float (default=0.5)
        Variance on the distance between a point and its neighbours
    sigma_n: float (default=0.5)
        Variance on the normal difference between the ones of a point and the ones of its neighbours
    neighbour_search_method_normals: str (default="ball")
        Neighbour search method to compute the normals at each point
    knn_normals: int (default=30)
        If "neighbour_search_method_normals" is "knn", number of neighbours to consider
    radius_normals: float (default=5.)
        If "neighbour_search_method_normals" is "ball", ball radius in which to find the neighbours
    weights_distance: bool (default=False)
        Whether to add a weighting to the neighbours on the distance information
    weights_color: bool (default=False)
        Whether to add a weighting to the neighbours on the color information
    workers: int (default=1)
        Number of workers to query the KDtree (neighbour search)
    use_open3d: bool (default=False)
        Whether to use open3d normal computation instead. No weighting is applied to neighbours in that case.

    Returns
    -------
    pcd: PointCloud
        Point cloud instance
    """

    # Compute normals
    pcd = compute_pcd_normals(pcd, neighbour_search_method_normals, knn=knn_normals, radius=radius_normals,
                              weights_distance=weights_distance, weights_color=weights_color, workers=workers,
                              use_open3d=use_open3d)

    # Build the KDTree for the normals
    normal_cloud = KDTree(pcd.df[["n_x", "n_y", "n_z"]].to_numpy())

    # Get point coordinates
    cloud = pcd.df.loc[:, ["x", "y", "z"]].values
    # Build the KDTree for the points
    cloud_tree = KDTree(cloud)

    # Request the indexes of the neighbours according to the spatial coordinates
    if neighbour_search_method == "knn":
        # Query the tree by knn for each point cloud data
        _, ind = cloud_tree.query(pcd.df[["x", "y", "z"]].to_numpy(), k=knn, workers=workers)
    elif neighbour_search_method == "ball":
        raise NotImplementedError("Due to memory consumption, scipy ball query is unusable: "
                                  "https://github.com/scipy/scipy/issues/12956.")
        # # Query the tree by radius for each point cloud data
        # ind = cloud_tree.query_ball_point(pcd.df[["x", "y", "z"]].to_numpy(), r=radius, workers=workers,
        #                                   return_sorted=False, return_length=False)
    else:
        raise NotImplementedError

    # Iterate over the points
    for k, row in tqdm(enumerate(ind)):
        # Euclidean distance from the point to its neighbors
        # The bigger it is, the lesser the weighting is
        distance = cloud_tree.data[row, :] - cloud_tree.data[k, :]
        d_d = [np.linalg.norm(i) for i in distance]

        # Cosinus between the normal of the point and the ones of its neighbors
        # The bigger it is, the lesser the weighting is
        d_n = np.dot(distance, normal_cloud.data[k, :])

        # Compute weighting of each neighbor according to
        # - its distance from the point
        # - its normal orientation
        w = np.multiply(weight_exp_2(d_d, sigma_d), weight_exp_2(d_n, sigma_n))
        delta_p = sum(w * d_n)
        sum_w = sum(w)

        # Change points' position along its normal as: p_new = p + w * n
        p_new = cloud_tree.data[k, :] + (delta_p / sum_w) * normal_cloud.data[k, :]
        pcd.df.loc[k, "x":"z"] = p_new

    return pcd


# def bilateral_denoising_radius(pcd: PointCloud,
#                                radius: float = 5,
#                                sigma_d: float = 0.5,
#                                sigma_n: float = 0.5,
#                                neighbour_search_method_normals: str = "ball",
#                                knn_normals: int = 50,
#                                radius_normals: float = 5.,
#                                weights_distance: bool = False,
#                                weights_color: bool = False,
#                                workers: int = 1,
#                                use_open3d: bool = False):
#     """
#     Bilateral denoising where neighbours are determined according to a ball of a user defined radius around the point.
#
#     Parameters
#     ----------
#     pcd: PointCloud
#         Point cloud instance
#     radius: float (default=5.)
#         If "neighbour_search_method" is "ball", ball radius in which to find the neighbours
#     sigma_d: float (default=0.5)
#         Variance on the distance between a point and its neighbours
#     sigma_n: float (default=0.5)
#         Variance on the normal difference between the ones of a point and the ones of its neighbours
#     neighbour_search_method_normals: str (default="ball")
#         Neighbour search method to compute the normals at each point
#     knn_normals: int (default=30)
#         If "neighbour_search_method_normals" is "knn", number of neighbours to consider
#     radius_normals: float (default=5.)
#         If "neighbour_search_method_normals" is "ball", ball radius in which to find the neighbours
#     weights_distance: bool (default=False)
#         Whether to add a weighting to the neighbours on the distance information
#     weights_color: bool (default=False)
#         Whether to add a weighting to the neighbours on the color information
#     workers: int (default=1)
#         Number of workers to query the KDtree (neighbour search)
#     use_open3d: bool (default=False)
#         Whether to use open3d normal computation instead. No weighting is applied to neighbours in that case.
#
#     Returns
#     -------
#     pcd: PointCloud
#         Point cloud instance
#     """
#
#     # Compute normals
#     pcd = compute_pcd_normals(pcd, neighbour_search_method_normals, knn=knn_normals, radius=radius_normals,
#                               weights_distance=weights_distance, weights_color=weights_color, workers=workers,
#                               use_open3d=use_open3d)
#
#     # Get point coordinates
#     cloud = pcd.df.loc[:, ["x", "y", "z"]].values
#     # Build the KDTree for the points
#     cloud_tree = KDTree(cloud)
#
#     # Iterate over the points
#     for idx, _ in tqdm(enumerate(cloud)):
#         neighbors_list = cloud_tree.query_ball_point(cloud[idx], radius)
#         distance = cloud_tree.data[neighbors_list, :] - cloud_tree.data[idx, :]
#         d_d = [np.linalg.norm(i) for i in distance]
#         d_n = np.dot(distance, normals[idx])
#         w = np.multiply(weight_exp_2(d_d, sigma_d), weight_exp_2(d_n, sigma_n))
#         delta_p = sum(w * d_n)
#         sum_w = sum(w)
#         p_new = cloud_tree.data[idx, :] + (delta_p / sum_w) * normals[idx]
#         df.loc[idx, 'x':'z'] = p_new
#         # ~ for neigh_idx in neighbors_list:
#
#     return df
            

    
    
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


# def main(df):
#     # ~ compute_normal_o3d(df)
#     # ~ compute_pcd_normals(df)
#     # ~ df_f = bilateral_denoising_knn(df)
#     df_f = bilateral_denoising_radius(df)
#     point_cloud_io.serialize_point_cloud("/home/data/bil_tlse2.las", df_f)
#
#
# if __name__ == "__main__":
#     fileName ='/home/code/stage/toulouse-points_color.pkl'
#     df = pd.read_pickle(fileName)
#     fileName2 ='/home/data/radiuso3dpyramidedekmin_04.las'
#     # ~ df = pd.read_pickle(fileName)
#     df2,_ = point_cloud_io.las2df(fileName2)
#     main(df)
