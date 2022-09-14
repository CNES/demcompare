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

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm

from ..tools.handlers import PointCloud


def compute_pcd_normals_o3d(
    pcd: PointCloud,
    neighbour_search_method: str = "knn",
    knn: int = 100,
    radius: float = 5.0,
) -> PointCloud:
    """
    Compute point cloud normals with open3d library

    Parameters
    ----------
    pcd: PointCloud
        Point cloud instance
    neighbour_search_method: str (default="knn")
        Neighbour search method
    knn: int (default=30)
        If "neighbour_search_method" is "knn", number of neighbours to consider
    radius: float (default=5.)
        If "neighbour_search_method" is "ball", ball radius in which to find the neighbours
    """

    if neighbour_search_method not in ["knn", "ball"]:
        raise ValueError(
            f"Neighbour search method should either be 'knn' or 'ball'. Here found "
            f"'{neighbour_search_method}'."
        )

    # Init
    if pcd.o3d_pcd is None:
        pcd.set_o3d_pcd_from_df()

    # Compute normals
    if neighbour_search_method == "knn":
        # Nearest neighbour search
        pcd.o3d_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamKNN(knn),
        )
    elif neighbour_search_method == "ball":
        # Ball search
        pcd.o3d_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamRadius(radius),
        )
    else:
        raise NotImplementedError

    # Assign it to the df
    normals = np.asarray(pcd.o3d_pcd.normals)
    pcd.df = pcd.df.assign(
        n_x=normals[:, 0], n_y=normals[:, 1], n_z=normals[:, 2]
    )

    return pcd


def compute_point_normal(
    point_coordinates: np.array, weights: float = None
) -> np.array:
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

    if point_coordinates.shape[0] <= 1:
        raise ValueError(
            f"The cluster of points from which to compute the local normal is empty or with just "
            f"one point. Increase the ball radius."
        )

    # Compute the centroid of the nearest neighbours
    centroid = np.mean(point_coordinates, axis=0)

    # Compute the covariance matrix
    cov_mat = np.cov(
        point_coordinates - centroid, rowvar=False, aweights=weights
    )

    # Find eigen values and vectors
    # use the Singular Value Decomposition A = U * S * V^T
    u, s, vT = np.linalg.svd(cov_mat)

    # TODO: find the right orientation for the normal

    # Extract local normal
    normal = u[:, -1]

    return normal


def weight_exp(distance: np.ndarray, mean_distance: np.ndarray) -> np.array:
    """Decreasing exponential function for weighting"""
    return np.exp(-(distance**2) / mean_distance**2)


# en entrée une liste (une valeur par voisin) et un chiffre, sortie liste
def weight_exp_2(d, sigma):
    out = [np.exp(-(val**2) / 2 * (sigma**2)) for val in d]
    return out


def compute_pcd_normals(
    pcd: PointCloud,
    neighbour_search_method: str = "knn",
    knn: int = 30,
    radius: float = 5.0,
    weights_distance: bool = False,
    weights_color: bool = False,
    workers: int = 1,
    use_open3d: bool = False,
) -> PointCloud:
    """
    Compute the normal for each point of the cloud

    Parameters
    ----------
    pcd: PointCloud
        Point cloud instance
    neighbour_search_method: str (default="knn")
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
        raise ValueError(
            f"Neighbour search method should either be 'knn' or 'ball'. Here found "
            f"'{neighbour_search_method}'."
        )

    if use_open3d:
        pcd = compute_pcd_normals_o3d(
            pcd, neighbour_search_method, knn=knn, radius=radius
        )

    else:
        # Init
        tree = KDTree(pcd.df[["x", "y", "z"]].to_numpy())
        weights = None
        results = np.zeros_like(pcd.df[["x", "y", "z"]].to_numpy())

        if neighbour_search_method == "knn":
            # Query the tree by knn for each point cloud data
            _, ind = tree.query(
                pcd.df[["x", "y", "z"]].to_numpy(), k=knn, workers=workers
            )
        elif neighbour_search_method == "ball":
            raise NotImplementedError(
                "Due to memory consumption, scipy ball query is unusable: "
                "https://github.com/scipy/scipy/issues/12956."
            )
            # # Query the tree by radius for each point cloud data
            # ind = tree.query_ball_point(pcd.df[["x", "y", "z"]].to_numpy(), r=radius, workers=workers,
            #                             return_sorted=False, return_length=False)
        else:
            raise NotImplementedError

        if weights_color:
            # Weighting of the variance according to the radiometric difference with the neighbours
            color_data = pcd.get_colors()

        # Loop on each point of the data to compute its normal
        for k, row in tqdm(enumerate(ind), desc="Normal computation by PCA per point"):

            if weights_distance:
                # Weighting of the variance according to the distance to the neighbours
                distance = tree.data[row, :] - tree.data[k, :]
                mean_distance = np.mean(distance)

                weights = weight_exp(distance, mean_distance)

            if weights_color:
                distance = color_data[row, :] - color_data[0, :]
                mean_distance = np.mean(distance)

                weights = (
                    weight_exp(distance, mean_distance)
                    if weights is None
                    else weights * weight_exp(distance, mean_distance)
                )

            # Compute the normal
            results[k, :] = compute_point_normal(tree.data[row, :], weights)

        # results = np.asarray(results)

        # Add normals information to the dataframe
        pcd.df = pcd.df.assign(
            n_x=results[:, 0], n_y=results[:, 1], n_z=results[:, 2]
        )

    return pcd


def bilateral_filtering(
    pcd: PointCloud,
    neighbour_search_method: str = "knn",
    knn: int = 30,
    radius: float = 5.0,
    sigma_d: float = 0.5,
    sigma_n: float = 0.5,
    neighbour_search_method_normals: str = "knn",
    knn_normals: int = 50,
    radius_normals: float = 5.0,
    weights_distance: bool = False,
    weights_color: bool = False,
    workers: int = 1,
    use_open3d: bool = False,
):
    """
    Bilateral denoising

    Parameters
    ----------
    pcd: PointCloud
        Point cloud instance
    neighbour_search_method: str (default="r")
        Neighbour search method
    knn: int (default=30)
        If "neighbour_search_method" is "knn", number of neighbours to consider
    radius: float (default=5.)
        If "neighbour_search_method" is "ball", ball radius in which to find the neighbours
    sigma_d: float (default=0.5)
        Variance on the distance between a point and its neighbours
    sigma_n: float (default=0.5)
        Variance on the normal difference between the ones of a point and the ones of its neighbours
    neighbour_search_method_normals: str (default="knn")
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
    pcd = compute_pcd_normals(
        pcd,
        neighbour_search_method_normals,
        knn=knn_normals,
        radius=radius_normals,
        weights_distance=weights_distance,
        weights_color=weights_color,
        workers=workers,
        use_open3d=use_open3d,
    )

    # Build the KDTree for the normals
    normal_cloud = KDTree(pcd.df[["n_x", "n_y", "n_z"]].to_numpy())

    # Get point coordinates
    cloud = pcd.df.loc[:, ["x", "y", "z"]].values
    # Build the KDTree for the points
    cloud_tree = KDTree(cloud)

    # Request the indexes of the neighbours according to the spatial coordinates
    if neighbour_search_method == "knn":
        # Query the tree by knn for each point cloud data
        _, ind = cloud_tree.query(
            pcd.df[["x", "y", "z"]].to_numpy(), k=knn, workers=workers
        )
    elif neighbour_search_method == "ball":
        raise NotImplementedError(
            "Due to memory consumption, scipy ball query is unusable: "
            "https://github.com/scipy/scipy/issues/12956."
        )
        # # Query the tree by radius for each point cloud data
        # ind = cloud_tree.query_ball_point(pcd.df[["x", "y", "z"]].to_numpy(), r=radius, workers=workers,
        #                                   return_sorted=False, return_length=False)
    else:
        raise NotImplementedError

    # Iterate over the points
    for k, row in tqdm(enumerate(ind), desc="Bilateral filtering per point"):
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
        p_new = (
            cloud_tree.data[k, :] + (delta_p / sum_w) * normal_cloud.data[k, :]
        )
        pcd.df.loc[k, "x":"z"] = p_new

    return pcd
