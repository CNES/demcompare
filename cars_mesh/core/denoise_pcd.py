#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2023 CNES.
#
# This file is part of cars-mesh

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
Denoising methods aiming at smoothing surfaces without losing genuine
high-frequency information.
"""
# pylint: disable=unsubscriptable-object


# Third party imports
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm

# Cars-mesh imports
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
        If "neighbour_search_method" is "ball", ball radius in which to find
        the neighbours
    """

    if neighbour_search_method not in ["knn", "ball"]:
        raise ValueError(
            f"Neighbour search method should either be 'knn' or 'ball'. "
            f"Here found '{neighbour_search_method}'."
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
    Compute unitary normal with the PCA approach from a point and its
    neighbours. The normal to a point on the surface of an object is
    approximated to the normal to the tangent plane
    defined by the point and its neighbours. It becomes a least square problem.
    See https://pcl.readthedocs.io/projects/tutorials/en/latest/
    normal_estimation.html

    The normal vector corresponds to the vector associated with the smallest
    eigen value of the neighborhood point covariance matrix.

    Parameters
    ----------
    point_coordinates: np.array
        Point coordinates
    weights: float (default=None)
        Absolute Weights for the covariance matrix (see numpy.cov
        documentation)

    Returns
    -------
    normal: np.array
        Local normal vector
    """

    if point_coordinates.shape[0] <= 1:
        raise ValueError(
            "The cluster of points from which to compute the local normal "
            "is empty or with just one point. Increase the ball radius."
        )

    # Compute the covariance matrix
    # The biased estimator is used for computation purposes (otherwise
    # we get a 0. value denominator for many cases)
    cov_mat = np.cov(point_coordinates, rowvar=False, aweights=weights, ddof=0)

    # Find eigen values and vectors
    # use the Singular Value Decomposition A = U * S * V^T
    u_arr, _, _ = np.linalg.svd(cov_mat)

    # TODO: find the right orientation for the normal

    # Extract local normal
    normal = u_arr[:, -1]

    return normal


def weight_exp(distance: np.ndarray, mean_distance: np.ndarray) -> np.ndarray:
    """Decreasing exponential function for weighting"""
    if np.any(mean_distance == 0.0):
        raise ValueError("Mean distance should be > 0.")
    return np.exp(-(distance**2) / mean_distance**2)


def weight_gaussian(distance: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Decreasing function inspired by the gaussian function for weighting"""
    if sigma == 0.0:
        raise ValueError("Sigma should be > 0.")
    return np.exp(-(np.asarray(distance) ** 2) / (2 * (sigma**2)))


def compute_pcd_normals(
    pcd: PointCloud,
    neighbour_search_method: str = "knn",
    knn: int = 30,
    radius: float = 5.0,
    weights_distance: bool = False,
    sigma_d: float = 0.5,
    weights_color: bool = False,
    sigma_c: float = 125.0,
    workers: int = 1,
    use_open3d: bool = True,
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
        If "neighbour_search_method" is "ball", ball radius in which to find
        the neighbours
    weights_distance: bool (default=False)
        Whether to add a weighting to the neighbours on the
        distance information
        It is only available if 'use_open3d' is False
    sigma_d: float (default=0.5)
        If 'weights_distance' is True, it is the standard deviation over the
        spatial distance to use in the exponential weighting function
    weights_color: bool (default=False)
        Whether to add a weighting to the neighbours on the color information
        It is only available if 'use_open3d' is False
    sigma_c: float (default=125.)
        If 'weights_color' is True, it is the standard deviation over the
        color distance to use in the exponential weighting function
    workers: int (default=1)
        Number of workers to query the KDtree (neighbour search)
    use_open3d: bool (default=False)
        Whether to use open3d normal computation instead. No weighting is
        applied to neighbours in that case.

    Returns
    -------
    pcd: PointCloud
        Point cloud instance
    """

    if neighbour_search_method not in ["knn", "ball"]:
        raise ValueError(
            f"Neighbour search method should either be 'knn' or 'ball'. "
            f"Here found '{neighbour_search_method}'."
        )

    if use_open3d:
        pcd = compute_pcd_normals_o3d(
            pcd, neighbour_search_method, knn=knn, radius=radius
        )

    else:
        # Init
        tree = KDTree(pcd.df[["x", "y", "z"]].to_numpy())
        results = np.zeros_like(pcd.df[["x", "y", "z"]].to_numpy())

        if neighbour_search_method == "knn":
            # Query the tree by knn for each point cloud data
            # ind is of shape (num_points, num_neighbours)
            _, ind = tree.query(
                pcd.df[["x", "y", "z"]].to_numpy(), k=knn, workers=workers
            )
        elif neighbour_search_method == "ball":
            raise NotImplementedError(
                "Due to memory consumption, scipy ball query is unusable: "
                "https://github.com/scipy/scipy/issues/12956."
            )
            # # Query the tree by radius for each point cloud data
            # ind = tree.query_ball_point(pcd.df[["x", "y", "z"]].to_numpy(),
            # r=radius, workers=workers, return_sorted=False,
            # return_length=False)
        else:
            raise NotImplementedError

        weights = None
        if weights_distance:
            # Weighting of the variance according to the distance to
            # the neighbours
            # Fetch points' nearest neighbours coordinates
            data_tmp = tree.data[ind]
            # Duplicate points' coordinates for vector operation
            data_duplicated = np.reshape(
                np.repeat(tree.data, data_tmp.shape[1], 0), data_tmp.shape
            )
            # Compute distance for each point of the cloud to each of its
            # neighbours
            distance = data_tmp - data_duplicated
            del data_tmp
            del data_duplicated
            distance = np.linalg.norm(distance, axis=-1)
            # Deduce the associated weights
            weights = weight_exp(distance, sigma_d)

        if weights_color:
            # Weighting of the variance according to the radiometric
            # difference with the neighbours
            # Fetch the points' colors
            color_data = pcd.get_colors().to_numpy()
            # Fetch points' nearest neighbours colors
            color_tmp = color_data[ind]
            # Duplicate points' colors for vector operation
            color_duplicated = np.reshape(
                np.repeat(color_data, color_tmp.shape[1], 0), color_tmp.shape
            )
            # Compute distance for each point of the cloud to each of its
            # neighbours' colors
            distance = color_tmp - color_duplicated
            del color_tmp
            del color_duplicated
            distance = np.linalg.norm(distance, axis=-1)
            # Deduce the associated weights
            weights = (
                weight_exp(distance, sigma_c)
                if weights is None
                else weights * weight_exp(distance, sigma_c)
            )

        # Loop on each point of the data to compute its normal
        for k, row in tqdm(
            enumerate(ind), desc="Normal computation by PCA per point"
        ):
            weights_k = weights if weights is None else weights[k]
            results[k, :] = compute_point_normal(tree.data[row, :], weights_k)

        # Add normals information to the dataframe
        pcd.df = pcd.df.assign(
            n_x=results[:, 0], n_y=results[:, 1], n_z=results[:, 2]
        )

    return pcd


def bilateral_filtering(
    pcd: PointCloud,
    num_iterations: int,
    neighbour_kdtree_dict: dict = None,
    sigma_d: float = 0.5,
    sigma_n: float = 0.5,
    neighbour_normals_dict: dict = None,
    num_chunks: int = 1,
):
    """
    Bilateral denoising

    Parameters
    ----------
    pcd: PointCloud
        Point cloud instance
    num_iterations: int
        Number of times to apply bilateral filtering in an iterative fashion
    neighbour_kdtree_dict: dict (default=None)
        Dictionary to define the kdtree options to encode the distance between
        points:

        * neighbour_search_method: str (default="knn").
          Neighbour search method

        * knn: int (default=30).
          If "neighbour_search_method" is "knn", number of neighbours to
          consider

        * radius: float (default=5.).
          If "neighbour_search_method" is "ball", ball radius in which to
          find the neighbours

        * num_workers_kdtree: int (default=1).
          Number of workers to query the KDtree (neighbour search)

    sigma_d: float (default=0.5)
        Variance on the distance between a point and its neighbours
    sigma_n: float (default=0.5)
        Variance on the normal difference between the ones of a point and
        the ones of its neighbours
    neighbour_normals_dict: dict (default=None)
        Dictionary to define the configuration to compute normals according
        to the ones of the neighbours

        * neighbour_search_method_normals: str (default="knn").
          Neighbour search method to compute the normals at each point

        * knn_normals: int (default=30).
          If "neighbour_search_method_normals" is "knn", number of neighbours
          to consider

        * radius_normals: float (default=5.).
          If "neighbour_search_method_normals" is "ball", ball radius in
          which to find the neighbours

        * weights_distance: bool (default=False).
          Whether to add a weighting to the neighbours on the distance
          information
          Only available if 'use_open3d' is False

        * weights_color: bool (default=False).
          Whether to add a weighting to the neighbours on the color
          information. Only available if 'use_open3d' is False

        * use_open3d: bool (default=False).
          Whether to use open3d normal computation instead. No weighting is
          applied to neighbours in that case.

    num_chunks: int (default=1)
        Number of chunks to apply bilateral processing (to fit in memory since
        it is optimized as vectorial calculus).

    Returns
    -------
    pcd: PointCloud
        Point cloud instance
    """
    ################################################################
    # Define default parameters and update them with the user inputs
    ################################################################

    # Defaults
    neighbour_kdtree_dict_default = {
        "neighbour_search_method": "knn",
        "knn": 30,
        "radius": 5.0,
        "num_workers_kdtree": -1,
    }

    neighbour_normals_dict_default = {
        "neighbour_search_method_normals": "knn",
        "knn_normals": 30,
        "radius_normals": 5.0,
        "weights_distance": True,
        "weights_color": True,
        "use_open3d": False,
    }

    # Updates
    if neighbour_kdtree_dict is None:
        neighbour_kdtree_dict = neighbour_kdtree_dict_default
    elif not isinstance(neighbour_kdtree_dict, dict):
        raise TypeError(
            f"'neighbour_kdtree_dict' should be a dict or `None`,"
            f" however found a"
            f" {isinstance(neighbour_kdtree_dict, dict)}."
        )
    else:
        neighbour_kdtree_dict = {
            key: neighbour_kdtree_dict.get(key, val)
            for key, val in neighbour_kdtree_dict_default.items()
        }
    del neighbour_kdtree_dict_default

    if neighbour_normals_dict is None:
        neighbour_normals_dict = neighbour_normals_dict_default
    elif not isinstance(neighbour_normals_dict, dict):
        raise TypeError(
            f"'neighbour_normals_dict' should be a dict or `None`,"
            f" however found a"
            f" {isinstance(neighbour_normals_dict, dict)}."
        )
    else:
        neighbour_normals_dict = {
            key: neighbour_normals_dict.get(key, val)
            for key, val in neighbour_normals_dict_default.items()
        }
    del neighbour_normals_dict_default

    #################
    # Compute normals
    #################

    pcd = compute_pcd_normals(
        pcd,
        neighbour_normals_dict["neighbour_search_method_normals"],
        knn=neighbour_normals_dict["knn_normals"],
        radius=neighbour_normals_dict["radius_normals"],
        weights_distance=neighbour_normals_dict["weights_distance"],
        weights_color=neighbour_normals_dict["weights_color"],
        workers=neighbour_kdtree_dict["num_workers_kdtree"],
        use_open3d=neighbour_normals_dict["use_open3d"],
    )

    # Make sure normals are unitary, otherwise normalize them
    if not pcd.are_normals_unitary:
        pcd.set_unitary_normals()

    normal_cloud = KDTree(pcd.df[["n_x", "n_y", "n_z"]].to_numpy())

    ###########################
    # Apply bilateral filtering
    ###########################
    # It is an iterative process (the filter is applied the number of times
    # specified by the user.
    # Computations are done in the vector space, but for memory consumption
    # considerations, the point cloud is cut in chunks (the number of chunks
    # being a user defined parameter).

    # Number of iterations per chunk
    iter_per_chunk = pcd.df.shape[0] / num_chunks
    # After rounding, how many iterations are left (if it is not an integer)
    delta_iter = pcd.df.shape[0] - np.around(iter_per_chunk) * (num_chunks - 1)
    # Compute the number of points processed by chunk
    num_points_per_chunk = [np.around(iter_per_chunk)] * (num_chunks - 1) + [
        delta_iter
    ]
    # Get the corresponding point indexes
    indexes_per_chunk = [0]
    for k in num_points_per_chunk:
        indexes_per_chunk += [int(indexes_per_chunk[-1]) + int(k)]

    # Bilateral filter
    def apply(idx_start: int, idx_end: int) -> None:
        """
        Compute the weights to apply to the normal vectors

        Parameters
        ----------
        idx_start: int
            Index of the first point of the list to process
        idx_end: int
            Index of the last point of the list to process
        """

        if idx_end <= idx_start:
            raise ValueError(
                f"Start index ({idx_start}) should be lower than "
                f"end index ({idx_end})."
            )

        # Request the indexes of the neighbours according to the
        # spatial coordinates
        if neighbour_kdtree_dict["neighbour_search_method"] == "knn":
            # Query the tree by knn for each point cloud data
            _, ind = cloud_tree.query(
                pcd.df.loc[idx_start : idx_end - 1, "x":"z"].to_numpy(),
                k=neighbour_kdtree_dict["knn"],
                workers=neighbour_kdtree_dict["num_workers_kdtree"],
            )

        elif neighbour_kdtree_dict["neighbour_search_method"] == "ball":
            raise NotImplementedError(
                "Due to memory consumption, scipy ball query is unusable: "
                "https://github.com/scipy/scipy/issues/12956. Morevover, code "
                "should be adapted because it is meant "
                "for a fixed number of neighbours."
            )
            # # Query the tree by radius for each point cloud data
            # ind = cloud_tree.query_ball_point(
            # pcd.df[["x", "y", "z"]].to_numpy(), r=radius, workers=workers,
            # return_sorted=False, return_length=False)
        else:
            raise NotImplementedError

        # Euclidean distance from the point to its neighbors
        # The bigger it is, the lesser the weighting is
        distances = cloud_tree.data[ind, :] - np.repeat(
            cloud_tree.data[idx_start:idx_end, None, :],
            repeats=neighbour_kdtree_dict["knn"],
            axis=1,
        )
        d_d = np.linalg.norm(distances, axis=-1)

        # Cosinus between the normal of the point and the ones of its neighbors
        # The bigger it is, the lesser the weighting is
        d_n = np.sum(
            np.multiply(
                distances,
                np.repeat(
                    normal_cloud.data[idx_start:idx_end, None, :],
                    neighbour_kdtree_dict["knn"],
                    axis=1,
                ),
            ),
            axis=-1,
        )
        del distances

        # Compute weighting of each neighbor according to
        # - its distance from the point
        # - its normal orientation
        weights = np.multiply(
            weight_gaussian(d_d, sigma_d), weight_gaussian(d_n, sigma_n)
        )
        delta_p = np.sum(weights * d_n, axis=1)
        sum_w = np.sum(weights, axis=1)

        del weights

        # Compute weights and apply to normal vectors (w * n)
        coeff = np.where(sum_w == 0.0, 0.0, delta_p / sum_w)
        w_n = (
            np.reshape(coeff, (-1, 1)) * normal_cloud.data[idx_start:idx_end, :]
        )

        # Change points' position along its normal as: p_new = p + w * n
        pcd.df.loc[idx_start : idx_end - 1, "x":"z"] = (
            cloud_tree.data[idx_start:idx_end, :] + w_n
        )

    # Iterations
    for _ in tqdm(
        range(num_iterations), position=0, leave=False, desc="Iterations"
    ):
        # Compute the KDTree at each iteration
        # Because by changing the point coordinates, you can change the k
        # nearest neighbours
        cloud_tree = KDTree(
            pcd.df.loc[:, ["x", "y", "z"]].to_numpy(), copy_data=True
        )

        # Number of workers for iterations should be adapted according to the
        # point cloud size, the number of knn and
        # the memory available
        for k in tqdm(
            range(num_chunks),
            position=1,
            leave=False,
            desc="Chunks per iteration",
        ):
            apply(indexes_per_chunk[k], indexes_per_chunk[k + 1])

    return pcd
