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
Filtering methods aiming at removing outliers or groups of outliers from the
point cloud.
"""

# Standard imports
import logging
from typing import Union

# Third party imports
import numpy as np
from scipy.spatial import KDTree

# cars-mesh imports
from ..tools.handlers import PointCloud

# cars v3
# from cars.steps.point_cloud import small_components_filtering, statistical_
# outliers_filtering


def statistical_filtering_outliers_o3d(
    pcd: PointCloud, nb_neighbors: int, std_factor: float
) -> PointCloud:
    """
    This method removes points which have mean distances with their k nearest
    neighbors that are greater than a distance threshold (dist_thresh).

    This threshold is computed from the mean (mean_distances) and
    standard deviation (stddev_distances) of all the points mean distances
    with their k nearest neighbors:

        dist_thresh = mean_distances + std_factor * stddev_distances

    Parameters
    ----------
    pcd: PointCloud
        Point cloud data
    nb_neighbors: int
        Number of neighbors
    std_factor: float
        Multiplication factor to use to compute the distance threshold

    Returns
    -------
    pcd: PointCloud
        Filtered point cloud data
    """

    # Check if open3d point cloud is initialized
    if pcd.o3d_pcd is None:
        pcd.set_o3d_pcd_from_df()

    pcd.o3d_pcd, ind_valid_pts = pcd.o3d_pcd.remove_statistical_outlier(
        nb_neighbors, std_ratio=std_factor
    )

    # Save the number of points before processing
    num_points_before = pcd.df.shape[0]

    # Get the point cloud filtered of the outlier points
    pcd.df = pcd.df.loc[ind_valid_pts]
    # Reset indexes
    pcd.df.reset_index(drop=True, inplace=True)

    # Check if output point cloud is empty (and thus cannot suffer other
    # processing)
    if pcd.df.empty:
        logging.error(
            "Point cloud output by the outlier filtering step is empty."
        )

    logging.info(
        f"{num_points_before - pcd.df.shape[0]} points "
        f"over {num_points_before} points were flagged as outliers "
        f"and removed."
    )

    return pcd


def radius_filtering_outliers_o3(
    pcd: PointCloud, radius: float, nb_points: int
) -> PointCloud:
    """
    This method removes points that have few neighbors in a given sphere
    around them. For each point, it computes the number of neighbors
    contained in a sphere of chosen radius, if this number is lower than
    nb_point, this point is deleted.

    Parameters
    ----------
    pcd: PointCloud
        Point cloud data
    radius: float
        Defines the radius of the sphere that will be used for counting the
        neighbors
    nb_points: int
        Defines the minimum amount of points that the sphere should contain

    Returns
    -------
    pcd: PointCloud
        Filtered point cloud data
    """

    # Check if open3d point cloud is initialized
    if pcd.o3d_pcd is None:
        pcd.set_o3d_pcd_from_df()

    # Apply radius filtering
    pcd.o3d_pcd, ind_valid_pts = pcd.o3d_pcd.remove_radius_outlier(
        nb_points, radius
    )

    # Save the number of points before processing
    num_points_before = pcd.df.shape[0]

    # Get the point cloud filtered of the outlier points
    pcd.df = pcd.df.loc[ind_valid_pts]
    # Reset indexes
    pcd.df.reset_index(drop=True, inplace=True)

    # Check if output point cloud is empty (and thus cannot suffer
    # other processing)
    if pcd.df.empty:
        logging.error(
            "Point cloud output by the outlier filtering step is empty."
        )

    logging.info(
        f"{num_points_before - pcd.df.shape[0]} points over "
        f"{num_points_before} points were flagged as outliers and removed."
    )

    return pcd


def local_density_analysis(
    pcd: PointCloud, nb_neighbors: int, proba_thresh: Union[None, float] = None
) -> PointCloud:
    """
    Compute the probability of a point to be an outlier based on the local
    density.

    Reference: Xiaojuan Ning, Fan Li,  Ge Tian, and Yinghui Wang (2018).
    "An efficient outlier removal method for scattered point cloud data".

    Parameters
    ----------
    pcd: PointCloud
        Point cloud data
    nb_neighbors: int
        Number of neighbors to consider
    proba_thresh: float (default = None)
        Probability threshold of a point to be an outlier. If 'None', then it
        is computed automatically per point as:
        proba_thresh_i = 0.1 * dist_average_i
        with dist_average_i: Average distance of the point i to its
        neighbours

    Returns
    -------
    pcd: PointCloud
        Filtered point cloud data
    """

    # Check that dataframe is initialized
    if pcd.df is None:
        pcd.set_df_from_o3d_pcd()

    # Save the number of points before processing
    num_points_before = pcd.df.shape[0]

    # Get point positions
    cloud_xyz = pcd.df.loc[:, ["x", "y", "z"]].to_numpy()

    # Build the neighbour tree
    cloud_tree = KDTree(cloud_xyz)

    # Init variables
    remove_pts_list = []
    dist_average_list = []
    proba_thresh_list = []

    # Loop over the point cloud
    for idx, _ in enumerate(cloud_xyz):
        # get the point nearest neighbours
        distances, _ = cloud_tree.query(cloud_xyz[idx], nb_neighbors)

        # compute the local density
        mean_neighbors_distances = np.sum(distances) / nb_neighbors
        density = (1 / nb_neighbors) * np.sum(
            np.exp(-distances / mean_neighbors_distances)
        )

        # define the probability of the point to be an outlier
        proba = 1 - density

        if proba_thresh is None:
            dist_average_list.append(proba)
            proba_thresh_i = 0.3 * mean_neighbors_distances
            proba_thresh_list.append(proba_thresh_i)

        else:
            proba_thresh_i = proba_thresh

        if proba > proba_thresh_i:
            remove_pts_list.append(idx)

    pcd.df = pcd.df.drop(index=pcd.df.index.values[remove_pts_list])
    # Reset indexes
    pcd.df.reset_index(drop=True, inplace=True)

    # Check if output point cloud is empty (and thus cannot suffer other
    # processing)
    if pcd.df.empty:
        logging.error(
            "Point cloud output by the outlier filtering step is empty."
        )

    # Update open3d pcd
    if pcd.o3d_pcd is not None:
        pcd.set_o3d_pcd_from_df()

    logging.info(
        f"{num_points_before - pcd.df.shape[0]} points over "
        f"{num_points_before} points were flagged as "
        f"outliers and removed."
    )

    return pcd


###################################
# DEPRECATED (compatible CARS 0.3)
###################################

# def statistical_filtering_outliers_cars(cloud, nb_neighbors, std_factor):
#     """
#     This methode removes points which have mean distances with their k
#     nearest neighbors
#     that are greater than a distance threshold (dist_thresh).
#
#     This threshold is computed from the mean (mean_distances) and
#     standard deviation (stddev_distances) of all the points mean distances
#     with their k nearest neighbors:
#
#         dist_thresh = mean_distances + std_factor * stddev_distances
#
#     :param cloud: cloud point, it should be a pandas DataFrame or a numpy
#     :param nb_neighbors: number of neighbors
#     :param std_factor: multiplication factor to use to compute the distance
#     threshold
#     :return: filtered pandas dataFrame cloud
#     """
#     if not (isinstance(cloud, pd.DataFrame) or isinstance(cloud,
#     np.ndarray)):
#         raise TypeError(f"Cloud is of an unknown type {type(cloud)}.
#         It should either be a pandas DataFrame or a
#         numpy "
#                         f"ndarray.")
#
#     pos,_ = points_cloud.statistical_outliers_filtering(cloud,nb_neighbors,
#     std_factor)
#
#     return pos
#
# def small_components_filtering_outliers_cars(cloud, radius, nb_points):
#     """
#     This method removes small components that have not enough points inside
#     For each point not yet processed, it computes the neighbors contained
#     in a sphere of choosen radius, and the
#     neighbors of neighbors until there are none left around. Those points
#     are considered as processed and the
#     identified cluster is added to a list For each cluster, if the number
#     of points inside is lower than nb_point,
#     this cluster is deleted
#
#     :param cloud: cloud point, it should be a pandas DataFrame or a numpy
#     :param radius: defines the radius of the sphere that will be used for
#     counting the neighbors
#     :param nb_points: defines the minimm amount of points that the sphere
#     should contain
#     :return cloud: filtered pandas dataFrame cloud
#     """
#     if not (isinstance(cloud, pd.DataFrame) or isinstance(cloud,
#     np.ndarray)):
#         raise TypeError(f"Cloud is of an unknown type {type(cloud)}.
#         It should either be a pandas DataFrame or a
#         numpy "
#                         f"ndarray.")
#     pos,_ = points_cloud.small_components_filtering(cloud,radius,nb_points)
#
#     return pos
