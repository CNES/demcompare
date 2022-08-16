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
Evaluation metrics
"""

import numpy as np
import open3d as o3d

from ..core.denoise_pcd import compute_pcd_normals
from ..tools.handlers import PointCloud


def mean_squared_distance(dist: np.array) -> float:
    return np.mean(dist ** 2).item()


def root_mean_squared_distance(dist: np.array) -> float:
    return np.linalg.norm(dist).item()


def mean_distance(dist: np.array) -> float:
    return np.mean(dist).item()


def median_distance(dist: np.array) -> float:
    return np.median(dist).item()


def hausdorff_asym_distance(dist: np.array) -> float:
    return np.amax(dist).item()


def hausdorff_sym_distance(dist_1: np.array, dist_2: np.array) -> float:
    return max(hausdorff_asym_distance(dist_1), hausdorff_asym_distance(dist_2))


def chamfer_distance(dist_1: np.array, dist_2: np.array) -> float:
    return np.mean(np.power(dist_1, 2)).item() + np.mean(np.power(dist_2, 2)).item()


def point_to_plane_distance(pcd_in, pcd_ref, knn=30, workers=1, use_open3d=True, **kwargs) -> np.ndarray:
    """
    The point-to-plane distance first computes the normal of the surface at every point in the reference point
    cloud as an indication of the local surface. The displacement of every corresponding point in the noisy
    point cloud is then projected onto the normal to calculate the point-to-plane distance.

    Source: Lang Zhou, Guoxing Sun, Yong Li, Weiqing Li, Zhiyong Su, Point cloud denoising review:
    from classical to deep learning-based approaches, Graphical Models, Volume 121, 2022, 101140, ISSN 1524-0703,
    https://doi.org/10.1016/j.gmod.2022.101140.
    """

    # Compute normals on the reference point cloud
    if not pcd_ref.has_normals:
        pcd_ref = compute_pcd_normals(pcd_ref, neighbour_search_method="knn", knn=knn, workers=workers,
                                      use_open3d=use_open3d, **kwargs)

    # Get indexes of the NN in the input point cloud from the reference point cloud
    pcd_in_indexes = []
    pcd_in_tree = o3d.geometry.KDTreeFlann(pcd_in.o3d_pcd)

    for i in range(pcd_ref.df.shape[0]):
        [_, idx, _] = pcd_in_tree.search_knn_vector_3d(pcd_ref.o3d_pcd.points[i], 1)
        pcd_in_indexes.append(idx)

    pcd_in_indexes = np.asarray(pcd_in_indexes)[:, 0]

    # Project each distance onto the local normal and retrieve the projected distance
    # Let:
    # * M be the point to project on the D line
    # * A be a point on D
    # * u be a direction vector of D (not necessarily unitary)
    # * P be the orthogonal projection of M
    #
    #     A             P
    # ----+=====>-------+------------------- D
    #         u         |
    #                   |
    #                   + M
    #
    # vect_AP = ( (vect_AM ∙ u) / ||u||² ) ∙ u = dist * u / ||u||

    vect_AM = pcd_in.df.loc[pcd_in_indexes, ["x", "y", "z"]].to_numpy() - pcd_ref.df.loc[:, ["x", "y", "z"]].to_numpy()
    # u is the normal vector that was computed before
    u = pcd_ref.df.loc[:, ["n_x", "n_y", "n_z"]].to_numpy()

    dist_point_to_plane = np.sum(vect_AM * u, axis=1)
    dist_point_to_plane /= np.linalg.norm(u, axis=1)

    return np.abs(dist_point_to_plane)


class PointCloudMetrics(object):
    """
    Compute metrics between two points clouds
    Nearest neighbours are computed during the initialisation step in order to share this information between
    different metrics and avoid a costly recomputing step.
    """
    def __init__(self, pcd_in: PointCloud, pcd_ref: PointCloud, **kwargs):
        self.pcd_in = pcd_in
        self.pcd_ref = pcd_ref

        self.modes = ["p2p", "p2s"]

        # Compute nearest neighbours for all the points with open3d
        # check if open3d pcd are initialized
        if self.pcd_in.o3d_pcd is None:
            self.pcd_in.set_o3d_pcd_from_df()
        if self.pcd_ref.o3d_pcd is None:
            self.pcd_ref.set_o3d_pcd_from_df()

        # Compute distance per point from nearest neighbours in -> ref
        self.dist_p2p_in_ref = np.asarray(self.pcd_in.o3d_pcd.compute_point_cloud_distance(self.pcd_ref.o3d_pcd))
        # Compute distance per point from nearest neighbours ref -> in
        self.dist_p2p_ref_in = np.asarray(self.pcd_ref.o3d_pcd.compute_point_cloud_distance(self.pcd_in.o3d_pcd))
        # Init distance point to plane
        # in -> ref
        self.dist_p2s_in_ref = point_to_plane_distance(self.pcd_ref, self.pcd_in, **kwargs)
        # ref -> in
        self.dist_p2s_ref_in = point_to_plane_distance(self.pcd_in, self.pcd_ref, **kwargs)

    def mean_squared_distance(self, mode) -> tuple:
        """
        Mean squared distance (or error, MSE)

        Parameters
        ----------
        mode: str
            Mode of computation. Either 'p2p' (point to point) or 'p2s' (point to surface):
        """
        if mode == "p2p":
            return mean_squared_distance(self.dist_p2p_in_ref), mean_squared_distance(self.dist_p2p_ref_in)
        elif mode == "p2s":
            return mean_squared_distance(self.dist_p2s_in_ref), mean_squared_distance(self.dist_p2s_ref_in)
        else:
            raise ValueError(f"Mode should be in '{self.modes}'. Here found '{mode}'.")

    def root_mean_squared_distance(self, mode) -> tuple:
        """
        Root mean squared distance (or error, RMSE)

        Parameters
        ----------
        mode: str
            Mode of computation. Either 'p2p' (point to point) or 'p2s' (point to surface):
        """
        if mode == "p2p":
            return root_mean_squared_distance(self.dist_p2p_in_ref), root_mean_squared_distance(self.dist_p2p_ref_in)
        elif mode == "p2s":
            return root_mean_squared_distance(self.dist_p2s_in_ref), root_mean_squared_distance(self.dist_p2s_ref_in)
        else:
            raise ValueError(f"Mode should be in '{self.modes}'. Here found '{mode}'.")

    def mean_distance(self, mode) -> tuple:
        """
        Mean distance (or error)

        Parameters
        ----------
        mode: str
            Mode of computation. Either 'p2p' (point to point) or 'p2s' (point to surface):
        """
        if mode == "p2p":
            return mean_distance(self.dist_p2p_in_ref), mean_distance(self.dist_p2p_ref_in)
        elif mode == "p2s":
            return mean_distance(self.dist_p2s_in_ref), mean_distance(self.dist_p2s_ref_in)
        else:
            raise ValueError(f"Mode should be in '{self.modes}'. Here found '{mode}'.")

    def median_distance(self, mode) -> tuple:
        """
        Median distance (or error)

        Parameters
        ----------
        mode: str
            Mode of computation. Either 'p2p' (point to point) or 'p2s' (point to surface):
        """
        if mode == "p2p":
            return median_distance(self.dist_p2p_in_ref), median_distance(self.dist_p2p_ref_in)
        elif mode == "p2s":
            return median_distance(self.dist_p2s_in_ref), median_distance(self.dist_p2s_ref_in)
        else:
            raise ValueError(f"Mode should be in '{self.modes}'. Here found '{mode}'.")

    def hausdorff_asym_distance(self, mode) -> tuple:
        """
        Hausdorff asymmetric distance

        Parameters
        ----------
        mode: str
            Mode of computation. Either 'p2p' (point to point) or 'p2s' (point to surface):
        """
        if mode == "p2p":
            return hausdorff_asym_distance(self.dist_p2p_in_ref), hausdorff_asym_distance(self.dist_p2p_ref_in)
        elif mode == "p2s":
            return hausdorff_asym_distance(self.dist_p2s_in_ref), hausdorff_asym_distance(self.dist_p2s_ref_in)
        else:
            raise ValueError(f"Mode should be in '{self.modes}'. Here found '{mode}'.")

    def hausdorff_sym_distance(self, mode) -> float:
        """
        Hausdorff symmetric distance

        Parameters
        ----------
        mode: str
            Mode of computation. Either 'p2p' (point to point) or 'p2s' (point to surface):
        """
        if mode == "p2p":
            return hausdorff_sym_distance(self.dist_p2p_in_ref, self.dist_p2p_ref_in)
        elif mode == "p2s":
            return hausdorff_sym_distance(self.dist_p2s_in_ref, self.dist_p2s_ref_in)
        else:
            raise ValueError(f"Mode should be in '{self.modes}'. Here found '{mode}'.")

    def chamfer_distance(self, mode) -> float:
        """
        Chamfer distance

        Parameters
        ----------
        mode: str
            Mode of computation. Either 'p2p' (point to point) or 'p2s' (point to surface):
        """
        if mode == "p2p":
            return chamfer_distance(self.dist_p2p_in_ref, self.dist_p2p_ref_in)
        elif mode == "p2s":
            return chamfer_distance(self.dist_p2s_in_ref, self.dist_p2s_ref_in)
        else:
            raise ValueError(f"Mode should be in '{self.modes}'. Here found '{mode}'.")
