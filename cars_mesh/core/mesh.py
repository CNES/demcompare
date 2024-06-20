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
Meshing methods to create a surface from the point cloud.
"""

# Standard imports
import logging
from typing import Union

# Third party imports
import matplotlib.tri as mtri
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

from ..core.denoise_pcd import compute_pcd_normals_o3d
from ..tools.handlers import Mesh, PointCloud


def ball_pivoting_reconstruction(
    pcd: PointCloud,
    radii: Union[list, float, None] = 0.6,
    normal_search_method: str = "knn",
    normal_nb_neighbor: int = 30,
    normal_radius: float = 2.0,
) -> Mesh:
    """
    Bernardini, Fausto et al. “The ball-pivoting algorithm for surface
    reconstruction.” IEEE Transactions on Visualization and Computer
    Graphics 5 (1999): 349-359.

    Function that computes a triangle mesh from a oriented PointCloud.
    This implements the Ball Pivoting algorithm proposed in F. Bernardini et
    al., “The ball-pivoting algorithm for surface reconstruction”, 1999.
    The implementation is also based on the algorithms outlined in Digne,
    “An Analysis and Implementation of a Parallel Ball Pivoting Algorithm”,
    2014. The surface reconstruction is done by rolling a ball with a given
    radius over the point cloud, whenever the ball touches three points
    a triangle is created.

    Radius is computed to be slightly larger than the average distance between
    points https://cs184team.github.io/cs184-final/writeup.html

    Parameters
    ----------
    pcd: PointCloud
        Point cloud object
    radii: Union[list, float, None], default=0.6
        Radius (unique or a list) of the ball.
    normal_search_method: str (default="knn")
        Search method for normal computation
    normal_nb_neighbor: int (defaul=30)
        Number of neighbours used by the KNN algorithm to compute the normals
        with Open3D.
    normal_radius: float (default=2.)
        Radius of search for neighbours for normal computation.

    Returns
    -------
    mesh: Mesh
        Mesh object
    """
    # Convert the point cloud with normals to an open3d format

    # init o3d point cloud
    if pcd.o3d_pcd is None:
        pcd.set_o3d_pcd_from_df()

    # add normals
    if not pcd.has_normals:
        logging.warning(
            "Some normal components are not included in the df point cloud "
            "(either 'n_x', or 'n_y', or 'n_z'). The normal computation "
            "will be run with open3d."
        )

        pcd = compute_pcd_normals_o3d(
            pcd,
            neighbour_search_method=normal_search_method,
            radius=normal_radius,
            knn=normal_nb_neighbor,
        )
        pcd.set_o3d_normals()

    else:
        pcd.o3d_pcd.normals = o3d.utility.Vector3dVector(
            np.ascontiguousarray(pcd.df[["n_x", "n_y", "n_z"]].to_numpy())
        )

    # Mesh point cloud
    if not isinstance(radii, list):
        radii = [radii]
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd.o3d_pcd, o3d.utility.DoubleVector(radii)
    )

    # Init Mesh object
    mesh = Mesh(pcd=pcd.df, o3d_pcd=pcd.o3d_pcd, o3d_mesh=o3d_mesh)
    mesh.set_df_from_o3d_mesh()

    return mesh


def poisson_reconstruction(
    pcd: PointCloud,
    depth: int = 8,
    width: float = 0.8,
    scale: float = 1.1,
    linear_fit: bool = False,
    n_threads: int = -1,
    normal_search_method: str = "knn",
    normal_nb_neighbor: int = 30,
    normal_radius: float = 2.0,
) -> Mesh:
    """
    Kazhdan, Michael M. et al. “Poisson surface reconstruction.” SGP '06 (2006).

    Function that computes a triangle mesh from a oriented PointCloud pcd.
    This implements the Screened Poisson Reconstruction proposed in Kazhdan
    and Hoppe, “Screened Poisson Surface Reconstruction”, 2013. This function
    uses the original implementation by Kazhdan.
    See https://github.com/mkazhdan/PoissonRecon

    Warning: Since the Poisson reconstruction is an optimisation process,
    the triangle vertices are not the initial point cloud points. Thus,
    "pcd" points will be different from "o3d..." and mesh points.

    Parameters
    ----------
    pcd: PointCloud
        Point cloud object
    depth: int (default=8)
        Maximum depth of the tree that will be used for surface reconstruction.
        Running at depth d corresponds to
        solving on a grid whose resolution is no larger than 2^d x 2^d x 2^d.
        Note that since the reconstructor adapts the octree to the sampling
        density, the specified reconstruction depth is only an upper bound.
    width: float (default=0.)
        Specifies the target width of the finest level octree cells.
        This parameter is ignored if depth is specified.
        It is expressed in the point cloud unit (for example in meters for
        utm data).
    scale: float (default=1.1)
        Specifies the ratio between the diameter of the cube used for
        reconstruction and the diameter of the samples’ bounding cube.
    linear_fit: bool (default=False)
        If true, the reconstructor will use linear interpolation to estimate
        the positions of iso-vertices.
    n_threads: int (default=-1)
        Number of threads used for reconstruction. Set to -1 to automatically
        determine it.
    normal_search_method: str (default="knn")
        Search method for normal computation
    normal_nb_neighbor: int (default=30)
        Number of neighbours used by the KNN algorithm to compute the normals
        with Open3D.
    normal_radius: float (default=2.)
        Radius of search for neighbours for normal computation.

    Returns
    -------
    mesh: Mesh
        Mesh object
    """
    # Convert the point cloud with normals to an open3d format

    # init o3d point cloud
    if pcd.o3d_pcd is None:
        pcd.set_o3d_pcd_from_df()

    # add normals
    if not pcd.has_normals:
        logging.warning(
            "Some normal components are not included in the df point cloud "
            "(either 'n_x', or 'n_y', or 'n_z'). The normal computation "
            "will be run with open3d."
        )

        pcd = compute_pcd_normals_o3d(
            pcd,
            neighbour_search_method=normal_search_method,
            radius=normal_radius,
            knn=normal_nb_neighbor,
        )
        pcd.set_o3d_normals()

    else:
        pcd.o3d_pcd.normals = o3d.utility.Vector3dVector(
            np.ascontiguousarray(pcd.df[["n_x", "n_y", "n_z"]].to_numpy())
        )

    # Mesh point cloud
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd.o3d_pcd, depth, width, scale, linear_fit, n_threads
    )

    # Init Mesh object
    # TODO: Check consistency between pcd.df list of points and o3d list of
    #  points Poisson creates new points
    mesh = Mesh(
        pcd=pcd.df,
        o3d_pcd=o3d.geometry.PointCloud(points=o3d_mesh[0].vertices),
        o3d_mesh=o3d_mesh[0],
    )
    mesh.set_df_from_o3d_mesh()

    return mesh


def delaunay_2d_reconstruction(
    pcd: PointCloud, method: str = "matplotlib"
) -> Mesh:
    """
    2.5D Delaunay triangulation: Delaunay triangulation on the planimetric
    points and add afterwards the z coordinates.

    Parameters
    ----------
    pcd: PointCloud
        Point cloud object
    method: str, default='matplotlib'
        Method to use for Delaunay 2.5D triangulation. Available methods are
        'matplotlib' and 'scipy'.

    Returns
    -------
    mesh: Mesh
        Mesh object
    """

    # Delaunay triangulation

    mesh = Mesh(pcd=pcd.df)

    if method == "scipy":
        mesh_data = Delaunay(pcd.df[["x", "y"]].to_numpy())
        mesh.set_df_from_vertices(mesh_data.simplices)
        return mesh

    if method == "matplotlib":
        mesh_data = mtri.Triangulation(pcd.df.x, pcd.df.y)
        mesh.set_df_from_vertices(mesh_data.triangles)
        return mesh

    raise NotImplementedError(
        f"Unknown library for Delaunay triangulation: '{method}'. "
        f"Should either be 'scipy' or 'matplotlib'."
    )
