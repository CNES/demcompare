"""
Meshing methods to create a surface from the point cloud.
"""

from typing import Union

import pandas as pd
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.tri as mtri

from mesh3d_lib.tools.point_cloud_handling import df2o3d


def ball_pivoting_reconstruction(df_pcd: pd.DataFrame, radii: Union[list, float, None] = 0.6) -> dict:
    """
    Bernardini, Fausto et al. “The ball-pivoting algorithm for surface reconstruction.” IEEE Transactions on
    Visualization and Computer Graphics 5 (1999): 349-359.

    Function that computes a triangle mesh from a oriented PointCloud. This implements the Ball Pivoting algorithm
    proposed in F. Bernardini et al., “The ball-pivoting algorithm for surface reconstruction”, 1999.
    The implementation is also based on the algorithms outlined in Digne, “An Analysis and Implementation of a
    Parallel Ball Pivoting Algorithm”, 2014. The surface reconstruction is done by rolling a ball with a given
    radius over the point cloud, whenever the ball touches three points a triangle is created.

    Radius is computed to be slightly larger than the average distance between points
    https://cs184team.github.io/cs184-final/writeup.html
    """
    # Convert the point cloud with normals to an open3d format

    # init o3d point cloud
    o3d_pcd = df2o3d(df_pcd)

    # add normals
    if ("n_x" not in df_pcd) or ("n_y" not in df_pcd) or ("n_z" not in df_pcd):
        raise ValueError("Some normal components are not included in the df point cloud (either 'n_x', or 'n_y', "
                         "or 'n_z'). Please launch again the normal computation.")
    else:
        o3d_pcd.normals = o3d.utility.Vector3dVector(df_pcd[["n_x", "n_y", "n_z"]].to_numpy())

    # Mesh point cloud
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3d_pcd, o3d.utility.DoubleVector(radii))

    return {"pcd": df_pcd, "mesh": np.asarray(o3d_mesh.triangles), "o3d_pcd": o3d_pcd, "o3d_mesh": o3d_mesh}


def poisson_reconstruction(df_pcd: pd.DataFrame, depth=8, width=0, scale=1.1, linear_fit=False, n_threads=-1) -> dict:
    """
    Kazhdan, Michael M. et al. “Poisson surface reconstruction.” SGP '06 (2006).

    Function that computes a triangle mesh from a oriented PointCloud pcd. This implements the Screened Poisson
    Reconstruction proposed in Kazhdan and Hoppe, “Screened Poisson Surface Reconstruction”, 2013. This function
    uses the original implementation by Kazhdan.
    See https://github.com/mkazhdan/PoissonRecon
    """
    # Convert the point cloud with normals to an open3d format

    # init o3d point cloud
    o3d_pcd = df2o3d(df_pcd)

    # add normals
    if ("n_x" not in df_pcd) or ("n_y" not in df_pcd) or ("n_z" not in df_pcd):
        raise ValueError("Some normal components are not included in the df point cloud (either 'n_x', or 'n_y', "
                         "or 'n_z'). Please launch again the normal computation.")
    else:
        o3d_pcd.normals = o3d.utility.Vector3dVector(df_pcd[["n_x", "n_y", "n_z"]].to_numpy())

    # Mesh point cloud
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcd,
                                                                         depth,
                                                                         width,
                                                                         scale,
                                                                         linear_fit,
                                                                         n_threads)

    return {"pcd": df_pcd, "mesh": np.asarray(o3d_mesh[0].triangles), "o3d_pcd": o3d_pcd, "o3d_mesh": o3d_mesh}


def delaunay_2d_reconstruction(df_pcd: pd.DataFrame, method="matplotlib") -> dict:
    """
    2.5D Delaunay triangulation: Delaunay triangulation on the planimetric points and add afterwards the z coordinates.
    """

    # Delaunay triangulation

    if method == "scipy":
        mesh = Delaunay(df_pcd[["x", "y"]].to_numpy())
        return {"pcd": df_pcd, "mesh": mesh.simplices}

    elif method == "matplotlib":
        mesh = mtri.Triangulation(df_pcd.x, df_pcd.y)
        return {"pcd": df_pcd, "mesh": mesh.triangles}

    else:
        raise NotImplementedError(f"Unknown library for Delaunay triangulation: '{method}'. Should either be 'scipy' "
                                  f"or 'matplotlib'.")


