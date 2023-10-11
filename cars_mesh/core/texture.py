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
Texturing methods to project radiometric information over surfaces to
provide a realistic rendering.
"""

# Standard imports
import logging
import os
import warnings
from typing import Union

# Third party imports
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window

# Cars-mesh imports
from ..tools.handlers import Mesh
from ..tools.point_cloud_io import change_frame
from ..tools.rpc import PleiadesRPC, apply_rpc_list

Image.MAX_IMAGE_PIXELS = 1000000000

warnings.filterwarnings(
    "ignore", category=rasterio.errors.NotGeoreferencedWarning
)


def tile_norm(
    arr: np.ndarray, q_percent: Union[tuple, list, np.ndarray] = None
):
    """
    Normalize an image to a 8-bit image by recomputing the color dynamic

    Parameters
    ----------
    arr: np.ndarray
        Image array
    q_percent: (2, ) list or tuple or np.ndarray
        Percentage of the respectively minimum and maximum values of the image
        from which to clip the dynamic
    """
    if q_percent is None:
        q_percent = [np.amin(arr), np.amax(arr)]
    else:
        arr = np.clip(arr, a_min=q_percent[0], a_max=q_percent[1])

    # Normalize
    a = 255.0 / (q_percent[1] - q_percent[0])
    b = -a * q_percent[0]

    arr = a * arr + b * np.ones_like(arr)
    return arr


def process_raster_by_tile(
    rio_infile,
    rio_outfile,
    fn_process_tile,
    col_off=0,
    row_off=0,
    tile_size=(1000, 1000),
    dst_bands=None,
    **kwargs,
):
    """
    Function that executes a user defined function using a tiling process.

    Parameters
    ----------
    rio_infile: rasterio.io.DatasetReader
    rio_outfile: rasterio.io.DatasetWriter
    fn_process_tile: Function
    col_off: int
    row_off: int
    tile_size: tuple or list or np.ndarray
        (row, col)
    dst_bands: list or tuple or np.ndarray or None
        List of bands to handle in input and output file
    kwargs
    """
    # Take all the bands if none specified
    if dst_bands is None:
        dst_bands = list(range(1, rio_infile.count + 1))

    # Clip tile size to the size of the image itself
    tile_size = np.clip(
        tile_size, a_min=[1, 1], a_max=[rio_outfile.height, rio_outfile.width]
    )

    # Compute the number of tiles in the (x, y) frame (==> (col, row))
    num_tiles_xy = np.asarray(
        [
            np.ceil(rio_outfile.width / tile_size[1]),
            np.ceil(rio_outfile.height / tile_size[0]),
        ],
        dtype=np.int64,
    )

    # Process by tile
    for j in range(num_tiles_xy[0]):  # col
        for i in range(num_tiles_xy[1]):  # row
            # Temporary tile size (row, col)
            tmp_tile_size = [
                min(tile_size[0], rio_outfile.height - i * tile_size[0]),
                min(tile_size[1], rio_outfile.width - j * tile_size[1]),
            ]

            # Read array
            arr = rio_infile.read(
                indexes=dst_bands,
                window=Window(
                    col_off + int(j * tile_size[1]),
                    row_off + int(i * tile_size[0]),
                    int(tmp_tile_size[1]),
                    int(tmp_tile_size[0]),
                ),
            )

            # Apply process
            arr = fn_process_tile(arr, **kwargs)

            # Write array in output file
            rio_outfile.write(
                arr,
                indexes=dst_bands,
                window=Window(
                    int(j * tile_size[1]),
                    int(i * tile_size[0]),
                    int(tmp_tile_size[1]),
                    int(tmp_tile_size[0]),
                ),
            )


def preprocess_image_texture(
    img_path: str,
    bbox: Union[tuple, list, np.ndarray],
    output_dir: str,
    tile_size: Union[tuple, list, np.ndarray] = (1000, 1000),
):
    """
    Function that transforms a 16-bit tif image into an 8-bit jpg cropped to
    a bbox extent

    Parameters
    ----------
    img_path: str
        Path to the TIF image to crop and normalize
    bbox: list or tuple or np.ndarray
        Bounding box extent for cropping: [top_left_col, top_left_row,
        bottom_right_col, bottom_right_row]
    output_dir: str
        Output directory path
    tile_size: (2, ) tuple or list or np.ndarray
        Tile size for the processing applied by tile for memory purpose
        (row, col)
    """

    # Define path of the output image
    output_8bit_path = os.path.join(
        output_dir, "8bit_" + os.path.basename(img_path).split(".")[0] + ".jpg"
    )

    # Get height and width of desired image texture as well as top left
    # corner offset
    x_off = int(bbox[0])
    y_off = int(bbox[1])
    width = int(bbox[2] - bbox[0])  # num rows
    height = int(bbox[3] - bbox[1])  # num cols

    # Open TIF image file
    with rasterio.open(img_path) as infile:
        # Bands handling
        if infile.count > 3:
            logging.debug(
                f"Texture image has more than three bands ({infile.count}). "
                f"Output texture image will be "
                f"generated with the first three bands."
            )

        dst_num_bands = min(infile.count, 3)
        dst_bands = list(range(1, dst_num_bands + 1))

        # Clip colors between 2% and 98%
        # TODO: Make it large scale by computing percentile on chunks
        #  (need reimplementation because numpy does not handle it). Or
        #  another idea: sample 20% of the points of the data and compute the
        #  percentile on it
        raster = infile.read(
            indexes=dst_bands, window=Window(x_off, y_off, width, height)
        )
        q_percent = np.percentile(raster, [2, 98])
        del raster

        # Create output jpg image
        with rasterio.open(
            output_8bit_path,
            mode="w",
            driver="JPEG",
            height=height,
            width=width,
            count=dst_num_bands,
            dtype=rasterio.uint8,
        ) as dst:
            # Apply normalization to the image by tile
            process_raster_by_tile(
                infile,
                dst,
                tile_norm,
                col_off=x_off,
                row_off=y_off,
                q_percent=q_percent,
                tile_size=tile_size,
                dst_bands=dst_bands,
            )

    return output_8bit_path, (width, height)


def generate_uvs(
    img_pts: Union[tuple, list, np.ndarray],
    triangles: Union[tuple, list, np.ndarray],
    bbox: Union[tuple, list, np.ndarray],
    img_texture_size: Union[tuple, list, np.ndarray],
) -> np.ndarray:
    """
    Function that computes the UVs coordinates

    Parameters
    ----------
    img_pts: (N, 2) list or tuple or np.ndarray
        Equivalent image coordinates of ground point cloud data
    triangles: (M, 3) list or tuple or np.ndarray
        Mesh triangle indexes. The indexes must correspond to the 'img_pts'
        indexes.
    bbox: (4, ) list or tuple or np.ndarray
        List of coordinates for respectively the top left and bottom right
        corners in image frame and in the order
        (col, row)
    img_texture_size: (2, ) list or tuple or np.ndarray
        Size (col, row) of the image texture

    Returns
    -------
    uvs: np.ndarray
        Normalized coordinates in the image texture of each triangle vertex.
        Normalization is done according to the image texture size and in [0, 1]
    """

    # Cast arguments to numpy arrays
    img_pts = np.asarray(img_pts)
    triangles = np.asarray(triangles)
    bbox = np.asarray(bbox)
    img_texture_size = np.asarray(img_texture_size)

    # Loop over triangles
    uvs = []
    # Y-axis is inverted
    for i in range(triangles.shape[0]):
        uvs.append(
            [
                (img_pts[triangles[i, 0], 0] - bbox[0]) / img_texture_size[0],
                1
                - (img_pts[triangles[i, 0], 1] - bbox[1]) / img_texture_size[1],
                (img_pts[triangles[i, 1], 0] - bbox[0]) / img_texture_size[0],
                1
                - (img_pts[triangles[i, 1], 1] - bbox[1]) / img_texture_size[1],
                (img_pts[triangles[i, 2], 0] - bbox[0]) / img_texture_size[0],
                1
                - (img_pts[triangles[i, 2], 1] - bbox[1]) / img_texture_size[1],
            ]
        )

    return np.asarray(uvs, dtype=np.float64)


def texturing(mesh: Mesh, cfg: dict) -> Mesh:
    """
    Function that creates the texture of a mesh.

    Parameters
    ----------
    mesh: Mesh
        Mesh object
    cfg: dict:
        Configuration dictionary. The algorithm retrieves the following
        information:

        * output_dir: Str
          Output directory to write the new texture image.

        * rpc_path: Str
          Path to the xml rpc file.

        * tif_img_path: Str
          Path to the TIF image.

        * utm_code: int
          UTM code of the point cloud

    Returns
    -------
    mesh: Mesh
        Mesh object with texture parameters
    """
    # Decode config
    output_dir = cfg["output_dir"]
    rpc_path = cfg["rpc_path"]
    tif_img_path = cfg["tif_img_path"]
    utm_code = cfg["utm_code"]
    image_offset = cfg["image_offset"]  # col, row

    # Compute RPC for inverse location
    rpc = PleiadesRPC(rpc_type="INVERSE", path_rpc=rpc_path)

    # If image_offset is given apply it to the RPC coefficients
    if image_offset is not None:
        if len(image_offset) != 2:
            raise ValueError(
                f"If specified, 'image_offset' should be a tuple or list of "
                f"2 elements (col, row)."
                f"Here: {image_offset}."
            )
        rpc.out_offset = np.asarray(rpc.out_offset)
        # RPC starts image pixel indexing at 1 whereas numpy arrays or QGIS
        # start at 0 Thus the line below removes (1, 1) to each dimension to
        # be consistent
        rpc.out_offset -= np.asarray(image_offset) - np.ones(2)

    # Convert vertices from UTM to geo (lon lat)
    vertices = mesh.pcd.df[["x", "y", "z"]].to_numpy()
    vertices_lon_lat = change_frame(
        pd.DataFrame(vertices, columns=["x", "y", "z"]), utm_code, 4326
    ).to_numpy()
    del vertices

    # Apply inverse location to get the equivalent image point coordinates
    img_pts = apply_rpc_list(rpc, vertices_lon_lat)

    # Process image to crop the TIF file to the extent of the mesh
    # Get the image bounding box rounding up to the nearest integer (to
    # avoid resampling)
    bbox = [
        np.floor(np.min(img_pts[:, 0])),  # left
        np.floor(np.min(img_pts[:, 1])),  # top
        np.ceil(np.max(img_pts[:, 0])),  # right
        np.ceil(np.max(img_pts[:, 1])),  # bottom
    ]

    if np.any(np.asarray(bbox) <= 0.0):
        raise ValueError(
            f"Some value is negative in the bbox top left and bottom right "
            f"corners (in image coordinates: {bbox}. Please check the "
            f"reference frame used (should be in UTM.) or RPCs."
        )

    image_texture_path, img_texture_size = preprocess_image_texture(
        tif_img_path, bbox, output_dir, tile_size=(2000, 2000)
    )

    # Compute UVs
    triangles = mesh.df[["p1", "p2", "p3"]].to_numpy()
    triangles_uvs = generate_uvs(img_pts, triangles, bbox, img_texture_size)
    del triangles

    # Add parameters to serialize the texture
    mesh.set_df_uvs(triangles_uvs)
    mesh.set_image_texture_path(image_texture_path)

    return mesh
