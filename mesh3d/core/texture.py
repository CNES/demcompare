#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 CNES.
#
# This file is part of mesh3d
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
Texturing methods to project radiometric information over surfaces to provide a realistic rendering.
"""

import os
from typing import Union

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window

from ..tools.handlers import Mesh
from ..tools.point_cloud_io import change_frame
from ..tools.rpc import PleiadesRPC, apply_rpc_list

Image.MAX_IMAGE_PIXELS = 1000000000


def preprocess_image_texture(
    img_path: str,
    bbox: Union[tuple, list, np.ndarray],
    output_dir: str,
    tile_size: Union[tuple, list, np.ndarray] = (1000, 1000),
):
    """
    Function that transforms a 16-bit tif image into an 8-bit jpg cropped to a bbox extent

    Parameters
    ----------
    img_path: str
        Path to the TIF image to crop and normalize
    bbox: list or tuple or np.ndarray
        Bounding box extent for cropping
    output_dir: str
        Output directory path
    tile_size: (2, ) tuple or list or np.ndarray
        Tile size for the processing applied by tile for memory purpose
    """

    # Define path of the output image
    output_8bit_path = os.path.join(
        output_dir, "8bit_" + os.path.basename(img_path).split(".")[0] + ".jpg"
    )

    # Define normalization function to be applied by tile (for memory consumption reasons, tiling is mandatory)
    def tile_norm(
        arr: np.ndarray, q_percent: Union[tuple, list, np.ndarray], **kwargs
    ):
        """
        Normalize an image to a 8-bit image by recomputing the color dynamic

        Parameters
        ----------
        arr: np.ndarray
            Image array
        q_percent: (2, ) list or tuple or np.ndarray
            Percentage of the respectively minimum and maximum values of the image from which to clip the dynamic
        """
        arr = np.clip(arr, a_min=q_percent[0], a_max=q_percent[1])

        # Normalize
        a = 255.0 / (q_percent[1] - q_percent[0])
        b = -a * q_percent[0]

        arr = a * arr + b * np.ones_like(arr)
        return arr

    # Define raster tiling process
    def process_raster_by_tile(
        rio_infile,
        rio_outfile,
        fn_process_tile,
        tile_size=(1000, 1000),
        **kwargs,
    ):
        """
        Function that executes a user defined function using a tiling process.
        """

        tile_size = np.clip(
            tile_size, a_min=[1, 1], a_max=[rio_infile.height, rio_infile.width]
        )

        # Compute the number of tiles
        num_tiles_xy = np.asarray(
            [
                np.ceil(rio_infile.width / tile_size[1]),
                np.ceil(rio_infile.height / tile_size[0]),
            ],
            dtype=np.int64,
        )

        kwargs["num_tiles_xy"] = num_tiles_xy

        for j in range(num_tiles_xy[0]):
            for i in range(num_tiles_xy[1]):
                tmp_tile_size = [
                    min(tile_size[0], rio_infile.height - i * tile_size[0]),
                    min(tile_size[1], rio_infile.width - j * tile_size[1]),
                ]

                arr = rio_infile.read(
                    1,
                    window=Window(
                        int(j * tile_size[1]),
                        int(i * tile_size[0]),
                        int(tmp_tile_size[1]),
                        int(tmp_tile_size[0]),
                    ),
                )

                kwargs["i"] = i
                kwargs["j"] = j
                kwargs["tmp_tile_size"] = tmp_tile_size
                arr = fn_process_tile(arr, **kwargs)
                rio_outfile.write(
                    arr,
                    indexes=1,
                    window=Window(
                        int(j * tile_size[1]),
                        int(i * tile_size[0]),
                        int(tmp_tile_size[1]),
                        int(tmp_tile_size[0]),
                    ),
                )

    # Open TIF image file
    with rasterio.open(img_path) as infile:
        raster = infile.read()

        # Create output jpg image
        with rasterio.open(
            output_8bit_path,
            mode="w",
            driver="JPEG",
            height=infile.height,
            width=infile.width,
            count=infile.count,
            dtype=rasterio.uint8,
            crs=infile.crs,
            transform=infile.transform,
        ) as dst:
            # Clip between 2% and 98%
            q_percent = np.percentile(raster, [2, 98])
            # Apply normalization to the image by tile
            process_raster_by_tile(
                infile, dst, tile_norm, q_percent=q_percent, tile_size=tile_size
            )

    # Crop image
    # Open it with PIL
    image = Image.open(output_8bit_path)
    # Crop with the bbox information
    im_crop = image.crop(bbox)
    # Save the cropped image to disk
    output_8bit_crop_path = os.path.join(
        output_dir,
        "8bit_crop_" + os.path.basename(img_path).split(".")[0] + ".jpg",
    )
    im_crop.save(output_8bit_crop_path, quality=100)

    return output_8bit_crop_path, np.size(im_crop)


def generate_uvs(img_pts, triangles, bbox, image_texture_size):
    """
    Function that computes the UVs coordinates

    Parameters
    ----------
    img_pts: (N, 2) list or tuple or np.ndarray
        Equivalent image coordinates of ground point cloud data
    triangles: (M, 3) list or tuple or np.ndarray
        Mesh triangle indexes. The indexes must correspond to the 'img_pts' indexes.
    image_texture_size: (2, ) list or tuple or np.ndarray
        Size (row, col) of the image texture

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
    image_texture_size = np.asarray(image_texture_size)

    # Loop over triangles
    uvs = []
    # Y-axis is inverted
    for i in range(triangles.shape[0]):
        uvs.append(
            [
                (img_pts[triangles[i, 0], 0] - bbox[0]) / image_texture_size[0],
                -(img_pts[triangles[i, 0], 1] - bbox[1])
                / image_texture_size[1],
                (img_pts[triangles[i, 1], 0] - bbox[0]) / image_texture_size[0],
                -(img_pts[triangles[i, 1], 1] - bbox[1])
                / image_texture_size[1],
                (img_pts[triangles[i, 2], 0] - bbox[0]) / image_texture_size[0],
                -(img_pts[triangles[i, 2], 1] - bbox[1])
                / image_texture_size[1],
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
        Configuration dictionary. The algorithm retrieves the following information:
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
                f"If specified, 'image_offset' should be a tuple or list of 2 elements (col, row)."
                f"Here: {image_offset}."
            )
        rpc.out_offset = np.asarray(rpc.out_offset)
        # RPC starts image pixel indexing at 1 whereas numpy arrays or QGIS start at 0
        # Thus the line below removes (1, 1) to each dimension to be consistent
        rpc.out_offset -= np.asarray(image_offset) - np.ones(2)

    # Open mesh, get vertices and triangles
    triangles = mesh.df[["p1", "p2", "p3"]].to_numpy()
    vertices = mesh.pcd.df[["x", "y", "z"]].to_numpy()

    # Convert vertices from UTM to geo (lon lat)
    vertices_lon_lat = change_frame(
        pd.DataFrame(vertices, columns=["x", "y", "z"]), utm_code, 4326
    ).to_numpy()

    # Apply inverse location to get the equivalent image point coordinates
    img_pts = apply_rpc_list(rpc, vertices_lon_lat)

    # Process image to crop the TIF file to the extent of the mesh
    # Get the image bounding box rounding up to the nearest integer (to avoid resampling)
    bbox = [
        np.floor(np.min(img_pts[:, 0])),
        np.floor(np.min(img_pts[:, 1])),
        np.ceil(np.max(img_pts[:, 0])),
        np.ceil(np.max(img_pts[:, 1])),
    ]
    image_texture_path, image_texture_size = preprocess_image_texture(
        tif_img_path, bbox, output_dir
    )

    # Compute UVs
    triangles_uvs = generate_uvs(img_pts, triangles, bbox, image_texture_size)

    # Add parameters to serialize the texture
    mesh.set_df_uvs(triangles_uvs)
    mesh.set_image_texture_path(image_texture_path)

    return mesh
