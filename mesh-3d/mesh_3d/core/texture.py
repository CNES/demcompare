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
Texturing methods to project radiometric information over surfaces to provide a realistic rendering.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from pyproj import Transformer
from tqdm import tqdm
from os.path import split
from PIL import Image 
import rasterio
from loguru import logger
from rasterio.windows import Window
from ..tools.handlers import Mesh


Image.MAX_IMAGE_PIXELS = 1000000000 

class RPC:
    """ RPC """

    def __init__(self, polynomials, degrees, rpc_type: str):

        if  (len(polynomials) != 4):
            raise ValueError("Output dimensions should be 4 (P1, Q1, P2, Q2)")

        for i in range(len(polynomials)):
            if ( len(degrees) != len(polynomials[i])):
                raise ValueError( "Number of monomes in the degrees array shall equal number of coefficients in the polynomial array.")

        self.polynomials = polynomials
        self.degrees = degrees
        self.rpc_type = rpc_type
        self.refoffset = []
        self.refscale = []
        self.outoffset = []
        self.outscale = []
    
    
    def set_normalisation_coefs(self, coefs):
        if len(coefs) != 10:
            raise ValueError("Normalisation and denormalisation coefficients shall be of size 10")

        if self.rpc_type== "INVERSE":
            self.refoffset = [coefs[1], coefs[3], coefs[5]]
            self.refscale = [coefs[0], coefs[2], coefs[4]]
            self.outoffset = [coefs[7], coefs[9]]
            self.outscale = [coefs[6], coefs[8]]


def process_raster_by_tile(ds_in, ds_out, fn_process_tile, tile_size=(1000, 1000), **kwargs):
    """
    Function that executes a user defined function using a tiling process.
    """

    tile_size = np.clip(tile_size, a_min=[1, 1], a_max=[ds_in.height, ds_in.width])
    logger.info(f"Tile size applied (row, col): {tile_size.tolist()}")

    # Compute the number of tiles
    num_tiles_xy = np.asarray([np.ceil(ds_in.width / tile_size[1]),
                               np.ceil(ds_in.height / tile_size[0])], dtype=np.int64)

    kwargs["num_tiles_xy"] = num_tiles_xy

    for j in tqdm(range(num_tiles_xy[0]), desc="Columns", position=0):
        for i in tqdm(range(num_tiles_xy[1]), desc="Rows", position=1, leave=False):
            tmp_tile_size = [min(tile_size[0], ds_in.height - i * tile_size[0]),
                             min(tile_size[1], ds_in.width - j * tile_size[1])]

            arr = ds_in.read(1, window=Window(int(j * tile_size[1]),
                                    int(i * tile_size[0]),
                                    int(tmp_tile_size[1]),
                                    int(tmp_tile_size[0])))

            kwargs["i"] = i
            kwargs["j"] = j
            kwargs["tmp_tile_size"] = tmp_tile_size
            arr = fn_process_tile(arr, **kwargs)
            ds_out.write(arr , indexes=1,
                window=Window(int(j * tile_size[1]), int(i * tile_size[0]),
                int(tmp_tile_size[1]), int(tmp_tile_size[0])))



def parser_rpc_xml(path_rpc):
    """
    Function that parses the xml file to get the RPC
    """
    tree = ET.parse(path_rpc)
    root = tree.getroot()

    coefs = []
    for coef_inv in root.iter('Inverse_Model'):
        for coef in coef_inv:
            coefs.append(float(coef.text))

    # coefs normalisation and denormalisation: 
    # [long_scale, long_offset, lat_scale, lat_offset, alt_scale, alt_offset, samp_scale, samp_offset, line_scale, line_offset]
    for coef_other in root.iter('RFM_Validity'):
        for coef in coef_other:
            if coef.tag == "LONG_SCALE":
                long_scale = float(coef.text)
            if coef.tag  == "LONG_OFF":
                long_offset = float(coef.text)
            if coef.tag  == "LAT_SCALE":
                lat_scale = float(coef.text)
            if coef.tag  == "LAT_OFF":
                lat_offset = float(coef.text)
            if coef.tag  == "HEIGHT_SCALE":
                alt_scale = float(coef.text)
            if coef.tag  == "HEIGHT_OFF":
                alt_offset = float(coef.text)
            if coef.tag  == "SAMP_SCALE":
                samp_scale = float(coef.text)
            if coef.tag  == "SAMP_OFF":
                samp_offset = float(coef.text)
            if coef.tag  == "LINE_SCALE":
                line_scale = float(coef.text)
            if coef.tag  == "LINE_OFF":
                line_offset = float(coef.text)

    # Change image convention from (1, 1) to (0.5, 0.5)
    return(
        coefs[0:20], coefs[20:40], coefs[40:60], coefs[60:80], 
        [long_scale, long_offset, lat_scale, lat_offset, 
        alt_scale, alt_offset, samp_scale, samp_offset-0.5,
        line_scale, line_offset-0.5])



def apply_rpc_list(rpc, input_coords):
    """
    Function that computes inverse locations using rpc
    """
    # normalize input
    norm_input = (np.array(input_coords) - np.array(rpc.refoffset)) / np.array(rpc.refscale)

    result = []
    for i in range(len(rpc.polynomials)):
        val = np.zeros(len(input_coords))
        for j in range(len(rpc.polynomials[0])):
            monomial = np.ones(len(input_coords))
            for k in range(3):
                monomial *= np.power(norm_input[:,k], rpc.degrees[j][k])
            val += rpc.polynomials[i][j] * monomial
        result.append(val)


    if 0 in result[1][:] or 0 in result[3][:]:
        raise ValueError("Divided by zero is not allowed")

    # [XNorm = P1/Q1, YNorm = P2/Q2]
    output = [result[0][:] / result[1][:], result[2][:] / result[3][:]]
    res = [[output[0][i], output[1][i]] for i in range(len(output[0]))]


    # denormalize output
    res = (np.array(res) * np.array(rpc.outscale)) + np.array(rpc.outoffset)

    return(res)


def conversion_data(coords):
    """
    Conversion points from epsg 32631 to epsg 4326
    """
    transformer = Transformer.from_crs("epsg:32631", "epsg:4326")
    out = transformer.transform(coords[:,0], coords[:,1], coords[:,2])

    return(np.dstack(out)[0])


def generate_uvs(img_pts, triangles, bbox, size_cropimg):
    """
    Function that computes the UVs coordinates
    """

    uvs = []
    # Y-axis is inverted
    for i in tqdm(range(len(triangles))):
        tmp = []
        tmp.append((img_pts[triangles[i][0]][0]-bbox[0])/size_cropimg[0])
        tmp.append(-(img_pts[triangles[i][0]][1]-bbox[1])/size_cropimg[1])
        tmp.append((img_pts[triangles[i][1]][0]-bbox[0])/size_cropimg[0])
        tmp.append(-(img_pts[triangles[i][1]][1]-bbox[1])/size_cropimg[1])
        tmp.append((img_pts[triangles[i][2]][0]-bbox[0])/size_cropimg[0])
        tmp.append(-(img_pts[triangles[i][2]][1]-bbox[1])/size_cropimg[1])
        uvs.append(tmp)

    return np.asarray(uvs, dtype=np.float64)


def pretreatment_tif(img, bbox, output_path):
    """
    Function that transforms a 16-bit tif image into an 8-bit jpg
    """

    png_path = os.path.join(output_path, os.path.basename(img).split('.')[0]+'.jpg')
    def tile_norm(arr, q_percent, **kwargs):
        arr = np.clip(arr, a_min=q_percent[0], a_max=q_percent[1])

        # Normalize
        a = 255. / (q_percent[1] - q_percent[0])
        b = - a * q_percent[0]

        arr = a * arr + b * np.ones_like(arr)
        return arr

    with rasterio.open(img) as infile:

        raster = infile.read()

        with rasterio.open(
                            png_path,
                            mode='w',
                            driver='JPEG',
                            height=infile.height,
                            width=infile.width,
                            count=infile.count,
                            dtype=rasterio.uint8,
                            crs=infile.crs,
                            transform=infile.transform
                            ) as dst:
            q_percent = np.percentile(raster, [2, 98])
            process_raster_by_tile(infile, dst, tile_norm, q_percent=q_percent)


    # crop image
    image = Image.open(png_path)
    im_crop = image.crop(bbox)
    path_tmp = split(png_path)[0] + '/crop_' + split(png_path)[1]
    im_crop.save(path_tmp, quality=100)
    size_cropimg = np.size(im_crop)

    return path_tmp, size_cropimg


def texturing(mesh: Mesh, dir_out: str, rpc_path: str, img_path: str):
    """
    Function that creates the texture of a mesh.
    

    Parameters
    ----------
    mesh: Mesh
        Mesh object
    dir_out: Str
        Output directory to write the new texture image.
    rpc_path: Str
        Path to the xml rpc file.
    img_path: Str
        Path to the TIF image.

    Returns
    -------
    mesh: Mesh
        Mesh object with texture parameters
    """

    # Compute RPC for inverse location
    P1, Q1, P2, Q2, coefs = parser_rpc_xml(rpc_path)
    pleiade_degrees = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [2,0,0], [0,2,0], [0,0,2], 
                       [1,1,1], [3,0,0], [1,2,0], [1,0,2], [2,1,0], [0,3,0], [0,1,2], [2,0,1], [0,2,1], [0,0,3]]
    rpc = RPC([P1, Q1, P2, Q2], pleiade_degrees, "INVERSE")
    rpc.set_normalisation_coefs(coefs)

    # Opening mes, get vertices and triangles
    triangles = mesh.df[["p1", "p2", "p3"]].to_numpy()
    vertices = mesh.pcd.df[["x", "y", "z"]].to_numpy()

    # Vertices convert in lat long -> long lat
    vertices_lon_lat = conversion_data(vertices)
    vertices_lon_lat[:,[1,0]] = vertices_lon_lat[:, [0,1]]

    # Compute image points
    img_pts = apply_rpc_list(rpc, vertices_lon_lat)

    # Image treatment 
    bbox = np.floor(np.min(img_pts[:,0])), np.floor(np.min(img_pts[:,1])), np.ceil(np.max(img_pts[:,0])), np.ceil(np.max(img_pts[:,1]))
    path_img, size_cropimg = pretreatment_tif(img_path, bbox, dir_out)

    # Compute UVs
    triangles_uvs = generate_uvs(img_pts, triangles, bbox, size_cropimg)

    # Parameters to serialize the texture
    for i, name in enumerate(["uv1_row", "uv1_col", "uv2_row", "uv2_col", "uv3_row", "uv3_col"]):
        mesh.df[name] = triangles_uvs[:,i]
    mesh.set_texture_parameters(path_img)

    return mesh
