#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of demcompare
# (see https://github.com/CNES/demcompare).
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
Mosaic part of demcompare
"""

# Standard imports
import argparse
import json
import os
import subprocess


def shellquote(string):
    return "'%s'" % string.replace("'", "'\\''")


garbage = []


def remove(target):
    try:
        os.remove(target)
    except OSError:
        pass


def garbage_cleanup():
    """
    Removes all the files listed in the global variable 'garbage'.
    """
    while garbage:
        remove(garbage.pop())


def read_tiles(tiles_file):
    """
    Read tiles function
    """
    tiles = []
    outdir = os.path.dirname(tiles_file)

    with open(tiles_file) as file:
        tiles = file.readlines()

    # Strip trailing \n
    tiles = list(map(str.strip, tiles))
    tiles = [os.path.join(outdir, t) for t in tiles]

    return tiles


def vrt_body_source(
    fname, band, src_x, src_y, src_w, src_h, dst_x, dst_y, dst_w, dst_h
):
    """
    Generate a source section in vrt body.

    Args:
        fname: Relative path to the source image
        band: index of the band to use as source
        src_x, src_y, src_w, src_h: source window (cropped from source image)
        dst_x, dst_y, dst_w, dst_h: destination window
                                    (where crop will be pasted)
    """

    body = "\t\t<SimpleSource>\n"
    body += (
        "\t\t\t<SourceFileName relativeToVRT='1'>%s</SourceFileName>\n" % fname
    )
    body += "\t\t\t<SourceBand>%i</SourceBand>\n" % band
    body += "\t\t\t<SrcRect xOff='%i' yOff='%i'" % (src_x, src_y)
    body += "xSize='%i' ySize='%i'/>\n" % (src_w, src_h)
    body += "\t\t\t<DstRect xOff='%i' yOff='%i'" % (dst_x, dst_y)
    body += "xSize='%i' ySize='%i'/>\n" % (dst_w, dst_h)
    body += "\t\t</SimpleSource>\n"

    return body


def vrt_header(size_w, size_h, data_type="Float32", band=1, color=False):
    """
    Generate vrt up header

    Args:
        w, h: size of the corresponding raster
    """
    header = '<VRTDataset rasterXSize="%i" rasterYSize="%i">\n' % (
        size_w,
        size_h,
    )
    header += vrt_bandheader(data_type=data_type, band=band, color=color)

    return header


def vrt_bandheader(data_type="Float32", band=1, color=False):
    """
    Generate vrt header for a monoband image

    Args:
        w,h: size of the corresponding raster
        band: band id
        color: color gray activation
        data_type: Type of the raster (default is Float32)

    """
    header = '\t<VRTRasterBand dataType="%s" band="%d">\n' % (data_type, band)
    if color is False:
        header += "\t\t<ColorInterp>Gray</ColorInterp>\n"

    return header


def vrt_footer():
    """
    Generate vrt footer
    """
    footer = vrt_bandfooter()
    footer += "</VRTDataset>\n"

    return footer


def vrt_bandfooter():
    """
    Generate vrt footer
    """
    footer = "\t</VRTRasterBand>\n"

    return footer


def global_extent(tiles):
    """
    Compute the global raster extent from a list of tiles
    Args:
        tiles: list of config files loaded from json files
    Returns:
         (min_x,max_x,min_y,max_y) tuple
    """
    min_x = None
    max_x = None
    min_y = None
    max_y = None

    # First loop is to compute global extent
    for tile in tiles:
        with open(tile, "r") as file:

            tile_cfg = json.load(file)

            size_x = tile_cfg["roi"]["x"]
            size_y = tile_cfg["roi"]["y"]
            size_w = tile_cfg["roi"]["w"]
            size_h = tile_cfg["roi"]["h"]

            if min_x is None or size_x < min_x:
                min_x = size_x
            if min_y is None or size_y < min_y:
                min_y = size_y
            if max_x is None or size_x + size_w > max_x:
                max_x = size_x + size_w
            if max_y is None or size_y + size_h > max_y:
                max_y = size_y + size_h

    return (min_x, max_x, min_y, max_y)


def write_row_vrts(
    outdir,
    tiles,
    sub_img,
    vrt_basename,
    min_x,
    max_x,
    nb_bands=1,
    data_type="Float32",
    color=False,
):
    """
    Write intermediate vrts (one per row)

    Args:
        outdir : output directory in which to store the vrts
        tiles: list of config files loaded from json files
        sub_img: Relative path of the sub-image to mosaic
            (for ex. height_map.tif)
        vrt_basename: basename of the output vrt
        min_x, max_x: col extent of the raster
    Returns:
        A dictionnary of vrt files with sections vrt_body, th and vrt_dir
    """
    vrt_row = {}

    # First loop, write all row vrts body section
    for tile in tiles:
        with open(tile, "r") as file:

            tile_cfg = json.load(file)

            size_x = tile_cfg["roi"]["x"]
            size_y = tile_cfg["roi"]["y"]
            size_w = tile_cfg["roi"]["w"]
            size_h = tile_cfg["roi"]["h"]

            tile_dir = os.path.dirname(tile)

            vrt_row.setdefault(
                size_y,
                {
                    "vrt_body": "",
                    "vrt_dir": outdir,
                    "vrt_name": "{}_row{}".format(vrt_basename, size_y),
                    "th": size_h,
                },
            )

            tile_sub_img = os.path.join(tile_dir, sub_img)

            # Check if source image exists
            if not os.path.exists(tile_sub_img):
                # print('Warning: ' + tile_sub_img +
                #       ' does not exist, skipping ...')
                continue

            relative_sub_img_dir = os.path.relpath(
                os.path.abspath(tile_sub_img), vrt_row[size_y]["vrt_dir"]
            )
            vrt_row[size_y]["vrt_body"] += vrt_body_source(
                relative_sub_img_dir,
                1,
                0,
                0,
                size_w,
                size_h,
                size_x - min_x,
                0,
                size_w,
                size_h,
            )

    # Second loop, write all row vrts
    # Do not use items()/iteritems() here because of python 2 and 3 compat
    for size_y in vrt_row:
        vrt_data = vrt_row[size_y]
        row_vrt_filename = os.path.join(
            vrt_data["vrt_dir"], vrt_data["vrt_name"]
        )

        with open(row_vrt_filename, "w") as row_vrt_file:
            # Write vrt header
            row_vrt_file.write(
                vrt_header(
                    max_x - min_x,
                    vrt_data["th"],
                    data_type=data_type,
                    color=color,
                )
            )

            for band in range(1, nb_bands + 1):
                if band > 1:
                    row_vrt_file.write(
                        vrt_bandheader(
                            band=band, data_type=data_type, color=color
                        )
                    )

                # Write vrt body
                if band > 1:
                    row_vrt_file.write(
                        vrt_data["vrt_body"].replace(
                            "<SourceBand>{}</SourceBand>".format(1),
                            "<SourceBand>{}</SourceBand>".format(band),
                        )
                    )
                else:
                    row_vrt_file.write(vrt_data["vrt_body"])

                if band != nb_bands:
                    row_vrt_file.write(vrt_bandfooter())

            # Write vrt footer
            row_vrt_file.write(vrt_footer())

        garbage.append(row_vrt_filename)

    return vrt_row


def write_main_vrt(
    vrt_row,
    vrt_name,
    min_x,
    max_x,
    min_y,
    max_y,
    nb_bands=1,
    data_type="Float32",
    color=False,
):
    """
    Write the main vrt file

    Args:
        vrt_row: The vrt files dictionnary from write_row_vrts()
        vrt_name: The output vrt_name
        min_x,max_x,min_y,max_y: Extent of the raster
    """
    vrt_dirname = os.path.dirname(vrt_name) or "."

    with open(vrt_name, "w") as main_vrt_file:
        main_vrt_file.write(
            vrt_header(
                max_x - min_x, max_y - min_y, data_type=data_type, color=color
            )
        )

        for band in range(1, nb_bands + 1):
            if band > 1:
                main_vrt_file.write(
                    vrt_bandheader(band=band, data_type=data_type, color=color)
                )
            # Do not use items()/iteritems() here
            # because of python 2 and 3 compat
            for size_y in vrt_row:
                vrt_data = vrt_row[size_y]
                relative_vrt_dir = os.path.relpath(
                    vrt_data["vrt_dir"], vrt_dirname
                )
                row_vrt_filename = os.path.join(
                    relative_vrt_dir, vrt_data["vrt_name"]
                )

                vrt_body_src = vrt_body_source(
                    row_vrt_filename,
                    band,
                    0,
                    0,
                    max_x - min_x,
                    vrt_data["th"],
                    0,
                    size_y - min_y,
                    max_x - min_x,
                    vrt_data["th"],
                )

                main_vrt_file.write(vrt_body_src)
            if band != nb_bands:
                main_vrt_file.write(vrt_bandfooter())
        main_vrt_file.write(vrt_footer())


def main(
    tiles_file, outfile, sub_img, nb_bands=1, data_type="Float32", color=False
):
    """
    Main mosaic program
    """
    outfile_basename = os.path.basename(outfile)
    outfile_dirname = os.path.dirname(outfile) or "."

    output_format = outfile_basename[-3:]

    # If output format is tif, we need to generate a temporary vrt
    # with the same name
    vrt_basename = outfile_basename

    if output_format in ("tif", "png"):
        vrt_basename = vrt_basename[:-3] + "vrt"
    elif output_format != "vrt":
        print(
            "Error: only vrt/tif/png extensions are allowed for output image."
        )
        return

    vrt_name = os.path.join(outfile_dirname, vrt_basename)

    # Read the tiles file
    if isinstance(tiles_file, list):
        tiles = tiles_file
    else:
        tiles = read_tiles(tiles_file)

    # Compute the global extent of the output image
    (min_x, max_x, min_y, max_y) = global_extent(tiles)

    # Now, write all row vrts
    vrt_row = write_row_vrts(
        outfile_dirname,
        tiles,
        sub_img,
        vrt_basename,
        min_x,
        max_x,
        nb_bands=nb_bands,
        color=color,
        data_type=data_type,
    )

    # Finally, write main vrt
    write_main_vrt(
        vrt_row,
        vrt_name,
        min_x,
        max_x,
        min_y,
        max_y,
        nb_bands=nb_bands,
        color=color,
        data_type=data_type,
    )

    # If Output format is tif, convert vrt file to tif
    if output_format == "tif":
        try:
            devnull = open(os.devnull, "w")
            cmd = [
                "gdal_translate",
                "-ot",
                "Float32",
                "-co",
                "TILES=YES",
                "-co",
                "BIGTIFF=IF_NEEDED",
                "{}".format(vrt_name),
                "{}".format(outfile),
            ]
            subprocess.check_call(
                cmd, stdout=devnull, stderr=subprocess.STDOUT, env=os.environ
            )
        except Exception as error:
            print("Error {}".format(error))

        garbage.append(vrt_name)
        garbage_cleanup()


FORMAT = ["Float32", "Byte"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("mosaic tool"))

    parser.add_argument(
        "tiles", metavar="tiles.txt", help=("path to the tiles.txt file")
    )
    parser.add_argument(
        "outfile",
        metavar="out.tif",
        help=("path to the output file." " File extension can be .tif or .vrt"),
    )
    parser.add_argument(
        "sub_img",
        metavar="pair_1/height_map.tif",
        help=(
            "path to the sub-image to mosaic."
            " Can be (but not limited to) height_map.tif,"
            " pair_n/height_map.tif, pair_n/rpc_err.tif,"
            " cloud_water_image_domain_mask.png."
            " Note that rectified_* files CAN NOT be used."
        ),
    )
    parser.add_argument("--format", type=str, choices=FORMAT, default="Float32")
    parser.add_argument(
        "--color",
        action="store_true",
        help=("deactivate color gray interpretation"),
    )
    args = parser.parse_args()

    main(
        args.tiles,
        args.outfile,
        args.sub_img,
        data_type=args.format,
        color=args.color,
    )
