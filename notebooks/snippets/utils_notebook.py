#!/usr/bin/env python
# coding: utf8
# Disable the protected-access to test the functions
# pylint:disable=protected-access
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
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
"""
utils functions for plotting in notebooks
"""

import itertools
import random

import numpy as np
import xarray as xr
from bokeh.colors import RGB
from bokeh.io import output_notebook, show
from bokeh.layouts import row
from bokeh.models import BasicTicker, ColorBar, Legend, LinearColorMapper
from bokeh.palettes import RdYlGn
from bokeh.plotting import figure
from matplotlib.colors import LinearSegmentedColormap


def demcompare_cmap(
    color: int = None,
) -> LinearSegmentedColormap:
    """
    Create a LinearSegmentedColormap from a list of colors.
    :param color: List of 4 colors
    :type color: list
    :return: LinearSegmentedColormap
    """
    colors = ["palegreen", "green", "lightcoral", "darkred"]
    nodes = [0.0, 0.4, 0.5, 1.0]
    cmap_shift = LinearSegmentedColormap.from_list(
        "mycmap", list(zip(nodes, colors))  # noqa: B905
    )

    if color is not None:
        return cmap_shift(color)

    return cmap_shift


def cmap_to_palette(cmap: LinearSegmentedColormap) -> list:
    """
    Transform cmap to colorlist
    """
    cmap_rgb = (255 * cmap(range(256))).astype("int")
    palette = [RGB(*tuple(rgb)).to_hex() for rgb in cmap_rgb]

    return palette


def stack_dems(
    input_ref: xr.Dataset,
    input_sec: xr.Dataset,
    title: str,
    input_extra: xr.Dataset = None,
    legend_ref: str = "Ref DEM",
    legend_sec: str = "Second DEM",
    legend_extra: str = "Sample Sec",
) -> figure:
    """
    Allows the user to view three stacked DEMs.
    :param input_ref: DEMS
    :type input_ref: dataset
    :param input_sec: DEMS
    :type input_extra: dataset
    :param title: Title
    :param input_extra: DEMS
    :type input_sec: dataset
    :type title: str
    :param legend_ref: Legend for reference DEM
    :type legend_ref: str
    :param legend_sec: Legend for second DEM
    :type legend_sec: str
    :param legend_extra: Legend for extra DEM
    :type legend_extra: str
    :return: figure
    """
    output_notebook()

    min_d = np.nanmin(input_ref["image"].data)
    max_d = np.nanmax(input_ref["image"].data)

    xlabel = "latitude"
    ylabel = "longitude"

    mapper_avec_opti = LinearColorMapper(
        palette=cmap_to_palette(demcompare_cmap()),
        low=min_d,
        high=max_d,
        nan_color=(0, 0, 0, 0),
    )

    fig = figure(
        title=title,
        width=800,
        height=450,
        tools=["reset", "pan", "box_zoom", "save"],
        output_backend="webgl",
        x_axis_label=xlabel,
        y_axis_label=ylabel,
    )

    dem_ref_img = fig.image(
        image=[np.flip(input_ref["image"].data, 0)],
        x=input_ref.bounds[0],
        y=input_ref.bounds[1],
        dw=input_ref.bounds[2] - input_ref.bounds[0],
        dh=input_ref.bounds[3] - input_ref.bounds[1],
        color_mapper=mapper_avec_opti,
    )

    dem_sec_img = fig.image(
        image=[np.flip(input_sec["image"].data, 0)],
        x=input_sec.bounds[0],
        y=input_sec.bounds[1],
        dw=input_sec.bounds[2] - input_sec.bounds[0],
        dh=input_sec.bounds[3] - input_sec.bounds[1],
        color_mapper=mapper_avec_opti,
    )

    legend_items = [(legend_ref, [dem_ref_img]), (legend_sec, [dem_sec_img])]

    if input_extra:
        dem_extra = fig.image(
            image=[np.flip(input_sec["image"].data, 0)],
            x=input_sec.bounds[0],
            y=input_sec.bounds[1],
            dw=input_sec.bounds[2] - input_sec.bounds[0],
            dh=input_sec.bounds[3] - input_sec.bounds[1],
            color_mapper=mapper_avec_opti,
        )

        legend_items.append((legend_extra, [dem_extra]))

    legend = Legend(items=legend_items, location="center", click_policy="hide")
    fig.add_layout(legend, "right")

    color_bar = ColorBar(
        color_mapper=mapper_avec_opti,
        ticker=BasicTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title="Altitudes (m)",
    )
    fig.add_layout(color_bar, "left")

    return fig


def side_by_side_fig(
    input_ref: xr.Dataset, input_sec: xr.Dataset, title_ref: str, title_sec: str
) -> row:
    """
    Show two figures side by side
    :param input_ref: DEMS
    :type input_sec: dataset
    :param input_sec: DEMS
    :type input_ref: dataset
    :param title_ref: Title
    :type title_ref: str
    :param title_sec: Title
    :type title_sec: str
    :param colorbar: Activate colobar
    :type colorbar: bool

    :return: figure
    """

    output_notebook()

    # looking for min and max for both datasets
    min_d = min(
        np.nanmin(input_ref["image"].data), np.nanmin(input_sec["image"].data)
    )
    max_d = max(
        np.nanmax(input_ref["image"].data), np.nanmax(input_sec["image"].data)
    )

    d_w = input_ref["image"].shape[1]
    d_h = input_ref["image"].shape[0]

    mapper_avec_opti = LinearColorMapper(
        palette=cmap_to_palette(demcompare_cmap()),
        low=min_d,
        high=max_d,
        nan_color=(0, 0, 0, 0),
    )

    first_fig = figure(
        title=title_ref,
        width=480,
        height=450,
        tools=["reset", "pan", "box_zoom", "save"],
        output_backend="webgl",
        x_axis_label="latitude",
        y_axis_label="longitude",
    )
    first_fig.image(
        image=[np.flip(input_ref["image"].data, 0)],
        x=input_ref.bounds[0],
        y=input_ref.bounds[1],
        dw=d_w,
        dh=d_h,
        color_mapper=mapper_avec_opti,
    )

    second_fig = figure(
        title=title_sec,
        width=480,
        height=450,
        tools=["reset", "pan", "box_zoom", "save"],
        output_backend="webgl",
        x_axis_label="latitude",
        y_axis_label="longitude",
    )
    second_fig.image(
        image=[np.flip(input_sec["image"].data, 0)],
        x=input_sec.bounds[0],
        y=input_sec.bounds[1],
        dw=d_w,
        dh=d_h,
        color_mapper=mapper_avec_opti,
    )

    color_bar = ColorBar(
        color_mapper=mapper_avec_opti,
        ticker=BasicTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title="Altitudes (m)",
    )
    first_fig.add_layout(color_bar, "left")
    second_fig.add_layout(color_bar, "left")

    layout = row([first_fig, second_fig])  # type:ignore

    return layout


def show_dem(
    input_ref: xr.Dataset,
    title_ref: str,
) -> None:
    """
    Show one figure
    :param input_ref: DEM
    :type input_ref: dataset
    :param title_ref: Title
    :type title_ref: str

    :return: figure
    """

    output_notebook()

    # looking for min and max for both datasets
    min_d = np.nanmin(input_ref["image"].data)
    max_d = np.nanmax(input_ref["image"].data)

    mapper_avec_opti = LinearColorMapper(
        palette=cmap_to_palette(demcompare_cmap()),
        low=min_d,
        high=max_d,
        nan_color=(0, 0, 0, 0),
    )

    first_fig = figure(
        title=title_ref,
        width=480,
        height=450,
        tools=["reset", "pan", "box_zoom", "save"],
        output_backend="webgl",
        x_axis_label="latitude",
        y_axis_label="longitude",
    )
    first_fig.image(
        image=[np.flip(input_ref["image"].data, 0)],
        x=input_ref.bounds[0],
        y=input_ref.bounds[1],
        dw=input_ref.bounds[2] - input_ref.bounds[0],
        dh=input_ref.bounds[3] - input_ref.bounds[1],
        color_mapper=mapper_avec_opti,
    )

    color_bar = ColorBar(
        color_mapper=mapper_avec_opti,
        ticker=BasicTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title="Altitudes (m)",
    )
    first_fig.add_layout(color_bar, "left")

    show(first_fig)


def plot_cdf_pdf(datas: np.ndarray, title_fig: str) -> None:
    """
    Plot cdf and pdf function
    :param datas: values for functions
    :type datas: np.ndarray
    :param title_fig: title
    :type title_fig: str
    """
    output_notebook()
    fig = figure(title=title_fig, tools=["reset", "pan", "box_zoom", "save"])

    if title_fig not in ("pdf", "cdf"):
        print(f"{title_fig} is not an available option")
        show(fig)
    else:
        fig.line(
            datas[0][1], np.insert(datas[0][0], 0, 0), legend_label=title_fig
        )
        if title_fig == "pdf":
            fig.xaxis.axis_label = (
                "Elevation difference (m) from - |p98| to |p98|"
            )
            fig.yaxis.axis_label = "Probability density"
        if title_fig == "cdf":
            fig.xaxis.axis_label = "Full absolute elevation differences (m)"
            fig.yaxis.axis_label = "Cumulative Probability"
        show(fig)


def plot_ratio_above_threshold(rat: np.ndarray, title: str) -> None:
    """
    Plot ratio above threshold function
    :param rat: values for functions
    :type rat: np.ndarray
    :param title: title
    :type title: str
    """

    fig = figure(title=title)
    fig.vbar(x=rat[0][1], top=rat[0][0], color="lightsteelblue", width=0.2)
    fig.xaxis.axis_label = "Altitudes (m)"
    fig.yaxis.axis_label = "% of pixel above the treshold"
    show(fig)


def plot_layers(
    processing_dataset: xr.Dataset,
    input_type: str,
    dem: xr.Dataset,
    title: str,
    name_layer: str,
) -> None:
    """
    plot layers dynamically
    :param processing_dataset: Processing dataset object
    :type processing_dataset: xr.Dataset
    :param input_type: "ref" or "sec"
    :type input_type: str
    :param dem: xarray.Dataset
    :type dem: dem
    :param title: figure's title
    :type title: str
    :param name_layer: name of layer to be shown
    :type name_layer: str
    """
    output_notebook()

    # Get image with mask layers
    number_layer = processing_dataset.classification_layers_names.index(
        name_layer
    )  # 1
    img = processing_dataset.classification_layers[number_layer].map_image[
        input_type
    ]
    img[img == dem.attrs["nodata"]] = np.nan

    d_w = img.shape[1]
    d_h = img.shape[0]

    # Get dem
    dem_image = dem.image.data
    dem_image[dem_image == 0] = np.nan

    # palette
    mapper_dem = LinearColorMapper(
        palette="Greys7",
        low=np.nanmin(dem_image),
        high=np.nanmax(dem_image),
        nan_color=(0, 0, 0, 0),
    )

    fig = figure(
        title=title,
        width=1000,
        height=1000,
        tools=["reset", "pan", "box_zoom", "save"],
        output_backend="webgl",
    )

    # Legend
    fig.image(
        image=[np.flip(dem_image, 0)],
        x=1,
        y=0,
        dw=d_w,
        dh=d_h,
        color_mapper=mapper_dem,
    )
    legend_items = []

    # create a color iterator
    colors = itertools.cycle(RdYlGn)
    # Points attributes
    size = 0.5
    alpha = 0.2

    # Get mask values and add them
    for iterateur, (key, _) in enumerate(
        processing_dataset.classification_layers[number_layer].classes.items()
    ):
        mask_layer = processing_dataset.classification_layers[
            number_layer
        ].classes_masks[input_type][iterateur]

        mask_x = np.where(mask_layer != 0)[1]
        mask_y = d_h - np.where(mask_layer != 0)[0]

        color = next(colors)
        mask = fig.circle(
            x=mask_x,
            y=mask_y,
            size=size,
            color=RdYlGn[color][random.choice([0, 1, 2])],
            alpha=alpha,
        )

        legend_items.append((key, [mask]))

    legend = Legend(items=legend_items, location="center", click_policy="hide")

    fig.add_layout(legend, "right")

    show(fig)
