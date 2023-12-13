#!/usr/bin/env python
# coding: utf8
# Disable the protected-access to test the functions
# pylint:disable=protected-access
# type: ignore
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

# pylint:disable=too-many-lines
"""
utils functions for plotting in notebooks
"""

import itertools
import random
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from bokeh.colors import RGB
from bokeh.io import output_notebook, show
from bokeh.layouts import row
from bokeh.models import BasicTicker, ColorBar, Legend, LinearColorMapper
from bokeh.palettes import RdYlGn
from bokeh.plotting import figure
from matplotlib.colors import LinearSegmentedColormap


def demcompare_cmap() -> LinearSegmentedColormap:
    """
    Create a LinearSegmentedColormap from a predefined list of colors.

    :return: LinearSegmentedColormap
    """
    nodes_colors = [
        (0.0, "palegreen"),
        (0.4, "green"),
        (0.5, "lightcoral"),
        (1.0, "darkred"),
    ]

    return LinearSegmentedColormap.from_list("mycmap", nodes_colors)


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
    )
    fig.add_layout(color_bar, "left")

    return fig


def side_by_side_fig(
    input_ref: xr.Dataset, input_sec: xr.Dataset, title_ref: str, title_sec: str
) -> Any:
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
    )
    first_fig.add_layout(color_bar, "left")
    second_fig.add_layout(color_bar, "left")

    layout = row([first_fig, second_fig])

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


def plot_slope_orientation_histogram(
    datas1: Tuple[np.ndarray, np.ndarray],
    datas2: Tuple[np.ndarray, np.ndarray],
    title_fig1: str,
    title_fig2: str,
):
    """
    Plot slope orientation histogram

    :param datas1: Tuple where the data is stored
    :type datas1: Tuple[np.ndarray]
    :param title_fig1: title of the figure 1
    :type title_fig1: str
    :param datas2: Tuple where the data is stored
    :type datas2: Tuple[np.ndarray]
    :param title_fig2: title of the figure 2
    :type title_fig2: str
    :return: None
    """

    output_notebook()

    plot = figure(
        title="Slope orientation histogram",
        tools=["reset", "pan", "box_zoom", "save"],
    )

    plot.axis.visible = False

    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None

    max_val = max(np.max(datas1[0]), np.max(datas2[0]))

    plot.annulus(
        x=0.0,
        y=0.0,
        inner_radius=max_val / 2.0,
        outer_radius=max_val,
        fill_alpha=0.0,
        alpha=0.5,
        line_color="black",
    )
    plot.annulus(
        x=0.0,
        y=0.0,
        inner_radius=max_val / 4.0,
        outer_radius=3 * max_val / 4.0,
        fill_alpha=0.0,
        alpha=0.5,
        line_color="black",
    )

    angles1 = datas1[1][:-1]
    x_1 = datas1[0] * np.cos(angles1)
    y_1 = datas1[0] * np.sin(angles1)
    plot.circle(x_1, y_1, line_width=2, legend_label=title_fig1)

    angles2 = datas2[1][:-1]
    x_2 = datas2[0] * np.cos(angles2)
    y_2 = datas2[0] * np.sin(angles2)
    plot.circle(x_2, y_2, line_width=2, color="red", legend_label=title_fig2)

    linspace_max_val = np.linspace(-max_val, max_val)

    linspace_max_val_diag = np.linspace(
        -max_val / np.sqrt(2), max_val // np.sqrt(2)
    )

    plot.line(
        linspace_max_val,
        np.zeros(linspace_max_val.shape[0]),
        line_width=2,
        line_color="black",
        alpha=0.25,
    )
    plot.line(
        np.zeros(linspace_max_val.shape[0]),
        linspace_max_val,
        line_width=2,
        line_color="black",
        alpha=0.25,
    )
    plot.line(
        linspace_max_val_diag,
        linspace_max_val_diag,
        line_width=2,
        line_color="black",
        alpha=0.25,
    )
    plot.line(
        linspace_max_val_diag,
        -linspace_max_val_diag,
        line_width=2,
        line_color="black",
        alpha=0.25,
    )

    # Add circles and labels at specific angles
    angles = (
        np.array(
            [
                0,
                np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4,
                np.pi,
                5 * np.pi / 4,
                3 * np.pi / 2,
                7 * np.pi / 4,
            ]
        )
        + 3 * np.pi / 4.0
    )
    labels = ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"][::-1]

    for angle, label in zip(angles, labels):  # noqa: B905
        c_x = 1.1 * max_val * np.cos(angle)
        c_y = 1.1 * max_val * np.sin(angle)
        plot.text(
            c_x, c_y, text=[label], text_baseline="middle", text_align="center"
        )

    show(plot)


def plot_visualizations(
    input_ref: xr.Dataset,
    input_sec: xr.Dataset,
    hillshade_ref: xr.Dataset,
    hillshade_sec: xr.Dataset,
    svf_ref: xr.Dataset,
    svf_sec: xr.Dataset,
    colorbar_range: list,
):
    """
    Show 6 figures: top three are elevation, hillshade and svf visualizations
        for the reference DEM. Bottom three are the sames for the second DEM.
    :param input_ref: DEMS
    :type input_ref: dataset
    :param input_sec: DEMS
    :type input_sec: dataset
    :param hillshade_ref: hillshade
    :type hillshade_ref: dataset
    :param hillshade_sec: hillshade
    :type hillshade_sec: dataset
    :param svf_ref: sky-view factor
    :type svf_ref: dataset
    :param svf_sec: sky-view factor
    :type svf_sec: dataset
    :param colorbar_range: min & max value for the 3 visualizations
    :type colorbar_range: list[6]

    :return: figure
    """

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(
        input_ref["image"],
        vmin=colorbar_range[0],
        vmax=colorbar_range[1],
        cmap="terrain",
    )
    plt.colorbar()
    plt.title("ref DEM")
    plt.subplot(2, 3, 2)
    plt.imshow(
        hillshade_ref["image"],
        vmin=colorbar_range[2],
        vmax=colorbar_range[3],
        cmap="Greys_r",
    )
    plt.colorbar()
    plt.title("ref DEM hillshade")
    plt.subplot(2, 3, 3)
    plt.imshow(
        svf_ref["image"],
        vmin=colorbar_range[4],
        vmax=colorbar_range[5],
        cmap="Greys_r",
    )
    plt.colorbar()
    plt.title("ref DEM svf")
    plt.subplot(2, 3, 4)
    plt.imshow(
        input_sec["image"],
        vmin=colorbar_range[0],
        vmax=colorbar_range[1],
        cmap="terrain",
    )
    plt.colorbar()
    plt.title("sec DEM")
    plt.subplot(2, 3, 5)
    plt.imshow(
        hillshade_sec["image"],
        vmin=colorbar_range[2],
        vmax=colorbar_range[3],
        cmap="Greys_r",
    )
    plt.colorbar()
    plt.title("sec DEM hillshade")
    plt.subplot(2, 3, 6)
    plt.imshow(
        svf_sec["image"],
        vmin=colorbar_range[4],
        vmax=colorbar_range[5],
        cmap="Greys_r",
    )
    plt.colorbar()
    plt.title("sec DEM svf")

    plt.show()


def plot_side_by_side(
    input_ref: xr.Dataset,
    input_sec: xr.Dataset,
    title_ref: str,
    title_sec: str,
    vmin_color_ref: float,
    vmax_color_ref: float,
    vmin_color_sec: float,
    vmax_color_sec: float,
):
    """
    Show both slope orientation histograms on the same plot.
    :param input_ref: DEMS
    :type input_ref: Tuple[np.ndarray]
    :param input_sec: DEMS
    :type input_sec: Tuple[np.ndarray]
    :param title_ref: title
    :type title_ref: str
    :param title_sec: title
    :type title_sec: str
    :param cmap: cmap parameter for the plot
    :type cmap: str
    :param vmin_color_ref: min value for the colorbar
    :type vmin_color_ref: float
    :param vmax_color_ref: max value for the colorbar
    :type vmax_color_ref: float
    :param vmin_color_sec: min value for the colorbar
    :type vmin_color_sec: float
    :param vmax_color_sec: max value for the colorbar
    :type vmax_color_sec: float

    :return: figure
    """

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(
        input_ref["image"], vmin=vmin_color_ref, vmax=vmax_color_ref, cmap="bwr"
    )
    plt.colorbar()
    plt.title(title_ref)
    plt.subplot(1, 2, 2)
    plt.imshow(
        input_sec["image"], vmin=vmin_color_sec, vmax=vmax_color_sec, cmap="bwr"
    )
    plt.colorbar()
    plt.title(title_sec)

    plt.show()


def plot_two_slope_orientation_histogram(
    angles_ref: np.ndarray,
    hist_ref: np.ndarray,
    angles_sec: np.ndarray,
    hist_sec: np.ndarray,
):
    """
    Show 2 plots side by side.
    :param angles_ref: angles
    :type angles_ref: ndarray
    :param hist_ref: angles
    :type hist_ref: ndarray
    :param angles_sec: bins
    :type angles_sec: ndarray
    :param hist_sec: bins
    :type hist_sec: ndarray

    :return: figure
    """

    angles_ref = angles_ref[:-1]
    angles_sec = angles_sec[:-1]

    # matplotlib displays the default 0° in the east direction,
    # and +90° in the direct trigonometric direction.
    angles_ref2 = -angles_ref
    angles_ref2 += np.pi / 2
    angles_sec2 = -angles_sec
    angles_sec2 += np.pi / 2

    plt.figure()
    a_x = plt.subplot(111, polar=True)
    a_x.plot(angles_ref2, hist_ref, ".", color="red", label="ref DEM")
    a_x.plot(angles_sec2, hist_sec, ".", color="blue", label="sec DEM")
    a_x.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 8, endpoint=False))
    new_label = [
        "90° \n (E)",
        "45°",
        "0° (N)",
        "315°",
        "270° \n (W)",
        "225°",
        "180° (S)",
        "135°",
    ]
    a_x.set_xticklabels(new_label)
    plt.legend()

    plt.title("Number of pixels as a function of the slope orientation.")

    c_f = plt.gcf()
    c_f.set_size_inches([16.8, 9.45])

    plt.show()


def plot_cdf_pdf_side_by_side(
    pdf: np.ndarray, cdf: np.ndarray, method: str
) -> None:
    """
    Plot cdf and pdf function
    :param pdf: values for pdf
    :type pdf: np.ndarray
    :param cdf: values for cdf
    :type cdf: np.ndarray
    :param method: method
    :type method: str
    """

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    a_x = plt.gca()
    center = (pdf[0][1][:-1] + pdf[0][1][1:]) / 2
    plt.bar(center, pdf[0][0])
    a_x.set_xlabel("Elevation difference (m)")
    a_x.set_ylabel("Probability density")
    plt.grid()
    plt.title("pdf for the " + method)

    plt.subplot(1, 2, 2)
    a_x = plt.gca()
    center = (cdf[0][1][:-1] + cdf[0][1][1:]) / 2
    plt.plot(center, cdf[0][0])
    a_x.set_xlabel("Full absolute elevation difference (m)")
    a_x.set_ylabel("Cumulative density")
    plt.grid()
    plt.title("cdf for the " + method)

    plt.show()


def plot_angular_diff(
    data: xr.Dataset,
    pdf: np.ndarray,
    cdf: np.ndarray,
    vmin_color: float,
    vmax_color: float,
) -> None:
    """
    Plot cdf and pdf function
    :param data: angular diff data
    :type data: dataset
    :param pdf: values for pdf
    :type pdf: np.ndarray
    :param cdf: values for cdf
    :type cdf: np.ndarray
    :param vmin_color: min value for the angular diff colorbar
    :type vmin_color: float
    :param vmax_color: max value for the angular diff colorbar
    :type vmax_color: float
    """

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(data["image"], vmin=vmin_color, vmax=vmax_color, cmap="Reds")
    plt.colorbar()
    plt.title("Angular difference (radian)")

    plt.subplot(1, 3, 2)
    a_x = plt.gca()
    center = (pdf[0][1][:-1] + pdf[0][1][1:]) / 2
    plt.bar(center, pdf[0][0], width=0.1)
    a_x.set_xlabel("Elevation difference (m)")
    a_x.set_ylabel("Probability density")
    plt.grid()
    plt.title("pdf for the angular difference")

    plt.subplot(1, 3, 3)
    a_x = plt.gca()
    center = (cdf[0][1][:-1] + cdf[0][1][1:]) / 2
    plt.plot(center, cdf[0][0])
    a_x.set_xlabel("Full absolute elevation difference (m)")
    a_x.set_ylabel("Cumulative density")
    plt.grid()
    plt.title("cdf for the angular difference")

    plt.show()
