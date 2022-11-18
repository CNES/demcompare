#!/usr/bin/env python
# coding: utf8
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
#
"""
Fichier contenant les items
"""
import json
import os
import pprint
from os.path import isfile, join

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import rasterio
from dash import dash_table, dcc

df = pd.read_csv(
    "./full_outputs/stats/Slope0/stats_results.csv"
)

PATH_RESULTS = "./full_outputs"
PATH_STATS = "./full_outputs/stats"

list_layers = [
    f for f in os.listdir(PATH_STATS) if not isfile(join(PATH_STATS, f))
]
img_diff_init = rasterio.open(f"{PATH_RESULTS}/initial_dem_diff.tif")
img_diff_final = rasterio.open(f"{PATH_RESULTS}/final_dem_diff.tif")

list_dem_diff = ["initial_dem_diff", "final_dem_diff"]
list_mode = ["Standard", "Intersection", "Exclusion"]
list_dem = ["original dem", "coregistered dem"]

title = dcc.Markdown(children="")
mygraph = dcc.Graph(figure={})
mydropdown = dcc.Dropdown(
    options=df.columns.values[1:], value="mean", clearable=False
)
mybuttons = dcc.RadioItems(
    list_layers, value=list_layers[0], labelStyle={"display": "block"}
)
mybuttons_table = dcc.RadioItems(
    list_layers, value=list_layers[0], labelStyle={"display": "block"}
)
mybuttons_mode = dcc.RadioItems(
    list_mode, value=list_mode[0], labelStyle={"display": "block"}
)

mybuttons_dem = dcc.RadioItems(
    list_dem, value=list_dem[0], labelStyle={"display": "block"}
)

# maps tab
mybuttons_dem_diff = dcc.RadioItems(
    list_dem_diff, value=list_dem_diff[0], labelStyle={"display": "block"}
)

my_graph_maps = dcc.Graph(id="maps_id", figure={})


init_alt_heatmap = go.Heatmap(
    z=img_diff_init.read(1),
    x0=img_diff_init.bounds[0],
    y0=img_diff_init.bounds[0],
    dx=img_diff_init.bounds[2] - img_diff_init.bounds[0],
    dy=img_diff_init.bounds[3] - img_diff_init.bounds[1],
    coloraxis="coloraxis"
)

final_alt_heatmap = go.Heatmap(
    z=img_diff_final.read(1),
    x0=img_diff_final.bounds[0],
    y0=img_diff_final.bounds[0],
    dx=img_diff_final.bounds[2] - img_diff_final.bounds[0],
    dy=img_diff_final.bounds[3] - img_diff_final.bounds[1],
    coloraxis="coloraxis"
)

my_table = dash_table.DataTable(
    data=df.to_dict("records"),
    columns=[{"name": i, "id": i} for i in df.columns]
)


# Mes histogrammes
item_histo = dcc.Graph(id="histo_id")

button_overlay = dbc.Button(
    "Overlay", id="submit-val", active=False, n_clicks=0
)


histo_fig = go.Figure()
histo_fig.add_trace(go.Histogram(x=img_diff_init.read(1).ravel(),
                                    name="Initial altitudes differences",
                                    histnorm="probability density"
                                 )
                    )
histo_fig.add_trace(go.Histogram(x=img_diff_final.read(1).ravel(),
                                name="Final altitudes differences",
                                histnorm="probability density"
                                 )
                    )

histo_fig.update_layout(barmode="overlay",
                        xaxis_title="Elevation difference (m) from -|p98| to |p98|",
                        title=f"Overlayed histograms on all pixels with and without coregistration")

histo_fig.update_traces(opacity=0.75)


overlay = dcc.Graph(id="overlay", figure=histo_fig)

# json_dico

with open("config_report.json", encoding="utf-8") as f:
    data = json.load(f)

JSON_str = pprint.pformat(data)

# Mes DEMs
gironde_ref = rasterio.open(f"{PATH_RESULTS}/coregistration/reproj_REF.tif")
gironde_sec = rasterio.open(f"{PATH_RESULTS}/coregistration/reproj_SEC.tif")
gironde_sec_coreg = rasterio.open(f"{PATH_RESULTS}/coregistration/reproj_coreg_SEC.tif")
gironde_ref_coreg = rasterio.open(f"{PATH_RESULTS}/coregistration/reproj_coreg_REF.tif")

gironde_ref_heatmap = go.Heatmap(
    z=gironde_ref.read(1),
    colorscale="Greys",
    x0=gironde_ref.bounds[0],
    y0=gironde_ref.bounds[0],
    dx=gironde_ref.bounds[2] - gironde_ref.bounds[0],
    dy=gironde_ref.bounds[3] - gironde_ref.bounds[1],
    coloraxis="coloraxis"
)

gironde_sec_heatmap = go.Heatmap(
    z=gironde_sec.read(1),
    x0=gironde_sec.bounds[0],
    y0=gironde_sec.bounds[0],
    dx=gironde_sec.bounds[2] - gironde_sec.bounds[0],
    dy=gironde_sec.bounds[3] - gironde_sec.bounds[1],
    coloraxis="coloraxis"
)

gironde_sec_coreg_heatmap = go.Heatmap(
    z=gironde_sec_coreg.read(1),
    x0=gironde_sec_coreg.bounds[0],
    y0=gironde_sec_coreg.bounds[0],
    dx=gironde_sec_coreg.bounds[2] - gironde_sec_coreg.bounds[0],
    dy=gironde_sec_coreg.bounds[3] - gironde_sec_coreg.bounds[1],
    coloraxis="coloraxis"
)

gironde_ref_coreg_heatmap = go.Heatmap(
    z=gironde_ref_coreg.read(1),
    colorscale="Greys",
    x0=gironde_ref_coreg.bounds[0],
    y0=gironde_ref_coreg.bounds[0],
    dx=gironde_ref_coreg.bounds[2] - gironde_ref_coreg.bounds[0],
    dy=gironde_ref_coreg.bounds[3] - gironde_ref_coreg.bounds[1],
    coloraxis="coloraxis"
)

item_dems = dcc.Graph(id="dems", figure={})

button_dem_origin = ["Reproj ref", "Reproj sec"]
button_dem_coreg = ["Reproj coreg ref", "Reproj coreg sec"]

mybutton_dem_origin = dcc.RadioItems(
    button_dem_origin, value=button_dem_origin[0], labelStyle={"display": "block"}
)

mybutton_dem_coreg = dcc.RadioItems(
    button_dem_coreg, value=button_dem_coreg[0], labelStyle={"display": "block"}
)

item_dems_origin = dcc.Graph(id="dems_origin", figure={})

item_dems_coreg = dcc.Graph(id="dems_coreg", figure={})