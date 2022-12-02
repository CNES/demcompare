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
Fichier contenant layers et callback
"""
from datetime import datetime
from jupyter_dash import JupyterDash
import assets.items as my_items
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

# App initialization
os.environ['no_proxy'] = "cnes.fr,sis.cnes.fr,localhost,127.0.0.1"

JupyterDash.infer_jupyter_proxy_config()

app = JupyterDash(__name__,
                 requests_pathname_prefix ='/user/USERNAME/proxy/8050/',
                 external_stylesheets=[
                                        dbc.themes.BOOTSTRAP,
                                        "https://codepen.io/chriddyp/pen/bWLwgP.css",
                ]
)
app.server.secret_key = os.environ.get('secret_key', 'secret')
# Title in tab
app.title = "Demcompare report"

# Création de mes petits onglets
tab_metrics = dcc.Tab(
    [
        dbc.Row([my_items.title]),
        dbc.Row(
            [
                dbc.Col([my_items.mybuttons], width=3),
                dbc.Col([my_items.mygraph], width=9),
            ]
        ),
        dbc.Row([dbc.Col([my_items.mydropdown], width=3)], justify="center"),
    ],
    label="Metrics",
    value="tab-1-metrics",
)

tab_alti_dif = dcc.Tab(
    [
        dbc.Row(html.H1(children="Maps and histogramms for altitudes differences")),
        dbc.Row(
            [
                dbc.Col([my_items.my_graph_maps], width=5),
                dbc.Col([my_items.mybuttons_dem_diff], width=1),
                dbc.Col([my_items.item_histo], width=6),
            ]
        ),
        dbc.Row([my_items.overlay]),
    ],
    label="Altitude differences",
    value="tab-3-layers",
)

tab_metrics_tab = dcc.Tab(
    [
        dbc.Row(html.H1(id="title-tables", children="")),
        dbc.Row(
            [
                dbc.Col([my_items.mybuttons_table], width=2),
                dbc.Col([my_items.mybuttons_mode], width=1),
                dbc.Col([my_items.my_table], width=9),
            ]
        ),
    ],
    label="Table metrics",
    value="tab-2-metrics",
)

tab_context = dcc.Tab(
    [
        dbc.Row(html.H1(children="Config used in demcompare")),
        dbc.Row(html.H3(children=f"Execution time : {str(datetime.now())}")),
        dbc.Row(html.H5(children=f"{my_items.JSON_str}")),
    ],
    label="Context",
    value="tab-context",
)

tab_dem = dcc.Tab(
    [
        dbc.Row([dbc.Col(html.H1(children="Reference DEM and second DEMs"))]),
        dbc.Row([dbc.Col(html.H2(children="DEMs are reprojected here"))]),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(my_items.item_dems),
                        dbc.Row(my_items.mybuttons_dem),
                    ],
                ),
                dbc.Col(
                    [
                        dbc.Row(my_items.item_dems_origin),
                        dbc.Row(my_items.mybutton_dem_origin),
                    ],
                ),
                dbc.Col(
                    [
                        dbc.Row(my_items.item_dems_coreg),
                        dbc.Row(my_items.mybutton_dem_coreg),
                    ],
                ),
            ]
        ),
    ],
    label="Digital Elevation Model",
    value="tab-dem",
)

# Titre + appel des onglets
app.layout = html.Div(
    [
        html.H1("Demcompare report"),
        dcc.Tabs(
            id="Tabs",
            value="tab-context",
            children=[
                tab_context,
                tab_dem,
                tab_metrics,
                tab_metrics_tab,
                tab_alti_dif,
            ],
        ),
        html.Div(id="tabs-content-example-graph"),
    ], style={'whiteSpace': 'pre-wrap'}
)


#############################################################################
# Callback allows components to interact
@app.callback(
    Output(my_items.mygraph, "figure"),
    Output(my_items.title, "children"),
    Input(my_items.mydropdown, "value"),
    Input(my_items.mybuttons, "value"),
)
def update_graph(
    column_name, button_name
):  # function arguments come from the component property of the Input
    """
    Update metrics graph
    """

    # datas
    dataframe_col = pd.read_csv(
        f"{my_items.PATH_STATS}/{button_name}/stats_results.csv"
    )

    fig = px.scatter(data_frame=dataframe_col, x="Set Name", y=column_name)

    return (
        fig,
        f"# Visualize {column_name} datas for "
        f"classification layers named {button_name}",
    )


# Callback allows components to interact
@app.callback(
    Output(my_items.my_table, "data"),
    Output(my_items.my_table, "columns"),
    Output(component_id="title-tables", component_property="children"),
    Input(my_items.mybuttons_table, "value"),
    Input(my_items.mybuttons_mode, "value"),
)
def update_table(
    button_name_layers, button_name_mode
):  # function arguments come from the component property of the Input
    """
    Update metrics table
    """

    if button_name_mode == "Standard":
        try:
            dataframe = pd.read_csv(
                f"{my_items.PATH_STATS}/{button_name_layers}/stats_results.csv"
            )
        except FileNotFoundError:
            dataframe = pd.DataFrame(
                columns=[
                    f"{button_name_mode} mode for "
                    f"{button_name_layers} layer doesn't exists"
                ]
            )
    if button_name_mode == "Intersection":
        try:
            dataframe = pd.read_csv(
                f"{my_items.PATH_STATS}/{button_name_layers}"
                f"/stats_results_intersection.csv"
            )
        except FileNotFoundError:
            dataframe = pd.DataFrame(
                columns=[
                    f"{button_name_mode} mode for "
                    f"{button_name_layers} layer doesn't exists"
                ]
            )

    if button_name_mode == "Exclusion":
        try:
            dataframe = pd.read_csv(
                f"{my_items.PATH_STATS}/{button_name_layers}"
                f"/stats_results_exclusion.csv"
            )
        except FileNotFoundError:
            dataframe = pd.DataFrame(
                columns=[
                    f"{button_name_mode} mode for "
                    f"{button_name_layers} layer doesn't exists"
                ]
            )

    data = dataframe.to_dict("records")
    columns = [{"name": i, "id": i} for i in dataframe.columns]
    title = f"Metrics table for {button_name_layers} layer and {button_name_mode} mode"

    return data, columns, title


@app.callback(
    Output(component_id="maps_id", component_property="figure"),
    Output(component_id="histo_id", component_property="figure"),
    Input(my_items.mybuttons_dem_diff, "value"),
)
def update_maps(chosen_diff):
    """
    Update maps in tab altidiff
    """

    fig_alt = go.Figure()
    fig_alt.add_trace(my_items.gironde_ref_heatmap)

    if chosen_diff == "final_dem_diff":
        fig_alt.add_trace(my_items.init_alt_heatmap)
        data_histo = my_items.img_diff_init.read(1).ravel()
        title_fig = "with"

    else:
        fig_alt.add_trace(my_items.final_alt_heatmap)
        data_histo = my_items.img_diff_final.read(1).ravel()
        title_fig = "without"

    fig_histo = px.histogram(
        data_histo,
        range_x=[np.nanmin(data_histo), np.nanmax(data_histo)],
        histnorm="probability density",
        title=f"Elevation difference histogram on all pixels {title_fig} coregistration",

    )
    fig_histo.update_layout(xaxis_title="Elevation difference (m) from -|p98| to |p98|")

    # the figure/plot created using the data filtered above
    fig_alt.update_layout(
        yaxis={"autorange": "reversed"}, width=500, height=500,
        title=f"Altitudes differences {title_fig} coregistration",
        coloraxis={'colorscale': 'BrBG_r'}
    )

    return fig_alt, fig_histo


@app.callback(
    Output(component_id="dems", component_property="figure"),
    Input(my_items.mybuttons_dem, "value"),
)
def update_dems(chosen_dem):
    """
    Update table dem first dems
    """
    fig_dems = go.Figure()
    fig_dems.add_trace(my_items.gironde_ref_heatmap)
    fig_dems.add_trace(my_items.gironde_sec_heatmap)

    if chosen_dem == "coregistered dem":
        fig_dems.add_trace(my_items.gironde_sec_coreg_heatmap)

    fig_dems.update_layout(
        title=f"Reprojected second {chosen_dem} on reference dem",
        yaxis={"autorange": "reversed"}, width=500, height=500,
        coloraxis={'colorscale':'BrBG_r'}
    )

    return fig_dems

@app.callback(
    Output(component_id="dems_origin", component_property="figure"),
    Input(my_items.mybutton_dem_origin, "value"),
)
def update_dems_origin(chosen_dem):
    """
    Update table dem second dems
    """
    fig_dems = go.Figure()
    fig_dems.add_trace(my_items.gironde_ref_heatmap)

    if chosen_dem == "Reproj sec":
        fig_dems.add_trace(my_items.gironde_sec_heatmap)

    fig_dems.update_layout(
        title=f"Reprojected second {chosen_dem} on reference dem",
        yaxis={"autorange": "reversed"}, width=500, height=500,
        coloraxis={'colorscale': 'BrBG_r'}
    )

    return fig_dems


@app.callback(
    Output(component_id="dems_coreg", component_property="figure"),
    Input(my_items.mybutton_dem_coreg, "value"),
)
def update_dems_coreg(chosen_dem):
    """
    Update table dem third dems
    """
    fig_dems = go.Figure()
    fig_dems.add_trace(my_items.gironde_ref_coreg_heatmap)

    if chosen_dem == "Reproj coreg sec":
        fig_dems.add_trace(my_items.gironde_sec_coreg_heatmap)

    fig_dems.update_layout(
        title=f"Reprojected second {chosen_dem} on reference dem",
        yaxis={"autorange": "reversed"}, width=500, height=500,
        coloraxis={'colorscale': 'BrBG_r'}
    )

    return fig_dems


# Run app
if __name__ == "__main__":
    app.run_server()
