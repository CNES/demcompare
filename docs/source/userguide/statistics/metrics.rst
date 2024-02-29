.. _metrics:

Metrics
=======

The following metrics are currently available on demcompare:

.. tabs::

  .. tab:: Scalar metrics

    - ``mean``
    - ``max``
    - ``min``
    - ``std`` (Standard Deviation)
    - ``rmse`` (Root Mean Squared Error)
    - ``median``
    - ``nmad`` (Normalized Median Absolute Deviation) = :math:`1.486*median(\lvert data - median(data)\rvert)`
    - ``sum``
    - ``squared_sum``
    - ``percentil_90``

  .. tab:: Vector metrics
      .. csv-table::
        :header: "Name", "Type", "Parameters", "Type", "Default value"
        :widths: auto
        :align: left

          ``'cdf'``\ Cumulative Density Function,vector,bin_step, "float", ``0.1``
          ,,output_csv_path, "string",``None``
          ,,output_plot_path, "string",``None``
          ``'pdf'``\ Probability Density Function,vector,bin_step, "float",``0.2``
          ,,width, "float",``0.7``
          ,,filter_p98, "boolean",``false``
          ,,output_csv_path, "string",``None``
          ,,output_plot_path, "string",``None``
          ``'ratio_above_threshold'``,vector,elevation_threshold, "List[float, int]", ":math:`[0.5, 1, 3]`"
          ,,original_unit, "string",``"m"``
          ,,output_csv_path, "string",``None``
          ``'slope-orientation-histogram'``,vector,output_plot_path, "string",``None``

  .. tab:: Matrix 2D metrics
      .. csv-table::
        :header: "Name", "Type", "Parameters", "Type", "Default value"
        :widths: auto
        :align: left

        ``'hillshade'``\ Hill shade,matrix,azimuth, "float", ``0.9``
        ,,angle_altitude, "float", ``45``
        ,,cmap, "str", ``Greys_r``
        ,,cmap_nodata, "str", ``royalblue``
        ,,colorbar_title, "str", ``Hill shade``
        ,,fig_title, "str", ``DEM hill shade``
        ,,plot_path, "str", ``None``
        ``'svf'``\ SkyViewFactor,matrix,filter_intensity, "float", ``315``
        ,,replication, "bool", true
        ,,quantiles, "List[float]", ":math:`[0.09, 0.91]`"
        ,,cmap, "str", ``Greys_r``
        ,,cmap_nodata, "str", ``royalblue``
        ,,colorbar_title, "str", ``Sky view factor``
        ,,fig_title, "str", ``DEM sky view factor``
        ,,plot_path, "str", ``None``

.. note::

    The metrics are always computed on **valid pixels**. Valid pixels are those whose value is different than NaN and the
    nodata value (-32768 by default if not specified in the input configuration or in the input DEM).

.. note::
    Apart from only considering the valid pixels, the user may also specify the ``remove_outliers`` option
    in the input configuration. This option will also **filter all DEM pixels outside (mu + 3 sigma) and (mu - 3 sigma)**,
    being *mu* the *mean* and *sigma* the *standard deviation* of all valid pixels in the DEM.

.. note::
    ``'ratio_above_threshold'`` and ``'slope-orientation-histogram'`` are not computed by default. They must be indicated in the configuration file in order to be used. An example on how to include them in the configuration is shown below.

.. note::
    ``'slope-orientation-histogram'`` should have the ``'output_plot_path'`` parameter specified, otherwise the plot will not be saved. An example on how to include it in the configuration is shown below.

.. note::

   More informations about the hillshade and the sky-view factor can be found in :ref:`hillshade_sky_view`.

.. code-block:: json

        {
            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "nodata": -9999.0
            },
            "input_sec": {
                "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
                "nodata": -32768
            }
            "statistics": {
                "alti-diff": {
                    "remove_outliers": true,
                    "metrics": ["mean", {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}}]
                },
                "ref": {
                    "remove_outliers": true,
                    "metrics": [
                        {
                            "slope-orientation-histogram": {
                                "output_plot_path": "path_to_plot"
                            }
                        }
                    ]
                }
            }
        }

.. note::
    ``'slope-orientation-histogram'``'s plots are not saved in the report.

.. warning::
   The combination of **DEM processing methods** and **metrics** may not be meaningful! 

See also:

.. toctree::
   :maxdepth: 4

   hillshade_sky_view.rst
   quality_measures.rst