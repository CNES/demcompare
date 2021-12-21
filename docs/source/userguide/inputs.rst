.. _inputs:

Inputs
======

Configuration parameters
************************


Here is the list of the parameters and the associated default value when it exists:

.. sourcecode:: text

    {
        "outputDir" : "./test_output/"
        "inputDSM" : {  "path", "./inputDSM.tif"
                        "zunit" : "meter",
                        "georef" : "WGS84",
                        "nodata" : }
        "inputRef" : {  "path", "inputRef.tif"
                        "zunit" : "meter",
                        "georef" : "WGS84",
                        "nodata" : }}
        "plani_opts" : {    "coregistration_method" : "nuth_kaab",
                            "disp_init" : {"x": 0, "y": 0}},
        "stats_opts" : {    "elevation_thresholds" : {"list": [0.5,1,3], "zunit": "meter"},
                            "remove_outliers": False,
                            "to_be_classification_layers": {"slope": {"ranges": [0, 10, 25, 50, 90],
                                                                      "ref": None,
                                                                      "dsm": None}},
                            "classification_layers": {}
    }

Where a valid `classification_layers` value could be:

.. sourcecode:: text

                            "classification_layers": {"land_cover": {"ref": 'None_or_path_to_land_cover_associated_with_the_ref',
                                                                     "dsm": 'None_or_path_to_land_cover_associated_with_the_dsm',
                                                                     "classes": {"forest": [31, 32], "urbain": [42]}}}
    }


If DEMs altitudes are to rely on **geoid**, configurations could be:

.. sourcecode:: text

    "inputDSM" : {  "path", "./inputDSM.tif"
                            "zunit" : "meter",
                            "georef" : "geoid",
                            "nodata" : }

In this case, **EGM96 geoid** will be used by default.

Otherwise, the absolute path to a locally available geoid model can be given, for instance:

.. sourcecode:: text

    "inputDSM" : {  "path", "./inputDSM.tif"
                            "zunit" : "meter",
                            "georef" : "geoid",
                            "geoid_path": "path/to/egm08_25.gtx"
                            "nodata" : }


