.. _dem_processing:

DEM processing module
=====================

**DemProcessing**: Implemented in `DemProcessing file <https://github.com/CNES/demcompare/blob/master/demcompare/dem_processing/dem_processing.py>`_

The `dem_processing`_ class creates the input needed to the :ref:`stats_processing_class`.

Several **DEM processing methods** can be used:

- DEM processing methods using only one DEM:

    - **ref**
    - **sec**
    - **ref-curvature**
    - **sec-curvature**

- DEM processing methods using two DEMs:

    - **alti-diff**
    - **alti-diff-slope-norm**
    - **angular-diff**

One can find here the full list of API functions available in the `dem_processing`_ module, as well as their description and
input and output parameters:
`DemProcessing API <https://demcompare.readthedocs.io/en/latest/api_reference/demcompare/dem_processing/index.html>`_
