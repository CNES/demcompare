.. _stats_modules:

Stats module
============

This section explains details of stats module conception and organization. 

In the following image we can find the classes that take part in demcompare's statistics computation step, along
with their relationship.

Statistics step architecture
----------------------------

The `stats_processing_class`_ module handles the API for the statistics computation. The :ref:`demcompare_module`
creates a `stats_processing_class`_ object when statistics are to be computed. The `stats_processing_class`_ object contains
different `classification_layer_class`_ objects, depending on the classification layers specified in the input configuration file.
Moreover, each `classification_layer_class`_ contains its own `metric_class`_ classes.

When statistics are computed, a `stats_dataset_class`_ object is obtained.


.. figure:: /images/schema_statistiques_class.png
    :width: 1100px
    :align: center

    Statistics classes relationship.


Stats processing
****************

.. _stats_processing_class:


**StatsProcessing**: Implemented in `StatsProcessing file <https://github.com/CNES/demcompare/blob/master/demcompare/stats_processing.py>`_

The `stats_processing_class`_ class handles the statistics computation for an input dem. Please notice that if the statistics
are to be computed on the difference between two dems, this difference must be given as the input dem.

The `stats_processing_class`_ class generates the different `classification_layer_class`_ objects to handle the statistics computation by class, and it
also generates the `stats_dataset_class`_ output object. It also has the API to compute the different available statistics on a chosen classification
layer and class.

As one can see in :ref:`demcompare_module`, the main demcompare module in __init__.py file uses the `stats_processing_class`_
class to perform the stats computation.


One can find here the full list of API functions available in the `stats_processing_class`_ module, as well as their description and
input and output parameters:
`StatsProcessing API <https://demcompare.readthedocs.io/en/latest/api_reference/demcompare/stats_processing/index.html>`_


Classification layer
********************

.. _classification_layer_class:


The **Classification Layer** class in demcompare is in charge of classifying the input DEM's pixels by classes and
obtains statistics by class.

All Classification Layer classes inherit from the **ClassificationLayerTemplate** abstract class. Currently, *segmentation*, *global*, *slope* and *fusion*
classification layers are available. For more details on the pixel classification of each classification layer type please see :ref:`statistics` :

- **SegmentationClassification**: Segmentation classification layer class. Implemented in `SegmentationClassification file <https://github.com/CNES/demcompare/blob/master/demcompare/classification_layer/segmentation_classification.py>`_

- **GlobalClassification**: Global classification layer class. Implemented in `GlobalClassification file <https://github.com/CNES/demcompare/blob/master/demcompare/classification_layer/global_classification.py>`_

- **SlopeClassification**: Slope classification layer class. Implemented in `SlopeClassification file <https://github.com/CNES/demcompare/blob/master/demcompare/classification_layer/slope_classification.py>`_

- **FusionClassification**: Fusion classification layer class. Implemented in `FusionClassification file <https://github.com/CNES/demcompare/blob/master/demcompare/classification_layer/fusion_classification.py>`_

Whereas the abstract class and the class Factory are implemented in :

- **ClassificationLayer**: The class Factory. Implemented in `ClassificationLayer file <https://github.com/CNES/demcompare/blob/master/demcompare/classification_layer/classification_layer.py>`_


- **ClassificationLayerTemplate**: The abstract class. Implemented in `ClassificationLayerTemplate file <https://github.com/CNES/demcompare/blob/master/demcompare/classification_layer/classification_layer_template.py>`_

Each classification layer contains the input DEM classified according to the classification layer type and inputs (ie. a segmentation map for SegmentationClassification, a slope range for SlopeClassification), and handles the statistics computation with the *compute_classif_stats* function.

To perform the metric computation, the `classification_layer_class`_ class creates each `metric_class`_ :ref:`statistics` object.

The computed metrics are stored in the input `stats_dataset_class`_ object and returned to the `stats_processing_class`_ module, which handles the API for statistics computation :ref:`statistics`.

One can find here the full list of API functions available in the `classification_layer_class`_ module, as well as their description and
input and output parameters: `ClassificationLayer API <https://demcompare.readthedocs.io/en/latest/api_reference/demcompare/classification_layer/classification_layer_template/index.html>`_


Metric
******

.. _metric_class:


The **Metric** class in demcompare is in charge of doing a statistics computation on a given *np.ndarray*.
All `metric_class`_ classes inherit from the **MetricTemplate** abstract class:

- **Metric**: The class Factory. Implemented in `Metric file <https://github.com/CNES/demcompare/blob/master/demcompare/metric/metric.py>`_
- **MetricTemplate**: The abstract class. Implemented in `MetricTemplate file <https://github.com/CNES/demcompare/blob/master/demcompare/metric/metric_template.py>`_

To avoid too many python files creation, and given the simplicity of some of the metric classes, they have been
grouped by type in *scalar_metrics.py* and *vector_metrics.py* :

- Metric classes implemented in `Scalar metrics file <https://github.com/CNES/demcompare/blob/master/demcompare/metric/scalar_metrics.py>`_

    - **Mean**
    - **Max**
    - **Min**
    - **Std**
    - **Rmse**
    - **Median**
    - **Nmad**
    - **Sum**
    - **Squared_sum**
    - **Percentil90**

Each scalar metric computes a scalar value based on the input data.

- Metric classes implemented in `Vector metrics file <https://github.com/CNES/demcompare/blob/master/demcompare/metric/vector_metrics.py>`_

    - **Cdf (Cumulative Distribution Function)**
    - **Pdf (Probability Density Function)**
    - **RatioAboveThreshold**

Each vector metric computes two arrays of values based on the input data.

For information on how to create a new metric, please see :ref:`tuto_new_metric`.

One can find here the full list of API functions available in the `classification_layer_class`_ module, as well as their description and
input and output parameters:
`Metric API <https://demcompare.readthedocs.io/en/latest/api_reference/demcompare/classification_layer/classification_layer_template/index.html>`_

Stats dataset
*************

.. _stats_dataset_class:

**StatsDataset**: Implemented in `StatsDataset file <https://github.com/CNES/demcompare/blob/master/demcompare/stats_dataset.py>`_

The `stats_dataset_class`_ stores the different statistics computed for an input DEM. It is generated by the `stats_processing_class`_ and its architecture
consists in a list of `xr.Dataset`, one for each `classification_layer_class`_ that has been used to compute the stats.
It also has the API to obtain the stored statistics.


The statistics of each classification layer are stored in the `xr.Dataset` with the following structure:

.. code-block:: text

    :image: 2D (row, col) input image as xarray.DataArray,

    :image_by_class: 3D (row, col, nb_classes)

        xarray.DataArray containing
        the image pixels belonging
        to each class considering the valid pixels

    :image_by_class_intersection: 3D (row, col, nb_classes)

        xarray.DataArray containing
        the image pixels belonging
        to each class considering the intersection mode

    :image_by_class_exclusion: 3D (row, col, nb_classes)

        xarray.DataArray containing
        the image pixels belonging
        to each class considering the exclusion mode

    :attributes:

                - name : name of the classification_layer. str

                - stats_by_class : dictionary containing
                  the stats per class considering the standard mode

                - stats_by_class_intersection : dictionary containing
                  the stats per class considering the intersection mode

                - stats_by_class_exclusion : dictionary containing
                  the stats per class considering the exclusion mode


One can find here the full list of API functions available in the `stats_dataset_class`_ module, as well as their description and
input and output parameters:
`StatsDataset API <https://demcompare.readthedocs.io/en/latest/api_reference/demcompare/stats_dataset/index.html>`_