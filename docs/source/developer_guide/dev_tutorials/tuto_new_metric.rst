.. _tuto_new_metric:

New metric implementation
=========================

Demcompare's architecture allows to easily implement a **new metric computation**.

To do so, a new class has to be implemented within `demcompare/scalar_metrics.py <https://github.com/CNES/demcompare/blob/master/demcompare/metric/scalar_metrics.py>`_ or `demcompare/vector_metrics.py <https://github.com/CNES/demcompare/blob/master/demcompare/metric/vector_metrics.py>`_ file, according to
the new metric's structure (see :ref:`stats_modules`).


Basic metric structure and functions
************************************

The new metric class inherits from the **MetricTemplate** class and must implement the **compute_metric** function. This
function takes a *np.array* as an entry and performs the corresponding metric computation on the input array. The computed metric can be
a float (that would be a scalar metric), a *Tuple[np.array, np.array]* (vector metric), or a *np.ndarray* (pixellic metric, none
implemented yet).

One may also implement the **__init__** function of the new metric class, mostly if this metric contains class attributes (ie. the RatioAboveThreshold
metric contains the *elevation_thresholds* attribute).

Hence, a basic *NewMetricClass* would be implemented with the following structure :

.. code-block:: bash

    @Metric.register("new_metric_class")
    class NewMetricClass(MetricTemplate):

        # Optional, only needed if the metric object has its own parameters
        def __init__(self, parameters: Dict = None):
            """
            Initialization the metric object

            :param parameters: optional input parameters
            :type parameters: dict
            :return: None
            """

        def compute_metric(
            self, data: np.ndarray
        ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
            """
            Metric computation method

            :param data: input data to compute the metric
            :type data: np.array
            :return: the computed mean
            :rtype: float
            """

Advanced metric functions
*************************

It is to be noticed that more functionalities regarding the output visualisation can be implemented within a class Metric.
An example of such functionalities would be the *ProbabilityDensityFunction* implemented in `demcompare/vector_metrics.py <https://github.com/CNES/demcompare/blob/master/demcompare/metric/vector_metrics.py>`_.
This metric implements, in addition to the **__init__** and **compute_metrics** function, the **save_csv_metric** and **save_plot_metric**
functions. Those functions will be called if the input metric configuration contains the arguments *"output_plot_path"* or *"output_csv_path"*,
allowing the user to obtain plot and csv output files for better analyzing the computed metric (see :ref:`statistics`).