.. _tuto_new_metric:

New metric implementation
=========================

Demcompare's architecture allows to easily implement a **new metric computation**.

To do so, a new class has to be implemented within *demcompare/scalar_metrics.py* or *demcompare/vector_metrics.py* file, according to
the new metric's structure.

The new metric class inherits from the **MetricTemplate** class and must implement the following functions:


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

