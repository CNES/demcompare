Overviews
=========

Configuration file
******************

The configuration file provides a list of parameters to Demcompare so that the processing pipeline can
run according to the parameters choosen by the user.

Demcompare works with JSON formatted data.

All configuration parameters are described in :ref:`inputs` and :ref:`step_by_step` chapters.

Example
*******

1. Install

For information about Demcompare's installation please refer to: :ref:`install`

2. Run demcompare with the example configuration file

Run the python script **demcompare** with a json configuration file as unique
argument (see *[`tests/test_config.json`](./tests/test_config.json)* as an example):

.. code-block:: bash

    cd tests/
    demcompare test_config.json


3. Visualize results

The results can be observed with:

.. code-block:: bash

    firefox test_output/doc/published_report/html/demcompare_report.html &
