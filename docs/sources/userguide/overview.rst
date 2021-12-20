Overviews
=========

Diagram
*******

The following interactive diagram highlights all coregistration steps avalaible in Demcompare.

Configuration file
******************

The configuration file provides a list of parameters to Demcompare so that the processing pipeline can
run according to the parameters choosen by the user.

Demcompare works with JSON formatted data.

All configuration parameters are described in :ref:`inputs` and :ref:`step_by_step` chapters.
As shown on the diagram, stereo steps must respect on order of priority, and can be called multiple times as explain on :ref:`sequencing` chapter.

Example
*******

1. Install

.. code-block:: bash

    pip install pandora[sgm]

2. Run demcompare with the example configuration file

Run the python script **demcompare** with a json configuration file as unique
argument (see [`tests/test_config.json`](./tests/test_config.json) as an example):
.. code-block:: bash
    cd tests/
    demcompare test_config.json



demcompare can be launched with a file containing its parameters (one per line) with "@" character:
.. code-block:: bash
demcompare @opts.txt


`opts.txt` example file:
.. code-block:: bash
    test_config.json
    --display

3. Visualize results

The results can be observed with:
.. code-block:: bash
    firefox test_output/doc/published_report/html/demcompare_report.html &
