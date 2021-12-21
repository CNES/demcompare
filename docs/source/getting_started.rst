Getting started
===============

Install
#######

For information about Demcompare's installation please refer to: :ref:`install`

First step
##########

Run the python script Demcompare with a json configuration file as unique
argument (see *tests/test_config.json* as an example):

.. code-block:: bash

    cd tests/
    demcompare test_config.json

The results can be observed with:

.. code-block:: bash

    firefox test_output/doc/published_report/html/demcompare_report.html &

Demcompare can be launched with a file containing its parameters (one per line) with "@" character:

.. code-block:: bash

    demcompare @opts.txt
    opts.txt example file:
    test_config.json
    --display

