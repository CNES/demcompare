.. _demcompare_cli:

Demcompare CLI
===============

This section describes the demcompare CLI section. 

The user can execute demcompare from the command line using an input configuration file. Different logging outputs can be choosen (ie. INFO or DEBUG modes)
giving the user different informations during the execution.

To have details on the input configuration file, see :ref:`command_line_execution`.

The demcompare CLI module is in demcompare.py file and handles argparse conversion to the :ref:`demcompare_module`, which will orchestrate the demcompare
API from the input configuration file.

- **demcompare.py** `demcompare CLI file <https://github.com/CNES/demcompare/blob/master/demcompare/demcompare.py>`_

This python file includes demcompare's main and input parser.

