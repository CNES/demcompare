.. _demcompare_cli:

Demcompare CLI
===============

Demcompare command line is the main tool entrance. 

The user can execute *demcompare* command line using an input configuration file.

Different logging outputs can be chosen (ie. INFO or DEBUG modes)
to give the user different information during the execution.

To have details on the input configuration file, see :ref:`command_line_execution`.

The demcompare CLI module is in <https://github.com/CNES/demcompare/blob/master/demcompare/demcompare.py>`_ file
and handles argparse conversion to the :ref:`demcompare_module`,
which will orchestrate the demcompare API from the input configuration file.

This python file includes demcompare's main and input parser.

