Tutorial
========

TODO: Provide edgelists for tutorial.

First, import :code:`uclasm`, the subgraph matching library.

.. code-block:: python

    >>> import uclasm

Load your template and world graphs from edgelist files.

.. code-block:: python

    >>> tmplt = uclasm.load_edgelist("template.csv",
    ...                              file_source_col="Source",
    ...                              file_target_col="Target",
    ...                              file_channel_col="eType")

    >>> world = uclasm.load_edgelist("world.csv",
    ...                              file_source_col="Source",
    ...                              file_target_col="Target",
    ...                              file_channel_col="eType")

Create a matching problem to search for instances of the template in the world.

.. code-block:: python

    >>> smp = uclasm.MatchingProblem(tmplt, world)

Run various filters to reduce the search space.

.. code-block:: python

    >>> smp.stats_filter()

View the state of the matching problem.

.. code-block:: python

    >>> print(smp)
    t1 has 2 candidates: w2, w4
    1 template nodes have 1 candidate: t2
