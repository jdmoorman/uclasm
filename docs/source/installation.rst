Installation
============

Installation from PyPI
----------------------

.. code-block:: bash

  $ pip install ucla-subgraph-matching

Development Installation
------------------------

You will need git_ and poetry_ to install for local development.

First, clone the repository using :code:`git`.

.. code-block:: bash

  $ git clone git@github.com:jdmoorman/uclasm.git
  $ cd uclasm

Now, install the project using :code:`poetry` and check that all tests are passing on your machine.

.. code-block:: bash

  $ poetry install
  $ poetry run pytest tests/

.. _git: https://git-scm.com/
.. _poetry: https://python-poetry.org/
