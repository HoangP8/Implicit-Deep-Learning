SIM (State-driven Implicit Models)
==================================

.. autoclass:: idl.sim.sim.SIM
   :members:
   :show-inheritance:
   :special-members: __call__

After defining a SIM model, you need to create a solver to solve the convex optimization problem. Several solvers are already provided in the following sections. However, you can also implement your own solver by inheriting from the :ref:`solver` class.

.. toctree::
   :maxdepth: 2
   :caption: Components:

   solvers/solver
   solvers/admm
   solvers/cvx
   solvers/ls
   solvers/gd