Audited reward variant
======================

``gymtorax/IterHybridAudited-v0`` (class ``IterHybridAuditedEnv``) is the ITER hybrid
environment described in :doc:`iter_env` with the same configuration, actions,
observations and time discretization, and a reward that differs in two places.
``gymtorax/IterHybrid-v0`` is unchanged, so results published with it remain valid.

Why a variant
-------------

In ``IterHybrid-v0`` the fusion-gain term :math:`Q/10` is not capped, while the
H98, :math:`q_{min}` and :math:`q_{95}` terms are capped at 1. A step counts as
H-mode when :math:`T_e(0) > 10` keV and :math:`T_i(0) > 10` keV, and the pedestal is
prescribed in time (0.5 keV until 100 s, 3 keV from 105 s). Switching the NBI and
ECRH power off after 105 s therefore keeps the H-mode condition true while
:math:`Q = P_{fus}/P_{aux}` grows without bound. ``examples/reward_exploit.py``
reproduces this with the open-loop reference policy and the heating switched off
from 105 s.

Changes
-------

1. **Capped fusion gain**: the fusion-gain term is :math:`\min(Q/10, 1)`, like the
   other three terms. :math:`Q` diverges as the auxiliary power goes to zero.
2. **L-H power condition**: a step counts as H-mode only if the temperature condition
   holds *and* :math:`P_{SOL} \geq P_{LH}`, with ``P_SOL_total`` and ``P_LH`` taken
   from the TORAX outputs. H-mode is only sustained while the power crossing the
   separatrix exceeds the L-H threshold.

Trajectories with :math:`Q \leq 10` and :math:`P_{SOL} \geq P_{LH}` in H-mode receive
exactly the ``IterHybrid-v0`` reward.

Baselines
---------

Undiscounted returns (:math:`\gamma = 1`) of the policies of :doc:`policies` and of
the heating-cut sequence, from ``examples/baselines_audited.py`` (20 random seeds).

gymtorax 1.0.0 with TORAX 1.0.3 (the versions of the published results):

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Policy
     - ``IterHybrid-v0``
     - ``IterHybridAudited-v0``
   * - PI controller
     - 3.79
     - 3.50
   * - Open-loop reference
     - 3.41
     - 3.41
   * - Random (mean ± std, 20 seeds, no failures)
     - 3.23 ± 0.06
     - 2.16 ± 0.06
   * - Open-loop, NBI and ECRH off from 105 s
     - 22.74
     - 1.91

gymtorax 1.1.1 with TORAX 1.4.2:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Policy
     - ``IterHybrid-v0``
     - ``IterHybridAudited-v0``
   * - PI controller (v1.0 gains)
     - -998.67 (failure)
     - -998.67 (failure)
   * - Open-loop reference
     - 3.26
     - 3.18
   * - Random (mean ± std, 20 seeds, no failures)
     - 3.15 ± 0.06
     - 2.09 ± 0.05
   * - Open-loop, NBI and ECRH off from 105 s
     - 20.96
     - 1.83

The full table, including two learned policies, is in ``examples/baselines_audited.md``.

Regenerate with:

.. code-block:: bash

    python examples/reward_exploit.py
    python examples/baselines_audited.py --seeds 20 --workers 8
