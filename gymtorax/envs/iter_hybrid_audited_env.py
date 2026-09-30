"""ITER hybrid scenario environment with an audited reward.

The configuration, action space, observation space and time discretization are
those of :class:`IterHybridEnv`. Only the reward differs, in two places:

1. The fusion-gain term is capped at the ITER target (``min(Q / 10, 1)``), like
   the H98, q_min and q95 terms already are.
2. A step counts as H-mode only if the core-temperature condition holds *and*
   the power crossing the separatrix exceeds the L-H threshold
   (``P_SOL_total >= P_LH``, both as computed by TORAX).

In ``IterHybridEnv`` the pedestal is prescribed in time, so the temperature
condition stays true after the auxiliary heating is switched off, and the
uncapped fusion-gain term then grows as Q = P_fusion / P_aux diverges. The two
changes remove that incentive without changing the reward of trajectories that
keep Q <= 10 and P_SOL >= P_LH.
"""

from .. import rewards as reward
from .iter_hybrid_env import IterHybridEnv


class IterHybridAuditedEnv(IterHybridEnv):
    """ITER hybrid scenario with a capped fusion-gain term and an L-H power check.

    See the module docstring for the two differences with :class:`IterHybridEnv`.
    All initialization parameters are passed to ``IterHybridEnv.__init__()``.

    Attributes:
        fusion_gain_cap (float): Upper bound on the normalized fusion gain
            ``Q / 10`` used in the reward. Defaults to ``1.0`` (Q = 10, the ITER
            target).
    """

    fusion_gain_cap = 1.0

    def _is_H_mode(self, next_state):  # noqa: N802
        """Return whether the plasma is counted as being in H-mode.

        Args:
            next_state (dict[str, Any]): state at time :math:`t+1`

        Returns:
            bool: ``True`` if the core is hot and ``P_SOL >= P_LH``.
        """
        hot_core = (
            next_state["profiles"]["T_e"][0] > 10
            and next_state["profiles"]["T_i"][0] > 10
        )
        # H-mode is only sustained while the power crossing the separatrix
        # exceeds the L-H threshold power.
        above_threshold = reward.get_P_SOL(next_state) >= reward.get_P_LH(next_state)
        return bool(hot_core and above_threshold)

    def _compute_reward(self, state, next_state, action):
        r"""Compute the audited reward.

        Same weighted sum as :meth:`IterHybridEnv._compute_reward`, with the
        fusion-gain term capped at ``fusion_gain_cap`` and H-mode detected by
        :meth:`_is_H_mode`.

        Args:
            state (dict[str, Any]): state at time :math:`t`
            next_state (dict[str, Any]): state at time :math:`t+1`
            action (numpy.ndarray): applied action at time :math:`t`

        Returns:
            float: reward associated to the transition (state, action, next_state)
        """
        weight_list = [1, 1, 1, 1]
        h_mode = self._is_H_mode(next_state)

        def _r_fusion_gain():
            # Q = P_fusion / P_aux diverges as P_aux -> 0, so it is capped like
            # the other terms.
            fusion_gain = min(
                reward.get_fusion_gain(next_state) / 10, self.fusion_gain_cap
            )  # Normalize with ITER target
            if h_mode:
                return fusion_gain
            else:
                return 0

        def _r_h98():
            h98 = reward.get_h98(next_state)
            if h_mode:
                if h98 <= 1:
                    return h98
                else:
                    return 1
            else:
                return 0

        def _r_q_min():
            q_min = reward.get_q_min(next_state)
            if q_min <= 1:
                return q_min
            elif q_min > 1:
                return 1

        def _r_q_95():
            q_95 = reward.get_q95(next_state)
            if q_95 / 3 <= 1:
                return q_95 / 3
            else:
                return 1

        # Calculate individual reward components
        r_fusion_gain = weight_list[0] * _r_fusion_gain() / 50
        r_h98 = weight_list[1] * _r_h98() / 50
        r_q_min = weight_list[2] * _r_q_min() / 150
        r_q_95 = weight_list[3] * _r_q_95() / 150

        total_reward = r_fusion_gain + r_h98 + r_q_min + r_q_95

        # Store reward breakdown for logging (attach to environment if it has reward_breakdown attribute)
        if hasattr(self, "reward_breakdown"):
            if not hasattr(self, "_reward_components"):
                self._reward_components = {
                    "fusion_gain": [],
                    "h98": [],
                    "q_min": [],
                    "q_edge": [],
                }

            self._reward_components["fusion_gain"].append(r_fusion_gain)
            self._reward_components["h98"].append(r_h98)
            self._reward_components["q_min"].append(r_q_min)
            self._reward_components["q_edge"].append(r_q_95)

        return total_reward
