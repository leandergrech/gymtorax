"""Tests for the audited ITER hybrid reward (``gymtorax/IterHybridAudited-v0``).

The unit tests call the reward functions on synthetic states and do not run
TORAX. The episode tests run a few full ITER hybrid episodes; the ones that
compare with the numbers published for Gym-TORAX 1.0 only run with TORAX 1.0.x,
the version those numbers were produced with.
"""

import os
from importlib.metadata import version

import gymnasium as gym
import numpy as np
import pytest

# Set matplotlib to a headless backend for CI environments. This must run before
# importing gymtorax, which imports matplotlib.pyplot at load time.
if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
    import matplotlib

    matplotlib.use("Agg")

import gymtorax  # noqa: F401  (import registers the environments with Gymnasium)
from gymtorax import IterHybridAgent, PIDAgent
from gymtorax.envs import IterHybridAuditedEnv, IterHybridEnv

TORAX_1_0 = version("torax").startswith("1.0.")
PAPER_PI_RETURN = 3.79  # Mouchamps et al., arXiv:2510.11283, Gym-TORAX 1.0 / TORAX 1.0
AUDITED_PI_RETURN = 3.50  # audited reward of the same PI episode (TORAX 1.0.3)
CUT_TIME = 105  # [s] heating is switched off from here in the exploit sequence


def _state(q=5.0, h98=0.9, q_min=1.2, q95=4.0, te0=12.0, ti0=12.0, p_sol=8e7, p_lh=6e7):
    return {
        "profiles": {"T_e": np.array([te0, 1.0]), "T_i": np.array([ti0, 1.0])},
        "scalars": {
            "Q_fusion": np.array([q]),
            "H98": np.array([h98]),
            "q_min": np.array([q_min]),
            "q95": np.array([q95]),
            "P_SOL_total": np.array([p_sol]),
            "P_LH": np.array([p_lh]),
        },
    }


def _reward(cls, next_state):
    # The reward only reads ``next_state``; no simulator is needed.
    env = object.__new__(cls)
    return cls._compute_reward(env, None, next_state, None)


class _TemperatureOnlyHMode(IterHybridAuditedEnv):
    """Audited reward without the P_SOL >= P_LH condition (Q cap only)."""

    def _is_H_mode(self, next_state):  # noqa: N802
        return (
            next_state["profiles"]["T_e"][0] > 10
            and next_state["profiles"]["T_i"][0] > 10
        )


class _UncappedFusionGain(IterHybridAuditedEnv):
    """Audited reward without the fusion-gain cap (P_SOL gate only)."""

    fusion_gain_cap = float("inf")


class TestAuditedReward:
    """Reward-level tests on synthetic states."""

    def test_same_as_v0_below_cap_and_above_threshold(self):
        state = _state(q=5.0, p_sol=8e7, p_lh=6e7)
        assert _reward(IterHybridAuditedEnv, state) == pytest.approx(
            _reward(IterHybridEnv, state)
        )

    def test_same_as_v0_in_l_mode(self):
        state = _state(te0=5.0, ti0=5.0)
        assert _reward(IterHybridAuditedEnv, state) == pytest.approx(
            _reward(IterHybridEnv, state)
        )

    def test_fusion_gain_is_capped(self):
        state = _state(q=200.0, h98=0.9, q_min=1.2, q95=4.0)
        expected = 1 / 50 + 0.9 / 50 + 1 / 150 + 1 / 150
        assert _reward(IterHybridAuditedEnv, state) == pytest.approx(expected)
        assert _reward(IterHybridEnv, state) > expected

    def test_h_mode_requires_p_sol_above_p_lh(self):
        state = _state(q=5.0, p_sol=1e7, p_lh=6e7)
        assert _reward(IterHybridAuditedEnv, state) == pytest.approx(2 / 150)
        assert _reward(IterHybridEnv, state) > 2 / 150


class _HeatingCutAgent:
    """Open-loop reference policy with NBI and ECRH switched off from ``cut_time``."""

    def __init__(self, action_space, cut_time):
        self.agent = IterHybridAgent(action_space)
        self.cut_time = cut_time
        self.time = 0

    def act(self, observation):
        action = self.agent.act(observation)
        if self.cut_time is not None and self.time >= self.cut_time:
            action["NBI"][0] = 0.0
            action["ECRH"][0] = 0.0
        self.time += 1
        return action


def _run(env_id, make_agent):
    """Run one episode; return the environment's return and the visited states."""
    env = gym.make(env_id, log_level="critical")
    agent = make_agent(env)
    observation, _ = env.reset(seed=0)
    total, states = 0.0, []
    terminated = truncated = False
    while not (terminated or truncated):
        observation, r, terminated, truncated, _ = env.step(agent.act(observation))
        total += r
        states.append(env.unwrapped.state)
    env.close()
    return total, states


def _pi_agent(env):
    return PIDAgent(
        env.action_space,
        lambda t: 0.2e6 + 0.4e6 + 1.4e6 * t / 100,
        env.unwrapped.action_handler.get_actions()["Ip"].ramp_rate[0],
        kp=0.700,
        ki=34.257,
        kd=0.0,
    )


@pytest.fixture(scope="module")
def exploit_episode():
    return _run(
        "gymtorax/IterHybridAudited-v0",
        lambda env: _HeatingCutAgent(env.action_space, CUT_TIME),
    )


@pytest.fixture(scope="module")
def reference_episode():
    return _run(
        "gymtorax/IterHybridAudited-v0",
        lambda env: _HeatingCutAgent(env.action_space, None),
    )


def _score(cls, states):
    return sum(_reward(cls, s) for s in states)


class TestAuditedEpisodes:
    """Full-episode tests of the exploit and of the published baselines."""

    def test_exploit_inflates_v0_return(self, exploit_episode, reference_episode):
        _, exploit_states = exploit_episode
        _, reference_states = reference_episode
        assert _score(IterHybridEnv, exploit_states) > 2 * _score(
            IterHybridEnv, reference_states
        )

    def test_audited_env_scores_exploit_below_reference(
        self, exploit_episode, reference_episode
    ):
        exploit_return, _ = exploit_episode
        reference_return, _ = reference_episode
        assert exploit_return < reference_return

    def test_q_cap_alone_changes_exploit_score(self, exploit_episode):
        _, states = exploit_episode
        assert _score(_TemperatureOnlyHMode, states) < _score(IterHybridEnv, states) - 1

    def test_p_sol_gate_alone_changes_exploit_score(self, exploit_episode):
        _, states = exploit_episode
        assert _score(_UncappedFusionGain, states) < _score(IterHybridEnv, states) - 1

    @pytest.mark.skipif(
        not TORAX_1_0, reason="paper numbers were produced with TORAX 1.0"
    )
    def test_v0_pi_return_matches_paper(self):
        pi_return, _ = _run("gymtorax/IterHybrid-v0", _pi_agent)
        assert pi_return == pytest.approx(PAPER_PI_RETURN, abs=0.01)

    @pytest.mark.skipif(
        not TORAX_1_0, reason="paper numbers were produced with TORAX 1.0"
    )
    def test_audited_pi_return(self, exploit_episode):
        pi_return, _ = _run("gymtorax/IterHybridAudited-v0", _pi_agent)
        assert pi_return == pytest.approx(AUDITED_PI_RETURN, abs=0.1)
        exploit_return, _ = exploit_episode
        assert exploit_return < pi_return
