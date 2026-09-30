"""Baseline returns on ``IterHybrid-v0`` and ``IterHybridAudited-v0``.

Runs the three policies of the Gym-TORAX example (PI controller, open-loop
reference, random policy) and the heating-cut sequence of
``examples/reward_exploit.py`` on both environments, and prints a Markdown table
of undiscounted returns (gamma = 1). The random policy is run for ``--seeds``
seeds; its mean, standard deviation and number of failed episodes (return
containing the -1000 failure reward) are reported.

Usage:
    python examples/baselines_audited.py --seeds 20 --workers 8
"""

import argparse
import json
import multiprocessing as mp
import time
from importlib.metadata import version

import numpy as np

ENV_IDS = ["gymtorax/IterHybrid-v0", "gymtorax/IterHybridAudited-v0"]
KP, KI = 0.700, 34.257  # PI gains of the Gym-TORAX example (optimized for v1.0)
CUT_TIME = 105

_ENVS = {}


def _env(env_id):
    import gymnasium as gym

    import gymtorax  # noqa: F401  (import registers the environments)

    if env_id not in _ENVS:
        _ENVS[env_id] = gym.make(env_id, log_level="critical")
    return _ENVS[env_id]


def _agent(env, policy, seed):
    from gymtorax import IterHybridAgent, PIDAgent, RandomAgent

    if policy == "pi":
        return PIDAgent(
            env.action_space,
            lambda t: 0.2e6 + 0.4e6 + 1.4e6 * t / 100,
            env.unwrapped.action_handler.get_actions()["Ip"].ramp_rate[0],
            kp=KP,
            ki=KI,
            kd=0.0,
        )
    if policy == "random":
        env.action_space.seed(seed)
        return RandomAgent(env.action_space)
    agent = IterHybridAgent(env.action_space)
    if policy == "open_loop":
        return agent

    class _HeatingCut:
        time = 0

        def act(self, observation):
            action = agent.act(observation)
            if self.time >= CUT_TIME:
                action["NBI"][0] = 0.0
                action["ECRH"][0] = 0.0
            self.time += 1
            return action

    return _HeatingCut()


def run_episode(task):
    """Run one episode of ``policy`` on ``env_id``; return its return and failure flag."""
    env_id, policy, seed = task
    env = _env(env_id)
    agent = _agent(env, policy, seed)
    observation, _ = env.reset(seed=seed)
    total, failed = 0.0, False
    terminated = truncated = False
    while not (terminated or truncated):
        observation, r, terminated, truncated, _ = env.step(agent.act(observation))
        total += r
        failed = failed or bool(r == -1000.0)
    return {
        "env_id": env_id,
        "policy": policy,
        "seed": seed,
        "return": float(total),
        "failed": failed,
    }


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20, help="random-policy seeds")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args()

    tasks = [(e, p, 0) for e in ENV_IDS for p in ("pi", "open_loop", "heating_cut")]
    tasks += [(e, "random", s) for e in ENV_IDS for s in range(args.seeds)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(run_episode, tasks, chunksize=1)
    minutes = (time.time() - t0) / 60

    def cell(env_id, policy):
        rows = [r for r in results if r["env_id"] == env_id and r["policy"] == policy]
        values = np.array([r["return"] for r in rows])
        n_failed = sum(r["failed"] for r in rows)
        if policy == "random":
            return f"{values.mean():.2f} ± {values.std():.2f} ({n_failed}/{len(rows)} failed)"
        return f"{values[0]:.2f}" + (" (failed)" if n_failed else "")

    labels = {
        "pi": f"PI controller (k_p = {KP}, k_i = {KI})",
        "open_loop": "Open-loop reference",
        "random": f"Random ({args.seeds} seeds)",
        "heating_cut": f"Open-loop, NBI and ECRH off from t = {CUT_TIME} s",
    }
    print(
        f"gymtorax {version('gymtorax')}, torax {version('torax')}, {minutes:.1f} min\n"
    )
    print("| Policy | IterHybrid-v0 | IterHybridAudited-v0 |")
    print("|---|---|---|")
    for policy in ("pi", "open_loop", "random", "heating_cut"):
        print(
            f"| {labels[policy]} | {cell(ENV_IDS[0], policy)} | {cell(ENV_IDS[1], policy)} |"
        )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "gymtorax": version("gymtorax"),
                    "torax": version("torax"),
                    "results": results,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
