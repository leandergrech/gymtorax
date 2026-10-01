# Baselines on `IterHybrid-v0` and `IterHybridAudited-v0`

Undiscounted episode returns (gamma = 1). One episode per deterministic policy; the random policy (uniform samples of the action space, `env.action_space.seed(seed)` and `env.reset(seed=seed)` for seeds 0-19) is reported as mean ± standard deviation over 20 episodes, with the number of episodes that ended with the -1000 failure reward. "Heating off from t = 105 s" is the open-loop reference with NBI and ECRH power set to zero from step 105 (`examples/reward_exploit.py`).

## gymtorax 1.0.0, TORAX 1.0.3 (the versions of the published results)

Branch `fix/audited-iter-hybrid-reward-v1.0` (this patch cherry-picked onto tag `v1.0.0`), dependencies from its `poetry.lock` (TORAX 1.0.3, JAX 0.7.1), Python 3.12.3.

| Policy | IterHybrid-v0 | IterHybridAudited-v0 |
|---|---|---|
| PI controller (k_p = 0.700, k_i = 34.257) | 3.79 | 3.50 |
| Open-loop reference | 3.41 | 3.41 |
| Random (20 seeds) | 3.23 ± 0.06 (0/20 failed) | 2.16 ± 0.06 (0/20 failed) |
| Open-loop, NBI and ECRH off from t = 105 s | 22.74 | 1.91 |
| TD3+BC trained offline on noisy PI logs ¹ | 4.01 | 3.56 |
| MBPO, full observation, seed 1, final policy ¹ | 5.63 | 3.69 |

The paper reports PI 3.79, open-loop 3.40 and random -10.79 (arXiv:2510.11283). The PI and open-loop values reproduce; the random value does not (no failure in 20 episodes here).

## gymtorax 1.1.1 (main), TORAX 1.4.2

Branch `fix/audited-iter-hybrid-reward`, dependencies from its `poetry.lock` (TORAX 1.4.2), Python 3.12.3.

| Policy | IterHybrid-v0 | IterHybridAudited-v0 |
|---|---|---|
| PI controller (k_p = 0.700, k_i = 34.257) | -998.67 (failed) | -998.67 (failed) |
| Open-loop reference | 3.26 | 3.18 |
| Random (20 seeds) | 3.15 ± 0.06 (0/20 failed) | 2.09 ± 0.05 (0/20 failed) |
| Open-loop, NBI and ECRH off from t = 105 s | 20.96 | 1.83 |

With TORAX 1.4 the PI gains optimized for v1.0 leave the observation bounds at step 105 (the failure reward is returned); the documentation already notes that the v1.0 results and gains are out of date for v1.1.

¹ Learned policies from <https://github.com/leandergrech/rl-tokamak-rampup> (trained on gymtorax 1.0.0 through a flat-vector wrapper). They were not re-trained for TORAX 1.4 and are only scored at the 1.0.0 pin. The MBPO policy was trained on the `IterHybrid-v0` reward.

## Regenerate

```bash
# gymtorax 1.0.0 / TORAX 1.0.3
git checkout fix/audited-iter-hybrid-reward-v1.0 && poetry install --with dev
poetry run python examples/reward_exploit.py
poetry run python examples/baselines_audited.py --seeds 20 --workers 8

# gymtorax 1.1.1 / TORAX 1.4.2
git checkout fix/audited-iter-hybrid-reward && poetry install --with dev
poetry run python examples/reward_exploit.py
poetry run python examples/baselines_audited.py --seeds 20 --workers 8

# learned policies (1.0.0 environment above, plus torch and the rl-tokamak-rampup package)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-deps -e /path/to/rl-tokamak-rampup
python /path/to/rl-tokamak-rampup/scripts/score_upstream_envs.py
```

Wall time of `baselines_audited.py` with 8 worker processes on a shared 16-thread laptop CPU (load average 25-35 from other jobs): 28.4 min for 1.0.0, 13.0 min for 1.1.1.
