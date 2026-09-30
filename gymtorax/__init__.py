from importlib.metadata import PackageNotFoundError, version

from gymnasium.envs.registration import register

from .agents import IterHybridAgent, PIDAgent, RandomAgent
from .envs import BaseEnv, IterHybridAuditedEnv, IterHybridEnv, TestEnv

try:
    __version__ = version("gymtorax")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "unknown"

# Register environments with Gymnasium
register(
    id="gymtorax/IterHybrid-v0",
    entry_point="gymtorax.envs:IterHybridEnv",
    kwargs={},
)

register(
    id="gymtorax/IterHybridAudited-v0",
    entry_point="gymtorax.envs:IterHybridAuditedEnv",
    kwargs={},
)

register(
    id="gymtorax/Test-v0",
    entry_point="gymtorax.envs:TestEnv",
    kwargs={},
)

__all__ = [
    "BaseEnv",
    "IterHybridEnv",
    "IterHybridAuditedEnv",
    "TestEnv",
    "PIDAgent",
    "IterHybridAgent",
    "RandomAgent",
]
