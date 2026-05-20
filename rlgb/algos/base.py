"""Base protocol for RL agents.

All five algorithm families implement this interface:

  A  NodeMove   – NeuroCUT / M1-policy (GraphSAGE + REINFORCE)
  B  Structured – WRT (Transformer + PPO + ring/wedge actions)
  C  CommunityRW – CLARE / SLRL (GIN + REINFORCE)
  D  DynamicAC  – AC2CD (GAT + A2C)
  E  Multicut   – SS2V-D3QN (subgraph NN + D3QN)

Each algo is self-contained: it owns the policy network, optimizer,
replay buffer (if needed), and implements the Trainer-facing methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Transition:
    """Single environment transition for replay / on-policy buffer."""
    obs: dict[str, np.ndarray]
    action: int | np.ndarray
    reward: float
    next_obs: dict[str, np.ndarray]
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class RLAgent(ABC):
    """Abstract base for all RL agents.

    The Trainer calls only these methods, so every algo family is
    plug-and-play with the same training loop.
    """

    # Subclass must set these
    name: str = "base"
    compatible_tasks: list[str] = []  # "partition" | "community_expand" | "dynamic_cd"

    @abstractmethod
    def select_action(
        self, obs: dict[str, np.ndarray], greedy: bool = False
    ) -> int | np.ndarray:
        """Sample an action from the policy given the current observation."""
        ...

    @abstractmethod
    def push_transition(self, t: Transition) -> None:
        """Store a transition in the internal buffer."""
        ...

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Perform one gradient update step. Returns metrics dict."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialise policy weights and config to disk."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore weights from a checkpoint."""
        ...

    def reset_episode(self) -> None:
        """Called at the start of each episode (optional hook)."""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called after each epoch (optional hook for LR scheduling etc.)."""
        pass

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        try:
            return next(self._policy.parameters()).device  # type: ignore[attr-defined]
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def to(self, device: str | torch.device) -> "RLAgent":
        try:
            self._policy.to(device)  # type: ignore[attr-defined]
        except AttributeError:
            pass
        return self


class EpisodeBuffer:
    """Simple on-policy episode buffer (used by REINFORCE-style agents)."""

    def __init__(self) -> None:
        self._transitions: list[Transition] = []

    def push(self, t: Transition) -> None:
        self._transitions.append(t)

    def drain(self) -> list[Transition]:
        batch = self._transitions
        self._transitions = []
        return batch

    def __len__(self) -> int:
        return len(self._transitions)


class ReplayBuffer:
    """Fixed-capacity circular replay buffer (used by DQN-style agents)."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._cap = capacity
        self._buf: list[Transition] = []
        self._pos = 0

    def push(self, t: Transition) -> None:
        if len(self._buf) < self._cap:
            self._buf.append(t)
        else:
            self._buf[self._pos] = t
        self._pos = (self._pos + 1) % self._cap

    def sample(self, n: int, rng: np.random.Generator) -> list[Transition]:
        idx = rng.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
        return [self._buf[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self._buf)
