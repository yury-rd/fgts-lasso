"""Bandit environments for the sparse-contextual-bandit benchmark.

Two environments:

* ``SyntheticEnv`` -- the standard sparse-linear-bandit setup used in the
  high-dimensional bandits literature (Bastani-Bayati 2020,
  Oh-Iyengar-Zeevi 2021, Ariu-Abe-Proutiere 2022).
* ``PowerSystemEnv`` -- a LinDistFlow-derived environment for the IEEE
  33-bus radial feeder. Each arm is a bus to be voltage-regulated; the
  ground-truth reward parameter is sparse on the path from the slack bus
  to that arm's bus (the natural electrical-distance support).

Both environments share the same stepping interface so the algorithms in
``algos.py`` are agnostic to the data source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from lindistflow import (
    R_tilde,
    BARAN_WU_LOADS_KW,
    BARAN_WU_LOADS_KVAR,
    path_to_root,
    S_BASE_MVA,
)


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class BanditEnv:
    """Base class. Subclasses must implement ``_sample_context`` and
    ``_reward(arm, x)`` and set ``self.K`` and ``self.d``."""

    K: int
    d: int
    theta_star: np.ndarray  # shape (K, d)
    supports: List[List[int]]

    def __init__(self, T: int):
        self.T = T
        self._rng: Optional[np.random.Generator] = None
        self._t = 0
        self._cum_pseudo_regret = 0.0
        self._regret_history: List[float] = []
        self._next_x: Optional[np.ndarray] = None

    # --- public interface --------------------------------------------------
    def reset(self, seed: int) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._cum_pseudo_regret = 0.0
        self._regret_history = []
        self._resample_problem()
        self._next_x = self._sample_context()
        return self._next_x.copy()

    def step(self, arm: int):
        x = self._next_x
        if x is None:
            raise RuntimeError("call reset() before step()")
        # observed (noisy) reward
        y_obs = self._reward(arm, x, with_noise=True)
        # pseudo-regret = expected reward gap
        opt_a = self.optimal_arm(x)
        pseudo_regret = self._reward(opt_a, x, with_noise=False) - self._reward(
            arm, x, with_noise=False
        )
        self._cum_pseudo_regret += float(pseudo_regret)
        self._regret_history.append(self._cum_pseudo_regret)
        info = {
            "t": self._t,
            "optimal_arm": int(opt_a),
            "pseudo_regret": float(pseudo_regret),
            "cum_pseudo_regret": float(self._cum_pseudo_regret),
        }
        info.update(self._extra_info(arm, x))
        self._t += 1
        # advance state
        self._next_x = self._sample_context()
        return self._next_x.copy(), float(y_obs), info

    def optimal_arm(self, x: np.ndarray) -> int:
        return int(np.argmax(self.theta_star @ x))

    def optimal_reward(self, x: np.ndarray) -> float:
        return float(np.max(self.theta_star @ x))

    def regret_so_far(self) -> float:
        return self._cum_pseudo_regret

    def regret_history(self) -> np.ndarray:
        return np.asarray(self._regret_history, dtype=float)

    # --- subclass hooks ----------------------------------------------------
    def _resample_problem(self) -> None:
        """Resample arm parameters / supports if they should depend on seed."""
        raise NotImplementedError

    def _sample_context(self) -> np.ndarray:
        raise NotImplementedError

    def _reward(self, arm: int, x: np.ndarray, with_noise: bool) -> float:
        raise NotImplementedError

    def _extra_info(self, arm: int, x: np.ndarray) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Synthetic environment
# ---------------------------------------------------------------------------

@dataclass
class SyntheticEnv(BanditEnv):
    """Synthetic sparse linear bandit.

    Parameters
    ----------
    d : int
        Ambient dimension.
    s : int
        Sparsity (size of the support of each arm parameter).
    K : int, default 5
        Number of arms.
    T : int, default 5000
        Horizon.
    noise_std : float, default 0.1
        Observation-noise standard deviation.
    """

    d: int = 50
    s: int = 5
    K: int = 5
    T: int = 5000
    noise_std: float = 0.1

    def __post_init__(self):
        BanditEnv.__init__(self, self.T)
        self.theta_star = np.zeros((self.K, self.d))
        self.supports: List[List[int]] = [[] for _ in range(self.K)]

    # subclass hooks
    def _resample_problem(self) -> None:
        rng = self._rng
        self.theta_star = np.zeros((self.K, self.d))
        self.supports = []
        for a in range(self.K):
            S = rng.choice(self.d, size=self.s, replace=False)
            S = sorted(S.tolist())
            signs = rng.choice([-1.0, 1.0], size=self.s)
            self.theta_star[a, S] = signs
            self.supports.append(S)

    def _sample_context(self) -> np.ndarray:
        rng = self._rng
        x = rng.standard_normal(self.d)
        nrm = float(np.linalg.norm(x))
        if nrm < 1e-12:
            x = np.ones(self.d) / np.sqrt(self.d)
        else:
            x = x / nrm
        return x

    def _reward(self, arm: int, x: np.ndarray, with_noise: bool) -> float:
        mean = float(self.theta_star[arm] @ x)
        if with_noise:
            return mean + float(self._rng.normal(0.0, self.noise_std))
        return mean


# ---------------------------------------------------------------------------
# Power-system environment (IEEE 33-bus LinDistFlow)
# ---------------------------------------------------------------------------

DEFAULT_BUS_LIST = (18, 22, 25, 33, 14)


@dataclass
class PowerSystemEnv(BanditEnv):
    """IEEE 33-bus radial feeder bandit.

    * d = 32 (non-slack buses).
    * K = 5 arms; each arm is a "voltage-regulation target" at a bus b_a.
    * theta*_a[k-2] = -2 * R_tilde[b_a-2, k-2] for k on path_to_root(b_a),
      and 0 otherwise. The negative sign turns the reward into a proxy for
      "negative voltage-deviation magnitude" -- larger reward means smaller
      voltage drop. Sparsity = path from slack to b_a (excluding the slack
      itself).
    * Context x_t in R^32 = stochastic load deviation from nominal in pu.
      Sampled as x_t = c * N(0, Sigma) with Sigma = diag(nominal load |s|),
      then standardised and clipped to ||x_t||_inf <= 0.3.
    * Reward y_t = theta*_{a_t}^T x_t + eps, eps ~ N(0, 0.005^2).
    """

    bus_list: tuple = DEFAULT_BUS_LIST
    K: int = 5
    T: int = 5000
    d: int = 32
    noise_std: float = 0.005
    inf_norm_clip: float = 0.3

    def __post_init__(self):
        BanditEnv.__init__(self, self.T)
        if len(self.bus_list) != self.K:
            raise ValueError("len(bus_list) must equal K")
        # Build sparse theta_star vectors; this is deterministic across seeds.
        self.theta_star = np.zeros((self.K, self.d))
        self.supports: List[List[int]] = []
        for a, b in enumerate(self.bus_list):
            ancestors = [k - 2 for k in path_to_root(int(b)) if k != 1]
            self.supports.append(sorted(ancestors))
            for k_idx in ancestors:
                self.theta_star[a, k_idx] = -2.0 * R_tilde[b - 2, k_idx]
        # Load-magnitude scale: nominal apparent-power magnitude in pu per bus.
        # |s_bus| (MVA) = sqrt(P^2 + Q^2) (MVA) / S_base. Using kW/kVAr:
        s_kva = np.sqrt(BARAN_WU_LOADS_KW ** 2 + BARAN_WU_LOADS_KVAR ** 2)
        self._sigma_pu = s_kva / 1000.0 / S_BASE_MVA  # pu, length 32
        # Scale so that *typical* fluctuations are ~30% of nominal.
        # We pick c so that E[|x_t|_inf] ~ inf_norm_clip *before* the clip.
        # Using c = 0.3 / max(sigma_pu) keeps things conservative and the clip
        # rarely binds.
        self._context_scale = self.inf_norm_clip / float(np.max(self._sigma_pu) + 1e-12)

    # subclass hooks
    def _resample_problem(self) -> None:
        # theta_star and supports are seed-independent for the power env;
        # only the noise / context streams depend on the rng.
        return None

    def _sample_context(self) -> np.ndarray:
        rng = self._rng
        z = rng.standard_normal(self.d)
        x = self._context_scale * self._sigma_pu * z
        # clip to physical bound
        np.clip(x, -self.inf_norm_clip, self.inf_norm_clip, out=x)
        return x

    def _reward(self, arm: int, x: np.ndarray, with_noise: bool) -> float:
        mean = float(self.theta_star[arm] @ x)
        if with_noise:
            return mean + float(self._rng.normal(0.0, self.noise_std))
        return mean

    # --- power-systems-specific telemetry ---------------------------------
    def _extra_info(self, arm: int, x: np.ndarray) -> dict:
        # "voltage deviation" under the chosen arm = -mean reward.
        # The chosen arm parameter is sparse on path_to_root; the actual
        # voltage deviation at the targeted bus is essentially the same
        # quantity scaled. We log the magnitude (|theta_a^T x|) and the RMS
        # deviation across all buses given x as the active-injection vector
        # (treating loads as the only injection, no reactive-control yet).
        v_dev_at_arm = float(abs(self.theta_star[arm] @ x))
        # RMS deviation across all 32 buses for this load vector p = -x:
        # V - 1 ~ 2 R_tilde @ p (we ignore Q for the RMS metric).
        v_full = 2.0 * (R_tilde @ (-x))
        v_rms = float(np.sqrt(np.mean(v_full ** 2)))
        return {
            "voltage_dev_arm": v_dev_at_arm,
            "voltage_rms": v_rms,
        }
