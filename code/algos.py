"""Bandit algorithms for the sparse-contextual benchmark.

Implementations:

* ``FGTSLasso``: forced-exploration + LASSO active-set + ridge Thompson
  sampling on the recovered support (proposed method).
* ``LassoBandit``: Bastani & Bayati 2020 "Online Decision Making with
  High-Dimensional Covariates", Operations Research 68(1).
* ``SALassoBandit``: Oh, Iyengar & Zeevi 2021 "Sparsity-Agnostic Lasso
  Bandit", ICML.
* ``ThresholdedLassoBandit``: Ariu, Abe & Proutiere 2022 "Thresholded
  Lasso Bandit", ICML.
* ``LinUCB``: Chu, Li, Reyzin & Schapire 2011 "Contextual Bandits with
  Linear Payoff Functions", AISTATS.

All algorithms expose the common ``act(x)`` / ``update(x, arm, r)`` API.
Numpy + scikit-learn (Lasso) only. Defensive against NaNs and rank-1 ridge
solves.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sklearn.linear_model import Lasso

# Suppress sklearn convergence noise; we are deliberately running short Lasso
# solves at every refit and tolerate non-converged warmstarted estimates.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Objective did not converge")
warnings.filterwarnings("ignore", message="Coordinate descent")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BanditAlgo(ABC):
    """Abstract base class for contextual-bandit algorithms.

    Subclasses must implement ``act`` (pick an arm given context) and
    ``update`` (incorporate the (x, arm, reward) triple).
    """

    name: str = "BanditAlgo"

    def __init__(self, d: int, K: int, T: int, seed: int = 0):
        self.d = d
        self.K = K
        self.T = T
        self.rng = np.random.default_rng(seed)
        # per-arm history (lists for cheap appends; dense matrices built on
        # demand at refit).
        self._X_per_arm: List[List[np.ndarray]] = [[] for _ in range(K)]
        self._y_per_arm: List[List[float]] = [[] for _ in range(K)]
        self._t = 0

    # ----- shared bookkeeping ---------------------------------------------
    def _record(self, x: np.ndarray, arm: int, r: float) -> None:
        self._X_per_arm[arm].append(np.asarray(x, dtype=float).copy())
        self._y_per_arm[arm].append(float(r))
        self._t += 1

    def _arm_data(self, arm: int):
        Xs = self._X_per_arm[arm]
        ys = self._y_per_arm[arm]
        if not Xs:
            return None, None
        return np.asarray(Xs), np.asarray(ys)

    # ----- abstract API ---------------------------------------------------
    @abstractmethod
    def act(self, x: np.ndarray) -> int: ...

    @abstractmethod
    def update(self, x: np.ndarray, arm: int, reward: float) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_lasso_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Fit a Lasso(alpha) and return the coefficient vector. Falls back to
    a zero vector if there are too few samples or numerical trouble."""
    if X.shape[0] < 2:
        return np.zeros(X.shape[1])
    alpha = max(float(alpha), 1e-6)
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000, tol=1e-4)
    try:
        model.fit(X, y)
        coef = np.asarray(model.coef_, dtype=float)
        if not np.all(np.isfinite(coef)):
            return np.zeros(X.shape[1])
        return coef
    except Exception:
        return np.zeros(X.shape[1])


def _ridge_solve(X: np.ndarray, y: np.ndarray, rho: float):
    """Return (mu, V_inv) from V = X^T X + rho I, mu = V^{-1} X^T y."""
    d = X.shape[1]
    V = X.T @ X + rho * np.eye(d)
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)
    mu = V_inv @ (X.T @ y)
    return mu, V_inv


# ---------------------------------------------------------------------------
# 1. FGTSLasso (proposed)
# ---------------------------------------------------------------------------

class FGTSLasso(BanditAlgo):
    """Forced-exploration Greedy Thompson Sampling on a Lasso-recovered support.

    Algorithm sketch:
      * Forced-exploration phase of length ``tau_0 = ceil(c1 * s_hint *
        log d)`` rounds, during which arms are pulled in round-robin order.
      * Periodic Lasso refit every ``tau_0`` rounds with regularisation
        ``lambda_t = 0.5 * sigma * sqrt(log(d) / n_a)``.
      * Active set ``hat_S_a = {j : |theta_a,j| > eps}``.
      * Ridge posterior on ``hat_S_a``: ``V_a = X_S^T X_S + rho I``,
        ``mu_a = V_a^{-1} X_S^T y``. Sample ``tilde_theta ~ N(mu, nu^2 V_a^{-1})``
        and pick ``argmax_a x_S^T tilde_theta``.
    """

    name = "FGTSLasso"

    def __init__(
        self,
        d: int,
        K: int,
        T: int,
        seed: int = 0,
        s_hint: int = 5,
        c1: float = 2.0,
        sigma: float = 0.1,
        rho: float = 1.0,
        nu: float = 0.2,
        coef_threshold: float = 1e-4,
    ):
        super().__init__(d, K, T, seed)
        self.s_hint = max(1, int(s_hint))
        self.c1 = float(c1)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.nu = float(nu)
        self.coef_thr = float(coef_threshold)
        self.tau_0 = max(K, int(np.ceil(c1 * self.s_hint * np.log(max(d, 2)))))
        # Cached active sets and ridge posteriors.
        self._S_hat: List[np.ndarray] = [np.arange(d) for _ in range(K)]
        self._mu: List[np.ndarray] = [np.zeros(d) for _ in range(K)]
        self._V_inv: List[np.ndarray] = [np.eye(d) for _ in range(K)]
        self._steps_since_refit = 0

    # -- main loop ---------------------------------------------------------
    def act(self, x: np.ndarray) -> int:
        # Forced exploration: round-robin
        if self._t < self.tau_0:
            return int(self._t % self.K)
        scores = np.full(self.K, -np.inf)
        for a in range(self.K):
            S = self._S_hat[a]
            if S.size == 0:
                # No active features; play a near-zero "explore" score with
                # tiny noise so that ties broken at random.
                scores[a] = float(self.rng.normal(0.0, 1e-3))
                continue
            mu_S = self._mu[a][S]
            V_inv_S = self._V_inv[a][np.ix_(S, S)]
            # Symmetrize for numerical safety
            V_inv_S = 0.5 * (V_inv_S + V_inv_S.T)
            try:
                # Cholesky-based sample
                L = np.linalg.cholesky(V_inv_S + 1e-9 * np.eye(S.size))
                z = self.rng.standard_normal(S.size)
                tilde = mu_S + self.nu * (L @ z)
            except np.linalg.LinAlgError:
                tilde = mu_S
            scores[a] = float(x[S] @ tilde)
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self._record(x, arm, reward)
        self._steps_since_refit += 1
        # Refit every tau_0 rounds, and also right at end of forced phase.
        if (
            self._t == self.tau_0
            or self._steps_since_refit >= self.tau_0
            or self._t == self.K  # earliest plausible refit
        ):
            self._refit_arm(arm)
            # Refit *all* arms periodically to keep posteriors fresh.
            for a in range(self.K):
                if a != arm:
                    self._refit_arm(a)
            self._steps_since_refit = 0
        else:
            # incremental update: just refit the arm we pulled
            self._refit_arm(arm)

    # -- per-arm refit -----------------------------------------------------
    def _refit_arm(self, arm: int) -> None:
        X, y = self._arm_data(arm)
        if X is None or X.shape[0] < 2:
            self._S_hat[arm] = np.arange(self.d)
            self._mu[arm] = np.zeros(self.d)
            self._V_inv[arm] = np.eye(self.d)
            return
        n_a = X.shape[0]
        lam = 0.1 * self.sigma * np.sqrt(np.log(max(self.d, 2)) / max(n_a, 1))
        coef = _safe_lasso_fit(X, y, lam)
        S = np.where(np.abs(coef) > self.coef_thr)[0]
        if S.size == 0:
            order = np.argsort(np.abs(coef))[::-1]
            top = min(self.s_hint, self.d)
            S = np.sort(order[:top])
        # Ridge posterior on support
        X_S = X[:, S]
        mu_S, V_inv_S = _ridge_solve(X_S, y, self.rho)
        # Embed back into d-dim arrays (zeros off-support).
        full_mu = np.zeros(self.d)
        full_mu[S] = mu_S
        full_V_inv = np.zeros((self.d, self.d))
        full_V_inv[np.ix_(S, S)] = V_inv_S
        self._S_hat[arm] = S
        self._mu[arm] = full_mu
        self._V_inv[arm] = full_V_inv


# ---------------------------------------------------------------------------
# 2. LassoBandit (Bastani & Bayati 2020)
# ---------------------------------------------------------------------------

class LassoBandit(BanditAlgo):
    """Simplified Bastani-Bayati Lasso Bandit.

    Two buffers per arm:
      * forced-sample buffer T_a (deterministic schedule),
      * all-sample buffer S_a.

    Action rule (simplified): use forced-sample Lasso to identify the top-2
    arms (by predicted reward); among those, pick the higher all-sample-Lasso
    predicted reward.

    Lambda schedules: lambda_t = lambda_0 * sqrt((log t + log d) / t)
    (eq. (5) in the paper, with a single base constant).
    """

    name = "LassoBandit"

    def __init__(
        self,
        d: int,
        K: int,
        T: int,
        seed: int = 0,
        lambda0: float = 0.01,
        h: float = 5.0,  # forced-sample density (q=1 in the paper's notation)
        K_top: int = 2,  # number of "good" arms to refine via all-sample lasso
    ):
        super().__init__(d, K, T, seed)
        self.lambda0 = float(lambda0)
        self.h = float(h)
        self.K_top = max(1, min(int(K_top), K))
        # Forced-sample sets (deterministic schedule).
        self._forced_X: List[List[np.ndarray]] = [[] for _ in range(K)]
        self._forced_y: List[List[float]] = [[] for _ in range(K)]
        self._forced_schedule = self._build_forced_schedule(T)
        # Per-arm coefficient caches.
        self._coef_forced = [np.zeros(d) for _ in range(K)]
        self._coef_all = [np.zeros(d) for _ in range(K)]

    def _build_forced_schedule(self, T: int) -> List[int]:
        """Return a length-T list assigning forced rounds to specific arms,
        or -1 for non-forced rounds. Schedule: every q rounds force an arm
        in round-robin (q chosen from h)."""
        q = max(1, int(round(self.h)))
        sched = [-1] * T
        a = 0
        # First K*h rounds: dense forced-pull schedule (warm start).
        # Then, every q rounds we force a round-robin arm.
        warmup = self.K * max(1, int(self.h))
        for t in range(min(warmup, T)):
            sched[t] = t % self.K
        for t in range(warmup, T):
            if (t - warmup) % q == 0:
                sched[t] = a
                a = (a + 1) % self.K
        return sched

    # ---------------------------------------------------------------------
    def act(self, x: np.ndarray) -> int:
        # If this round is a forced-sample round, play that arm.
        if self._t < len(self._forced_schedule):
            forced = self._forced_schedule[self._t]
            if forced >= 0:
                return int(forced)
        # forced-set scores
        forced_scores = np.array(
            [float(self._coef_forced[a] @ x) for a in range(self.K)]
        )
        # candidate set: top-K_top by forced-set score.
        cand = np.argsort(forced_scores)[-self.K_top:]
        if cand.size == 0:
            return int(self.rng.integers(self.K))
        all_scores = np.array(
            [float(self._coef_all[a] @ x) for a in cand]
        )
        return int(cand[int(np.argmax(all_scores))])

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        # All-sample buffer
        self._record(x, arm, reward)
        # If this was a forced round for this arm, also add to forced buffer.
        # (Use the round we were just *about to* play, i.e. self._t-1 after
        # _record incremented.)
        t_pre = self._t - 1
        if t_pre < len(self._forced_schedule) and self._forced_schedule[t_pre] == arm:
            self._forced_X[arm].append(np.asarray(x, dtype=float).copy())
            self._forced_y[arm].append(float(reward))
        # Refit both Lassos for this arm
        self._refit_forced(arm)
        self._refit_all(arm)

    def _refit_forced(self, arm: int) -> None:
        Xs = self._forced_X[arm]
        ys = self._forced_y[arm]
        if len(Xs) < 2:
            return
        X = np.asarray(Xs)
        y = np.asarray(ys)
        n = X.shape[0]
        lam = self.lambda0 * np.sqrt((np.log(max(n, 2)) + np.log(self.d)) / n)
        self._coef_forced[arm] = _safe_lasso_fit(X, y, lam)

    def _refit_all(self, arm: int) -> None:
        X, y = self._arm_data(arm)
        if X is None or X.shape[0] < 2:
            return
        n = X.shape[0]
        lam = self.lambda0 * np.sqrt((np.log(max(n, 2)) + np.log(self.d)) / (2 * n))
        self._coef_all[arm] = _safe_lasso_fit(X, y, lam)


# ---------------------------------------------------------------------------
# 3. SALassoBandit (Oh, Iyengar & Zeevi 2021)
# ---------------------------------------------------------------------------

class SALassoBandit(BanditAlgo):
    """Sparsity-Agnostic Lasso Bandit (Oh-Iyengar-Zeevi 2021).

    Single Lasso per arm with adaptive ``lambda_t = sigma * sqrt((4 log t +
    2 log d) / t)``. Greedy action ``argmax_a x^T theta_a``. A short
    forced-exploration phase of length ``2K`` warms up each arm.
    """

    name = "SALassoBandit"

    def __init__(
        self,
        d: int,
        K: int,
        T: int,
        seed: int = 0,
        sigma: float = 0.1,
        lambda_mult: float = 0.1,
    ):
        super().__init__(d, K, T, seed)
        self.sigma = float(sigma)
        self.lambda_mult = float(lambda_mult)
        self._coef = [np.zeros(d) for _ in range(K)]
        self._warmup = 2 * K

    def act(self, x: np.ndarray) -> int:
        if self._t < self._warmup:
            return int(self._t % self.K)
        scores = np.array([float(self._coef[a] @ x) for a in range(self.K)])
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self._record(x, arm, reward)
        self._refit(arm)

    def _refit(self, arm: int) -> None:
        X, y = self._arm_data(arm)
        if X is None or X.shape[0] < 2:
            return
        n = X.shape[0]
        t_use = max(n, 2)
        lam = self.lambda_mult * self.sigma * np.sqrt((4 * np.log(t_use) + 2 * np.log(self.d)) / t_use)
        self._coef[arm] = _safe_lasso_fit(X, y, lam)


# ---------------------------------------------------------------------------
# 4. ThresholdedLassoBandit (Ariu, Abe & Proutiere 2022)
# ---------------------------------------------------------------------------

class ThresholdedLassoBandit(BanditAlgo):
    """Lasso + hard thresholding + ridge re-fit on the kept support.

    Threshold ``tau = c * sqrt(s_hint * log d / n)``. Greedy action.
    """

    name = "ThresholdedLassoBandit"

    def __init__(
        self,
        d: int,
        K: int,
        T: int,
        seed: int = 0,
        s_hint: int = 5,
        c_thr: float = 1.0,
        sigma: float = 0.1,
        rho: float = 1.0,
    ):
        super().__init__(d, K, T, seed)
        self.s_hint = int(s_hint)
        self.c_thr = float(c_thr)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self._coef = [np.zeros(d) for _ in range(K)]
        self._warmup = 2 * K

    def act(self, x: np.ndarray) -> int:
        if self._t < self._warmup:
            return int(self._t % self.K)
        scores = np.array([float(self._coef[a] @ x) for a in range(self.K)])
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self._record(x, arm, reward)
        self._refit(arm)

    def _refit(self, arm: int) -> None:
        X, y = self._arm_data(arm)
        if X is None or X.shape[0] < 2:
            return
        n = X.shape[0]
        lam = 0.5 * self.sigma * np.sqrt(np.log(max(self.d, 2)) / max(n, 1))
        coef = _safe_lasso_fit(X, y, lam)
        tau = self.c_thr * np.sqrt(self.s_hint * np.log(max(self.d, 2)) / max(n, 1))
        S = np.where(np.abs(coef) > tau)[0]
        if S.size == 0:
            # Fall back to top-s_hint by magnitude
            order = np.argsort(np.abs(coef))[::-1]
            S = np.sort(order[: self.s_hint])
        X_S = X[:, S]
        mu_S, _V_inv_S = _ridge_solve(X_S, y, self.rho)
        full = np.zeros(self.d)
        full[S] = mu_S
        self._coef[arm] = full


# ---------------------------------------------------------------------------
# 5. LinUCB (Chu, Li, Reyzin & Schapire 2011)
# ---------------------------------------------------------------------------

class LinUCB(BanditAlgo):
    """Disjoint-arm LinUCB.

    Per-arm ``A_a = lambda I + sum_t x_t x_t^T``, ``b_a = sum_t r_t x_t``,
    ``theta_a = A_a^{-1} b_a``. Action: ``argmax_a x^T theta_a + beta *
    sqrt(x^T A_a^{-1} x)``. Default ``beta = 1.0``.
    """

    name = "LinUCB"

    def __init__(
        self,
        d: int,
        K: int,
        T: int,
        seed: int = 0,
        beta: Optional[float] = None,
        reg: float = 1.0,
    ):
        super().__init__(d, K, T, seed)
        self.reg = float(reg)
        self.beta = float(beta) if beta is not None else 1.0
        self._A = [self.reg * np.eye(d) for _ in range(K)]
        self._A_inv = [(1.0 / self.reg) * np.eye(d) for _ in range(K)]
        self._b = [np.zeros(d) for _ in range(K)]
        self._theta = [np.zeros(d) for _ in range(K)]

    def act(self, x: np.ndarray) -> int:
        scores = np.empty(self.K)
        for a in range(self.K):
            mean = float(self._theta[a] @ x)
            var = float(x @ self._A_inv[a] @ x)
            var = max(var, 0.0)
            scores[a] = mean + self.beta * np.sqrt(var)
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self._record(x, arm, reward)
        x = np.asarray(x, dtype=float)
        # Sherman-Morrison update of A_inv
        Ai = self._A_inv[arm]
        Aix = Ai @ x
        denom = 1.0 + float(x @ Aix)
        self._A_inv[arm] = Ai - np.outer(Aix, Aix) / denom
        self._A[arm] = self._A[arm] + np.outer(x, x)
        self._b[arm] = self._b[arm] + reward * x
        self._theta[arm] = self._A_inv[arm] @ self._b[arm]


# ---------------------------------------------------------------------------
# 6. KernelUCB (Valko et al. 2013) -- RBF kernel
# ---------------------------------------------------------------------------

class KernelUCB(BanditAlgo):
    """Kernel-UCB with an RBF kernel, per-arm.

    Maintains, per arm a, the kernel-ridge predictor
        mu_a(x)     = k_x^T (K_a + lam * I)^{-1} y_a,
        sigma_a(x)  = sqrt(k(x,x) - k_x^T (K_a + lam * I)^{-1} k_x),
    selects argmax_a (mu_a + beta * sigma_a).

    For tractability we cap the per-arm history at ``max_n`` and recompute
    the kernel matrix lazily (every ``refit_period`` updates).
    """

    name = "KernelUCB"

    def __init__(
        self,
        d: int,
        K: int,
        T: int,
        seed: int = 0,
        beta: float = 1.0,
        gamma: Optional[float] = None,
        reg: float = 1.0,
        max_n: int = 800,
        refit_period: int = 25,
    ):
        super().__init__(d, K, T, seed)
        self.beta = float(beta)
        self.reg = float(reg)
        self.gamma = float(gamma) if gamma is not None else 1.0 / max(d, 1)
        self.max_n = int(max_n)
        self.refit_period = int(refit_period)
        self._alpha = [None for _ in range(K)]   # (K_a + lam I)^-1 y_a
        self._K_inv = [None for _ in range(K)]   # (K_a + lam I)^-1
        self._steps_since_refit = 0

    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X); Y = np.atleast_2d(Y)
        sq = (np.sum(X ** 2, axis=1)[:, None]
              + np.sum(Y ** 2, axis=1)[None, :]
              - 2.0 * X @ Y.T)
        return np.exp(-self.gamma * np.maximum(sq, 0.0))

    def _refit_arm(self, arm: int) -> None:
        X, y = self._arm_data(arm)
        if X is None or X.shape[0] < 1:
            self._alpha[arm] = None
            self._K_inv[arm] = None
            return
        if X.shape[0] > self.max_n:
            X = X[-self.max_n:]; y = y[-self.max_n:]
        Kmat = self._kernel(X, X) + self.reg * np.eye(X.shape[0])
        try:
            Kinv = np.linalg.inv(Kmat)
        except np.linalg.LinAlgError:
            Kinv = np.linalg.pinv(Kmat)
        self._K_inv[arm] = (X, Kinv)
        self._alpha[arm] = (X, Kinv @ y)

    def act(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float)
        scores = np.empty(self.K)
        for a in range(self.K):
            if self._alpha[a] is None:
                scores[a] = self.beta  # uniform prior
                continue
            X_a, alpha = self._alpha[a]
            kx = self._kernel(x.reshape(1, -1), X_a).ravel()
            mu = float(kx @ alpha)
            X_a2, K_inv = self._K_inv[a]
            kxx = float(self._kernel(x.reshape(1, -1), x.reshape(1, -1)).ravel()[0])
            var = max(kxx - float(kx @ K_inv @ kx), 1e-9)
            scores[a] = mu + self.beta * np.sqrt(var)
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self._record(x, arm, reward)
        self._steps_since_refit += 1
        if self._steps_since_refit >= self.refit_period or self._alpha[arm] is None:
            for a in range(self.K):
                self._refit_arm(a)
            self._steps_since_refit = 0


# ---------------------------------------------------------------------------
# 7. GP-TS (Gaussian-Process Thompson Sampling) -- RBF kernel
# ---------------------------------------------------------------------------

class GPTSBandit(BanditAlgo):
    """GP-TS with an RBF kernel. Per-arm GP posterior; sample one draw at
    each context and pick the argmax.

    Cap per-arm history at ``max_n`` and refit lazily for tractability.
    """

    name = "GPTS"

    def __init__(
        self,
        d: int,
        K: int,
        T: int,
        seed: int = 0,
        gamma: Optional[float] = None,
        sigma: float = 0.1,
        reg: float = 1.0,
        max_n: int = 800,
        refit_period: int = 25,
    ):
        super().__init__(d, K, T, seed)
        self.gamma = float(gamma) if gamma is not None else 1.0 / max(d, 1)
        self.sigma = float(sigma)
        self.reg = float(reg)
        self.max_n = int(max_n)
        self.refit_period = int(refit_period)
        self._cache = [None for _ in range(K)]   # (X_a, K_inv, alpha)
        self._steps_since_refit = 0

    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X); Y = np.atleast_2d(Y)
        sq = (np.sum(X ** 2, axis=1)[:, None]
              + np.sum(Y ** 2, axis=1)[None, :]
              - 2.0 * X @ Y.T)
        return np.exp(-self.gamma * np.maximum(sq, 0.0))

    def _refit_arm(self, arm: int) -> None:
        X, y = self._arm_data(arm)
        if X is None or X.shape[0] < 1:
            self._cache[arm] = None
            return
        if X.shape[0] > self.max_n:
            X = X[-self.max_n:]; y = y[-self.max_n:]
        Kmat = self._kernel(X, X) + (self.sigma ** 2 + self.reg * 1e-6) * np.eye(X.shape[0])
        try:
            K_inv = np.linalg.inv(Kmat)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(Kmat)
        alpha = K_inv @ y
        self._cache[arm] = (X, K_inv, alpha)

    def act(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float)
        scores = np.empty(self.K)
        for a in range(self.K):
            if self._cache[a] is None:
                scores[a] = float(self.rng.standard_normal())
                continue
            X_a, K_inv, alpha = self._cache[a]
            kx = self._kernel(x.reshape(1, -1), X_a).ravel()
            mu = float(kx @ alpha)
            kxx = 1.0  # RBF k(x,x)=1
            var = max(kxx - float(kx @ K_inv @ kx), 1e-9)
            scores[a] = mu + np.sqrt(var) * float(self.rng.standard_normal())
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self._record(x, arm, reward)
        self._steps_since_refit += 1
        if self._steps_since_refit >= self.refit_period or self._cache[arm] is None:
            for a in range(self.K):
                self._refit_arm(a)
            self._steps_since_refit = 0


# ---------------------------------------------------------------------------
# 8. LinTS (Ambient-space linear Thompson Sampling, Agrawal-Goyal 2013)
# ---------------------------------------------------------------------------

class LinTS(BanditAlgo):
    """Linear Thompson Sampling on the full ambient space (no sparsity).

    Maintains, per arm a, ridge posterior
        V_a = X_a^T X_a + reg * I,    mu_a = V_a^{-1} X_a^T y_a.
    At each round, draws theta_a ~ N(mu_a, nu^2 V_a^{-1}) for every arm and
    picks argmax_a x^T theta_a.

    Equivalent to FGTS-LASSO with no support recovery -- included to isolate
    the contribution of LASSO support recovery from the contribution of
    randomized exploration on the per-arm posterior.
    """

    name = "LinTS"

    def __init__(
        self,
        d: int,
        K: int,
        T: int,
        seed: int = 0,
        nu: float = 0.5,
        reg: float = 1.0,
    ):
        super().__init__(d, K, T, seed)
        self.nu = float(nu)
        self.reg = float(reg)
        self._A = [self.reg * np.eye(d) for _ in range(K)]
        self._A_inv = [(1.0 / self.reg) * np.eye(d) for _ in range(K)]
        self._b = [np.zeros(d) for _ in range(K)]
        self._mu = [np.zeros(d) for _ in range(K)]

    def act(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float)
        scores = np.empty(self.K)
        for a in range(self.K):
            cov = self.nu ** 2 * self._A_inv[a]
            cov = 0.5 * (cov + cov.T)
            try:
                L = np.linalg.cholesky(cov + 1e-9 * np.eye(self.d))
                z = self.rng.standard_normal(self.d)
                tilde = self._mu[a] + L @ z
            except np.linalg.LinAlgError:
                tilde = self._mu[a]
            scores[a] = float(x @ tilde)
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self._record(x, arm, reward)
        x = np.asarray(x, dtype=float)
        Ai = self._A_inv[arm]
        Aix = Ai @ x
        denom = 1.0 + float(x @ Aix)
        self._A_inv[arm] = Ai - np.outer(Aix, Aix) / denom
        self._A[arm] = self._A[arm] + np.outer(x, x)
        self._b[arm] = self._b[arm] + reward * x
        self._mu[arm] = self._A_inv[arm] @ self._b[arm]


ALGO_REGISTRY = {
    "FGTSLasso": FGTSLasso,
    "LassoBandit": LassoBandit,
    "SALassoBandit": SALassoBandit,
    "ThresholdedLassoBandit": ThresholdedLassoBandit,
    "LinUCB": LinUCB,
    "KernelUCB": KernelUCB,
    "GPTS": GPTSBandit,
    "LinTS": LinTS,
}
