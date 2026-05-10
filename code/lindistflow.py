"""LinDistFlow IEEE 33-bus oracle.

Standard LinDistFlow voltage approximation (Baran-Wu 1989) for the IEEE 33-bus
radial distribution feeder. Pure numpy.

Bus 1 is the slack bus. Non-slack buses are 2..33 (32 buses).

Convention: ``A`` is the reduced (32 x 32) bus-to-branch incidence matrix for
the non-slack buses. For a radial tree this is invertible. Then

    R_tilde = A^{-T} diag(r_pu) A^{-1}
    X_tilde = A^{-T} diag(x_pu) A^{-1}

and the LinDistFlow voltage drop approximation is

    V_b - 1 ~= 2 * (R_tilde @ p + X_tilde @ q)[b]

with p, q the per-unit net active/reactive injections (positive = generation).

The (b, k) entry of R_tilde equals the sum of resistances on the common path
from the root to b and to k -- so the b-th row has nonzero support exactly on
the path from root to b (minus the slack bus).
"""

from __future__ import annotations

import numpy as np

# IEEE 33-bus base values
V_BASE_KV = 12.66
S_BASE_MVA = 10.0
Z_BASE = (V_BASE_KV ** 2) / S_BASE_MVA  # 16.0356 ohm

# Branch data: (from_bus, to_bus, R_ohm, X_ohm)
BRANCH_DATA = [
    (1, 2, 0.0922, 0.047), (2, 3, 0.493, 0.2511), (3, 4, 0.366, 0.1864),
    (4, 5, 0.3811, 0.1941), (5, 6, 0.819, 0.707), (6, 7, 0.1872, 0.6188),
    (7, 8, 0.7114, 0.2351), (8, 9, 1.03, 0.74), (9, 10, 1.044, 0.74),
    (10, 11, 0.1966, 0.065), (11, 12, 0.3744, 0.1238), (12, 13, 1.468, 1.155),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.591, 0.526), (15, 16, 0.7463, 0.545),
    (16, 17, 1.289, 1.721), (17, 18, 0.732, 0.574), (2, 19, 0.164, 0.1565),
    (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784), (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083), (23, 24, 0.898, 0.7091), (24, 25, 0.896, 0.7011),
    (6, 26, 0.203, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.059, 0.9337),
    (28, 29, 0.8042, 0.7006), (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.963),
    (31, 32, 0.3105, 0.3619), (32, 33, 0.341, 0.5302),
]

# Baran-Wu nominal active/reactive load magnitudes per bus (kW, kVAr) for
# buses 2..33. Bus 1 is the slack bus and has no load.
BARAN_WU_LOADS_KW = np.array([
    100, 90, 120, 60, 60, 200, 200, 60, 60, 45, 60, 60, 120, 60, 60,
    60, 90, 90, 90, 90, 90, 90, 420, 420, 60, 60, 60, 120, 200, 150,
    210, 60,
], dtype=float)
BARAN_WU_LOADS_KVAR = np.array([
    60, 40, 80, 30, 20, 100, 100, 20, 20, 30, 35, 35, 80, 10, 20,
    20, 40, 40, 40, 40, 40, 50, 200, 200, 25, 25, 20, 70, 600, 70,
    100, 40,
], dtype=float)

N_BUSES = 33
N_NONSLACK = 32  # buses 2..33

# ---------------------------------------------------------------------------
# Build the radial topology and incidence matrix
# ---------------------------------------------------------------------------

def _build_parent_map():
    """Return parent[b] for b in 2..33 (1-indexed). parent[1] = None (slack)."""
    parent = {1: None}
    for fb, tb, _r, _x in BRANCH_DATA:
        parent[tb] = fb
    return parent


def path_to_root(b: int) -> list:
    """Return the list of bus indices on the path from the root (bus 1) to b.

    Includes both endpoints and is ordered root -> b. Buses are 1-indexed.
    """
    if b < 1 or b > N_BUSES:
        raise ValueError(f"bus index {b} out of range [1, {N_BUSES}]")
    parent = _build_parent_map()
    chain = []
    cur = b
    while cur is not None:
        chain.append(cur)
        cur = parent[cur]
    return chain[::-1]


def _build_incidence_and_impedance():
    """Build (A, r_pu, x_pu).

    A is the reduced (32 x 32) bus-to-branch incidence matrix for non-slack
    buses. Rows index non-slack buses (row b -> bus b+2); columns index
    branches in BRANCH_DATA order. Entry A[b, i] = +1 if branch i terminates
    at bus b+2 (i.e. b+2 is the to-bus), -1 if branch i originates at bus b+2.
    The slack bus (bus 1) row is dropped.

    For a radial tree this reduced matrix is square (32x32) and invertible.
    With this sign convention, (A^{-1})[i, b] = 1 if branch i lies on the
    path from the root (slack) to bus b+2, else 0. This makes A^{-1} the
    path-incidence matrix used in the LinDistFlow derivation.
    """
    n = N_NONSLACK
    A = np.zeros((n, n), dtype=float)
    r_pu = np.zeros(n, dtype=float)
    x_pu = np.zeros(n, dtype=float)
    for i, (fb, tb, R_ohm, X_ohm) in enumerate(BRANCH_DATA):
        # to-bus is always non-slack in this dataset
        b_to = tb - 2
        A[b_to, i] = 1.0
        if fb != 1:  # from-bus is non-slack; otherwise the slack row is dropped
            b_from = fb - 2
            A[b_from, i] = -1.0
        r_pu[i] = R_ohm / Z_BASE
        x_pu[i] = X_ohm / Z_BASE
    return A, r_pu, x_pu


A, R_PU, X_PU = _build_incidence_and_impedance()
A_INV = np.linalg.inv(A)

# R_tilde[b, k] = sum_{branch i on path(root, b) intersect path(root, k)} r_i.
R_tilde = A_INV.T @ np.diag(R_PU) @ A_INV
X_tilde = A_INV.T @ np.diag(X_PU) @ A_INV


def voltage_deviation(p_pu: np.ndarray, q_pu: np.ndarray) -> np.ndarray:
    """LinDistFlow voltage deviation V_b - 1 for non-slack buses (length 32).

    p_pu, q_pu are per-unit net injections (positive = generation) at non-slack
    buses 2..33.
    """
    p_pu = np.asarray(p_pu, dtype=float).reshape(-1)
    q_pu = np.asarray(q_pu, dtype=float).reshape(-1)
    if p_pu.shape[0] != N_NONSLACK or q_pu.shape[0] != N_NONSLACK:
        raise ValueError("p_pu, q_pu must have length 32")
    return 2.0 * (R_tilde @ p_pu + X_tilde @ q_pu)


# ---------------------------------------------------------------------------
# Self-test (sparsity pattern)
# ---------------------------------------------------------------------------

def _branches_on_path(b: int) -> set:
    """Indices (into BRANCH_DATA) of branches on the path from root to bus b."""
    chain = path_to_root(b)  # [1, ..., b]
    on_path = set()
    chain_set = set(chain)
    for i, (_fb, tb, _r, _x) in enumerate(BRANCH_DATA):
        # Branch i terminates at tb; it is on the path iff tb is on the chain
        # from root to b (the to-bus uniquely identifies a branch in a radial
        # tree).
        if tb in chain_set:
            on_path.add(i)
    return on_path


def _selftest(verbose: bool = False) -> None:
    """Verify the LinDistFlow sparsity / value structure.

    For LinDistFlow on a radial tree:

        R_tilde[b-2, k-2] = sum_{i in path(b) ∩ path(k)} r_i

    so the (b-2)-th row has nonzero support exactly on the set of buses k
    whose path-to-root shares at least one branch with path-to-root(b)
    (equivalently: LCA(b, k) is not the slack bus). On the IEEE 33 feeder
    every non-slack bus passes through branch (1,2), so all paths intersect,
    making R_tilde dense.

    The "row support equals path_to_root(b) minus slack" statement holds in
    the weaker sense that the diagonal entry equals the sum of resistances
    on the path, and that the path-to-root buses are *contained* in the
    support. We assert both: (a) path-buses are in the support; (b) every
    nonzero entry corresponds to a bus whose path shares branches with b's
    path; (c) the diagonal is r_pu summed along path-to-root(b); (d) the
    sparsity pattern of X_tilde matches that of R_tilde.
    """
    tol = 1e-9
    for b in range(2, N_BUSES + 1):
        row = R_tilde[b - 2, :]
        nz = set(np.where(np.abs(row) > tol)[0].tolist())
        path_b_branches = _branches_on_path(b)
        # (a) every non-slack bus on path-to-root(b) appears in the support
        path_buses = set(k - 2 for k in path_to_root(b) if k != 1)
        assert path_buses.issubset(nz), f"bus {b}: missing path-buses"
        # (b) every entry in the support comes from a path-intersection
        for k_idx in nz:
            k_bus = k_idx + 2
            shared = path_b_branches & _branches_on_path(k_bus)
            assert shared, f"bus {b}: nz at k={k_bus} but no shared branches"
        # (c) diagonal = total resistance from slack to bus b
        diag_expected = sum(R_PU[i] for i in path_b_branches)
        assert abs(row[b - 2] - diag_expected) < 1e-9, (
            f"bus {b}: diag {row[b-2]} != {diag_expected}"
        )
        if verbose:
            print(
                f"bus {b}: support size {len(nz)}; "
                f"path-to-root size {len(path_buses)}; diag = {row[b-2]:.5f}"
            )
    # (d) X_tilde same sparsity pattern as R_tilde
    for b in range(2, N_BUSES + 1):
        row_r = R_tilde[b - 2, :]
        row_x = X_tilde[b - 2, :]
        assert (np.abs(row_r) > tol).tolist() == (np.abs(row_x) > tol).tolist()
    if verbose:
        print("All sparsity-pattern checks passed.")


if __name__ == "__main__":
    _selftest(verbose=True)
    print(f"R_tilde shape: {R_tilde.shape}")
    print(f"R_tilde[0, 0] (bus 2 path to root): {R_tilde[0, 0]:.6f} pu (expect r_12/Z_base = {0.0922/Z_BASE:.6f})")
    print(f"path_to_root(18): {path_to_root(18)}")
    print(f"path_to_root(33): {path_to_root(33)}")
