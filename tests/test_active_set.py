"""Tests for training values, tie points, and active set."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.active_set import (
    check_active_set_constancy,
    compute_training_values,
    find_active_branches,
    find_active_set,
    find_tie_points,
)
from src.core.candidates import build_candidate_family
from src.core.piecewise import TrainingLoss


def _make_simple_quadratic():
    """f = (w - alpha)^2, w in [-3, 3], alpha in [0.5, 2.5].
    Unique minimizer w* = alpha with training value 0.
    Endpoints have training values (w_min - alpha)^2 and (w_max - alpha)^2.
    """
    c = np.zeros((3, 3))
    c[2, 0] = 1.0
    c[1, 1] = -2.0
    c[0, 2] = 1.0
    return TrainingLoss(pieces=[(-3.0, 3.0, c)], alpha_range=(0.5, 2.5), w_range=(-3.0, 3.0))


def _make_symmetric_quartic():
    """f = (w^2 - alpha)^2, w in [-2, 2], alpha in [0.1, 3].
    Two global minimizers w = +/- sqrt(alpha), both with training value 0.
    The w=0 branch has training value alpha^2.
    """
    c = np.zeros((3, 5))
    c[2, 0] = 1.0
    c[1, 2] = -2.0
    c[0, 4] = 1.0
    return TrainingLoss(pieces=[(-2.0, 2.0, c)], alpha_range=(0.1, 3.0), w_range=(-2.0, 2.0))


def test_training_values_simple():
    f = _make_simple_quadratic()
    alpha_grid = np.linspace(0.5, 2.5, 500)
    cands = build_candidate_family(f, alpha_grid)
    r = compute_training_values(f, cands, alpha_grid)

    # Interior branch (w = alpha) should have training value 0
    interior = [i for i, c in enumerate(cands) if c.kind == "interior"]
    assert len(interior) == 1
    assert np.allclose(r[interior[0]], 0.0, atol=1e-6)

    # Endpoint w_min = -3: r = (-3 - alpha)^2
    w_min_idx = 0
    expected = (-3.0 - alpha_grid) ** 2
    assert np.allclose(r[w_min_idx], expected, atol=1e-6)


def test_training_values_quartic():
    f = _make_symmetric_quartic()
    alpha_grid = np.linspace(0.1, 3.0, 1000)
    cands = build_candidate_family(f, alpha_grid)
    r = compute_training_values(f, cands, alpha_grid)

    # Sort interior branches by midpoint w-value
    interiors = [(i, c) for i, c in enumerate(cands) if c.kind == "interior"]
    interiors.sort(key=lambda x: x[1].w_values[len(alpha_grid) // 2])

    # w = -sqrt(alpha) and w = +sqrt(alpha) both have training value 0
    assert np.allclose(r[interiors[0][0]], 0.0, atol=1e-6), "Negative branch should have r=0"
    assert np.allclose(r[interiors[2][0]], 0.0, atol=1e-6), "Positive branch should have r=0"

    # w = 0 has training value alpha^2
    assert np.allclose(r[interiors[1][0]], alpha_grid**2, atol=1e-6), "Zero branch should have r=alpha^2"


def test_active_set_simple():
    """For f = (w - alpha)^2, only the interior branch is active (achieves r=0)."""
    f = _make_simple_quadratic()
    alpha_grid = np.linspace(0.5, 2.5, 500)
    cands = build_candidate_family(f, alpha_grid)
    r = compute_training_values(f, cands, alpha_grid)

    active = find_active_set(r)
    active_idx = find_active_branches(r)

    interior_idx = [i for i, c in enumerate(cands) if c.kind == "interior"]
    assert list(active_idx) == interior_idx


def test_active_set_quartic():
    """For f = (w^2 - alpha)^2, the two sqrt branches are active (both achieve r=0).
    The w=0 branch and endpoints are not active.
    """
    f = _make_symmetric_quartic()
    alpha_grid = np.linspace(0.1, 3.0, 1000)
    cands = build_candidate_family(f, alpha_grid)
    r = compute_training_values(f, cands, alpha_grid)

    active_idx = find_active_branches(r)

    # Should be exactly the two sqrt branches (not w=0, not endpoints)
    interiors = [(i, c) for i, c in enumerate(cands) if c.kind == "interior"]
    interiors.sort(key=lambda x: x[1].w_values[len(alpha_grid) // 2])

    neg_idx, zero_idx, pos_idx = interiors[0][0], interiors[1][0], interiors[2][0]
    assert neg_idx in active_idx
    assert pos_idx in active_idx
    assert zero_idx not in active_idx


def test_tie_points_simple():
    """For f = (w - alpha)^2, the unique minimizer never ties with endpoints
    (since r_interior = 0 < r_endpoint for all alpha in range). No tie points.
    """
    f = _make_simple_quadratic()
    alpha_grid = np.linspace(0.5, 2.5, 500)
    cands = build_candidate_family(f, alpha_grid)
    r = compute_training_values(f, cands, alpha_grid)

    ties = find_tie_points(r, alpha_grid)
    assert len(ties) == 0


def test_tie_points_two_piece():
    """Two-piece: f1 = (w + alpha/2)^2 on [-3,0], f2 = (w - alpha)^2 on [0,3].
    Interior branches: w1 = -alpha/2 (r=0), w2 = alpha (r=0).
    Both achieve r=0 for all alpha, so they're tied everywhere.
    But the boundary w*=0 has r = f1(alpha, 0) = alpha^2/4 and
    r = f2(alpha, 0) = alpha^2. These tie with r=0 at alpha=0 (outside range).
    So within [0.5, 2.5]: the two interior branches are always tied (both r=0),
    and no branch ties with them. Tie points should be empty or just between
    the two r=0 branches (which are identically tied, not crossing).
    """
    c1 = np.zeros((3, 3))
    c1[2, 0] = 0.25
    c1[1, 1] = 1.0
    c1[0, 2] = 1.0

    c2 = np.zeros((3, 3))
    c2[2, 0] = 1.0
    c2[1, 1] = -2.0
    c2[0, 2] = 1.0

    f = TrainingLoss(
        pieces=[(-3.0, 0.0, c1), (0.0, 3.0, c2)],
        alpha_range=(0.5, 2.5),
        w_range=(-3.0, 3.0),
    )
    alpha_grid = np.linspace(0.5, 2.5, 500)
    cands = build_candidate_family(f, alpha_grid)
    r = compute_training_values(f, cands, alpha_grid)

    # The two interior branches both have r=0, so diff is identically 0.
    # find_tie_points looks for sign changes, so it should find none.
    ties = find_tie_points(r, alpha_grid)

    # There might be ties between non-active branches (e.g. endpoint vs boundary),
    # but no ties that involve the active branches crossing each other.
    # The key check: active set should still be constant.
    intervals = check_active_set_constancy(r, alpha_grid, ties)
    for iv in intervals:
        assert iv["constant"], f"Active set not constant on {iv['alpha_range']}"


def test_active_set_constancy_quartic():
    """Theorem 25 check: active set constant on tame intervals for symmetric quartic."""
    f = _make_symmetric_quartic()
    alpha_grid = np.linspace(0.1, 3.0, 1000)
    cands = build_candidate_family(f, alpha_grid)
    r = compute_training_values(f, cands, alpha_grid)

    ties = find_tie_points(r, alpha_grid)
    intervals = check_active_set_constancy(r, alpha_grid, ties)

    for iv in intervals:
        assert iv["constant"], (
            f"Active set not constant on {iv['alpha_range']}: "
            f"active = {iv['active_indices']}"
        )


if __name__ == "__main__":
    test_training_values_simple()
    test_training_values_quartic()
    test_active_set_simple()
    test_active_set_quartic()
    test_tie_points_simple()
    test_tie_points_two_piece()
    test_active_set_constancy_quartic()
    print("All active set tests passed.")
