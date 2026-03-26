"""Tests for numerical branch tracking."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.branches import find_interior_branches, find_roots_in_interval
from src.core.piecewise import TrainingLoss


def test_find_roots_quadratic():
    """Roots of w^2 - 1 = 0 in (-2, 2) should be {-1, 1}."""
    roots = find_roots_in_interval(lambda w: w**2 - 1, -2, 2)
    assert len(roots) == 2
    roots = sorted(roots)
    assert abs(roots[0] - (-1.0)) < 1e-10
    assert abs(roots[1] - 1.0) < 1e-10


def test_find_roots_cubic():
    """Roots of w(w-1)(w+1) = w^3 - w in (-2, 2) should be {-1, 0, 1}."""
    roots = find_roots_in_interval(lambda w: w**3 - w, -2, 2)
    assert len(roots) == 3
    roots = sorted(roots)
    assert abs(roots[0] - (-1.0)) < 1e-10
    assert abs(roots[1] - 0.0) < 1e-10
    assert abs(roots[2] - 1.0) < 1e-10


def test_symmetric_quartic_branches():
    """f(alpha, w) = (w^2 - alpha)^2, domain alpha in [0.1, 3], w in [-2, 2].

    d_w f = 4w(w^2 - alpha), so interior critical points are w = +/- sqrt(alpha)
    (the root w=0 is a local max for alpha > 0, but it's still a critical point).
    Non-degenerate roots: d_ww f = 12w^2 - 4*alpha.
      At w = +/- sqrt(alpha): d_ww = 12*alpha - 4*alpha = 8*alpha != 0. Good.
      At w = 0: d_ww = -4*alpha < 0. Also non-degenerate.
    So we expect 3 branches: w = -sqrt(alpha), w = 0, w = +sqrt(alpha).
    """
    c = np.zeros((3, 5))
    c[2, 0] = 1.0  # alpha^2
    c[1, 2] = -2.0  # -2*alpha*w^2
    c[0, 4] = 1.0  # w^4

    f = TrainingLoss(
        pieces=[(-2.0, 2.0, c)], alpha_range=(0.1, 3.0), w_range=(-2.0, 2.0)
    )
    alpha_grid = np.linspace(0.1, 3.0, 1000)

    branches = find_interior_branches(f, alpha_grid)

    assert len(branches) == 3, f"Expected 3 branches, got {len(branches)}"

    mid = len(alpha_grid) // 2
    branches.sort(key=lambda b: b["w_values"][mid])

    # Branch 0: w = -sqrt(alpha)
    expected_neg = -np.sqrt(alpha_grid)
    assert np.allclose(branches[0]["w_values"], expected_neg, atol=1e-6), (
        "Negative branch mismatch"
    )

    # Branch 1: w = 0
    assert np.allclose(branches[1]["w_values"], 0.0, atol=1e-6), "Zero branch mismatch"

    # Branch 2: w = +sqrt(alpha)
    expected_pos = np.sqrt(alpha_grid)
    assert np.allclose(branches[2]["w_values"], expected_pos, atol=1e-6), (
        "Positive branch mismatch"
    )


def test_simple_quadratic_single_branch():
    """f(alpha, w) = (w - alpha)^2, domain alpha in [0.5, 2.5], w in [-3, 3].

    d_w f = 2(w - alpha), so unique interior critical point w = alpha.
    d_ww f = 2 != 0. One branch.
    """
    c = np.zeros((3, 3))
    c[2, 0] = 1.0
    c[1, 1] = -2.0
    c[0, 2] = 1.0

    f = TrainingLoss(
        pieces=[(-3.0, 3.0, c)], alpha_range=(0.5, 2.5), w_range=(-3.0, 3.0)
    )
    alpha_grid = np.linspace(0.5, 2.5, 500)

    branches = find_interior_branches(f, alpha_grid)
    assert len(branches) == 1

    expected = alpha_grid
    assert np.allclose(branches[0]["w_values"], expected, atol=1e-6)


def test_two_piece_training():
    """Two-piece quadratic:
      w in [-3, 0]: f = (w + alpha/2)^2
      w in [0, 3]:  f = (w - alpha)^2

    Piece 1 branch: w = -alpha/2 (in (-3, 0) when alpha in (0, 6))
    Piece 2 branch: w = alpha    (in (0, 3) when alpha in (0, 3))
    """
    # Piece 1: f = w^2 + alpha*w + alpha^2/4
    c1 = np.zeros((3, 3))
    c1[2, 0] = 0.25
    c1[1, 1] = 1.0
    c1[0, 2] = 1.0

    # Piece 2: f = w^2 - 2*alpha*w + alpha^2
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

    branches = find_interior_branches(f, alpha_grid)
    assert len(branches) == 2, f"Expected 2 branches, got {len(branches)}"

    mid = len(alpha_grid) // 2
    branches.sort(key=lambda b: b["w_values"][mid])

    # Branch 0 (piece 1): w = -alpha/2
    expected_neg = -alpha_grid / 2
    assert np.allclose(branches[0]["w_values"], expected_neg, atol=1e-5), (
        "Piece 1 branch mismatch"
    )

    # Branch 1 (piece 2): w = alpha
    expected_pos = alpha_grid
    assert np.allclose(branches[1]["w_values"], expected_pos, atol=1e-5), (
        "Piece 2 branch mismatch"
    )


if __name__ == "__main__":
    test_find_roots_quadratic()
    test_find_roots_cubic()
    test_symmetric_quartic_branches()
    test_simple_quadratic_single_branch()
    test_two_piece_training()
    print("All branch tracking tests passed.")
