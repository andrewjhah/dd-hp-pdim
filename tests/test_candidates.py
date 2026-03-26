"""Tests for candidate family assembly."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.candidates import build_candidate_family
from src.core.piecewise import TrainingLoss


def test_single_piece_quadratic():
    """f = (w - alpha)^2, w in [-3, 3].
    Candidates: w_min=-3, w_max=3, one interior branch w=alpha.
    Total: 3 (Corollary 11: 2 + 0 + 1*(2-1) = 3).
    """
    c = np.zeros((3, 3))
    c[2, 0] = 1.0
    c[1, 1] = -2.0
    c[0, 2] = 1.0

    f = TrainingLoss(
        pieces=[(-3.0, 3.0, c)], alpha_range=(0.5, 2.5), w_range=(-3.0, 3.0)
    )
    alpha_grid = np.linspace(0.5, 2.5, 500)

    cands = build_candidate_family(f, alpha_grid)

    endpoints = [c for c in cands if c.kind == "endpoint"]
    boundaries = [c for c in cands if c.kind == "boundary"]
    interiors = [c for c in cands if c.kind == "interior"]

    assert len(endpoints) == 2
    assert len(boundaries) == 0
    assert len(interiors) == 1
    assert len(cands) == 3

    # Check endpoint values
    assert np.all(endpoints[0].w_values == -3.0)
    assert np.all(endpoints[1].w_values == 3.0)

    # Check interior branch
    assert np.allclose(interiors[0].w_values, alpha_grid, atol=1e-6)


def test_symmetric_quartic():
    """f = (w^2 - alpha)^2, w in [-2, 2].
    Candidates: w_min, w_max, 3 interior branches (w = -sqrt(a), 0, +sqrt(a)).
    Total: 5 (Corollary 11: 2 + 0 + 1*(4-1) = 5).
    """
    c = np.zeros((3, 5))
    c[2, 0] = 1.0
    c[1, 2] = -2.0
    c[0, 4] = 1.0

    f = TrainingLoss(
        pieces=[(-2.0, 2.0, c)], alpha_range=(0.1, 3.0), w_range=(-2.0, 2.0)
    )
    alpha_grid = np.linspace(0.1, 3.0, 1000)

    cands = build_candidate_family(f, alpha_grid)

    endpoints = [c for c in cands if c.kind == "endpoint"]
    interiors = [c for c in cands if c.kind == "interior"]

    assert len(endpoints) == 2
    assert len(interiors) == 3
    assert len(cands) == 5


def test_two_piece_with_boundary():
    """Two-piece training with boundary at w=0.
    Candidates: w_min=-3, w_max=3, boundary w*=0, plus 2 interior branches.
    Total: 5 (Corollary 11: 2 + 1*2 + 2*(2-1) = 6, but T_f*Delta counts
    roots of boundary polys, not boundaries themselves. Here T_f=1.)
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

    endpoints = [c for c in cands if c.kind == "endpoint"]
    boundaries = [c for c in cands if c.kind == "boundary"]
    interiors = [c for c in cands if c.kind == "interior"]

    assert len(endpoints) == 2
    assert len(boundaries) == 1
    assert boundaries[0].w_values[0] == 0.0
    assert len(interiors) == 2
    assert len(cands) == 5

    # Every global minimizer should be in the candidate family.
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5]:
        w_star, _ = f.minimize_over_w(alpha, n_grid=5000)
        idx = np.argmin(np.abs(alpha_grid - alpha))
        cand_w_values = [c.w_values[idx] for c in cands]
        dists = [abs(w_star - cw) for cw in cand_w_values]
        assert min(dists) < 0.05, (
            f"At alpha={alpha}, minimizer w*={w_star:.4f} not close to any candidate: {cand_w_values}"
        )


def test_candidate_labels():
    """Check that labels and kinds are set correctly."""
    c = np.zeros((3, 3))
    c[2, 0] = 1.0
    c[1, 1] = -2.0
    c[0, 2] = 1.0

    f = TrainingLoss(
        pieces=[(-3.0, 3.0, c)], alpha_range=(0.5, 2.5), w_range=(-3.0, 3.0)
    )
    alpha_grid = np.linspace(0.5, 2.5, 100)

    cands = build_candidate_family(f, alpha_grid)

    assert cands[0].label == "w_min"
    assert cands[0].is_constant()
    assert cands[1].label == "w_max"
    assert cands[1].is_constant()
    assert not cands[2].is_constant()
    assert cands[2].piece_idx == 0


if __name__ == "__main__":
    test_single_piece_quadratic()
    test_symmetric_quartic()
    test_two_piece_with_boundary()
    test_candidate_labels()
    print("All candidate family tests passed.")
