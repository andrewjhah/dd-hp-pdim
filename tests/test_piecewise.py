"""Tests for piecewise.py."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.piecewise import TrainingLoss, ValidationLoss


def test_single_piece_quadratic():
    """f(alpha, w) = (w - alpha)^2 = w^2 - 2*alpha*w + alpha^2.
    Minimizer: w* = alpha. Min value: 0.
    """
    c = np.zeros((3, 3))
    c[2, 0] = 1.0  # alpha^2
    c[1, 1] = -2.0  # -2*alpha*w
    c[0, 2] = 1.0  # w^2

    f = TrainingLoss(
        pieces=[(-3.0, 3.0, c)],
        alpha_range=(0.5, 2.5),
        w_range=(-3.0, 3.0),
    )

    assert f.M_f == 1
    assert f.T_f == 0
    assert f.boundary_w_values() == []

    assert abs(f.eval(1.0, 1.0)) < 1e-12
    assert abs(f.eval(2.0, 2.0)) < 1e-12

    assert abs(f.eval(1.0, 3.0) - 4.0) < 1e-12  # (3-1)^2

    # Brute-force minimization
    w_star, f_star = f.minimize_over_w(1.5, n_grid=5000)
    assert abs(w_star - 1.5) < 0.01
    assert abs(f_star) < 0.01


def test_two_piece_training():
    """Two-piece quadratic training loss with boundary at w=0.
    For w >= 0: f = (w - alpha)^2
    For w < 0:  f = (w + alpha/2)^2
    """
    # Piece 1: w in [-3, 0], f = w^2 + alpha*w + alpha^2/4
    c1 = np.zeros((3, 3))
    c1[2, 0] = 0.25  # alpha^2/4
    c1[1, 1] = 1.0  # alpha*w
    c1[0, 2] = 1.0  # w^2

    # Piece 2: w in [0, 3], f = w^2 - 2*alpha*w + alpha^2
    c2 = np.zeros((3, 3))
    c2[2, 0] = 1.0  # alpha^2
    c2[1, 1] = -2.0  # -2*alpha*w
    c2[0, 2] = 1.0  # w^2

    f = TrainingLoss(
        pieces=[(-3.0, 0.0, c1), (0.0, 3.0, c2)],
        alpha_range=(0.5, 2.5),
        w_range=(-3.0, 3.0),
    )

    assert f.M_f == 2
    assert f.T_f == 1
    assert f.boundary_w_values() == [0.0]

    # Piece 2 minimizer at w = alpha
    assert abs(f.eval(1.0, 1.0)) < 1e-12
    # Piece 1 minimizer at w = -alpha/2
    assert abs(f.eval(1.0, -0.5)) < 1e-12

    # For alpha=1: piece 2 min = 0 at w=1, piece 1 min = 0 at w=-0.5
    w_star, f_star = f.minimize_over_w(1.0, n_grid=5000)
    assert abs(f_star) < 0.01


def test_validation_single_piece():
    """g(alpha, w) = w^2 - 2w + 1 = (w-1)^2."""
    c = np.zeros((1, 3))
    c[0, 0] = 1.0
    c[0, 1] = -2.0
    c[0, 2] = 1.0

    g = ValidationLoss(
        pieces=[(-2.0, 2.0, c)],
        alpha_range=(0.1, 3.0),
        w_range=(-2.0, 2.0),
    )

    assert g.M_g == 1
    assert abs(g.eval(1.0, 1.0)) < 1e-12  # (1-1)^2 = 0
    assert abs(g.eval(1.0, 0.0) - 1.0) < 1e-12  # (0-1)^2 = 1


def test_validation_with_alpha():
    """g(alpha, w) = w^2 + alpha*w."""
    c = np.zeros((2, 3))
    c[0, 2] = 1.0  # w^2
    c[1, 1] = 1.0  # alpha*w

    g = ValidationLoss(
        pieces=[(-3.0, 3.0, c)],
        alpha_range=(0.5, 4.0),
        w_range=(-3.0, 3.0),
    )

    # At alpha=2, w=1: 1 + 2 = 3
    assert abs(g.eval(2.0, 1.0) - 3.0) < 1e-12
    # At alpha=2, w=-1: 1 - 2 = -1
    assert abs(g.eval(2.0, -1.0) - (-1.0)) < 1e-12


def test_validation_fails_outside():
    """Querying outside the piece range should raise."""
    c = np.zeros((1, 2))
    c[0, 1] = 1.0

    g = ValidationLoss(
        pieces=[(0.0, 1.0, c)],
        alpha_range=(0.0, 1.0),
        w_range=(0.0, 1.0),
    )

    try:
        g.eval(0.5, 2.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_training_piece_lookup():
    """get_piece returns the correct piece index."""
    c1 = np.array([[1.0]])
    c2 = np.array([[2.0]])
    c3 = np.array([[3.0]])

    f = TrainingLoss(
        pieces=[(-1.0, 0.0, c1), (0.0, 1.0, c2), (1.0, 2.0, c3)],
        alpha_range=(0.0, 1.0),
        w_range=(-1.0, 2.0),
    )

    idx, c = f.get_piece(-0.5)
    assert idx == 0 and c[0, 0] == 1.0

    idx, c = f.get_piece(0.5)
    assert idx == 1 and c[0, 0] == 2.0

    idx, c = f.get_piece(1.5)
    assert idx == 2 and c[0, 0] == 3.0

    idx, c = f.get_piece(0.0)
    assert idx in [0, 1]  # either adjacent piece is fine


if __name__ == "__main__":
    test_single_piece_quadratic()
    test_two_piece_training()
    test_validation_single_piece()
    test_validation_with_alpha()
    test_validation_fails_outside()
    test_training_piece_lookup()
    print("All piecewise tests passed.")
