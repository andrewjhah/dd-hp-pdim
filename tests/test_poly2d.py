"""Tests for poly2d.py."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.poly2d import (
    eval_poly2d,
    eval_poly2d_grid,
    partial_alpha,
    partial_alpha_w,
    partial_w,
    partial_ww,
    total_degree,
    trim,
)


def test_eval_constant():
    """p(alpha, w) = 5."""
    c = np.array([[5.0]])
    assert eval_poly2d(c, 0.0, 0.0) == 5.0
    assert eval_poly2d(c, 1.0, 2.0) == 5.0


def test_eval_linear():
    """p(alpha, w) = 3*alpha + 2*w.
    c[1,0] = 3, c[0,1] = 2.
    """
    c = np.array([[0.0, 2.0], [3.0, 0.0]])
    assert eval_poly2d(c, 1.0, 0.0) == 3.0
    assert eval_poly2d(c, 0.0, 1.0) == 2.0
    assert eval_poly2d(c, 1.0, 1.0) == 5.0
    assert eval_poly2d(c, 2.0, 3.0) == 12.0  # 3*2 + 2*3


def test_eval_quadratic():
    """p(alpha, w) = alpha^2 - 2*alpha*w + w^2 = (alpha - w)^2.
    c[2,0]=1, c[1,1]=-2, c[0,2]=1.
    """
    c = np.array([[0.0, 0.0, 1.0], [0.0, -2.0, 0.0], [1.0, 0.0, 0.0]])
    assert eval_poly2d(c, 3.0, 3.0) == 0.0
    assert eval_poly2d(c, 5.0, 3.0) == 4.0  # (5-3)^2
    assert eval_poly2d(c, 1.0, 4.0) == 9.0  # (1-4)^2


def test_eval_quartic_training():
    """f(alpha, w) = (w^2 - alpha)^2 = w^4 - 2*alpha*w^2 + alpha^2.
    c[0,4]=1, c[1,2]=-2, c[2,0]=1.
    """
    c = np.zeros((3, 5))
    c[2, 0] = 1.0  # alpha^2
    c[1, 2] = -2.0  # -2*alpha*w^2
    c[0, 4] = 1.0  # w^4

    # At w^2 = alpha, should be 0
    assert abs(eval_poly2d(c, 4.0, 2.0)) < 1e-12  # 2^2 = 4
    assert abs(eval_poly2d(c, 4.0, -2.0)) < 1e-12
    assert abs(eval_poly2d(c, 1.0, 1.0)) < 1e-12

    # At w=0, should be alpha^2
    assert abs(eval_poly2d(c, 3.0, 0.0) - 9.0) < 1e-12


def test_eval_grid():
    """Test vectorized grid evaluation matches pointwise."""
    c = np.zeros((3, 5))
    c[2, 0] = 1.0
    c[1, 2] = -2.0
    c[0, 4] = 1.0

    alpha_arr = np.array([0.5, 1.0, 2.0, 4.0])
    w_arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    grid = eval_poly2d_grid(c, alpha_arr, w_arr)

    for ia, a in enumerate(alpha_arr):
        for iw, w in enumerate(w_arr):
            expected = eval_poly2d(c, a, w)
            assert abs(grid[ia, iw] - expected) < 1e-12, (
                f"Mismatch at alpha={a}, w={w}: {grid[ia, iw]} vs {expected}"
            )


def test_partial_w_linear():
    """p = 3*alpha + 2*w  =>  d/dw p = 2."""
    c = np.array([[0.0, 2.0], [3.0, 0.0]])
    dw = partial_w(c)
    assert dw.shape == (2, 1)
    assert dw[0, 0] == 2.0
    assert dw[1, 0] == 0.0


def test_partial_w_quartic():
    """f = w^4 - 2*alpha*w^2 + alpha^2  =>  d/dw f = 4w^3 - 4*alpha*w."""
    c = np.zeros((3, 5))
    c[2, 0] = 1.0
    c[1, 2] = -2.0
    c[0, 4] = 1.0

    dw = partial_w(c)
    # dw should have shape (3, 4): coefficients of alpha^i * w^j
    # d/dw(w^4) = 4*w^3, d/dw(-2*alpha*w^2) = -4*alpha*w
    assert dw.shape == (3, 4)
    assert dw[0, 3] == 4.0  # 4*w^3
    assert dw[1, 1] == -4.0  # -4*alpha*w
    # All others should be 0
    for i in range(3):
        for j in range(4):
            if (i, j) not in [(0, 3), (1, 1)]:
                assert dw[i, j] == 0.0, f"dw[{i},{j}] = {dw[i, j]}, expected 0"

    # Check evaluation: at alpha=1, w=1: 4(1)^3 - 4(1)(1) = 0
    assert abs(eval_poly2d(dw, 1.0, 1.0)) < 1e-12
    # At alpha=1, w=2: 4(8) - 4(1)(2) = 32 - 8 = 24
    assert abs(eval_poly2d(dw, 1.0, 2.0) - 24.0) < 1e-12


def test_partial_alpha_quartic():
    """f = w^4 - 2*alpha*w^2 + alpha^2  =>  d/dalpha f = -2*w^2 + 2*alpha."""
    c = np.zeros((3, 5))
    c[2, 0] = 1.0
    c[1, 2] = -2.0
    c[0, 4] = 1.0

    da = partial_alpha(c)
    assert da.shape == (2, 5)
    assert da[1, 0] == 2.0  # 2*alpha
    assert da[0, 2] == -2.0  # -2*w^2

    # At alpha=2, w=2: -2(4) + 2(2) = -8 + 4 = -4
    assert abs(eval_poly2d(da, 2.0, 2.0) - (-4.0)) < 1e-12


def test_partial_ww_quartic():
    """f = w^4 - 2*alpha*w^2 + alpha^2  =>  d^2/dw^2 f = 12*w^2 - 4*alpha."""
    c = np.zeros((3, 5))
    c[2, 0] = 1.0
    c[1, 2] = -2.0
    c[0, 4] = 1.0

    dww = partial_ww(c)
    # d/dw(4*w^3 - 4*alpha*w) = 12*w^2 - 4*alpha
    assert dww[0, 2] == 12.0  # 12*w^2
    assert dww[1, 0] == -4.0  # -4*alpha

    # At alpha=3, w=1: 12 - 12 = 0
    assert abs(eval_poly2d(dww, 3.0, 1.0)) < 1e-12
    # At alpha=3, w=2: 48 - 12 = 36
    assert abs(eval_poly2d(dww, 3.0, 2.0) - 36.0) < 1e-12


def test_partial_alpha_w():
    """f = w^4 - 2*alpha*w^2 + alpha^2  =>  d^2/(dalpha dw) f = -4*w."""
    c = np.zeros((3, 5))
    c[2, 0] = 1.0
    c[1, 2] = -2.0
    c[0, 4] = 1.0

    daw = partial_alpha_w(c)
    # d/dalpha(4*w^3 - 4*alpha*w) = -4*w
    assert abs(eval_poly2d(daw, 0.0, 3.0) - (-12.0)) < 1e-12
    assert abs(eval_poly2d(daw, 5.0, 3.0) - (-12.0)) < 1e-12  # no alpha dependence


def test_partial_w_constant():
    """d/dw of a constant is 0."""
    c = np.array([[7.0]])
    dw = partial_w(c)
    assert dw.shape == (1, 1)
    assert dw[0, 0] == 0.0


def test_total_degree():
    c = np.zeros((3, 5))
    c[2, 0] = 1.0  # alpha^2: degree 2
    c[1, 2] = -2.0  # alpha*w^2: degree 3
    c[0, 4] = 1.0  # w^4: degree 4
    assert total_degree(c) == 4


def test_trim():
    c = np.zeros((5, 5))
    c[0, 0] = 1.0
    c[1, 2] = 3.0
    t = trim(c)
    assert t.shape == (2, 3)
    assert t[0, 0] == 1.0
    assert t[1, 2] == 3.0


def test_derivatives_against_finite_differences():
    """Cross-check all derivatives against finite differences on a random polynomial."""
    rng = np.random.default_rng(42)
    c = rng.standard_normal((4, 4))  # random degree-3 polynomial in each variable

    dw_c = partial_w(c)
    da_c = partial_alpha(c)
    dww_c = partial_ww(c)
    daw_c = partial_alpha_w(c)

    h1 = 1e-7  # step for first derivatives
    h2 = 1e-4  # larger step for second derivatives (better numerical conditioning)
    for _ in range(20):
        a = rng.uniform(0.5, 3.0)
        w = rng.uniform(-2.0, 2.0)

        # d/dw via finite difference
        fd_dw = (eval_poly2d(c, a, w + h1) - eval_poly2d(c, a, w - h1)) / (2 * h1)
        exact_dw = eval_poly2d(dw_c, a, w)
        assert abs(fd_dw - exact_dw) < 1e-5, f"d/dw: fd={fd_dw}, exact={exact_dw}"

        # d/dalpha
        fd_da = (eval_poly2d(c, a + h1, w) - eval_poly2d(c, a - h1, w)) / (2 * h1)
        exact_da = eval_poly2d(da_c, a, w)
        assert abs(fd_da - exact_da) < 1e-5, f"d/da: fd={fd_da}, exact={exact_da}"

        # d^2/dw^2
        fd_dww = (
            eval_poly2d(c, a, w + h2)
            - 2 * eval_poly2d(c, a, w)
            + eval_poly2d(c, a, w - h2)
        ) / h2**2
        exact_dww = eval_poly2d(dww_c, a, w)
        assert abs(fd_dww - exact_dww) < 1e-3, (
            f"d^2/dw^2: fd={fd_dww}, exact={exact_dww}"
        )

        # d^2/(dalpha dw)
        fd_daw = (
            eval_poly2d(c, a + h2, w + h2)
            - eval_poly2d(c, a + h2, w - h2)
            - eval_poly2d(c, a - h2, w + h2)
            + eval_poly2d(c, a - h2, w - h2)
        ) / (4 * h2**2)
        exact_daw = eval_poly2d(daw_c, a, w)
        assert abs(fd_daw - exact_daw) < 1e-3, (
            f"d^2/(da dw): fd={fd_daw}, exact={exact_daw}"
        )


if __name__ == "__main__":
    test_eval_constant()
    test_eval_linear()
    test_eval_quadratic()
    test_eval_quartic_training()
    test_eval_grid()
    test_partial_w_linear()
    test_partial_w_quartic()
    test_partial_alpha_quartic()
    test_partial_ww_quartic()
    test_partial_alpha_w()
    test_partial_w_constant()
    test_total_degree()
    test_trim()
    test_derivatives_against_finite_differences()
    print("All poly2d tests passed.")
