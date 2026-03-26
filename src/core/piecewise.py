"""
Piecewise polynomial functions on A \times W.

Each piece is defined by a w-interval [w_lo, w_hi] and a 2d coefficient array.
"""

import numpy as np

from .poly2d import eval_poly2d, partial_alpha, partial_alpha_w, partial_w, partial_ww


class TrainingLoss:
    """Piecewise polynomial training loss f(alpha, w) on A x W.
    Pieces are ordered by w-interval, and adjacent pieces
    share a boundary value (pieces[i].w_hi == pieces[i+1].w_lo).
    """

    def __init__(self, pieces, alpha_range, w_range):
        """
        Parameters
        ----------
        pieces : list of (w_lo, w_hi, coeff)
        alpha_range : (float, float)
            Hyperparameter domain A = [alpha_min, alpha_max].
        w_range : (float, float)
            Weight domain W = [w_min, w_max].
        """
        self.pieces = pieces
        self.alpha_min, self.alpha_max = alpha_range
        self.w_min, self.w_max = w_range
        self._validate()

    def _validate(self):
        sorted_pieces = sorted(self.pieces, key=lambda p: p[0])
        assert abs(sorted_pieces[0][0] - self.w_min) < 1e-12, (
            f"First piece starts at {sorted_pieces[0][0]}, expected w_min={self.w_min}"
        )
        assert abs(sorted_pieces[-1][1] - self.w_max) < 1e-12, (
            f"Last piece ends at {sorted_pieces[-1][1]}, expected w_max={self.w_max}"
        )
        for i in range(len(sorted_pieces) - 1):
            assert abs(sorted_pieces[i][1] - sorted_pieces[i + 1][0]) < 1e-12, (
                f"Gap between piece {i} (ends {sorted_pieces[i][1]}) and piece {i + 1} (starts {sorted_pieces[i + 1][0]})"
            )

    @property
    def M_f(self):
        """Number of training pieces."""
        return len(self.pieces)

    @property
    def alpha_range(self):
        return (self.alpha_min, self.alpha_max)

    @property
    def w_range(self):
        return (self.w_min, self.w_max)

    def boundary_w_values(self):
        """Roots of b_k^f in the interior of W."""
        boundaries = set()
        for w_lo, w_hi, _ in self.pieces:
            boundaries.add(w_lo)
            boundaries.add(w_hi)
        boundaries.discard(self.w_min)
        boundaries.discard(self.w_max)
        return sorted(boundaries)

    @property
    def T_f(self):
        """Number of internal training boundaries."""
        return len(self.boundary_w_values())

    def get_piece(self, w):
        """Return (piece_index, coeff) for the piece containing w."""
        for idx, (w_lo, w_hi, coeff) in enumerate(self.pieces):
            if w_lo - 1e-12 <= w <= w_hi + 1e-12:
                return idx, coeff
        return None, None

    def eval(self, alpha, w):
        """Evaluate f(alpha, w)."""
        _, coeff = self.get_piece(w)
        if coeff is None:
            raise ValueError(f"w={w} outside all pieces (w_range={self.w_range})")
        return eval_poly2d(coeff, alpha, w)

    def eval_w_array(self, alpha, w_arr):
        return np.array([self.eval(alpha, w) for w in w_arr])

    def minimize_over_w(self, alpha, n_grid=1000):
        """Brute-force minimize f(alpha, w) over W for a fixed alpha."""
        w_arr = np.linspace(self.w_min, self.w_max, n_grid)
        f_arr = self.eval_w_array(alpha, w_arr)
        idx = np.argmin(f_arr)
        return w_arr[idx], f_arr[idx]


class ValidationLoss:
    """Piecewise polynomial validation loss g(alpha, w) on A x W."""

    def __init__(self, pieces, alpha_range, w_range):
        self.pieces = pieces
        self.alpha_min, self.alpha_max = alpha_range
        self.w_min, self.w_max = w_range

    @property
    def M_g(self):
        return len(self.pieces)

    def get_piece(self, alpha, w):
        for idx, (w_lo, w_hi, coeff) in enumerate(self.pieces):
            if w_lo - 1e-12 <= w <= w_hi + 1e-12:
                return idx, coeff
        return None, None

    def eval(self, alpha, w):
        _, coeff = self.get_piece(alpha, w)
        if coeff is None:
            raise ValueError(f"(alpha={alpha}, w={w}) outside all pieces")
        return eval_poly2d(coeff, alpha, w)
