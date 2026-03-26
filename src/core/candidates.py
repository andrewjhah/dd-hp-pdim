"""
Collects the three types of candidate branches on a given alpha grid:
  1. Endpoint branches: w(alpha) = w_min or w_max (constant)
  2. Boundary branches: w(alpha) = w_k* for each internal training boundary value
  3. Interior critical branches: smooth curves from d_w p_m = 0
"""

import numpy as np

from .branches import find_interior_branches


class CandidateBranch:
    """A single candidate branch w(alpha)."""

    def __init__(self, kind, label, w_values, piece_idx=None):
        """
        kind: 'endpoint', 'boundary', or 'interior'
        label: human-readable name
        w_values: array of w(alpha) values on the alpha grid
        piece_idx: training piece index (for interior branches)
        """
        self.kind = kind
        self.label = label
        self.w_values = w_values
        self.piece_idx = piece_idx

    def is_constant(self):
        return self.kind in ("endpoint", "boundary")

    def __repr__(self):
        return f"CandidateBranch({self.kind}, {self.label!r})"


def build_candidate_family(training_loss, alpha_grid, n_scan=500):
    """Build the full candidate family B_I for a training loss on alpha_grid.

    Returns a list of CandidateBranch objects.
    """
    n = len(alpha_grid)
    candidates = []

    # Endpoint branches
    candidates.append(
        CandidateBranch(
            kind="endpoint",
            label="w_min",
            w_values=np.full(n, training_loss.w_min),
        )
    )
    candidates.append(
        CandidateBranch(
            kind="endpoint",
            label="w_max",
            w_values=np.full(n, training_loss.w_max),
        )
    )

    # Boundary branches
    for w_star in training_loss.boundary_w_values():
        candidates.append(
            CandidateBranch(
                kind="boundary",
                label=f"w*={w_star:.4g}",
                w_values=np.full(n, w_star),
            )
        )

    # Interior critical branches
    interior = find_interior_branches(training_loss, alpha_grid, n_scan)
    for i, branch in enumerate(interior):
        candidates.append(
            CandidateBranch(
                kind="interior",
                label=f"interior_m{branch['piece_idx']}_{i}",
                w_values=branch["w_values"],
                piece_idx=branch["piece_idx"],
            )
        )

    return candidates
