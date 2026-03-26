"""
Training values, tie loci, and active set identification.

Given candidates, we compute:
  - r_w(alpha) = f(alpha, w(alpha)) for each candidate
  - Training-value tie points where r_w = r_u for distinct w, u
  - The active set A_I: candidates achieving the training minimum
"""

import numpy as np
from scipy.optimize import brentq


def compute_training_values(training_loss, candidates, alpha_grid):
    """Compute r_w(alpha) for each candidate branch.
    Returns array of shape (n_candidates, len(alpha_grid)).
    """
    n_cands = len(candidates)
    n_alpha = len(alpha_grid)
    r = np.zeros((n_cands, n_alpha))

    for i, cand in enumerate(candidates):
        for j, alpha in enumerate(alpha_grid):
            r[i, j] = training_loss.eval(alpha, cand.w_values[j])

    return r


def find_tie_points(training_values, alpha_grid):
    """Find alpha-values where two candidates have equal training values.

    Returns a sorted list of approximate tie-point alpha values.
    """
    n_cands = training_values.shape[0]
    ties = []

    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            diff = training_values[i] - training_values[j]
            for k in range(len(diff) - 1):
                if diff[k] * diff[k + 1] < 0:
                    # Linear interpolation for the crossing
                    alpha_tie = alpha_grid[k] - diff[k] * (
                        alpha_grid[k + 1] - alpha_grid[k]
                    ) / (diff[k + 1] - diff[k])
                    ties.append(alpha_tie)

    if not ties:
        return []
    ties = sorted(ties)
    deduped = [ties[0]]
    for t in ties[1:]:
        if t - deduped[-1] > 1e-8:
            deduped.append(t)
    return deduped


def find_active_set(training_values, tol=1e-8):
    """Identify active candidates at each alpha point.

    A candidate is active at alpha if its training value equals the minimum.
    Returns boolean array of shape (n_candidates, n_alpha).
    """
    r_min = np.min(training_values, axis=0)  # (n_alpha,)
    return np.abs(training_values - r_min[np.newaxis, :]) < tol


def find_active_branches(training_values, tol=1e-8):
    """Return indices of candidates that are active for at least one alpha."""
    active_mask = find_active_set(training_values, tol)
    return np.where(np.any(active_mask, axis=1))[0]


def check_active_set_constancy(training_values, alpha_grid, tie_points, tol=1e-8):
    """Verify that the active set is constant on each tame interval.

    Returns a list of dicts, one per tame interval, each with:
      'alpha_range': (lo, hi) of the interval
      'active_indices':set of active candidate indices
      'constant': whether the active set was indeed constant
    """
    alpha_lo = alpha_grid[0]
    alpha_hi = alpha_grid[-1]

    boundaries = [alpha_lo] + tie_points + [alpha_hi]
    intervals = []

    active_mask = find_active_set(training_values, tol)

    for seg in range(len(boundaries) - 1):
        lo, hi = boundaries[seg], boundaries[seg + 1]
        # Find grid points strictly inside this interval
        margin = (alpha_grid[1] - alpha_grid[0]) * 2
        in_interval = (alpha_grid > lo + margin) & (alpha_grid < hi - margin)
        if not np.any(in_interval):
            continue

        idx = np.where(in_interval)[0]
        # Active set at each point in the interval
        active_sets = [frozenset(np.where(active_mask[:, j])[0]) for j in idx]
        is_constant = len(set(active_sets)) == 1

        intervals.append(
            {
                "alpha_range": (lo, hi),
                "active_indices": active_sets[0] if active_sets else frozenset(),
                "constant": is_constant,
            }
        )

    return intervals
