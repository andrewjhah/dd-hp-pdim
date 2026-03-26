"""
Branch tracking for interior critical points.

Given a training piece p_m(alpha, w), an interior critical branch is a smooth
curve w(alpha) satisfying d_w p_m(alpha, w(alpha)) = 0 with d_ww p_m != 0.

We find branches at a starting alpha by root-scanning, then track them
across alpha using Newton's method.
"""

import numpy as np
from scipy.optimize import brentq

from .poly2d import eval_poly2d, partial_w, partial_ww


def find_roots_in_interval(func, lo, hi, n_scan=500):
    """Find all roots of func(w) = 0 in (lo, hi)."""
    w_scan = np.linspace(lo, hi, n_scan)
    vals = np.array([func(w) for w in w_scan])

    roots = []
    for k in range(len(vals) - 1):
        if vals[k] * vals[k + 1] < 0:
            try:
                root = brentq(func, w_scan[k], w_scan[k + 1])
                roots.append(root)
            except ValueError:
                pass
    return roots


def track_branch(dw_func, dww_func, w0, alpha_grid, w_lo, w_hi, margin=1e-8):
    """Track a single branch w(alpha) starting from w0 at alpha_grid[0].
    Returns an array of w-values.
    """
    w_track = np.full(len(alpha_grid), np.nan)
    w_track[0] = w0

    for i in range(1, len(alpha_grid)):
        alpha = alpha_grid[i]
        w_prev = w_track[i - 1]
        if np.isnan(w_prev):
            break

        # Newton iteration
        w_curr = w_prev
        converged = False
        for _ in range(30):
            fval = dw_func(alpha, w_curr)
            dfval = dww_func(alpha, w_curr)
            if abs(dfval) < 1e-14:
                break
            w_next = w_curr - fval / dfval
            if abs(w_next - w_curr) < 1e-12:
                w_curr = w_next
                converged = True
                break
            w_curr = w_next

        if (
            converged
            and w_lo + margin < w_curr < w_hi - margin
            and abs(dw_func(alpha, w_curr)) < 1e-8
        ):
            w_track[i] = w_curr
            continue

        delta = max(abs(w_curr - w_prev) * 3, 0.05)
        lo = max(w_lo + margin, w_prev - delta)
        hi = min(w_hi - margin, w_prev + delta)
        try:
            root = brentq(lambda w: dw_func(alpha, w), lo, hi)
            w_track[i] = root
        except (ValueError, RuntimeError):
            break

    return w_track


def find_interior_branches(training_loss, alpha_grid, n_scan=500):
    """Find all interior critical branches for each training piece.
    Returns a list of dicts, each with piece_idx and w_values.
    """
    branches = []

    for m, (w_lo, w_hi, coeff) in enumerate(training_loss.pieces):
        dw_coeff = partial_w(coeff)
        dww_coeff = partial_ww(coeff)

        def dw_func(a, w, _c=dw_coeff):
            return eval_poly2d(_c, a, w)

        def dww_func(a, w, _c=dww_coeff):
            return eval_poly2d(_c, a, w)

        alpha0 = alpha_grid[0]
        margin = 1e-8
        roots = find_roots_in_interval(
            lambda w: dw_func(alpha0, w), w_lo + margin, w_hi - margin, n_scan
        )

        # Keep only non-degenerate roots (d_ww != 0)
        roots = [r for r in roots if abs(dww_func(alpha0, r)) > 1e-10]

        for w0 in roots:
            w_values = track_branch(dw_func, dww_func, w0, alpha_grid, w_lo, w_hi)
            # Only keep if we tracked most of the way
            valid_frac = np.sum(~np.isnan(w_values)) / len(alpha_grid)
            if valid_frac > 0.9:
                branches.append({"piece_idx": m, "w_values": w_values})

    return branches
