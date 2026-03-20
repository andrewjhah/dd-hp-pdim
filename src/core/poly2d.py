"""
Representing and performing arithmetic on 2d polynomials.

We store a 2d polynomial
    p(alpha, w) = sum_{i,j} c[i,j] * alpha^i * w^j
as a 2d numpy array of coefficients, where c[i,j] is the
coefficient of alpha^i * w^j.

Shape: c has shape (deg_alpha + 1, deg_w + 1).
"""

import numpy as np


def eval_poly2d(coeff, alpha, w):
    result = 0.0
    for i in range(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            if coeff[i, j] != 0:
                result += coeff[i, j] * alpha**i * w**j
    return result


def eval_poly2d_grid(coeff, alpha_arr, w_arr):
    alpha_powers = np.array([alpha_arr**i for i in range(coeff.shape[0])])
    w_powers = np.array([w_arr**j for j in range(coeff.shape[1])])
    # result[a, w] = sum_ij c[i,j] * alpha^i * w^j = alpha_powers^T @ coeff @ w_powers
    return alpha_powers.T @ coeff @ w_powers


def partial_w(coeff):
    if coeff.shape[1] <= 1:
        return np.zeros((coeff.shape[0], 1))
    new = np.zeros((coeff.shape[0], coeff.shape[1] - 1))
    for j in range(1, coeff.shape[1]):
        new[:, j - 1] = j * coeff[:, j]
    return new


def partial_alpha(coeff):
    if coeff.shape[0] <= 1:
        return np.zeros((1, coeff.shape[1]))
    new = np.zeros((coeff.shape[0] - 1, coeff.shape[1]))
    for i in range(1, coeff.shape[0]):
        new[i - 1, :] = i * coeff[i, :]
    return new


def partial_ww(coeff):
    return partial_w(partial_w(coeff))


def partial_alpha_w(coeff):
    return partial_alpha(partial_w(coeff))


def total_degree(coeff):
    max_deg = 0
    for i in range(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            if coeff[i, j] != 0:
                max_deg = max(max_deg, i + j)
    return max_deg


def trim(coeff):
    """For removing trailing zero rows and columns."""
    last_row = 0
    for i in range(coeff.shape[0]):
        if np.any(coeff[i, :] != 0):
            last_row = i
    last_col = 0
    for j in range(coeff.shape[1]):
        if np.any(coeff[:, j] != 0):
            last_col = j
    return coeff[: last_row + 1, : last_col + 1]
