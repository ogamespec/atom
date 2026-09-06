# -*- coding: utf-8 -*-
"""Радиальная задача Шрёдингера на сетке (базовая для обеих моделей).

Подстановка psi = u(r)/r приводит радиальное уравнение к виду
    [-0.5 d^2/dr^2 + l(l+1)/(2r^2) + V(r)] u(r) = E u(r),
    u(0) = 0, u(R_max) = 0.
Дискретизация на равномерной сетке r_i = i*h (i = 1..N):
    (u_{i+1} - 2u_i + u_{i-1})/h^2 = 2 (V_i + l(l+1)/(2r_i^2) - E) u_i
даёт трёхдиагональную матрицу, диагонализуемую точно.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal


def build_radial(h, N, l, Vfun):
    """Диагональ d, поддиагональ e трёхдиагональной матрицы и сетка r."""
    r = h * np.arange(1, N + 1)
    d = 1.0 / h ** 2 + l * (l + 1) / (2.0 * r ** 2) + Vfun(r)
    e = np.full(N - 1, -0.5 / h ** 2)
    return d, e, r


def solve_radial(h, N, l, Vfun, k=6):
    """k низших собственных значений и собственных функций u_i(r)."""
    d, e, r = build_radial(h, N, l, Vfun)
    E, U = eigh_tridiagonal(d, e, select='i', select_range=(0, k - 1))
    return E, U, r


def normalize_u(U, h):
    """Нормировка int u^2 dr = 1."""
    for j in range(U.shape[1]):
        U[:, j] /= np.sqrt(np.sum(U[:, j] ** 2) * h)
    return U
