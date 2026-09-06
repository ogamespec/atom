# -*- coding: utf-8 -*-
"""Точная аналитическая модель атома водорода (нерелятивистская).

Уравнение Шрёдингера (атомные единицы):
    H psi = E psi,   H = -0.5 * nabla^2 - 1/r

Энергии:               E_n = -0.5 / n^2
Радиальная функция:    R_nl(r) = N_nl * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho),
                       rho = 2r/n, L — обобщённые полиномы Лагерра
Угловая часть:         Y_lm(theta, phi) — сферические гармоники
"""
import numpy as np
from scipy.special import eval_genlaguerre, sph_harm, factorial


def energy_au(n):
    """E_n = -0.5/n^2 (хартри)."""
    return -0.5 / n ** 2


def radial_R(n, l, r):
    """Радиальная волновая функция R_nl(r), атомные единицы."""
    r = np.asarray(r, dtype=float)
    rho = 2.0 * r / n
    norm = np.sqrt((2.0 / n) ** 3
                   * factorial(n - l - 1, exact=True)
                   / (2 * n * factorial(n + l, exact=True)))
    L = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    return norm * np.exp(-rho / 2.0) * rho ** l * L


def radial_prob(n, l, r):
    """Радиальная плотность вероятности r^2 R_nl(r)^2."""
    R = radial_R(n, l, r)
    return r ** 2 * R ** 2


def psi_nlm_on_grid(n, l, m, X, Y, Z):
    """Волновая функция psi_nlm на 3D-сетке (вещественная при m = 0)."""
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    theta = np.arccos(np.clip(Z / np.maximum(R, 1e-300), -1.0, 1.0))
    phi = np.arctan2(Y, X)
    rad = radial_R(n, l, R)
    ang = sph_harm(m, l, phi, theta)
    psi = rad * ang
    return psi.real if m == 0 else psi


def mom_1s(p):
    """Импульсная волновая функция 1s: phi(p) = (2*sqrt(2)/pi)/(1+p^2)^2."""
    p = np.asarray(p, dtype=float)
    return (2.0 * np.sqrt(2.0) / np.pi) / (1.0 + p ** 2) ** 2


def mom_prob_1s(p):
    """|phi_1s(p)|^2 = (8/pi^2)/(1+p^2)^4."""
    p = np.asarray(p, dtype=float)
    return (8.0 / np.pi ** 2) / (1.0 + p ** 2) ** 4


def expectation_values(n, l):
    """Средние значения (а.е.) для состояния (n,l):
    <r>, <r^2>, <1/r>, <p^2>, <T>, <V>. Проверка вириала: <T> = -E, <V> = 2E.
    """
    r = (3 * n ** 2 - l * (l + 1)) / 2.0
    r2 = n ** 2 * (5 * n ** 2 + 1 - 3 * l * (l + 1)) / 2.0
    inv_r = 1.0 / n ** 2
    p2 = 1.0 / n ** 2
    T = p2 / 2.0
    V = -inv_r
    return {'r': r, 'r2': r2, 'inv_r': inv_r, 'p2': p2, 'T': T, 'V': V}
