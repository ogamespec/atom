# -*- coding: utf-8 -*-
"""Вычислительная модель: 3D-решётка ячеек, дискретное уравнение Шрёдингера.

Постулаты модели (см. отчёт):
  * пространство — кубическая решётка с шагом a (топология — тор в расчётах,
    свободное пространство в пределе L -> inf);
  * время — глобальный тактовый генератор CLK, такт tau = a/c;
  * ячейка хранит комплексную амплитуду (fixed-point, 2B бит) и значения полей;
  * динамика амплитуд — дискретное уравнение Шрёдингера
        i dpsi/dt = -0.5 L psi + V psi,   L — дискретный лапласиан;
  * поле протона — неподвижная точка локального правила (уравнение Пуассона),
    V_lat(x) = -G(x), где G — функция Грина решётки: (-L)G = 4*pi*delta,
    G(x) -> 1/|x| при |x| >> a, G(0) = 2*pi*W = 3.1759119... (интеграл Ватсона).
"""
import numpy as np
from scipy import sparse

MADELUNG_SC = 2.8372974794806       # постоянная Маделунга ПК-решётки
WATSON_W = 0.505462019717326        # интеграл Ватсона
G0_FREE = 2.0 * np.pi * WATSON_W    # функция Грина свободной решётки в нуле (a=1)


def lap_apply(f, a):
    """Дискретный лапласиан L f = sum_{+-mu} (f(x+e_mu) - f(x)) / a^2 (тор)."""
    s = np.zeros_like(f)
    for ax in range(3):
        s += np.roll(f, 1, ax) + np.roll(f, -1, ax)
    return (s - 6.0 * f) / a ** 2


def kin_apply(f, a):
    """Кинетический оператор T = -0.5 L."""
    return -0.5 * lap_apply(f, a)


def energy_expect(psi, V, a):
    """<psi|H|psi> для действительной psi, норма <psi|psi> = 1 (метрика a^3)."""
    n = np.sum(psi * psi) * a ** 3
    return np.sum(psi * (kin_apply(psi, a) + V * psi)) * a ** 3 / n


def ortho(psi, states):
    """Ортогонализация Грамма-Шмидта против уже найденных состояний."""
    for st in states:
        psi = psi - np.sum(st * psi) * st
    return psi


def quantize_fixed(x, B):
    """Квантование в fixed-point: B бит на компоненту, глобальная шкала = max|x|."""
    s = float(np.max(np.abs(x)))
    if s == 0.0:
        return x
    lev = 2.0 ** (B - 1)
    return np.round(x * (lev / s)) * (s / lev)


def imag_time(psi, V, a, dbeta, steps, B=None, states=(), log_every=200,
              callback=None):
    """Мнимое время (локальный явный шаг) с необязательной битовой квантизацией.

    psi <- psi - dbeta*(T + V) psi, нормировка на каждом такте.
    Возвращает финальное состояние и историю энергий.
    """
    psi = psi.copy()
    log = []
    for s in range(1, steps + 1):
        psi = psi - dbeta * (kin_apply(psi, a) + V * psi)
        if B is not None:
            psi = quantize_fixed(psi, B)
        if states:
            psi = ortho(psi, states)
        psi /= np.sqrt(np.sum(psi * psi))
        if log_every and s % log_every == 0:
            E = float(energy_expect(psi, V, a))
            log.append((s, E))
            if callback:
                callback(s, E)
    return psi, log


def lattice_green_fft(a, N):
    """Периодическая функция Грина решётки на торе N^3: (-L)G = 4*pi*delta - фон.

    G_per(0) = G_free(0) - c/N + O(1/N^3), константа c измеряется численно
    (см. scripts/run_model.py).
    """
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=a)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    denom = 4.0 * (np.sin(0.5 * kx * a) ** 2
                   + np.sin(0.5 * ky * a) ** 2
                   + np.sin(0.5 * kz * a) ** 2) / a ** 2
    ghat = np.zeros_like(denom)
    m = denom > 0
    ghat[m] = 4.0 * np.pi / denom[m]
    # Обратное преобразование с нормировкой (1/a³): G(x) = (1/N³a³) Σ_k Ĝ(k) e^{ikx}
    return np.fft.ifftn(ghat).real / a ** 3


def torus_potential(a, N):
    """Потенциал протона в центре тора N^3: V = -G_per, источник в центре."""
    G = lattice_green_fft(a, N)
    G = np.roll(G, shift=(N // 2, N // 2, N // 2), axis=(0, 1, 2))
    return -G


def splitstep_real_time(psi0, V, a, tau, steps, record_every):
    """Вещественное время: расщепление Троттера e^{-iH tau} ~ e^{-iV tau/2} e^{-iT tau} e^{-iV tau/2}.

    Кинетическая часть точна в импульсном представлении (БПФ).
    Возвращает (t, C(t)), C(t) = <psi0|psi(t)>.
    """
    N = psi0.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=a)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    kin_k = (2.0 / a ** 2) * (np.sin(0.5 * kx * a) ** 2
                              + np.sin(0.5 * ky * a) ** 2
                              + np.sin(0.5 * kz * a) ** 2)
    eiv = np.exp(-0.5j * V * tau)
    eik = np.exp(-1j * kin_k * tau)
    psi = psi0.astype(complex)
    ts, Cs = [], []
    for s in range(1, steps + 1):
        psi *= eiv
        psi = np.fft.ifftn(eik * np.fft.fftn(psi))
        psi *= eiv
        if s % record_every == 0:
            ts.append(s * tau)
            Cs.append(np.sum(np.conj(psi0) * psi) * a ** 3)
    return np.array(ts), np.array(Cs)


def grid_coords(a, N):
    """Координаты узлов x,y,z с центром в (N//2, N//2, N//2)."""
    L = N * a
    x = a * np.arange(N) - L / 2.0
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return X, Y, Z


def enumerate_shells(K):
    """(УСТАРЕЛО: грубые оболочки round(|x|) не сходятся к континууму.
    Используйте точные оболочки: r2_sieve, r3_multiplicities,
    exact_shell_hamiltonian.)"""
    c = np.indices((2 * K + 1, 2 * K + 1, 2 * K + 1)).reshape(3, -1).T - K
    X, Y, Z = c[:, 0], c[:, 1], c[:, 2]
    s2 = X * X + Y * Y + Z * Z
    shell = np.round(np.sqrt(s2)).astype(int)
    ok = shell <= K + 1
    m = np.bincount(shell[ok], minlength=K + 2)
    B = np.zeros((K + 2, K + 2))
    for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        ns2 = (X + dx) ** 2 + (Y + dy) ** 2 + (Z + dz) ** 2
        nshell = np.round(np.sqrt(ns2)).astype(int)
        both = ok & (nshell <= K + 1)
        for i in range(K + 2):
            for j in range(K + 2):
                B[i, j] += np.sum(both & (shell == i) & (nshell == j))
    B = B + B.T
    return m, B


def r2_sieve(Smax):
    """r2(n) — число упорядоченных представлений n = x² + y² (x,y ∈ Z).

    r2(n) = 4·Σ_{d|n} χ₄(d), χ₄(d) = 0 (d чётное), +1 (d≡1 mod 4), −1 (d≡3 mod 4).
    """
    r2 = np.zeros(Smax + 1, dtype=np.int64)
    r2[0] = 1
    for d in range(1, Smax + 1, 2):
        c = 1 if d % 4 == 1 else -1
        r2[d::d] += c
    r2[1:] *= 4
    return r2


def r3_multiplicities(Smax, r2):
    """m_s = r3(s) — число узлов с |x|² = s: m_s = Σ_q w_q·r2(s−q²), w_0=1, w_{q≥1}=2."""
    m = np.zeros(Smax + 1, dtype=np.int64)
    qmax = int(np.sqrt(Smax))
    for q in range(qmax + 1):
        w = 1 if q == 0 else 2
        m[q * q:] += w * r2[:Smax - q * q + 1]
    return m


def exact_shell_hamiltonian(h, Smax, V_s, m, r2):
    """Обобщённая задача M g = E D g на ТОЧНЫХ оболочках s = |x|² (s ≤ Smax).

    M — разреженная (CSR), D = diag(m_s); V_s(s) — потенциал оболочки (хартри).
    Связи: B[s, s'] = 6·r2(s−q²) при s' = s+2q+1 (q ∈ Z, s−q² ≥ 0);
    диагональ: m_s·(V_s + 3/h²) (все 6 соседей узла лежат вне его оболочки).
    """
    shells = np.nonzero(m[:Smax + 1] > 0)[0]
    nsh = len(shells)
    idx = np.full(Smax + 1, -1, dtype=np.int64)
    idx[shells] = np.arange(nsh)
    rows = [np.arange(nsh)]
    cols = [np.arange(nsh)]
    diag = m[shells].astype(float) * (V_s(shells) + 3.0 / h ** 2)
    data = [diag]
    qmax = int(np.sqrt(Smax))
    for q in range(0, qmax + 1):            # s' = s + 2q + 1 > s — пара один раз
        lo = q * q
        hi = Smax - (2 * q + 1)
        if hi < lo:
            continue
        s = shells[(shells >= lo) & (shells <= hi)]
        w = r2[s - q * q]
        keep = w > 0
        s, w = s[keep], w[keep]
        sp = s + 2 * q + 1
        j = idx[sp]
        keep = j >= 0
        s, w, j = s[keep], w[keep], j[keep]
        if len(s) == 0:
            continue
        si = idx[s]
        val = -3.0 * w.astype(float) / h ** 2
        rows.append(np.concatenate([si, j]))
        cols.append(np.concatenate([j, si]))
        data.append(np.concatenate([val, val]))
    M = sparse.csr_matrix((np.concatenate(data),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nsh, nsh))
    D = m[shells].astype(float)
    return M, D, shells


def free_space_potential(a, N):
    """Поле свободного пространства: V = −1/|x|, узел источника −G(0)/a."""
    X, Y, Z = grid_coords(a, N)
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    V = -np.divide(1.0, R, out=np.zeros_like(R), where=R > 0)
    V[N // 2, N // 2, N // 2] = -G0_FREE / a
    return V


def radial_density(psi, a):
    """Радиальная плотность |psi(r)|^2, усреднённая по сферам."""
    N = psi.shape[0]
    L = N * a
    x = a * np.arange(N) - L / 2.0
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    nbins = N // 2
    edges = np.linspace(0.0, L / 2.0, nbins + 1)
    hist, _ = np.histogram(R.ravel(), bins=edges, weights=(psi ** 2).ravel())
    rmid = 0.5 * (edges[1:] + edges[:-1])
    vol = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    return rmid, hist / vol
