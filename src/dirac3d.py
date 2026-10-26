# -*- coding: utf-8 -*-
"""Квантовый клеточный автомат Дирака в 3+1D (Бялыницкий-Бирула, 1994).

Вейлевский автомат (безмассовая частица, 2-компонентный спинор phi(x)):
    phi(x, t+dt) = sum_h W(h) . phi(x + h, t),  h пробегает 8 углов куба {+-1,+-1,+-1},
    W_{+++} = q+ P1,  W_{++-} = q- P2,  W_{+-+} = q- P1,  W_{+--} = q+ P2,
    W_{-++} = q- P3,  W_{-+-} = q+ P4,  W_{--+} = q+ P3,  W_{---} = q- P4,
    q+ = (1+i)/4,  q- = (1-i)/4,
    P1 = [[1,0],[1,0]],  P2 = [[0,1],[0,1]],  P3 = [[1,0],[-1,0]],  P4 = [[0,-1],[0,1]].
Континуальный предел:  d phi/dt = c (sigma . grad) phi  (уравнение Вейля, c = a/dt).

Массивный автомат Дирака (4-компонентный спинор psi = (phi, chi)):
    psi(x, t+dt) = sum_h D(h) . psi(x + h, t),
    D(h) = [[ cos(ma) W(h),   i sin(ma) W(-h) ],
            [ i sin(ma) W(h),  cos(ma) W(-h) ]].
Континуальный предел:  d psi/dt = (rho3 sigma.grad + i rho1 m) psi (уравнение Дирака).

Свойства (проверяются численно в scripts/run_dirac3d.py): правило унитарно,
обратимо, локально (носитель одного такта — куб 3x3x3), причинно (групповая
скорость <= c), без фермионного удвоения (единственная низкоэнергетическая
мода при k = 0), дисперсия omega = sqrt(m^2 + k^2) в континуальном пределе,
спин-1/2 — две внутренние компоненты, Zitterbewegung на частоте 2m.

Теорема Бялыницкого-Бирулы: всякий локальный унитарный автомат с
2-компонентными волновыми функциями, который (A) не меняет однородную волну
и (B) сохраняет симметрию кубической решётки, даёт в континуальном пределе
уравнение Вейля. Это частичный ответ на вопрос «почему правило именно такое»
(основной отчёт, раздел 6): локальность + унитарность + симметрия решётки
однозначно выделяют уравнение Вейля/Дирака.
"""
import numpy as np

QP = (1.0 + 1.0j) / 4.0          # q+
QM = (1.0 - 1.0j) / 4.0          # q-

P1 = np.array([[1.0, 0.0], [1.0, 0.0]])
P2 = np.array([[0.0, 1.0], [0.0, 1.0]])
P3 = np.array([[1.0, 0.0], [-1.0, 0.0]])
P4 = np.array([[0.0, -1.0], [0.0, 1.0]])

# Восемь углов куба и соответствующие матрицы W(h)
HS = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]
W_LIST = [QP * P1, QM * P2, QM * P1, QP * P2,
          QM * P3, QP * P4, QP * P3, QM * P4]


def roll3(f, dx, dy, dz):
    """Циклический сдвиг поля на (dx, dy, dz) узлов (тор)."""
    f = np.roll(f, dx, axis=1)
    f = np.roll(f, dy, axis=2)
    f = np.roll(f, dz, axis=3)
    return f


def weyl_step(phi):
    """Один такт Вейлевского автомата: phi(x,t+dt) = sum_h W(h)·phi(x+h,t). phi: (2,N,N,N).

    Внимание: np.roll(f, s)[i] = f[i−s], поэтому для значения phi(x+h)
    используется сдвиг −h.
    """
    out = np.zeros_like(phi)
    for (dx, dy, dz), W in zip(HS, W_LIST):
        out += np.einsum('ij,jklm->iklm', W, roll3(phi, -dx, -dy, -dz))
    return out


def weyl_inv_step(phi):
    """Обратный такт (обратимость): phi(x,t) = sum_h W(h)^+ phi(x−h, t+dt)."""
    out = np.zeros_like(phi)
    for (dx, dy, dz), W in zip(HS, W_LIST):
        out += np.einsum('ij,jklm->iklm', W.conj().T, roll3(phi, dx, dy, dz))
    return out


def _w_minus(h):
    """W(-h) по списку HS."""
    return W_LIST[HS.index((-h[0], -h[1], -h[2]))]


def dirac_step(psi, m):
    """Один такт массивного автомата Дирака: psi(x,t+dt) = sum_h D(h)·psi(x+h,t).

    psi: (4, N, N, N), компоненты (phi0, phi1, chi0, chi1).
    """
    cm, sm = np.cos(m), np.sin(m)
    out = np.zeros_like(psi)
    for h, W in zip(HS, W_LIST):
        Wm = _w_minus(h)
        shp = roll3(psi[:2], -h[0], -h[1], -h[2])
        shc = roll3(psi[2:], -h[0], -h[1], -h[2])
        out[:2] += cm * np.einsum('ij,jklm->iklm', W, shp) \
            + 1j * sm * np.einsum('ij,jklm->iklm', Wm, shc)
        out[2:] += 1j * sm * np.einsum('ij,jklm->iklm', W, shp) \
            + cm * np.einsum('ij,jklm->iklm', Wm, shc)
    return out


def dirac_inv_step(psi, m):
    """Обратный такт массивного автомата Дирака: psi(x,t) = sum_h D(h)^+ psi(x−h, t+dt)."""
    cm, sm = np.cos(m), np.sin(m)
    out = np.zeros_like(psi)
    for h, W in zip(HS, W_LIST):
        Wm = _w_minus(h)
        shp = roll3(psi[:2], h[0], h[1], h[2])
        shc = roll3(psi[2:], h[0], h[1], h[2])
        out[:2] += cm * np.einsum('ij,jklm->iklm', W.conj().T, shp) \
            - 1j * sm * np.einsum('ij,jklm->iklm', W.conj().T, shc)
        out[2:] += - 1j * sm * np.einsum('ij,jklm->iklm', Wm.conj().T, shp) \
            + cm * np.einsum('ij,jklm->iklm', Wm.conj().T, shc)
    return out


def weyl_dispersion(k, m=0.0):
    """Собственные значения W̃(k) = sum_h W(h) e^{ik.h}; возвращает набор |arg|."""
    M = np.zeros((2, 2), dtype=complex)
    for (hx, hy, hz), W in zip(HS, W_LIST):
        M += W * np.exp(1j * (hx * k[0] + hy * k[1] + hz * k[2]))
    lam = np.linalg.eigvals(M)
    return np.abs(np.angle(lam))


def dirac_dispersion(k, m):
    """Собственные значения D̃(k) (4x4); возвращает набор |arg| — ветви +-omega."""
    cm, sm = np.cos(m), np.sin(m)
    Wp = np.zeros((2, 2), dtype=complex)
    Wm = np.zeros((2, 2), dtype=complex)
    for (hx, hy, hz), W in zip(HS, W_LIST):
        ph = np.exp(1j * (hx * k[0] + hy * k[1] + hz * k[2]))
        Wp += W * ph
        Wm += W * np.conj(ph)
    D = np.zeros((4, 4), dtype=complex)
    D[:2, :2] = cm * Wp
    D[:2, 2:] = 1j * sm * Wm
    D[2:, :2] = 1j * sm * Wp
    D[2:, 2:] = cm * Wm
    lam = np.linalg.eigvals(D)
    return np.abs(np.angle(lam))


def omega(k, m):
    """Частота правила omega(k) (положительная ветвь, частицы)."""
    ws = dirac_dispersion(k, m)
    return float(np.median(ws))


def dirac_eigenvector(k, m):
    """Положительно-частотный собственный спинор D̃(k) (для пакета).

    Физическая положительная частота ω эволюционирует как e^{−iωt},
    т.е. соответствует собственному значению e^{−iω} (отрицательный arg).
    """
    cm, sm = np.cos(m), np.sin(m)
    Wp = np.zeros((2, 2), dtype=complex)
    Wm = np.zeros((2, 2), dtype=complex)
    for (hx, hy, hz), W in zip(HS, W_LIST):
        ph = np.exp(1j * (hx * k[0] + hy * k[1] + hz * k[2]))
        Wp += W * ph
        Wm += W * np.conj(ph)
    D = np.zeros((4, 4), dtype=complex)
    D[:2, :2] = cm * Wp
    D[:2, 2:] = 1j * sm * Wm
    D[2:, :2] = 1j * sm * Wp
    D[2:, 2:] = cm * Wm
    lam, vec = np.linalg.eig(D)
    j = int(np.argmin(np.angle(lam)))
    return vec[:, j]
