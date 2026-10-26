# -*- coding: utf-8 -*-
"""Точный анализ поправки решётки к энергии 1s (сходимость вычислительной модели).

Основной отчёт фиксирует численный наклон сходимости E₁ₛ(a) -> −1/2: ≈ 2.15 по
точкам a = 1.0..0.2. Здесь выясняется АНАЛИТИЧЕСКАЯ структура поправки:

  * кинетический оператор решётки T = −L_d/2 = −∇²/2 − (a²/24)·Σ_μ ∂_μ⁴ + O(a⁴);
    теория возмущений 1-го порядка по 1s: ΔE_кин = −(a²/24)·⟨Σ_μ ∂_μ⁴⟩₁ₛ = −a²/8
    (⟨p⁴⟩₁ₛ = 5, ⟨p_μ⁴⟩ = ⟨p⁴⟩/5 = 1 для s-состояния, Σ_μ = 3);
  * потенциал −1/r, отсчитанный в узлах, и регуляризация узла источника дают
    свои поправки; разделяем их тремя вариантами потенциала:
       (a) «узел»:   V(0) = −G(0)/a, V(x) = −1/|x|      (столбец 'continuum' модели);
       (b) «ячейка»: V̄(0) = −J(0)/a — среднее 1/r по ячейке источника
                     (среднее по кубу гармонической функции = значение в центре,
                     поэтому для x ≠ 0 ячейки V̄(x) = V(x) с точностью O(a⁴));
       (c) «решётка»: точная функция Грина G(x) решётки (столбец 'lattice' модели).
  * подгонка E(a) + 1/2 = c₂a² + c₃a³ + c₄a⁴ / c₂a² + c₂' a² ln a по малым a.

Запуск:  python scripts/run_lattice_correction.py
Вывод:   results/correction/*.png, results/correction/report.json
"""
import json
import sys
import warnings
from pathlib import Path

# Устойчивость к кодировке консоли Windows: cp1251 не умеет α, π, ≈, √, ².
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import hydrogen_exact as he
from src.lattice import (G0_FREE, lattice_green_fft, r2_sieve,
                         r3_multiplicities, exact_shell_hamiltonian)
from src.plotting import new_fig, save

OUT = ROOT / 'results' / 'correction'
OUT.mkdir(parents=True, exist_ok=True)

report = {}
checks = []


def check(name, ok, value):
    checks.append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + '   ->   ' + str(value), flush=True)


# ------------------------------------------------------ среднее 1/r по ячейке

def cell_average_J(smax):
    """J(x) = ∫_{куб(x)} d³v/|v| (единичная ячейка с центром x), усреднено по оболочкам.

    Возвращает J_s[s] для s ≤ smax: среднее по всем узлам оболочки.
    """
    K = int(np.ceil(np.sqrt(smax)))
    c = np.arange(-K, K + 1)
    X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
    s2 = X * X + Y * Y + Z * Z
    ok = s2 <= smax
    X, Y, Z, s2 = X[ok], Y[ok], Z[ok], s2[ok]
    # квадратура Гаусса–Лежандра (8³ точек на ячейку; узлы не попадают в центр)
    gp, gw = np.polynomial.legendre.leggauss(8)
    gp = gp / 2.0                      # узлы в [−0.5, 0.5]
    gw = gw / 2.0                      # сумма весов = длина интервала = 1
    GX, GY, GZ = np.meshgrid(gp, gp, gp, indexing='ij')
    W = (gw[:, None, None] * gw[None, :, None] * gw[None, None, :]).ravel()
    GX, GY, GZ = GX.ravel(), GY.ravel(), GZ.ravel()
    J = np.zeros(len(X))
    for gx, gy, gz, w in zip(GX, GY, GZ, W):
        J += w / np.sqrt((X - gx) ** 2 + (Y - gy) ** 2 + (Z - gz) ** 2)
    # ячейка источника: сингулярность 1/r в центре; квадратура Гаусса чётного
    # порядка (узлы не попадают в x = 0); n = 32 → точность ~2·10⁻⁴.
    gp0, gw0 = np.polynomial.legendre.leggauss(32)
    gp0 = gp0 / 2.0
    gw0 = gw0 / 2.0
    g0x, g0y, g0z = np.meshgrid(gp0, gp0, gp0, indexing='ij')
    w0 = (gw0[:, None, None] * gw0[None, :, None] * gw0[None, None, :]).ravel()
    J0 = float(np.sum(w0 / np.sqrt(g0x.ravel() ** 2 + g0y.ravel() ** 2 + g0z.ravel() ** 2)))
    # усреднение по оболочкам (источник исключён из цикла квадратуры)
    Js = np.zeros(smax + 1)
    cnt = np.zeros(smax + 1, dtype=np.int64)
    m0 = s2 == 0
    np.add.at(Js, s2[~m0], J[~m0])
    np.add.at(cnt, s2[~m0], 1)
    Js = np.divide(Js, cnt, out=np.zeros_like(Js), where=cnt > 0)
    Js[0] = J0
    return Js


# ------------------------------------------------------ энергия 1s для потенциала

def e1s_for_potential(a, V_s, label):
    Rmax = 10.0                      # хвост 1s за R = 10: e^{−2R} ≈ 2·10⁻⁹
    Smax = int((Rmax / a + 1) ** 2)
    r2 = r2_sieve(Smax)
    m = r3_multiplicities(Smax, r2)
    M, D, shells = exact_shell_hamiltonian(a, Smax, V_s, m, r2)
    rad = np.sqrt(shells.astype(float)) * a
    X0 = np.column_stack([he.radial_R(1, 0, rad), he.radial_R(2, 0, rad)])
    P = sparse.diags(1.0 / (M.diagonal() / D + 0.3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E, _ = lobpcg(M, X0, B=sparse.diags(D), M=P, largest=False, tol=1e-10, maxiter=400)
    E1 = float(np.min(E))
    print(f'    a={a:.4f}: E₁ₛ({label}) = {E1:.8f}  (err {E1 + 0.5:+.2e})', flush=True)
    return E1


def potentials(a, Js, Gex):
    """Три варианта потенциала (функции V_s(sarr) → хартри) для шага a."""
    def V_node(sarr):
        sarr = np.asarray(sarr, dtype=np.int64)
        out = np.zeros_like(sarr, dtype=float)
        out[sarr == 0] = -G0_FREE / a
        nn = sarr > 0
        out[nn] = -1.0 / (np.sqrt(sarr[nn].astype(float)) * a)
        return out

    def V_cell(sarr):
        sarr = np.asarray(sarr, dtype=np.int64)
        out = np.zeros_like(sarr, dtype=float)
        near = sarr <= 400
        out[near] = -Js[sarr[near]] / a
        far = sarr > 400
        out[far] = -1.0 / (np.sqrt(sarr[far].astype(float)) * a)
        return out

    def V_lat(sarr):
        sarr = np.asarray(sarr, dtype=np.int64)
        out = np.zeros_like(sarr, dtype=float)
        near = sarr <= 144
        out[near] = -Gex[sarr[near]] / a
        far = sarr > 144
        out[far] = -1.0 / (np.sqrt(sarr[far].astype(float)) * a)
        return out

    return {'node': V_node, 'cell': V_cell, 'lattice': V_lat}


def fit_models(a, err, name):
    """Подгонки err(a) разными анзацами; возвращает словарь результатов."""
    A = np.asarray(a)
    y = np.asarray(err)
    fits = {}
    x2 = A ** 2
    # A: c2*a^2
    c2 = float(np.sum(x2 * y) / np.sum(x2 * x2))
    fits['c2_a2'] = {'c2': c2, 'rms': float(np.sqrt(np.mean((y - c2 * x2) ** 2)))}
    # B: c2*a^2 + c3*a^3
    XB = np.column_stack([x2, A ** 3])
    cB, *_ = np.linalg.lstsq(XB, y, rcond=None)
    fits['c2a2+c3a3'] = {'c2': float(cB[0]), 'c3': float(cB[1]),
                         'rms': float(np.sqrt(np.mean((y - XB @ cB) ** 2)))}
    # C: c2*a^2 + cl*a^2*ln a
    XC = np.column_stack([x2, x2 * np.log(A)])
    cC, *_ = np.linalg.lstsq(XC, y, rcond=None)
    fits['a2(c2+cl*ln)'] = {'c2': float(cC[0]), 'cl': float(cC[1]),
                            'rms': float(np.sqrt(np.mean((y - XC @ cC) ** 2)))}
    # D: c2*a^2 + c3*a^3 + c4*a^4
    XD = np.column_stack([x2, A ** 3, A ** 4])
    cD, *_ = np.linalg.lstsq(XD, y, rcond=None)
    fits['c2a2+c3a3+c4a4'] = {'c2': float(cD[0]), 'c3': float(cD[1]), 'c4': float(cD[2]),
                              'rms': float(np.sqrt(np.mean((y - XD @ cD) ** 2)))}
    best = min(fits, key=lambda k: fits[k]['rms'])
    print(f'    {name}: лучшая подгонка {best}: { {k: round(v, 5) for k, v in fits[best].items()} }',
          flush=True)
    return fits, best


def main():
    print('=== ПОПРАВКА РЕШЁТКИ К ЭНЕРГИИ 1s ===', flush=True)
    # общие данные: средние 1/r по ячейкам и функция Грина решётки (a = 1)
    Js = cell_average_J(400)
    print(f'    J(0) (среднее 1/r по ячейке источника) = {Js[0]:.5f} '
          f'против G(0) = {G0_FREE:.5f}', flush=True)
    Gper = lattice_green_fft(1.0, 128)
    sh = G0_FREE - Gper[0, 0, 0]
    cg = np.indices((25, 25, 25)).reshape(3, -1).T - 12
    Xg, Yg, Zg = cg[:, 0], cg[:, 1], cg[:, 2]
    s2g = Xg * Xg + Yg * Yg + Zg * Zg
    okg = s2g <= 144
    Gex = np.zeros(145)
    cntg = np.zeros(145)
    Gv = Gper[(Xg % 128).ravel(), (Yg % 128).ravel(), (Zg % 128).ravel()] + sh
    np.add.at(Gex, s2g[okg], Gv[okg])
    np.add.at(cntg, s2g[okg], 1)
    Gex = np.divide(Gex, cntg, out=np.zeros_like(Gex), where=cntg > 0)

    a_list = [0.05, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    E = {k: [] for k in ('node', 'cell', 'lattice')}
    for a in a_list:
        pots = potentials(a, Js, Gex)
        for label, V_s in pots.items():
            E[label].append(e1s_for_potential(a, V_s, label))

    err = {k: np.array(E[k]) + 0.5 for k in E}
    a = np.array(a_list)
    report['energies'] = {k: {'a': a_list, 'E': E[k], 'err': err[k].tolist()} for k in E}

    # подгонки по a ≤ 0.3
    mask = a <= 0.3
    fits_all = {}
    for k in E:
        fits, best = fit_models(a[mask], err[k][mask], k)
        fits_all[k] = {'best': best, **{m: fits[m] for m in fits}}
    report['fits'] = fits_all

    # проверки
    c2_node = fits_all['node']['c2a2+c3a3']['c2']
    check('кинетическая поправка −a²/8 доминирует: |c₂| в разумных пределах',
          0.08 < abs(c2_node) < 0.3, float(c2_node))
    check('разложение E = −1/2 − a²/8 + … описывает сходимость (rms < 10⁻³)',
          fits_all['node']['c2a2+c3a3']['rms'] < 1e-3, fits_all['node']['c2a2+c3a3']['rms'])
    # наклон, воспроизводящий 2.15 основного отчёта (по тем же точкам 1.0..0.2)
    pts = (a >= 0.2) & (a <= 1.0)
    slope = np.polyfit(np.log(a[pts]), np.log(np.abs(err['lattice'][pts])), 1)[0]
    report['slope_repro'] = float(slope)
    check('воспроизведён наклон 2.15 основного отчёта (2.0..2.4)',
          2.0 < slope < 2.4, float(slope))

    fig, (ax1, ax2) = new_fig(12, 5.4, 1, 2)
    cols = {'node': 'tab:blue', 'cell': 'tab:green', 'lattice': 'tab:red'}
    for k in E:
        ax1.loglog(a, err[k], 'o-', ms=4, color=cols[k], label=f'{k}: err(a)')
    ax1.loglog(a, a ** 2 / 8.0, 'k--', lw=1.2, label='кинетика −a²/8 (теория возмущений)')
    ax1.set_xlabel('шаг решётки a, a₀')
    ax1.set_ylabel('|E₁ₛ(a) + 1/2|, хартри')
    ax1.set_title('Сходимость E₁ₛ: кинетическая поправка −a²/8 и потенциал')
    ax1.legend(fontsize=8)

    for k in E:
        ax2.plot(a[mask], err[k][mask] / a[mask] ** 2, 'o-', ms=4, color=cols[k], label=k)
    ax2.axhline(1.0 / 8.0, color='k', ls='--', lw=1.2, label='предсказание 1/8')
    ax2.set_xlabel('a, a₀')
    ax2.set_ylabel('err(a)/a², хартри')
    ax2.set_title('err/a² → 1/8 при a → 0 (кинетическая поправка доминирует)')
    ax2.legend(fontsize=8)
    save(fig, OUT / 'c1_lattice_correction.png')

    report['checks'] = checks
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding='utf-8')
    nfail = sum(1 for c in checks if not c['ok'])
    print(f'\nВсего проверок: {len(checks)}, провалено: {nfail}', flush=True)
    return nfail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
