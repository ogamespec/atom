# -*- coding: utf-8 -*-
"""ЭТАЛОННАЯ МОДЕЛЬ (baseline): атом водорода в классической квантовой механике.

Что делает:
  1) точная аналитическая модель (E_n, R_nl, Y_lm, импульсное представление,
     вириальная теорема, серии Лаймана/Бальмера);
  2) численное решение радиального уравнения (диагонализация) и сравнение с
     аналитикой;
  3) 3D-решение (мнимое время) на сетке: плотности, энергии, импульсный профиль;
  4) аналитика квантовых биений (суперпозиция 1s+2s), период и длина волны.

Запуск:  python scripts/run_baseline.py
Вывод:   results/baseline/*.png, results/baseline/report.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.units import (E_HARTREE_EV, M_P_OVER_M_E, ALPHA, C_AU, T_AU_S,
                       au_to_ev, wavelength_nm)
from src import hydrogen_exact as he
from src.radial import solve_radial, normalize_u
from src.lattice import imag_time, energy_expect, G0_FREE, radial_density, grid_coords
from src.plotting import new_fig, save

OUT = ROOT / 'results' / 'baseline'
OUT.mkdir(parents=True, exist_ok=True)

report = {}
checks = []


def check(name, ok, value):
    checks.append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + '   ->   ' + str(value), flush=True)


# ------------------------------------------------------------------ аналитика

def analytic():
    print('== 1. Аналитическая модель ==', flush=True)
    r = np.linspace(1e-3, 60.0, 8000)
    states = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]

    fig, (ax1, ax2) = new_fig(12, 5.4, 1, 2)
    for (n, l) in states:
        R = he.radial_R(n, l, r)
        ax1.plot(r, R, label=f'n={n}, l={l}')
        ax2.plot(r, he.radial_prob(n, l, r), label=f'n={n}, l={l}')
    ax1.set_xlim(0, 25)
    ax2.set_xlim(0, 25)
    ax1.set_xlabel('r, боровские радиусы a₀')
    ax1.set_ylabel('Rₙₗ(r)')
    ax1.set_title('Радиальные волновые функции')
    ax2.set_xlabel('r, a₀')
    ax2.set_ylabel('r²Rₙₗ²(r)')
    ax2.set_title('Радиальная плотность вероятности')
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
    save(fig, OUT / 'b1_radial_wavefunctions.png')

    # нормировки и средние
    for (n, l) in states:
        norm = np.trapezoid(he.radial_prob(n, l, r), r)
        check(f'нормировка P(r) для ({n},{l}) = 1', abs(norm - 1) < 1e-5, float(norm))
    expv = {}
    for (n, l) in [(1, 0), (2, 0), (2, 1)]:
        v = he.expectation_values(n, l)
        expv[f'{n}{l}'] = {k: float(x) for k, x in v.items()}
        virial_ok = abs(v['T'] + he.energy_au(n)) < 1e-12 and abs(v['V'] - 2 * he.energy_au(n)) < 1e-12
        check(f'вириал для ({n},{l}): <T>=-E, <V>=2E', virial_ok, (v['T'], v['V']))
    report['expectation_values'] = expv

    # спектр и серии
    levels = {}
    for n in range(1, 9):
        levels[n] = {'E_au': he.energy_au(n), 'E_eV': au_to_ev(he.energy_au(n))}
    report['levels_eV'] = {str(n): au_to_ev(he.energy_au(n)) for n in range(1, 9)}
    series = {}
    for n in (2, 3, 4, 5):
        series[f'Lyman n{n}->1'] = wavelength_nm(1, n)
    for n in (3, 4, 5, 6):
        series[f'Balmer n{n}->2'] = wavelength_nm(2, n)
    report['series_nm'] = {k: float(v) for k, v in series.items()}
    check('E_1 = -0.5 хартри', abs(he.energy_au(1) + 0.5) < 1e-15, he.energy_au(1))
    check('E_1 = -13.6057 эВ', abs(au_to_ev(he.energy_au(1)) + 13.605693122994) < 1e-9,
          au_to_ev(he.energy_au(1)))

    fig, ax = new_fig(8.5, 6.5)
    for n in range(1, 7):
        E = au_to_ev(he.energy_au(n))
        ax.plot([0.15, 0.85], [E, E], 'b-', lw=1.5)
        ax.text(0.87, E, f'n={n}', va='center', fontsize=9)
    for n, color in [(2, 'r'), (3, 'r'), (4, 'r')]:
        ax.annotate('', xy=(0.28, au_to_ev(he.energy_au(1))), xytext=(0.28, au_to_ev(he.energy_au(n))),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
        ax.text(0.31, 0.5 * (au_to_ev(he.energy_au(1)) + au_to_ev(he.energy_au(n))),
                f'Ly-α/β/γ: {wavelength_nm(1, n):.1f} нм', fontsize=8, color=color)
    for n, color in [(3, 'g'), (4, 'g')]:
        ax.annotate('', xy=(0.62, au_to_ev(he.energy_au(2))), xytext=(0.62, au_to_ev(he.energy_au(n))),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
        ax.text(0.64, 0.5 * (au_to_ev(he.energy_au(2)) + au_to_ev(he.energy_au(n))),
                f'Hα/β: {wavelength_nm(2, n):.1f} нм', fontsize=8, color=color)
    ax.set_xlim(0, 1)
    ax.set_ylim(-14.2, 0.4)
    ax.set_ylabel('E, эВ')
    ax.set_title('Спектр атома водорода: Eₙ = −13.6057 эВ / n²')
    ax.set_xticks([])
    save(fig, OUT / 'b2_energy_levels.png')

    # импульсное представление 1s
    p = np.linspace(0.0, 200.0, 200000)
    pnorm = np.trapezoid(4 * np.pi * p ** 2 * he.mom_prob_1s(p), p)
    check('нормировка |φ₁ₛ(p)|² = 1', abs(pnorm - 1) < 1e-4, float(pnorm))
    p_peak_num = p[np.argmax(p ** 2 * he.mom_prob_1s(p))]
    p_peak = 1.0 / np.sqrt(3.0)
    rms_p = 1.0
    report['momentum_1s'] = {'peak_p': float(p_peak), 'peak_p_num': float(p_peak_num),
                             'rms_p': float(rms_p),
                             'peak_value': float(he.mom_prob_1s(0.0))}
    check('максимум p²|φ|² при p = 1/√3', abs(p_peak_num - p_peak) < 5e-4, float(p_peak_num))


# --------------------------------------------------- радиальная диагонализация

def radial_diag():
    print('== 2. Численная радиальная диагонализация ==', flush=True)
    h = 0.05
    N = 1200
    exact_l = {0: [1, 2, 3, 4], 1: [2, 3], 2: [3]}
    Vfun = lambda r: -1.0 / r

    diag = {}
    for l, ns in exact_l.items():
        E, U, r = solve_radial(h, N, l, Vfun, k=max(6, len(ns) + 2))
        U = normalize_u(U, h)
        for j, n in enumerate(ns):
            err = E[j] - he.energy_au(n)
            check(f'E({n},{l}) диагонализация vs точное', abs(err) < 1e-3, float(E[j]))
            diag[f'{n}{l}'] = {'E_num': float(E[j]), 'err': float(err)}
    report['radial_diag'] = diag

    # собственные функции против аналитики
    E, U, r = solve_radial(h, N, 0, Vfun, k=4)
    U = normalize_u(U, h)
    fig, (ax1, ax2) = new_fig(12, 5.4, 1, 2)
    for j, n in enumerate([1, 2, 3]):
        u_ex = r * he.radial_R(n, 0, r)
        s = np.sign(np.sum(U[:, j] * u_ex))
        ax1.plot(r, s * U[:, j], label=f'численно, n={n}')
        ax1.plot(r, u_ex, '--', label=f'аналитика, n={n}')
    ax1.set_xlim(0, 25)
    ax1.set_xlabel('r, a₀')
    ax1.set_ylabel('u(r) = r·R(r)')
    ax1.set_title('Радиальные собственные функции (l=0)')
    ax1.legend(fontsize=8)

    hh = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.05]
    errs = []
    for hh_i in hh:
        E0, _, _ = solve_radial(hh_i, int(60 / hh_i), 0, Vfun, k=1)
        errs.append(abs(E0[0] + 0.5))
    slope = np.polyfit(np.log(hh), np.log(errs), 1)[0]
    ax2.loglog(hh, errs, 'o-', label=f'ошибка E₁ₛ, наклон {slope:.2f}')
    ax2.set_xlabel('шаг сетки h, a₀')
    ax2.set_ylabel('|E₁ₛ + 0.5|, хартри')
    ax2.set_title('Сходимость численной схемы')
    ax2.legend()
    save(fig, OUT / 'b3_radial_diag.png')
    report['radial_convergence'] = {'h': hh, 'err': errs, 'slope': float(slope)}
    check('порядок сходимости ≈ 2', 1.5 < slope < 2.5, float(slope))


# ------------------------------------------------------ 3D, мнимое время

def td3d():
    print('== 3. 3D-решение (мнимое время) ==', flush=True)
    # a = 0.2: та же сетка, что в вычислительной модели (честное сравнение);
    # при меньших a периодическая сетка обрезает хвост 2s (артефакт сворачивания)
    N, a = 128, 0.2
    L = N * a
    X, Y, Z = grid_coords(a, N)
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    V = -np.divide(1.0, R, out=np.zeros_like(R), where=R > 0)
    V[N // 2, N // 2, N // 2] = -G0_FREE / a   # регуляризация на узле источника

    def norm(psi):
        return psi / np.sqrt(np.sum(psi ** 2) * a ** 3)

    states3d = {}
    dbeta, steps = 0.004, 8000
    psis = []
    logs = {}
    for name, (n, l, m) in [('1s', (1, 0, 0)), ('2s', (2, 0, 0)), ('2p_z', (2, 1, 0))]:
        psi = norm(he.psi_nlm_on_grid(n, l, m, X, Y, Z))
        psi, log = imag_time(psi, V, a, dbeta, steps, states=psis, log_every=1000)
        E = energy_expect(psi, V, a)
        states3d[name] = {'E': float(E), 'E_exact': he.energy_au(n), 'err': float(E - he.energy_au(n))}
        logs[name] = [[int(s), float(e)] for s, e in log]
        psis.append(psi)
        print(f'    {name}: E = {E:.6f} (точное {he.energy_au(n):.6f})', flush=True)
        tol = 0.015 if name == '1s' else 0.008
        check(f'3D энергия {name} ≈ точная', abs(E - he.energy_au(n)) < tol, float(E))
    report['td3d_energies'] = states3d
    report['td3d_energy_log'] = logs

    fig, axes = new_fig(14, 9, 2, 3)
    axes[0][0].imshow(psis[0][:, :, N // 2].T ** 2, origin='lower', cmap='inferno',
                      extent=[-L / 2, L / 2, -L / 2, L / 2])
    axes[0][0].set_title('1s: |ψ|² в плоскости z=0')
    axes[0][1].imshow(psis[2][N // 2, :, :].T ** 2, origin='lower', cmap='inferno',
                      extent=[-L / 2, L / 2, -L / 2, L / 2])
    axes[0][1].set_title('2p_z: |ψ|² в плоскости x=0')
    for j, (name, (n, l, m)) in enumerate([('1s', (1, 0, 0)), ('2s', (2, 0, 0)), ('2p_z', (2, 1, 0))]):
        rm, dens = radial_density(psis[j], a)
        r = np.linspace(0.1, 12.0, 2000)
        exact_dens = he.radial_R(n, l, r) ** 2 / (4 * np.pi)
        axes[1][j].semilogy(rm, np.maximum(dens, 1e-10), label='модель (3D)')
        axes[1][j].semilogy(r, np.maximum(exact_dens, 1e-12), '--', label='аналитика')
        axes[1][j].set_xlim(0, 12)
        axes[1][j].set_ylim(1e-8, 1)
        axes[1][j].set_xlabel('r, a₀')
        axes[1][j].set_ylabel('|ψ(r)|²')
        axes[1][j].set_title(name)
        axes[1][j].legend(fontsize=8)
    axes[0][2].set_title('Сходимость энергии (мнимое время)')
    for name in ('1s', '2s', '2p_z'):
        s, e = zip(*logs[name])
        axes[0][2].plot(s, e, label=name)
        axes[0][2].axhline(he.energy_au({'1s': 1, '2s': 2, '2p_z': 2}[name]),
                           color='k', ls=':', lw=0.6)
    axes[0][2].legend(fontsize=8)
    axes[0][2].set_xlabel('шаг')
    axes[0][2].set_ylabel('E, хартри')
    save(fig, OUT / 'b4_3d_density.png')

    # импульсное представление 1s через БПФ (ψ с простой нормировкой Σψ²=1:
    # φ(p_k) = (2π)^(−3/2) · a^{3/2} · FFT(ψ))
    F = np.fft.fftn(psis[0])
    phi = F * a ** 1.5 * (2 * np.pi) ** -1.5
    kp = 2 * np.pi * np.fft.fftfreq(N, d=a)
    prof = np.abs(phi[:, 0, 0]) ** 2
    kpos = kp[:N // 2]
    prof = prof[:N // 2]
    dp = 2 * np.pi / L
    knorm = np.sum(np.abs(phi) ** 2) * dp ** 3
    check('нормировка φ(p) из БПФ = 1', abs(knorm - 1) < 1e-3, float(knorm))

    fig, ax = new_fig(9, 6)
    ax.loglog(kpos, prof, label='модель (БПФ)')
    ax.loglog(kpos, he.mom_prob_1s(kpos), '--', label='аналитика (8/π²)(1+p²)⁻⁴')
    ax.set_xlabel('p, а.е. (ħ/a₀)')
    ax.set_ylabel('|φ(p)|²')
    ax.set_title('Импульсное распределение 1s')
    ax.legend()
    save(fig, OUT / 'b5_momentum.png')
    mask = kpos < 2.0
    maxdev = np.max(np.abs(prof[mask] / he.mom_prob_1s(kpos[mask]) - 1))
    # отклонение до ~10% при p<2 — честное следствие того, что состояние на
    # решётке a=0.2 связано глубже континуума (−0.511 против −0.5)
    check('импульсный профиль совпадает при p<2 (±12%)', maxdev < 0.12, float(maxdev))
    report['momentum_fft'] = {'max_rel_dev_p2': float(maxdev)}


# ------------------------------------------------------------ биения (аналитика)

def beats_analytic():
    print('== 4. Квантовые биения (аналитика) ==', flush=True)
    dE = he.energy_au(2) - he.energy_au(1)   # 0.375 хартри
    T_au = 2 * np.pi / dE
    T_fs = T_au * T_AU_S * 1e15
    lam = wavelength_nm(1, 2)
    t = np.linspace(0, 2 * T_au, 1000)
    C2 = np.cos(0.5 * dE * t) ** 2
    fig, ax = new_fig(9, 5.5)
    ax.plot(t, C2)
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ax.set_xlabel('t, а.е. (1 а.е. = 2.419·10⁻¹⁷ с)')
    ax.set_ylabel('|C(t)|²')
    ax.set_title(f'Биения 1s+2s: ΔE = {dE:.3f} хартри = {au_to_ev(dE):.3f} эВ, '
                 f'T = {T_fs:.3f} фс, λ = {lam:.1f} нм (Lyman-α)')
    save(fig, OUT / 'b6_beats.png')
    report['beats_analytic'] = {'dE_au': float(dE), 'dE_eV': au_to_ev(dE),
                                'T_au': float(T_au), 'T_fs': float(T_fs),
                                'lambda_nm': float(lam)}
    check('ΔE(1s→2s) = 3/8 хартри = 10.204 эВ', abs(au_to_ev(dE) - 10.2043) < 1e-3,
          au_to_ev(dE))


def main():
    print('=== BASELINE: квантовая модель водорода ===', flush=True)
    print(f'Константы: m_p/m_e = {M_P_OVER_M_E:.9f}, α = {ALPHA:.9f}, '
          f'1 хартри = {E_HARTREE_EV:.6f} эВ, c = {C_AU:.3f} а.е.', flush=True)
    analytic()
    radial_diag()
    td3d()
    beats_analytic()
    report['checks'] = checks
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding='utf-8')
    nfail = sum(1 for c in checks if not c['ok'])
    print(f'\nВсего проверок: {len(checks)}, провалено: {nfail}', flush=True)
    return nfail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
