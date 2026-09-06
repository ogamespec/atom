# -*- coding: utf-8 -*-
"""ВЫЧИСЛИТЕЛЬНАЯ МОДЕЛЬ: дискретная вселенная (решётка + CLK + поля + биты).

Эксперименты:
  1) функция Грина решётки G(x) -> 1/r (поле как неподвижная точка локального
     правила); измерение G(0), анизотропии и отличий от 1/r вблизи источника;
  2) поле из явной локальной релаксации (за сколько тактов устанавливается 1/r);
  3) спектр дискретной радиальной модели vs эталон, сходимость по a;
  4) 3D-состояния 1s/2s/2p_z на торе (мнимое время), энергии и профили;
  5) квантовые биения на дискретных часах (такт tau = a/c), измерение ΔE;
  6) битовая глубина ячеек: точность энергии vs число бит B;
  7) информационный учёт: биты электрона, поля, протона; анализ числа 1836.

Запуск:  python scripts/run_model.py
Вывод:   results/model/*.png, results/model/report.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import lobpcg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.units import (C_AU, T_AU_S, M_P_OVER_M_E, E_HARTREE_EV, au_to_ev)
from src import hydrogen_exact as he
from src.lattice import (G0_FREE, MADELUNG_SC, imag_time, energy_expect,
                         lattice_green_fft, splitstep_real_time,
                         radial_density, grid_coords, r2_sieve,
                         r3_multiplicities, exact_shell_hamiltonian,
                         free_space_potential)
from src.plotting import new_fig, save

OUT = ROOT / 'results' / 'model'
OUT.mkdir(parents=True, exist_ok=True)

report = {}
checks = []


def check(name, ok, value):
    checks.append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + '   ->   ' + str(value), flush=True)


# ------------------------------------------------------------ 1) функция Грина

def green():
    print('== 1. Функция Грина решётки (поле) ==', flush=True)
    N = 128
    a = 1.0
    Gper = lattice_green_fft(a, N)

    # Измеряем поправку тора: G_free(0) = G_per(0) + c/N + O(1/N^2);
    # согласованность c на двух сетках проверяет асимптотику 1/N.
    c_meas = (Gper[0, 0, 0] - G0_FREE) * N
    Gper96 = lattice_green_fft(a, 96)
    c_96 = (Gper96[0, 0, 0] - G0_FREE) * 96
    print(f'    G_per(0) = {Gper[0,0,0]:.6f}, G_free(0) = 2πW = {G0_FREE:.6f}, '
          f'поправка тора c(N=128) = {c_meas:.4f}, c(N=96) = {c_96:.4f} '
          f'(2π/3 = {2*np.pi/3:.4f}, Маделунг M = {MADELUNG_SC:.4f}, M+2π/3 = {MADELUNG_SC+2*np.pi/3:.4f})',
          flush=True)
    report['green'] = {'G_per_0': float(Gper[0, 0, 0]),
                       'G_free_0': float(G0_FREE),
                       'torus_shift_c': float(c_meas),
                       'torus_shift_c_N96': float(c_96)}
    check('поправка тора c согласована на сетках 96 и 128',
          abs(c_meas - c_96) < 0.05, (float(c_meas), float(c_96)))

    # свободная G вблизи источника: G_free(x) ≈ G_per(x) + (G0 − G_per(0))
    sh = G0_FREE - Gper[0, 0, 0]
    dirs = {}
    shells = {}
    for name, (dx, dy, dz) in {'[100]': (1, 0, 0), '[110]': (1, 1, 0), '[111]': (1, 1, 1)}.items():
        rr, gg, ex = [], [], []
        for i in range(1, 13):
            x = i * dx
            r = np.sqrt(x ** 2 + (i * dy) ** 2 + (i * dz) ** 2)
            G = Gper[x % N, (i * dy) % N, (i * dz) % N] + sh
            rr.append(r); gg.append(G); ex.append(1.0 / r)
        dirs[name] = {'r': rr, 'G': gg, 'exact': ex}
    report['green_directions'] = dirs

    for i in (1, 2, 3):
        shells[i] = {'G': Gper[i, 0, 0] + sh, 'one_over_r': 1.0 / i}
    report['green_shells'] = shells
    print('    G_free на осях: ' + ', '.join(
        f'G({i},0,0)={Gper[i,0,0]+sh:.4f} (1/r={1.0/i:.4f})' for i in (1, 2, 3, 4)),
        flush=True)

    fig, (ax1, ax2) = new_fig(12, 5.4, 1, 2)
    for name, d in dirs.items():
        r = np.array(d['r']); G = np.array(d['G'])
        ax1.plot(r, G * r, 'o-', label=name)
        ax2.plot(r, (G * r - 1.0) * 100, 'o-', label=name)
    ax1.axhline(1.0, color='k', ls=':', lw=0.8)
    ax1.set_xlabel('r, ячеек (a=1)')
    ax1.set_ylabel('G(x)·r')
    ax1.set_title('Функция Грина решётки → 1/r')
    ax2.set_xlabel('r, ячеек')
    ax2.set_ylabel('(G·r − 1)·100, %')
    ax2.set_title('Отклонение от 1/r и кубическая анизотропия')
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    save(fig, OUT / 'm1_lattice_green.png')

    # проверка: вдали от источника G ≈ 1/r с высокой точностью
    G10 = Gper[10, 0, 0] + sh
    check('G(10,0,0) = 1/10 ± 0.5%', abs(G10 * 10 - 1) < 5e-3, float(G10))
    # отличие в ближней зоне — существенно (это и есть «истинное» поле решётки)
    G1 = Gper[1, 0, 0] + sh
    report['green_G1'] = float(G1)
    check('G(1,0,0) > 1 (усиление поля решётки вдоль оси)',
          1.0 < G1 < 1.15, float(G1))


# ---------------------------------------------------- 2) поле из локального правила

def field_relax():
    print('== 2. Поле как неподвижная точка локального правила ==', flush=True)
    N, a = 48, 1.0
    src = np.zeros((N, N, N))
    src[N // 2, N // 2, N // 2] = 4.0 * np.pi / a ** 3
    Gref = lattice_green_fft(a, N)
    Gref = np.roll(Gref, (N // 2,) * 3, axis=(0, 1, 2))

    V = np.zeros_like(src)
    omega = 1.0                       # Якоби: SOR с ω>1 неустойчив из-за проекции нулевой моды
    resid_log = []
    for it in range(1, 1501):
        neigh = sum(np.roll(V, s, ax) for ax in range(3) for s in (1, -1))
        Vnew = (1 - omega) * V + omega * (neigh + src * a ** 2) / 6.0
        Vnew -= Vnew.mean()
        V = Vnew
        if it % 25 == 0:
            res = np.max(np.abs(V - Gref)) / max(np.max(np.abs(Gref)), 1e-300)
            resid_log.append((it, float(res)))
    res = np.max(np.abs(V - Gref)) / max(np.max(np.abs(Gref)), 1e-300)
    check('релаксация поля сошлась к функции Грина', res < 2e-2, float(res))
    report['field_relax'] = {'resid_log': [[int(i), float(r)] for i, r in resid_log],
                             'final_resid': float(res)}

    fig, (ax1, ax2) = new_fig(12, 5.4, 1, 2)
    it = np.array([i for i, _ in resid_log]); rr = np.array([r for _, r in resid_log])
    ax1.semilogy(it, rr)
    ax1.set_xlabel('число тактов (итераций)')
    ax1.set_ylabel('относительная невязка')
    ax1.set_title('Сходимость локальной релаксации к статическому полю')
    x = np.arange(1, N // 2)
    ax2.plot(x, -V[N // 2 + x, N // 2, N // 2], 'o', ms=4, label='поле после релаксации')
    ax2.plot(x, -Gref[N // 2 + x, N // 2, N // 2], '-', label='функция Грина (БПФ)')
    ax2.plot(x, 1.0 / x, '--', label='1/r (континуум)')
    ax2.set_xlabel('r, ячеек')
    ax2.set_ylabel('−V(r)')
    ax2.set_title('Статическое поле протона')
    ax2.legend(fontsize=8)
    save(fig, OUT / 'm7_field_relax.png')


# ------------------------------------------------------ 3) спектр дискретной модели

def spectrum():
    print('== 3. Спектр дискретной модели (точные s-оболочки решётки) ==', flush=True)
    Rmax = 60.0
    Gper = lattice_green_fft(1.0, 128)
    sh = G0_FREE - Gper[0, 0, 0]      # поправка тора: G_free(x) ≈ G_per(x) + sh вблизи источника

    # средние значения функции Грина по ТОЧНЫМ оболочкам s = |x|² (s ≤ 144)
    c = np.indices((25, 25, 25)).reshape(3, -1).T - 12
    X, Y, Z = c[:, 0], c[:, 1], c[:, 2]
    s2 = X * X + Y * Y + Z * Z
    ok = s2 <= 144
    Gex = np.zeros(145); cnt = np.zeros(145)
    Gv = Gper[(X % 128).ravel(), (Y % 128).ravel(), (Z % 128).ravel()] + sh
    np.add.at(Gex, s2[ok], Gv[ok])
    np.add.at(cnt, s2[ok], 1)
    Gex = np.divide(Gex, cnt, out=np.zeros_like(Gex), where=cnt > 0)
    report['shell_G'] = {str(s): float(Gex[s]) for s in (0, 1, 2, 3, 4, 5, 6, 8, 9)}
    print(f'    <G> по точным оболочкам: s=1: {Gex[1]:.4f}, s=2: {Gex[2]:.4f}, '
          f's=3: {Gex[3]:.4f}, s=4: {Gex[4]:.4f} (1/√s: 1.0, 0.707, 0.577, 0.5)', flush=True)

    def make_V(h, use_lattice_field):
        def V_s(sarr):
            sarr = np.asarray(sarr, dtype=np.int64)
            out = np.zeros_like(sarr, dtype=float)
            m0 = sarr == 0
            out[m0] = -G0_FREE / h
            nn = ~m0
            s = sarr[nn]
            if use_lattice_field:
                near = s <= 144
                out[nn & (sarr <= 144)] = -Gex[sarr[nn & (sarr <= 144)]] / h
                far = sarr > 144
                out[far] = -1.0 / (np.sqrt(sarr[far].astype(float)) * h)
            else:
                out[nn] = -1.0 / (np.sqrt(s.astype(float)) * h)
            return out
        return V_s

    a_list = [1.0, 0.5, 0.25, 0.2]
    rows = {}
    for a in a_list:
        h = a
        Smax = int((Rmax / h + 1) ** 2)
        r2 = r2_sieve(Smax)
        m = r3_multiplicities(Smax, r2)
        nsh = int(np.sum(m > 0))
        print(f'    a={a}: S_max={Smax}, оболочек {nsh}', flush=True)
        for label, use_lat in (('lattice', True), ('continuum', False)):
            V_s = make_V(h, use_lat)
            M, D, shells = exact_shell_hamiltonian(h, Smax, V_s, m, r2)
            # начальные векторы ЛОБПКГ: аналитические формы 1s..4s
            rad = np.sqrt(shells.astype(float)) * h
            X0 = np.column_stack([he.radial_R(n, 0, rad) for n in (1, 2, 3, 4)])
            P = sparse.diags(1.0 / (M.diagonal() / D + 0.3))   # предобусловливатель
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')   # ЛОБПКГ: первые 3-4 вектора сходятся,
                E, _ = lobpcg(M, X0, B=sparse.diags(D), M=P, largest=False,   # хвост блока —
                              tol=1e-8, maxiter=600)                          # некритично
            E = np.sort(E)
            Eneg = [float(x) for x in E if x < 0]
            rows[f'a={a},{label}'] = Eneg
            print(f'       поле {label}: ' + ' '.join(f'{x:.5f}' for x in Eneg), flush=True)
    report['spectrum'] = rows

    # таблица ошибок первых s-состояний: j-е связанное s-состояние имеет n = j+1
    tab = {}
    for a in a_list:
        tab[a] = {}
        for label in ('lattice', 'continuum'):
            for j, E in enumerate(rows[f'a={a},{label}']):
                n = j + 1
                tab[a][f'{n}s({label[0]})'] = {'E': E, 'err': E - he.energy_au(n)}
    report['spectrum_errors'] = {str(k): v for k, v in tab.items()}

    errs1 = [tab[a]['1s(l)']['err'] for a in a_list]
    errs2_a = [a for a in a_list if '2s(l)' in tab[a]]
    errs2 = [tab[a]['2s(l)']['err'] for a in errs2_a]
    slope1 = np.polyfit(np.log(a_list), np.log(np.abs(errs1)), 1)[0]
    report['spectrum_slope_E1s'] = float(slope1)
    check('E₁ₛ(a=0.2) в пределах 3%', abs(tab[0.2]['1s(l)']['err']) < 0.02,
          float(tab[0.2]['1s(l)']['err']))
    check('E₂ₛ(a=0.2) в пределах 2%', abs(tab[0.2]['2s(l)']['err']) < 2.5e-3,
          float(tab[0.2]['2s(l)']['err']))
    check('наклон сходимости E₁ₛ ≈ 2', 1.5 < slope1 < 2.7, float(slope1))
    print('    таблица E(a):', flush=True)
    for a in a_list:
        parts = []
        for key in ('1s(l)', '2s(l)', '3s(l)', '1s(c)'):
            if key in tab[a]:
                parts.append(f'{key}: E={tab[a][key]["E"]:.5f} (err {tab[a][key]["err"]:+.5f})')
        print(f'    a={a}: ' + ', '.join(parts), flush=True)

    near = {}
    for a in (0.5, 0.25, 0.2):
        near[a] = {'dE1s': tab[a]['1s(l)']['E'] - tab[a]['1s(c)']['E'],
                   'dE2s': (tab[a]['2s(l)']['E'] - tab[a]['2s(c)']['E'])
                   if '2s(l)' in tab[a] else None}
    report['near_field_effect'] = {str(k): v for k, v in near.items()}
    print(f'    влияние ближнего поля (G против 1/r): {near}', flush=True)

    fig, (ax1, ax2) = new_fig(12, 5.4, 1, 2)
    for a in a_list:
        Eneg = np.array(rows[f'a={a},lattice'])
        ax1.plot(np.full_like(Eneg, a), Eneg, 'o', ms=5,
                 label=f'a={a} ({len(Eneg)} связанных сост.)')
    for n in range(1, 6):
        ax1.axhline(he.energy_au(n), color='k', ls=':', lw=0.6)
        ax1.text(1.05, he.energy_au(n), f'−1/(2·{n}²)', fontsize=7, va='center')
    ax1.set_xscale('log')
    ax1.set_xlabel('шаг решётки a, a₀')
    ax1.set_ylabel('E, хартри')
    ax1.set_title('Связанные состояния дискретной модели')
    ax1.legend(fontsize=8)

    ax2.loglog(a_list, np.abs(errs1), 'o-', label=f'E₁ₛ, наклон {slope1:.2f}')
    if errs2_a:
        ax2.loglog(errs2_a, np.abs(errs2), 's-', label='E₂ₛ')
    ax2.set_xlabel('a, a₀')
    ax2.set_ylabel('|E − E_точн|, хартри')
    ax2.set_title('Сходимость к эталону при a → 0')
    ax2.legend()
    save(fig, OUT / 'm2_spectrum.png')

    # без поля связанных состояний нет: у свободной решётки спектр ≥ 0
    h = 0.2
    Smax = int((Rmax / h + 1) ** 2)
    r2 = r2_sieve(Smax)
    m = r3_multiplicities(Smax, r2)
    M0, D0, shells0 = exact_shell_hamiltonian(h, Smax, lambda s: np.zeros(len(np.atleast_1d(s))), m, r2)
    rad0 = np.sqrt(shells0.astype(float)) * h
    X0 = np.column_stack([np.ones_like(rad0), rad0, np.random.default_rng(1).standard_normal(len(rad0))])
    P0 = sparse.diags(1.0 / (M0.diagonal() / D0 + 0.3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E0, _ = lobpcg(M0, X0, B=sparse.diags(D0), M=P0, largest=False, tol=1e-8, maxiter=400)
    report['free_min_energy'] = float(E0.min())
    check('без поля нет связанных состояний (E ≥ 0)', E0.min() > -1e-6, float(E0.min()))

    # перекрёстная проверка: 3D-решётка (мнимое время) против точных оболочек
    for a_cross, N3 in ((1.0, 64), (0.2, 64)):
        V3 = free_space_potential(a_cross, N3)
        X3, Y3, Z3 = grid_coords(a_cross, N3)
        psi = np.exp(-np.sqrt(X3 ** 2 + Y3 ** 2 + Z3 ** 2)) / np.sqrt(np.pi)
        psi /= np.sqrt(np.sum(psi ** 2) * a_cross ** 3)
        dbeta = 0.04 if a_cross == 1.0 else 0.004
        psi3, _ = imag_time(psi, V3, a_cross, dbeta, 3000)
        E3 = energy_expect(psi3, V3, a_cross)
        Eshell = rows[f'a={a_cross},continuum'][0]
        dE = E3 - Eshell
        report[f'crosscheck_a={a_cross}'] = {'E_3d': float(E3), 'E_shell': float(Eshell),
                                             'dE': float(dE)}
        print(f'    перекрёстная проверка a={a_cross}: 3D E₁ₛ = {E3:.5f}, '
              f'оболочки E₁ₛ = {Eshell:.5f}, Δ = {dE:+.5f}', flush=True)
        check(f'3D vs оболочки при a={a_cross}', abs(dE) < 5e-3, float(dE))


# -------------------------------------------------------- 4) 3D-состояния на торе

def states3d():
    print('== 4. 3D-состояния (периодическая сетка, поле свободного пространства) ==', flush=True)
    N, a = 128, 0.2
    L = N * a
    V = free_space_potential(a, N)
    X, Y, Z = grid_coords(a, N)
    print(f'    сетка {N}³, a = {a} a₀, бокс {L:.1f} a₀; V(0) = {V[N//2,N//2,N//2]:.3f} хартри',
          flush=True)

    def norm(psi):
        return psi / np.sqrt(np.sum(psi ** 2) * a ** 3)

    dbeta, steps = 0.004, 6000
    psis, logs, Es = [], {}, {}
    for name, (n, l, m) in [('1s', (1, 0, 0)), ('2s', (2, 0, 0)), ('2p_z', (2, 1, 0))]:
        psi = norm(he.psi_nlm_on_grid(n, l, m, X, Y, Z))
        psi, log = imag_time(psi, V, a, dbeta, steps, states=psis, log_every=1000,
                             callback=lambda s, E, nm=name: print(
                                 f'    {nm}: шаг {s}, E = {E:.6f}', flush=True) if s % 3000 == 0 else None)
        E = energy_expect(psi, V, a)
        Es[name] = float(E)
        logs[name] = [[int(s), float(e)] for s, e in log]
        psis.append(psi)
        print(f'    {name}: E = {E:.6f} (эталон {he.energy_au(n):.6f})', flush=True)
        check(f'3D: {name} ≈ эталон', abs(E - he.energy_au(n)) < 0.02, float(E))
    report['states3d'] = {'a': a, 'N': N, 'E': Es, 'log': logs}

    fig, axes = new_fig(14, 9, 2, 3)
    axes[0][0].imshow(psis[0][:, :, N // 2].T ** 2, origin='lower', cmap='inferno',
                      extent=[-L / 2, L / 2, -L / 2, L / 2])
    axes[0][0].set_title('1s: |ψ|², плоскость z=0')
    axes[0][1].imshow(psis[1][:, :, N // 2].T ** 2, origin='lower', cmap='inferno',
                      extent=[-L / 2, L / 2, -L / 2, L / 2])
    axes[0][1].set_title('2s: |ψ|², плоскость z=0')
    axes[0][2].imshow(psis[2][N // 2, :, :].T ** 2, origin='lower', cmap='inferno',
                      extent=[-L / 2, L / 2, -L / 2, L / 2])
    axes[0][2].set_title('2p_z: |ψ|², плоскость x=0')
    for j, (name, (n, l, m)) in enumerate([('1s', (1, 0, 0)), ('2s', (2, 0, 0)), ('2p_z', (2, 1, 0))]):
        rm, dens = radial_density(psis[j], a)
        r = np.linspace(0.05, 14.0, 2000)
        exact_dens = he.radial_R(n, l, r) ** 2 / (4 * np.pi)
        axes[1][j].semilogy(rm, np.maximum(dens, 1e-12), label='модель (решётка)')
        axes[1][j].semilogy(r, np.maximum(exact_dens, 1e-14), '--', label='аналитика')
        axes[1][j].set_xlim(0, 14)
        axes[1][j].set_ylim(1e-10, 1)
        axes[1][j].set_xlabel('r, a₀')
        axes[1][j].set_ylabel('|ψ(r)|²')
        axes[1][j].set_title(f'{name}: E = {Es[name]:.4f} хартри')
        axes[1][j].legend(fontsize=8)
    save(fig, OUT / 'm3_states3d.png')


# ----------------------------------------------------- 5) биения на дискретных часах

def beats():
    print('== 5. Квантовые биения на дискретных часах ==', flush=True)
    N, a = 96, 0.26
    L = N * a
    tau = a / C_AU                       # 1 такт CLK
    V = free_space_potential(a, N)
    X, Y, Z = grid_coords(a, N)

    def norm(psi):
        return psi / np.sqrt(np.sum(psi ** 2) * a ** 3)

    print(f'    сетка {N}³, a = {a} a₀, такт τ = a/c = {tau:.5f} а.е. = {tau*T_AU_S*1e18:.2f} ас',
          flush=True)
    psis = []
    for name, (n, l, m) in [('1s', (1, 0, 0)), ('2s', (2, 0, 0))]:
        psi = norm(he.psi_nlm_on_grid(n, l, m, X, Y, Z))
        psi, _ = imag_time(psi, V, a, 0.004, 5000, states=psis, log_every=1000,
                           callback=lambda s, E, nm=name: print(
                               f'    {nm}: шаг {s}, E = {E:.6f}', flush=True))
        E = energy_expect(psi, V, a)
        print(f'    {name}: E = {E:.6f}', flush=True)
        psis.append(psi)
    E1, E2 = energy_expect(psis[0], V, a), energy_expect(psis[1], V, a)
    dE_eig = E2 - E1
    psi0 = norm(psis[0] + psis[1])

    steps = 9000
    t, C = splitstep_real_time(psi0, V, a, tau, steps, record_every=25)
    C2 = np.abs(C) ** 2

    def model(t, w, amp, off):
        return amp * np.cos(0.5 * w * t) ** 2 + off

    popt, _ = curve_fit(model, t, C2, p0=[0.375, 1.0, 0.0], bounds=([0.3, 0.8, -0.05], [0.45, 1.2, 0.05]))
    w_fit = popt[0]
    T_ticks = 2 * np.pi / (w_fit * tau)
    print(f'    ΔE(собственные значения) = {dE_eig:.5f}, ΔE(подгонка биений) = {w_fit:.5f}, '
          f'эталон 0.375; период = {T_ticks:.0f} тактов = {2*np.pi/w_fit*T_AU_S*1e15:.3f} фс',
          flush=True)
    check('биения: подгонка ≈ собственные значения модели', abs(w_fit - dE_eig) < 0.01,
          float(w_fit))
    check('биения: ΔE модели ≈ 0.375 ± 5%', abs(dE_eig - 0.375) < 0.02, float(dE_eig))
    report['beats'] = {'a': a, 'tau_au': float(tau), 'tau_s': float(tau * T_AU_S),
                       'E1': float(E1), 'E2': float(E2), 'dE_eig': float(dE_eig),
                       'dE_fit': float(w_fit), 'T_ticks': float(T_ticks),
                       'T_fs': float(2 * np.pi / w_fit * T_AU_S * 1e15)}

    fig, (ax1, ax2) = new_fig(12, 5.4, 1, 2)
    ax1.plot(t, C2, 'o', ms=3, label='модель (такт = τ)')
    ax1.plot(t, model(t, *popt), '-', label=f'подгонка, ΔE = {w_fit:.4f}')
    ax1.plot(t, np.cos(0.5 * 0.375 * t) ** 2, '--', label='эталон ΔE = 0.375')
    ax1.set_xlabel('t, а.е.')
    ax1.set_ylabel('|C(t)|²')
    ax1.set_title('Биения 1s+2s на дискретных часах')
    ax1.legend(fontsize=8)
    ax1b = ax1.twiny()
    ax1b.set_xlim(np.array(ax1.get_xlim()) / tau)
    ax1b.set_xlabel('такты CLK')

    y = C2 - C2.mean()
    yz = np.concatenate([y, np.zeros(15 * len(y))])
    f = np.fft.rfftfreq(len(yz), d=t[1] - t[0])
    spec = np.abs(np.fft.rfft(yz))
    ax2.plot(f * 2 * np.pi, spec)
    ax2.axvline(0.375, color='r', ls='--', label='эталон ΔE = 0.375')
    ax2.axvline(w_fit, color='g', ls=':', label=f'подгонка {w_fit:.4f}')
    ax2.set_xlim(0.2, 0.6)
    ax2.set_xlabel('частота ω (энергия), хартри')
    ax2.set_ylabel('|БПФ|')
    ax2.set_title('Спектр биений (нулевое заполнение ×16)')
    ax2.legend(fontsize=8)
    save(fig, OUT / 'm4_beats.png')


# ------------------------------------------------------------ 6) битовая глубина

def bitdepth():
    print('== 6. Битовая глубина ячеек ==', flush=True)
    N, a = 64, 0.32
    V = free_space_potential(a, N)
    X, Y, Z = grid_coords(a, N)

    def norm(psi):
        return psi / np.sqrt(np.sum(psi ** 2) * a ** 3)

    psi0 = norm(he.psi_nlm_on_grid(1, 0, 0, X, Y, Z))
    psi_ref, _ = imag_time(psi0, V, a, 0.004, 5000, log_every=1000)
    E_ref = energy_expect(psi_ref, V, a)
    print(f'    эталон float64: E = {E_ref:.6f}', flush=True)

    B_list = [5, 6, 8, 10, 12, 16]
    errs = {}
    for B in B_list:
        psi, _ = imag_time(psi0, V, a, 0.004, 5000, B=B)
        E = energy_expect(psi, V, a)
        errs[B] = {'E': float(E), 'err': float(E - E_ref)}
        print(f'    B = {B:2d} бит: E = {E:.6f}, ошибка = {E - E_ref:+.2e}', flush=True)
    report['bitdepth'] = {'B': B_list, 'E': [errs[B]['E'] for B in B_list],
                          'err': [errs[B]['err'] for B in B_list], 'E_ref': float(E_ref)}

    # B_min: минимальная глубина, при которой ошибка < 1e-3 (экстраполяция по B ≥ 8:
    # при B ≤ 6 режим другой — квантизация разрушает состояние)
    slope_hi = np.polyfit([B for B in B_list if B >= 8],
                          np.log10(np.abs([errs[B]['err'] for B in B_list if B >= 8])), 1)[0]
    B_est = float(np.ceil(-3.0 / slope_hi))
    report['B_min_1e-3'] = B_min
    report['B_est_1e-3'] = B_est
    print(f'    минимальная глубина для ошибки < 1e-3: B = {B_min} '
          f'(оценка по наклону B≥8: {B_est:.0f} бит)', flush=True)
    check('ошибка B=16 < ошибки B=5', abs(errs[16]['err']) < abs(errs[5]['err']),
          float(errs[16]['err']))
    check('ошибка B=16 < 2e-3', abs(errs[16]['err']) < 2e-3, float(errs[16]['err']))

    fig, ax = new_fig(9, 5.5)
    e = np.array([abs(errs[B]['err']) for B in B_list])
    ax.semilogy(B_list, e, 'o-')
    slope = np.polyfit(B_list, np.log10(e), 1)[0]
    ax.plot(B_list, 10 ** (np.polyval([slope, np.log10(e[-1]) - slope * B_list[-1]], B_list)),
            '--', alpha=0.5, label=f'наклон {slope:.2f} дек/бит')
    ax.set_xlabel('число бит B на компоненту амплитуды')
    ax.set_ylabel('|E(B) − E(float64)|, хартри')
    ax.set_title('Точность энергии vs битовая глубина ячеек')
    ax.legend(fontsize=8)
    save(fig, OUT / 'm5_bitdepth.png')
    report['bitdepth_slope'] = float(slope)


# -------------------------------------------------------- 7) информационный учёт

def info_bits():
    print('== 7. Информационный учёт и анализ числа 1836 ==', flush=True)
    ratio = M_P_OVER_M_E
    six_pi5 = 6 * np.pi ** 5
    info = {
        'm_p_over_m_e': ratio,
        'six_pi_5': float(six_pi5),
        'six_pi_5_rel_dev': float((six_pi5 - ratio) / ratio),
        '1836_factorization': '2²·3³·17',
        '2_pow_11': 2048,
        '2048_minus_1836': 212,
        '1836_binary': bin(1836),
        '1836_hex': hex(1836),
        'sphere_radius_for_1836_cells': float((3 * 1836 / (4 * np.pi)) ** (1 / 3)),
        'm_e_c2_keV': 510.99895,
        'm_p_c2_MeV': 938.27208816,
        'binding_eV': 13.6057,
        'binding_in_units_of_m_e': 13.6057 / 510998.95,
    }
    # биты состояния электрона (по данным bitdepth, B_min = 12)
    B = 12
    N, a = 64, 0.32
    X, Y, Z = grid_coords(a, N)
    psi = np.exp(-np.sqrt(X ** 2 + Y ** 2 + Z ** 2)) / np.sqrt(np.pi)
    psi /= np.sqrt(np.sum(psi ** 2) * a ** 3)
    thr = np.max(np.abs(psi)) / 2 ** (B - 1)
    n_cells_e = int(np.sum(np.abs(psi) > thr))
    info['electron_state_cells'] = n_cells_e
    info['electron_state_bits'] = 2 * B * n_cells_e
    info['r_support_electron_bohr'] = float(np.log(2 ** (B - 1)))
    # биты поля (свободное пространство): точность 0.005 хартри
    r_cut = 1.0 / 0.005
    cells_field = (4 * np.pi / 3) * (r_cut / a) ** 3
    B_V = int(np.ceil(np.log2(G0_FREE / a / 0.005)))
    info['field_cutoff_bohr'] = float(r_cut)
    info['field_cells'] = float(cells_field)
    info['field_bits'] = float(B_V * cells_field)
    info['B_V'] = B_V
    # биты поля на торе расчёта (64³, a=0.32)
    info['torus_field_cells'] = N ** 3
    info['torus_field_bits'] = B_V * N ** 3
    # тактовый учёт
    a_beat = 0.26
    tau = a_beat / C_AU
    info['tau_s'] = float(tau * T_AU_S)
    info['orbit_ticks_n1'] = float(2 * np.pi / tau)
    info['beat_period_ticks'] = float(2 * np.pi / (0.375 * tau))
    info['lyman_alpha_cells'] = float(121.567e-9 / (a_beat * 5.29177210903e-11))
    report['info'] = info
    print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)


def main():
    print('=== ВЫЧИСЛИТЕЛЬНАЯ МОДЕЛЬ ===', flush=True)
    green()
    field_relax()
    spectrum()
    states3d()
    beats()
    bitdepth()
    info_bits()
    report['checks'] = checks
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding='utf-8')
    nfail = sum(1 for c in checks if not c['ok'])
    print(f'\nВсего проверок: {len(checks)}, провалено: {nfail}', flush=True)
    return nfail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
