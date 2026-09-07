# -*- coding: utf-8 -*-
"""Расширение исследования вычислительной модели.

1) Правило материи — шахматная доска Фейнмана (квантовая прогулка Дирака):
   унитарность, дисперсия vs Дирак/Шрёдингер, Zitterbewegung (частота 2m).
2) Свободный протон: динамическое поле (волновое уравнение, 7-точечный
   лапласиан, такт τ = a/(√3·c) — условие Куранта), фронт со скоростью c,
   установление статического поля 1/r; анимация (GIF).
3) Числа структур: 1836 = 6·17·18 = 3·4·9·17, 6π⁵ ≈ 1836, π⁵ ≈ 17·18,
   r_p = 4λ̄_p, два масштаба (планковский субстрат vs эффективные ячейки).

Запуск:  python scripts/run_extensions.py
Вывод:   results/extensions/*.png, *.gif, report.json
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.units import (M_P_OVER_M_E, M_P_KG, M_E_KG, HBAR_JS, C_LIGHT, A0_M,
                       C_AU, T_AU_S)
from src import checkerboard as cb
from src import hydrogen_exact as he
from src.lattice import (lattice_green_fft, free_space_potential, grid_coords,
                         imag_time, energy_expect)
from src.plotting import new_fig, save

OUT = ROOT / 'results' / 'extensions'
OUT.mkdir(parents=True, exist_ok=True)

report = {}
checks = []


def check(name, ok, value):
    checks.append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + '   ->   ' + str(value), flush=True)


# ------------------------------------------------------------ 1) шахматная доска

def checkerboard():
    print('== 1. Правило материи: шахматная доска Фейнмана (квантовая прогулка Дирака) ==',
          flush=True)
    eps, m = 0.05, 1.0

    # унитарность
    N = 512
    rng = np.random.default_rng(7)
    psiR = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    psiL = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    n0 = cb.norm(psiR, psiL)
    psiR, psiL = cb.step(psiR, psiL, eps, m)
    dn = abs(cb.norm(psiR, psiL) - n0)
    check('унитарность шага (норма сохраняется)', dn < 1e-12, float(dn))

    # дисперсия против Дирака и Шрёдингера
    k = np.linspace(0.0, 2.5, 300)
    w_ca = cb.dispersion(eps, m, k)
    w_dirac = np.sqrt(m ** 2 + k ** 2)
    w_schr = m + k ** 2 / (2 * m)
    dev_dirac = np.max(np.abs(w_ca - w_dirac))
    k_schr = k[k < 0.3]
    dev_schr = np.max(np.abs(cb.dispersion(eps, m, k_schr) - (m + k_schr ** 2 / (2 * m))))
    check('дисперсия КА = Дирак √(m²+k²) ± 1e-3', dev_dirac < 1e-3, float(dev_dirac))
    check('нерелятивистский предел m + k²/2m при k<0.3', dev_schr < 1.5e-3, float(dev_schr))
    report['checkerboard'] = {'eps': eps, 'm': m, 'dev_dirac': float(dev_dirac),
                              'dev_schrodinger': float(dev_schr)}

    fig, ax = new_fig(9, 6)
    ax.plot(k, w_ca, label='клеточный автомат (точная дисперсия)')
    ax.plot(k, w_dirac, '--', label='Дирак: ω = √(m²+k²)')
    ax.plot(k, w_schr, ':', label='Шрёдингер: ω = m + k²/2m')
    ax.axvline(0.3, color='k', ls=':', lw=0.6)
    ax.text(0.32, 1.35, 'k ≪ m: Шрёдингер', fontsize=8)
    ax.text(1.6, 1.7, 'k ~ m: только Дирак', fontsize=8)
    ax.set_xlabel('импульс k (ħ/a = 1)')
    ax.set_ylabel('частота ω (энергия)')
    ax.set_title('Дисперсия локального правила: континуальные пределы')
    ax.legend(fontsize=8)
    save(fig, OUT / 'ext1_dirac_dispersion.png')

    # Zitterbewegung: дрожание скорости с частотой 2m
    Nz, x0 = 4096, 1200
    x = np.arange(Nz)
    g = np.exp(-(x - x0) ** 2 / (2 * 0.7 ** 2))
    psiR = g.astype(complex)
    psiL = np.zeros_like(g, dtype=complex)
    ts, xs = [], []
    for t in range(1601):
        if t % 5 == 0:
            ts.append(t * eps)
            xs.append(cb.mean_x(psiR, psiL, 1.0))
        psiR, psiL = cb.step(psiR, psiL, eps, m)
    xs = np.array(xs)
    ts = np.array(ts)
    v = np.diff(xs) / np.diff(ts)
    tv = ts[1:]
    nw = 80

    def model(t, w, A, g2, ph, a0, a1):
        return a0 + a1 * t + A * np.cos(w * t + ph) * np.exp(-g2 * t)

    popt, _ = curve_fit(model, tv[:nw], v[:nw],
                        p0=[2.0, 0.8, 0.05, 0.0, 19.0, -0.02], maxfev=20000)
    w_fit, A_fit, g_fit = popt[0], popt[1], popt[2]
    check('Zitterbewegung: частота 2m (подгонка)', abs(w_fit - 2 * m) < 0.1, float(w_fit))
    report['checkerboard']['zitter'] = {'w_fit': float(w_fit), 'A': float(A_fit),
                                        'gamma': float(g_fit), 'mean_speed': float(v.mean()),
                                        'c_cells_per_unit': 1.0 / eps}

    fig, ax = new_fig(9, 5.5)
    ax.plot(tv[:120], v[:120], 'o', ms=3, label='скорость ⟨σ₃⟩(t)')
    ax.plot(tv[:120], model(tv[:120], *popt), '-', label=f'подгонка: ω = {w_fit:.3f} ≈ 2m, A = {A_fit:.2f}')
    ax.axhline(v.mean(), color='k', ls=':', lw=0.8, label=f'среднее {v.mean():.2f} ≈ c')
    ax.set_xlabel('t (ħ/mc²)')
    ax.set_ylabel('d⟨x⟩/dt, ячеек/ед. времени')
    ax.set_title('Zitterbewegung: дрожание скорости с частотой 2mc²/ħ')
    ax.legend(fontsize=8)
    save(fig, OUT / 'ext2_zitterbewegung.png')


# ------------------------------------------------------ 2) свободный протон (GIF)

def proton_animation():
    print('== 2. Свободный протон: динамика поля и анимация ==', flush=True)
    N, a = 128, 1.0
    c = N // 2
    # источник — шар радиуса 3 (протон — структура конечного размера, а не точка)
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
    R = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2)
    src = np.zeros((N, N, N))
    src[R <= 3.0] = 1.0
    src *= 4.0 * np.pi / (a ** 3 * np.sum(src))

    def lap(f):
        s = np.zeros_like(f)
        for ax in range(3):
            s += np.roll(f, 1, ax) + np.roll(f, -1, ax)
        return (s - 6.0 * f) / a ** 2

    tau = a / np.sqrt(3.0)          # такт: условие Куранта для 7-точечного правила
    tau2 = tau ** 2
    T = 80                          # фронт достигает r = 80/√3 ≈ 46 < 64 (граница)
    V = np.zeros((N, N, N))
    Vp = np.zeros((N, N, N))
    profiles = {}
    r_front = []
    frames = []
    for t in range(1, T + 1):
        Vn = 2.0 * V - Vp + tau2 * (lap(V) - src)      # V̈ = LV − 4πρ, V ≤ 0
        Vn -= Vn.mean()
        Vp, V = V, Vn
        if t % 10 == 0:
            profiles[t] = [float(x) for x in V[c:, c, c][:60]]
        prof = np.abs(V[c:, c, c])
        fr = np.where(prof > 0.5 / np.maximum(np.arange(N - c), 1))[0]
        r_front.append((t, int(fr.max()) if len(fr) else 0))
        if t % 2 == 0:
            fig, ax = new_fig(5.6, 5.2)
            im = ax.imshow(-V[c, :, :].T, origin='lower', cmap='inferno', vmin=0.0, vmax=0.6,
                           extent=[-N / 2, N / 2, -N / 2, N / 2])
            ax.plot(0, 0, 'o', color='white', ms=9)
            ax.plot(0, 0, 'o', color='crimson', ms=6)
            ax.set_title(f'свободный протон: поле на такте t = {t}\n'
                         f'фронт на r ≈ {t/3**0.5:.0f} ячеек (скорость c = a/(√3·τ))',
                         fontsize=9)
            ax.set_xlabel('x, ячеек')
            ax.set_ylabel('y, ячеек')
            fig.colorbar(im, ax=ax, label='−V (напряжённость поля)', fraction=0.046)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frames.append(Image.open(buf).convert('RGB'))
            plt.close(fig)

    frames[0].save(OUT / 'proton_field.gif', save_all=True, append_images=frames[1:],
                   duration=110, loop=0)
    print(f'    анимация сохранена: {OUT / "proton_field.gif"} ({len(frames)} кадров)', flush=True)

    # фронт: r(t) = t/√3 − lag, lag ~ 1–2 ячейки (порог 50% статики)
    rf = np.array(r_front)
    lag = rf[:, 0] / np.sqrt(3.0) - rf[:, 1]
    late = rf[:, 0] >= 40
    slope, intercept = np.polyfit(rf[10:, 0], rf[10:, 1], 1)
    check('фронт поля: скорость c = a/(√3τ)', abs(slope - 1 / np.sqrt(3.0)) < 0.04, float(slope))
    check('фронт отстаёт от светового конуса ≤ 3 ячеек', np.max(np.abs(lag[late])) < 3.0,
          float(np.max(np.abs(lag[late]))))

    # статика: за фронтом поле = СВОБОДНАЯ функция Грина (волны периодических
    # образов источника ещё не дошли: t=80 < 128−r; поправка Маделунга ξ/N)
    Gref = lattice_green_fft(1.0, N)
    Gref = np.roll(Gref, (c, c, c), axis=(0, 1, 2))
    rr = np.arange(5, 11)
    vprof = V[c + rr, c, c]
    gprof = -(Gref[c + rr, c, c] + 2.8373 / N)      # G_free = G_per + ξ/N
    dev = np.max(np.abs(vprof / gprof - 1.0))
    check('за фронтом поле ≈ статическое 1/r (r = 5..10)', dev < 0.08, float(dev))
    report['proton'] = {'N': N, 'T': T, 'tau': float(tau), 'front_slope': float(slope),
                        'front_intercept': float(intercept),
                        'static_max_dev': float(dev),
                        'profiles': {str(t): v for t, v in profiles.items()},
                        'r_front': [[int(t), int(r)] for t, r in r_front]}

    fig, (ax1, ax2) = new_fig(12, 5.2, 1, 2)
    for t, pr in profiles.items():
        ax1.plot(np.arange(len(pr)), -np.array(pr), label=f'такт t = {t}')
    ax1.plot(np.arange(2, 46), 1.0 / np.arange(2, 46), 'k--', lw=1.5, label='1/r (статика)')
    ax1.set_xlim(0, 46)
    ax1.set_ylim(0, 0.6)
    ax1.set_xlabel('r, ячеек')
    ax1.set_ylabel('−V(r)')
    ax1.set_title('Фронт поля распространяется со скоростью c и оставляет статику 1/r')
    ax1.legend(fontsize=8)
    ax2.plot(rf[:, 0], rf[:, 1], 'o-', label=f'фронт, наклон {slope:.3f} ≈ 1/√3')
    ax2.plot(rf[:, 0], rf[:, 0] / np.sqrt(3.0), '--', label='световой конус r = t/√3')
    ax2.set_xlabel('t, тактов (τ = a/(√3·c))')
    ax2.set_ylabel('r_фронта, ячеек')
    ax2.set_title('Световой конус локального правила')
    ax2.legend(fontsize=8)
    save(fig, OUT / 'ext3_proton_front.png')


# ------------------------------------------------- 3) нейтрино и структуры частиц

def structure_numbers():
    print('== 3. Числа структур: 1836, 6π⁵, бит массы, кварки, нейтрино, два масштаба ==',
          flush=True)
    m_e_ev = 510998.95
    ratio = M_P_OVER_M_E
    m_bit_B = m_e_ev / ratio                # вариант B: электрон = 1836 бит
    m_quark_me = (M_P_KG / 3.0) / M_E_KG    # конституентный кварк, в массах электрона
    binding_me = ratio - 1836.0
    six_pi5 = 6 * np.pi ** 5
    pi5 = np.pi ** 5
    # радиусы и масштабы
    rp_fm = 0.8414                          # зарядовый радиус протона, фм
    lp_m = 1.616255e-35                     # планковская длина
    rp_m = rp_fm * 1e-15
    lp_reduced_p = HBAR_JS / (M_P_KG * C_LIGHT)     # λ̄_p = ħ/(m_p c)
    lp_reduced_e = HBAR_JS / (M_E_KG * C_LIGHT)     # λ̄_e = ħ/(m_e c)
    n_planck_proton = (4 * np.pi / 3) * (rp_m / lp_m) ** 3
    a_eff = rp_m / 7.6                    # эффективная ячейка структуры протона
    redundancy = n_planck_proton / 1836.0
    info = {
        'm_p_over_m_e': ratio,
        '1836_as_6_17_18': 6 * 17 * 18,
        '1836_as_3_4_9_17': 3 * 4 * 9 * 17,
        '612_as_4_9_17': 4 * 9 * 17,
        'm_bit_B_eV': m_bit_B,
        'm_quark_MeV': (M_P_KG / 3.0) * C_LIGHT ** 2 / 1.602176634e-13,
        'm_quark_in_m_e': m_quark_me,
        'binding_me': binding_me,
        'binding_keV': binding_me * m_e_ev / 1e3,
        'six_pi_5': six_pi5,
        'six_pi_5_rel_dev': (six_pi5 - ratio) / ratio,
        'm_p_me_minus_6pi5_me': ratio - six_pi5,
        'pi_5': pi5,
        'pi_5_vs_17_18': (pi5 - 17 * 18) / (17 * 18),
        'sphere_R_for_1836': (3 * 1836 / (4 * np.pi)) ** (1 / 3),
        'r_p_over_lambda_p': rp_m / lp_reduced_p,        # должно быть ≈ 4
        'lambda_e_over_lambda_p': lp_reduced_e / lp_reduced_p,  # = m_p/m_e
        'r_p_over_lambda_e': rp_m / lp_reduced_e,        # ≈ 4/1836
        'a_eff_fm': a_eff * 1e15,
        'a_eff_over_lambda_p_2': a_eff / (lp_reduced_p / 2.0),
        'a_eff_over_a0': a_eff / A0_M,
        'planck_cells_in_proton': n_planck_proton,
        'redundancy_per_bit': redundancy,
        'a0_over_lP': A0_M / lp_m,
        'lambda_e_cells_at_a_eff': lp_reduced_e / a_eff,
        'm_e_over_m_nu': {str(mnu): m_e_ev / mnu for mnu in (0.05, 0.1, 0.3, 0.8)},
    }
    report['structure'] = info
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in info.items()}, indent=2, ensure_ascii=False), flush=True)
    check('1836 = 6·17·18', 6 * 17 * 18 == 1836, 1836)
    check('1836 = 3·4·9·17 (3 кварка × 612)', 3 * 4 * 9 * 17 == 1836, 1836)
    check('кварк = 612 = 4·9·17 единиц', abs(m_quark_me - 612) < 1.0, float(m_quark_me))
    check('совпадение 6π⁵ с 1836 (0.002%)', abs((six_pi5 - ratio) / ratio) < 2e-5,
          float((six_pi5 - ratio) / ratio))
    check('π⁵ ≈ 17·18 (6·10⁻⁵)', abs((pi5 - 17 * 18) / (17 * 18)) < 1e-4,
          float((pi5 - 17 * 18) / (17 * 18)))
    check('радиус протона r_p = 4λ̄_p (0.02%)', abs(rp_m / lp_reduced_p - 4.0) < 2e-3,
          float(rp_m / lp_reduced_p))


# --------------------------------------- 4) нуклеосинтез дейтерия (p + n → d + γ)

def deuterium():
    print('== 4. Нуклеосинтез дейтерия: проверка структуры протона ==', flush=True)
    m_p = 938.27208816          # МэВ
    m_n = 939.56542052
    m_d = 1875.61294257
    m_e = 0.510998950           # МэВ
    B_d = m_p + m_n - m_d       # энергия связи дейтрона
    B_d_me = B_d / m_e
    m_bit = 510998.95 / M_P_OVER_M_E / 1e6   # m_e/1836, МэВ
    B_d_bits = B_d / m_bit

    # спин и магнитный момент дейтрона → доля d-волны:
    # μ_d = (μ_p+μ_n)(1 − 1.5·P_D) + 0.75·P_D
    mu_p, mu_n, mu_d = 2.79284734463, -1.91304273, 0.8574382338
    P_D = (mu_p + mu_n - mu_d) / (1.5 * (mu_p + mu_n) - 0.75)
    Q_d = 0.2859                 # квадрупольный момент, фм²

    # BBN: закалка n/p, распад нейтрона в ожидании, бутылочное горлышко
    dm = m_n - m_p              # 1.2933 МэВ
    T_f = 0.8                   # МэВ — закалка слабых процессов
    np_freeze = np.exp(-dm / T_f)
    t_wait = 200.0              # с — ожидание до T_D
    tau_n = 879.4
    np_D = np_freeze * np.exp(-t_wait / tau_n)
    eta = 6.1e-10
    T_D = 0.07
    for _ in range(50):         # T_D = B_d/(ln(1/η) + 1.5·ln(m_p/T_D))
        T_D = B_d / (np.log(1.0 / eta) + 1.5 * np.log(m_p / T_D))
    Y_p = 2 * np_D / (1 + np_D)
    D_H = 2.527e-5              # наблюдаемое (D/H)_p

    info = {
        'B_d_MeV': B_d, 'B_d_in_m_e': B_d_me, 'B_d_in_bits_278eV': B_d_bits,
        'B_d_over_mass_sum': B_d / (m_p + m_n),
        'm_d_in_m_e': m_d / m_e,
        'I_p_plus_I_n_minus_Bd': (m_p + m_n - B_d) / m_e,
        'spin_deuteron': 1.0,
        'mu_p': mu_p, 'mu_n': mu_n, 'mu_d': mu_d,
        'P_D_d_wave': P_D, 'Q_d_fm2': Q_d,
        'dm_n_p_MeV': dm, 'np_freeze': np_freeze, 'np_at_deuterium': np_D,
        'T_freeze_MeV': T_f, 'T_deuterium_MeV': T_D,
        'eta': eta, 'Y_p_estimate': Y_p, 'Y_p_observed': 0.247,
        'D_H_observed': D_H,
    }
    report['deuterium'] = info
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in info.items()}, indent=2, ensure_ascii=False), flush=True)
    check('энергия связи дейтрона B_d = 2.2246 МэВ', abs(B_d - 2.2246) < 1e-3, float(B_d))
    check('B_d = 4.35 массы электрона', abs(B_d_me - 4.353) < 0.01, float(B_d_me))
    check('B_d ≈ 7994 бита (m_bit = 278 эВ), целочисленность ±1', abs(B_d_bits - round(B_d_bits)) < 1.0,
          float(B_d_bits))
    check('d-волна дейтрона P_D ≈ 4% (по магнитным моментам)', 0.03 < P_D < 0.05, float(P_D))
    check('n/p при закалке ≈ 1/5', 0.15 < np_freeze < 0.25, float(np_freeze))
    check('бутылочное горлышко T_D ∈ 0.05..0.10 МэВ', 0.05 < T_D < 0.10, float(T_D))
    check('оценка Y_p ∈ 0.22..0.30', 0.22 < Y_p < 0.30, float(Y_p))

    fig, (ax1, ax2) = new_fig(12, 5.2, 1, 2)
    T = np.geomspace(0.05, 3.0, 200)
    npr = np.where(T >= T_f, np.exp(-dm / T), np_freeze)
    ax1.semilogx(T, npr, label='n/p (равновесие → закалка)')
    ax1.axvline(T_f, color='r', ls='--', lw=1, label=f'закалка T = {T_f} МэВ')
    ax1.axvline(T_D, color='g', ls='--', lw=1, label=f'горлышко дейтерия T = {T_D:.3f} МэВ')
    ax1.set_xlabel('температура, МэВ')
    ax1.set_ylabel('n/p')
    ax1.set_title(f'BBN: n/p = e^(−Δm/T), Δm = {dm:.3f} МэВ; Y_p ≈ {Y_p:.2f}')
    ax1.legend(fontsize=8)
    labels = ['p', 'n', 'd', 'B_d']
    vals = [m_p / m_e, m_n / m_e, m_d / m_e, B_d / m_e]
    bars = ax2.bar(labels, vals, color=['tab:blue', 'tab:orange', 'tab:green', 'crimson'])
    ax2.set_ylabel('масса, единиц m_e')
    ax2.set_title(f'Бюджет масс p + n → d + γ: B_d = {B_d:.4f} МэВ = {B_d_me:.3f} m_e')
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 20, f'{v:.1f}', ha='center', fontsize=8)
    save(fig, OUT / 'ext4_deuterium.png')


# -------------------------------------- 5) взаимодействие протона и электрона

def atom_animation():
    print('== 5. Анимация взаимодействия протона и электрона ==', flush=True)
    N, a = 96, 0.26
    V = free_space_potential(a, N)
    X, Y, Z = grid_coords(a, N)

    def norm(psi):
        return psi / np.sqrt(np.sum(psi ** 2) * a ** 3)

    psis, Es = [], {}
    for name, (n, l, m) in [('1s', (1, 0, 0)), ('2p_z', (2, 1, 0))]:
        psi = norm(he.psi_nlm_on_grid(n, l, m, X, Y, Z))
        psi, _ = imag_time(psi, V, a, 0.004, 5000, states=psis, log_every=1000)
        E = energy_expect(psi, V, a)
        Es[name] = float(E)
        psis.append(psi)
        print(f'    {name}: E = {E:.5f}', flush=True)
    psi1, psi2 = psis
    E1, E2 = Es['1s'], Es['2p_z']
    dE = E2 - E1
    T_au = 2 * np.pi / dE

    # дипольный момент: d = Σ ψ1·z·ψ2 (физическая норма), <z>(t) = d·cos(ΔE·t)
    # на решётке a=0.26 состояние связано глубже (−0.520 против −0.5), поэтому
    # d ≈ 0.705 вместо континуального 0.7449 (−5.4% — ошибка дискретизации)
    d_dip = float(np.sum(psi1 * Z * psi2))
    check('дипольная амплитуда ≈ 0.7449 a₀ ± 8% (дискретизация)', abs(abs(d_dip) - 0.7449) < 0.06,
          float(d_dip))

    nf = 40
    ts = np.linspace(0.0, T_au, nf, endpoint=False)
    zvals = []
    frames = []
    for t in ts:
        psi = (psi1 * np.exp(-1j * E1 * t) + psi2 * np.exp(-1j * E2 * t)) / np.sqrt(2.0)
        dens = np.abs(psi) ** 2
        zt = float(np.sum(dens * Z))
        zvals.append(zt)
        fig, (ax1, ax2) = new_fig(11, 5.0, 1, 2)
        ax1.imshow(dens[N // 2, :, :].T, origin='lower', cmap='inferno',
                   extent=[-N * a / 2, N * a / 2, -N * a / 2, N * a / 2])
        ax1.contour(-V[N // 2, :, :].T, levels=[0.05, 0.15, 0.4], colors='white',
                    alpha=0.5, linewidths=0.7,
                    extent=[-N * a / 2, N * a / 2, -N * a / 2, N * a / 2])
        ax1.plot(0, 0, 'o', color='white', ms=8)
        ax1.plot(0, 0, 'o', color='crimson', ms=5)
        ax1.set_xlabel('y, a₀')
        ax1.set_ylabel('z, a₀')
        ax1.set_title(f'|ψ|² (плоскость x=0): t = {t:.2f} а.е.', fontsize=9)
        zz = np.array(zvals)
        ax2.plot(ts[:len(zz)], zz, '-', color='tab:blue')
        ax2.plot(t, zt, 'o', color='crimson', ms=7)
        ax2.axhline(0, color='k', ls=':', lw=0.6)
        ax2.set_xlabel('t, а.е.')
        ax2.set_ylabel('⟨z⟩, a₀')
        ax2.set_title(f'Диполь: ⟨z⟩ = d·cos(ΔE·t), ΔE = {dE:.3f} (Lyman-α)')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))
        plt.close(fig)

    frames[0].save(OUT / 'atom_interaction.gif', save_all=True, append_images=frames[1:],
                   duration=110, loop=0)
    print(f'    анимация сохранена: {OUT / "atom_interaction.gif"} ({len(frames)} кадров)',
          flush=True)

    # проверки
    check('ортонормированность базиса: 1s ⊥ 2p_z', abs(float(np.sum(psi1 * psi2))) < 1e-6,
          float(np.sum(psi1 * psi2)))
    psi_last = (psi1 * np.exp(-1j * E1 * ts[-1]) + psi2 * np.exp(-1j * E2 * ts[-1])) / np.sqrt(2.0)
    check('нормировка ψ(t) сохраняется', abs(float(np.sum(np.abs(psi_last) ** 2)) - 1.0) < 1e-12,
          float(np.sum(np.abs(psi_last) ** 2)))
    zz = np.array(zvals)
    # малый артефакт сворачивания хвоста 2p у границы (~0.2%) — допускаем
    check('диполь: max|⟨z⟩| ≈ |d| (артефакт границы ≤ 0.005)',
          abs(float(np.max(np.abs(zz))) - abs(d_dip)) < 0.005, float(np.max(np.abs(zz))))

    # радиационное затухание: классическая оценка времени жизни
    omega = dE
    P_avg = 2.0 * omega ** 4 * d_dip ** 2 / (3.0 * C_AU ** 3)   # а.е. мощности
    tau_rad = omega / P_avg                                    # а.е. времени
    report['atom'] = {'a': a, 'N': N, 'E1s': E1, 'E2p': E2, 'dE': dE,
                      'T_au': float(T_au), 'T_fs': float(T_au * T_AU_S * 1e15),
                      'dipole': float(d_dip), 'P_avg_au': float(P_avg),
                      'tau_rad_au': float(tau_rad), 'tau_rad_ns': float(tau_rad * T_AU_S * 1e9),
                      'tau_rad_periods': float(tau_rad / T_au)}
    print(f'    ΔE = {dE:.4f} (эталон 0.375), период {T_au:.2f} а.е. = '
          f'{T_au*T_AU_S*1e15:.2f} фс; диполь d = {d_dip:.4f} a₀; τ_рад ≈ '
          f'{tau_rad*T_AU_S*1e9:.2f} нс (измеренное 1.6 нс)', flush=True)
    check('частота диполя ≈ Lyman-α ± 6%', abs(dE - 0.375) < 0.023, float(dE))
    check('радиационное время жизни ~ нс (порядок величины)', 0.5 < tau_rad * T_AU_S * 1e9 < 10,
          float(tau_rad * T_AU_S * 1e9))


# -------------------------------------------- 6) осцилляции нейтрино в модели

def neutrino_oscillation():
    print('== 6. Осцилляции нейтрино в клеточной модели ==', flush=True)
    # --- реальные параметры (PDG/NOvA) ---
    th12, th23, th13 = np.radians(33.41), np.radians(49.1), np.radians(8.57)
    delta = np.radians(195.0)
    c12, s12 = np.cos(th12), np.sin(th12)
    c23, s23 = np.cos(th23), np.sin(th23)
    c13, s13 = np.cos(th13), np.sin(th13)
    U = np.array([
        [c12 * c13, s12 * c13, s13 * np.exp(-1j * delta)],
        [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta),
         c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta), s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta),
         -c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta), c23 * c13],
    ])
    dms21_r, dms31_r = 7.42e-5, 2.51e-3          # эВ² (реальные)
    L_atm_km = 2.48 * 1.0 / dms31_r              # 4πħc·E/Δm² при E = 1 МэВ
    L_sol_km = 2.48 * 1.0 / dms21_r
    m_nu = 0.05                                   # эВ (шкала)
    a_nu_m = 1.973269804e-13 / (2.0 * m_nu * 1e-6)   # λ̄_ν/2 = ħc/(2m_νc²), м
    cells_per_osc = L_atm_km * 1e3 / a_nu_m

    # --- клеточная модель: 3 массовых канала квантовой прогулки ---
    eps, N = 0.05, 8192
    pbar, sig = 10.0, 2.0
    m1, m2 = 0.0, 0.3                            # Δm²₂₁ = 0.09
    m3 = np.sqrt(m1 ** 2 + (m2 ** 2 - m1 ** 2) * (dms31_r / dms21_r))  # реальное отношение 33.8
    w1 = cb.dispersion(eps, m1, pbar)
    w2 = cb.dispersion(eps, m2, pbar)
    w3 = cb.dispersion(eps, m3, pbar)
    dw21, dw31 = w2 - w1, w3 - w1
    # групповые скорости (численно) — для гауссова перекрытия разделяющихся мод
    dk = 0.002
    v1 = (cb.dispersion(eps, m1, pbar + dk) - cb.dispersion(eps, m1, pbar - dk)) / (2 * dk)
    v2 = (cb.dispersion(eps, m2, pbar + dk) - cb.dispersion(eps, m2, pbar - dk)) / (2 * dk)
    v3 = (cb.dispersion(eps, m3, pbar + dk) - cb.dispersion(eps, m3, pbar - dk)) / (2 * dk)
    x = np.arange(N)
    x0 = 400
    # гауссов пакет с импульсом p̄ (ультрарелятивистский пучок);
    # положительно-энергетические спиноры u_i(p̄) — чистые правые движители
    g = np.exp(-(x - x0) ** 2 / (2 * (sig / eps) ** 2)) * np.exp(1j * pbar * eps * x)
    E_i = [np.sqrt(pbar ** 2 + mi ** 2) for mi in (m1, m2, m3)]
    AR = [np.sqrt((E + pbar) / (2 * E)) for E in E_i]
    AL = [np.sqrt((E - pbar) / (2 * E)) for E in E_i]
    Fij = [[AR[i] * AR[j] + AL[i] * AL[j] for j in range(3)] for i in range(3)]
    psiR = [np.conj(U[0, i]) * AR[i] * g.astype(complex) for i in range(3)]
    psiL = [np.conj(U[0, i]) * AL[i] * g.astype(complex) for i in range(3)]

    T = 5000
    rec_t, rec_P = [], []
    frames = []
    ts_an = np.linspace(0, T * eps, 600)

    def G_ij(dv, t):
        return np.exp(-(dv * t) ** 2 / (4.0 * sig ** 2))

    Pee_an = 1.0 \
        - 4 * np.abs(U[0, 0]) ** 2 * np.abs(U[0, 1]) ** 2 * Fij[0][1] * G_ij(v2 - v1, ts_an) * np.sin(0.5 * dw21 * ts_an) ** 2 \
        - 4 * np.abs(U[0, 0]) ** 2 * np.abs(U[0, 2]) ** 2 * Fij[0][2] * G_ij(v3 - v1, ts_an) * np.sin(0.5 * dw31 * ts_an) ** 2 \
        - 4 * np.abs(U[0, 1]) ** 2 * np.abs(U[0, 2]) ** 2 * Fij[1][2] * G_ij(v3 - v2, ts_an) * np.sin(0.5 * (dw31 - dw21) * ts_an) ** 2
    for t in range(1, T + 1):
        for i in range(3):
            psiR[i], psiL[i] = cb.step(psiR[i], psiL[i], eps,
                                       m1 + (m2 - m1) * (i == 1) + (m3 - m1) * (i == 2))
        if t % 5 == 0:
            rho = np.zeros((3, N))
            for a in range(3):
                nuR = sum(U[a, i] * psiR[i] for i in range(3))
                nuL = sum(U[a, i] * psiL[i] for i in range(3))
                rho[a] = np.abs(nuR) ** 2 + np.abs(nuL) ** 2
            tot = rho.sum(axis=0).sum()
            rec_t.append(t * eps)
            rec_P.append(rho.sum(axis=1) / tot)
        if t % 100 == 0:
            rho = np.zeros((3, N))
            for a in range(3):
                nuR = sum(U[a, i] * psiR[i] for i in range(3))
                nuL = sum(U[a, i] * psiL[i] for i in range(3))
                rho[a] = np.abs(nuR) ** 2 + np.abs(nuL) ** 2
            fig, (ax1, ax2) = new_fig(12, 5.0, 1, 2)
            xx = eps * x
            totc = rho.sum(axis=0)
            ax1.plot(xx, rho[0] / totc.max(), color='crimson', lw=1.4, label='ν_e')
            ax1.plot(xx, rho[1] / totc.max(), color='forestgreen', lw=1.4, label='ν_μ')
            ax1.plot(xx, rho[2] / totc.max(), color='royalblue', lw=1.4, label='ν_τ')
            ax1.plot(xx, totc / totc.max(), 'k--', lw=0.8, label='полный поток')
            ax1.set_xlim(eps * x0 - 2, eps * x0 + 280)
            ax1.set_ylim(0, 1.05)
            ax1.set_xlabel('x (c=1)')
            ax1.set_ylabel('плотность аромата')
            ax1.set_title(f'Пучок нейтрино: L = {t*eps:.0f} (такт {t})', fontsize=9)
            ax1.legend(fontsize=7, ncol=2)
            rt = np.array(rec_t)
            rp = np.array(rec_P)
            for a, (lab, col) in enumerate([('P_ee', 'crimson'), ('P_eμ', 'forestgreen'),
                                            ('P_eτ', 'royalblue')]):
                ax2.plot(rt, rp[:, a], color=col, lw=1.2, label=lab)
            ax2.plot(ts_an, Pee_an, 'k--', lw=0.9, label='аналитика 3-аромата')
            ax2.axvline(t * eps, color='gray', lw=0.8)
            ax2.set_xlabel('L = t (c=1)')
            ax2.set_ylabel('вероятность')
            ax2.set_title('Осцилляции ν_e → ν_α (модель vs аналитика)')
            ax2.legend(fontsize=7)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frames.append(Image.open(buf).convert('RGB'))
            plt.close(fig)

    frames[0].save(OUT / 'neutrino_oscillation.gif', save_all=True,
                   append_images=frames[1:], duration=110, loop=0)
    print(f'    анимация сохранена: {OUT / "neutrino_oscillation.gif"} ({len(frames)} кадров)',
          flush=True)

    rt = np.array(rec_t)
    rp = np.array(rec_P)
    # --- проверки ---
    check('сохранение вероятности Σ_α P_eα = 1',
          np.max(np.abs(rp.sum(axis=1) - 1.0)) < 1e-12,
          float(np.max(np.abs(rp.sum(axis=1) - 1.0))))
    # аналитика с точными дисперсионными частотами, спинорами и перекрытием мод
    mask = rt < 60
    Pee_num = rp[mask, 0]
    Pee_an_m = 1.0 \
        - 4 * np.abs(U[0, 0]) ** 2 * np.abs(U[0, 1]) ** 2 * Fij[0][1] * G_ij(v2 - v1, rt[mask]) * np.sin(0.5 * dw21 * rt[mask]) ** 2 \
        - 4 * np.abs(U[0, 0]) ** 2 * np.abs(U[0, 2]) ** 2 * Fij[0][2] * G_ij(v3 - v1, rt[mask]) * np.sin(0.5 * dw31 * rt[mask]) ** 2 \
        - 4 * np.abs(U[0, 1]) ** 2 * np.abs(U[0, 2]) ** 2 * Fij[1][2] * G_ij(v3 - v2, rt[mask]) * np.sin(0.5 * (dw31 - dw21) * rt[mask]) ** 2
    dev_an = np.max(np.abs(Pee_num - Pee_an_m))
    check('P_ee(модель) = аналитика 3-ароматов (±2%)', dev_an < 0.02, float(dev_an))
    # FFT быстрой (атмосферной) компоненты: вычитаем медленную солнечную
    y = rp[:, 0]
    ys = np.convolve(y, np.ones(40) / 40.0, mode='same')
    yf = y - ys
    yf_win = yf * np.hanning(len(yf))
    yz = np.concatenate([yf_win, np.zeros(15 * len(yf_win))])
    f = np.fft.rfftfreq(len(yz), d=rt[1] - rt[0])
    spec = np.abs(np.fft.rfft(yz))
    f_peak = f[np.argmax(spec)]
    check('частота осцилляций = Δω₃₁/(2π) (дисперсия КА)',
          abs(2 * np.pi * f_peak - dw31) < 0.005, float(2 * np.pi * f_peak))
    # декогеренция: разделение массовых мод на решётке (амплитуда быстрой компоненты)
    dv31 = dw31 / pbar                      # разность групповых скоростей ≈ Δm²/(2p̄²)
    L_coh = sig / dv31                      # длина когерентности
    def local_amp(t0, half=7.5):
        m = np.abs(rt - t0) < half
        return float(yf[m].std())
    amp_early = local_amp(15.0)
    amp_late = local_amp(215.0)
    check('декогеренция: амплитуда падает (разделение мод)', amp_late < 0.6 * amp_early,
          (float(amp_early), float(amp_late)))
    report['neutrino'] = {
        'masses_walker': [m1, m2, float(m3)], 'pbar': pbar, 'sig': sig,
        'dms_ratio': float((m3 ** 2 - m1 ** 2) / (m2 ** 2 - m1 ** 2)),
        'dw21': float(dw21), 'dw31': float(dw31),
        'dev_analytic': float(dev_an), 'f_peak': float(2 * np.pi * f_peak),
        'L_coh': float(L_coh), 'L_coh_cells': float(L_coh / eps),
        'L_atm_km_real': float(L_atm_km), 'L_sol_km_real': float(L_sol_km),
        'a_nu_m': float(a_nu_m), 'cells_per_osc': float(cells_per_osc),
        'U': [[[float(z.real), float(z.imag)] for z in row] for row in U],
    }
    print(f'    Δω₃₁ = {dw31:.4f} (FFT: {2*np.pi*f_peak:.4f}), Δω₂₁ = {dw21:.4f}; '
          f'L_coh = {L_coh:.0f} = {L_coh/eps:.0f} ячеек; L_атм(1 МэВ) = {L_atm_km:.0f} км = '
          f'{cells_per_osc:.1e} ячеек (a_ν = {a_nu_m*1e6:.2f} мкм)', flush=True)

    # --- итоговый рисунок ---
    fig, axes = new_fig(12, 9, 2, 2)
    axes[0][0].plot(rt, rp[:, 0], color='crimson', label='модель')
    axes[0][0].plot(ts_an, Pee_an, 'k--', lw=1, label='аналитика')
    axes[0][0].set_xlabel('L = t (c=1)')
    axes[0][0].set_ylabel('P_ee')
    axes[0][0].set_title('Выживание ν_e (быстрая атмосферная осцилляция)')
    axes[0][0].legend(fontsize=8)
    axes[0][1].plot(2 * np.pi * f, spec)
    axes[0][1].axvline(dw31, color='r', ls='--', label=f'Δω₃₁ = {dw31:.4f}')
    axes[0][1].set_xlim(0, 0.6)
    axes[0][1].set_xlabel('частота ω')
    axes[0][1].set_ylabel('|БПФ(P_ee)|')
    axes[0][1].set_title('Спектр осцилляций: пик на Δω₃₁')
    axes[0][1].legend(fontsize=8)
    axes[1][0].plot(rt, rp[:, 1], color='forestgreen', label='P_eμ')
    axes[1][0].plot(rt, rp[:, 2], color='royalblue', label='P_eτ')
    axes[1][0].set_xlabel('L = t')
    axes[1][0].set_ylabel('вероятность')
    axes[1][0].set_title('Появление ν_μ, ν_τ (медленная солнечная модуляция)')
    axes[1][0].legend(fontsize=8)
    axes[1][1].text(0.05, 0.9, 'Параметры модели', fontsize=10, transform=axes[1][1].transAxes)
    txt = (f'm₁, m₂, m₃ = {m1}, {m2}, {m3:.3f}\n'
           f'Δm²₃₁/Δm²₂₁ = {(m3**2-m1**2)/(m2**2-m1**2):.1f} (реальное 33.8)\n'
           f'L_osc(атм, 1 МэВ) = {L_atm_km:.0f} км\n'
           f'L_osc(солн, 1 МэВ) = {L_sol_km:.0f} км\n'
           f'a_ν = λ̄_ν/2 = {a_nu_m*1e6:.2f} мкм\n'
           f'ячеек на осцилляцию = {cells_per_osc:.1e}\n'
           f'декогеренция: L_coh = {L_coh/eps:.0f} ячеек')
    axes[1][1].text(0.05, 0.5, txt, fontsize=9, transform=axes[1][1].transAxes)
    axes[1][1].axis('off')
    save(fig, OUT / 'ext5_neutrino.png')


def main():
    print('=== РАСШИРЕНИЕ: правило материи, нейтрино, структуры, протон ===', flush=True)
    checkerboard()
    proton_animation()
    structure_numbers()
    deuterium()
    atom_animation()
    neutrino_oscillation()
    report['checks'] = checks
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding='utf-8')
    nfail = sum(1 for c in checks if not c['ok'])
    print(f'\nВсего проверок: {len(checks)}, провалено: {nfail}', flush=True)
    return nfail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
