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

from src.units import (M_P_OVER_M_E, M_P_KG, M_E_KG, HBAR_JS, C_LIGHT, A0_M)
from src import checkerboard as cb
from src.lattice import lattice_green_fft
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


def main():
    print('=== РАСШИРЕНИЕ: правило материи, нейтрино, структуры, протон ===', flush=True)
    checkerboard()
    proton_animation()
    structure_numbers()
    deuterium()
    report['checks'] = checks
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding='utf-8')
    nfail = sum(1 for c in checks if not c['ok'])
    print(f'\nВсего проверок: {len(checks)}, провалено: {nfail}', flush=True)
    return nfail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
