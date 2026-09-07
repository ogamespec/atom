# -*- coding: utf-8 -*-
"""Визуальный словарь понятий исследования: новые анимации и схемы.

1) ca_rule.gif           — клеточный автомат, бит, CLK, локальность (правило 90);
2) imagtime_collapse.gif — мнимое время: облако электрона стягивается в 1s;
3) checkerboard_walk.gif — квантовая прогулка: шахматная доска, спин, Zitter;
4) scales.png            — два масштаба: планковский субстрат vs эффективные биты;
5) torus.png             — тор, периодические образы, постоянная Маделунга;
6) six_pi5.png           — 6π⁵ ≈ 6·17·18: геометрия «6 лучей × 17 колец × 18 ячеек».

Запуск:  python scripts/run_glossary.py
Вывод:   results/glossary/*.png, *.gif, report.json
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import checkerboard as cb
from src.lattice import free_space_potential, grid_coords, energy_expect
from src.plotting import new_fig, save

OUT = ROOT / 'results' / 'glossary'
OUT.mkdir(parents=True, exist_ok=True)

report = {}
checks = []


def check(name, ok, value):
    checks.append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + '   ->   ' + str(value), flush=True)


def gif(frames, path, duration=110):
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    print(f'    {path.name} ({len(frames)} кадров)', flush=True)


# -------------------------------------------------- 1) клеточный автомат (правило 90)

def ca_rule():
    print('== 1. Клеточный автомат: бит, CLK, локальность ==', flush=True)
    N, T = 240, 80
    s = np.zeros(N, dtype=int)
    s[N // 2] = 1
    history = [s.copy()]
    for t in range(T):
        s = (np.roll(s, -1) ^ np.roll(s, 1)).astype(int)     # правило 90
        history.append(s.copy())
    H = np.array(history)
    frames = []
    for t in range(2, T + 1, 2):
        fig, ax = new_fig(8, 5.5)
        ax.imshow(H[:t], cmap='binary', aspect='auto', origin='upper')
        ax.set_xlabel('ячейки (биты)')
        ax.set_ylabel('такты CLK')
        ax.set_title(f'Правило 90: s′(i) = s(i−1) XOR s(i+1) — локальное обновление, такт {t}')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))
        plt.close(fig)
    gif(frames, OUT / 'ca_rule.gif')
    # проверка: при t = 2^k из одного бита — ровно 2 живые ячейки (треугольник Серпинского)
    live = int(H[64].sum())
    check('правило 90: треугольник Серпинского (2 бита на такте 64)', live == 2, live)


# ----------------------------------------------------- 2) мнимое время: стягивание в 1s

def imagtime_collapse():
    print('== 2. Мнимое время: облако электрона стягивается в 1s ==', flush=True)
    N, a = 64, 0.3
    V = free_space_potential(a, N)
    X, Y, Z = grid_coords(a, N)
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    psi = np.exp(-(R - 6.0) ** 2 / (2 * 0.5 ** 2))       # широкое «облако»-оболочка
    psi /= np.sqrt(np.sum(psi ** 2))
    dbeta, steps = 0.004, 5000
    frames, es = [], []

    def lap(f):
        s = np.zeros_like(f)
        for ax in range(3):
            s += np.roll(f, 1, ax) + np.roll(f, -1, ax)
        return (s - 6.0 * f) / a ** 2

    for t in range(1, steps + 1):
        psi = psi - dbeta * (-0.5 * lap(psi) + V * psi)
        psi /= np.sqrt(np.sum(psi * psi))
        if t % 125 == 0:
            E = energy_expect(psi, V, a)
            es.append((t, E))
            fig, (ax1, ax2) = new_fig(11, 5.0, 1, 2)
            ax1.imshow(psi[N // 2, :, :].T ** 2, origin='lower', cmap='inferno',
                       extent=[-N * a / 2, N * a / 2, -N * a / 2, N * a / 2])
            ax1.plot(0, 0, 'o', color='white', ms=7)
            ax1.plot(0, 0, 'o', color='crimson', ms=4)
            ax1.set_xlabel('y, a₀')
            ax1.set_ylabel('z, a₀')
            ax1.set_title(f'|ψ|²: шаг мнимого времени {t}', fontsize=9)
            ts = [s for s, _ in es]
            ee = [e for _, e in es]
            ax2.plot(ts, ee, 'o-', color='tab:blue')
            ax2.axhline(-0.5, color='k', ls='--', lw=1, label='E₁ₛ = −0.5 (эталон)')
            ax2.set_xlabel('шаг')
            ax2.set_ylabel('E = ⟨ψ|H|ψ⟩')
            ax2.set_title('Спуск к основному состоянию')
            ax2.legend(fontsize=8)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frames.append(Image.open(buf).convert('RGB'))
            plt.close(fig)
    gif(frames, OUT / 'imagtime_collapse.gif')
    E_final = energy_expect(psi, V, a)
    check('мнимое время: сходимость к 1s (E ≈ −0.53 при a=0.3)',
          abs(E_final + 0.533) < 0.01, float(E_final))
    report['imagtime'] = {'E_final': float(E_final), 'steps': steps}


# ----------------------------------------------- 3) квантовая прогулка: спин и Zitter

def checkerboard_walk():
    print('== 3. Квантовая прогулка: шахматная доска, спин, Zitter ==', flush=True)
    eps, N, m, pbar, sig = 0.05, 2048, 1.0, 10.0, 2.0
    x = np.arange(N)
    x0 = 300
    E = np.sqrt(pbar ** 2 + m ** 2)
    AR = np.sqrt((E + pbar) / (2 * E))
    AL = np.sqrt((E - pbar) / (2 * E))
    g = np.exp(-(x - x0) ** 2 / (2 * (sig / eps) ** 2)) * np.exp(1j * pbar * eps * x)
    psiR = AR * g
    psiL = AL * g
    T = 2000
    frames = []
    history = np.zeros((40, N))
    for t in range(1, T + 1):
        psiR, psiL = cb.step(psiR, psiL, eps, m)
        if t % 50 == 0:
            k = t // 50 - 1
            history[k] = np.abs(psiR) ** 2 + np.abs(psiL) ** 2
            fig, (ax1, ax2) = new_fig(11, 5.0, 1, 2)
            xx = eps * x
            ax1.plot(xx, np.abs(psiR) ** 2, color='crimson', lw=1.3, label='|ψR|² (спин →)')
            ax1.plot(xx, np.abs(psiL) ** 2, color='royalblue', lw=1.3, label='|ψL|² (спин ←)')
            ax1.set_xlim(eps * x0, eps * x0 + 100)
            ax1.set_xlabel('x')
            ax1.set_ylabel('плотность')
            ax1.set_title(f'Пакет: двухкомпонентный спинор, такт {t}', fontsize=9)
            ax1.legend(fontsize=8)
            ax2.imshow(history, cmap='inferno', aspect='auto', origin='upper',
                       extent=[eps * x0, eps * (x0 + 100), T, 0])
            ax2.set_xlabel('x')
            ax2.set_ylabel('такт')
            ax2.set_title('Пространственно-временная диаграмма (траектория ~ c)')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frames.append(Image.open(buf).convert('RGB'))
            plt.close(fig)
    gif(frames, OUT / 'checkerboard_walk.gif')
    # проверка: групповая скорость ≈ 1 − m²/(2p̄²)
    xm = eps * (np.sum(x * history[20]) / np.sum(history[20]) -
                np.sum(x * history[5]) / np.sum(history[5]))
    dt = (20 - 5) * 50 * eps
    v = xm / dt
    v_theory = 1.0 - m ** 2 / (2 * pbar ** 2)
    check('квантовая прогулка: групповая скорость = 1 − m²/2p̄²', abs(v - v_theory) < 0.02,
          float(v))
    report['walk'] = {'v': float(v), 'v_theory': float(v_theory)}


# --------------------------------------------------------- 4) два масштаба (схема)

def scales():
    print('== 4. Схема: два масштаба информации ==', flush=True)
    fig, axes = new_fig(14, 4.6, 1, 3)
    # планковский субстрат
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((1200, 2)) * 0.28
    axes[0].scatter(pts[:, 0], pts[:, 1], s=2, color='tab:gray')
    axes[0].add_patch(plt.Circle((0, 0), 1.0, fill=False, color='crimson', lw=2))
    axes[0].text(0, -1.35, 'протон = 5.9·10⁵⁹ планковских ячеек', ha='center', fontsize=8)
    axes[0].set_title('Планковский субстрат (a = l_P)')
    axes[0].set_xlim(-1.5, 1.5)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].axis('equal')
    axes[0].axis('off')
    # эффективная структура: 6 лучей × 17 колец
    for rr in range(1, 18):
        axes[1].add_patch(plt.Circle((0, 0), rr / 18.0, fill=False,
                                     color='tab:blue', lw=0.5, alpha=0.7))
    for a6 in range(6):
        th = a6 * np.pi / 3
        axes[1].plot([0, np.cos(th)], [0, np.sin(th)], color='tab:blue', lw=1.2)
    for a6 in range(6):
        th = a6 * np.pi / 3 + np.pi / 6
        axes[1].plot(np.cos(th) * 0.97, np.sin(th) * 0.97, 'o', ms=4, color='crimson')
    axes[1].text(0, -1.3, '1836 = 6 лучей × 17 колец × 18 ячеек', ha='center', fontsize=8)
    axes[1].set_title('Эффективная структура (a = 0.11 фм)')
    axes[1].axis('equal')
    axes[1].axis('off')
    # атом
    axes[2].add_patch(plt.Circle((0, 0), 0.15, color='crimson'))
    cloud = np.exp(-np.linspace(0, 2.5, 60) ** 2)
    for i in range(4):
        axes[2].add_patch(plt.Circle((0, 0), 0.3 + 0.55 * i, fill=False,
                                     color='tab:blue', alpha=0.25 + 0.2 * i, lw=1.5))
    axes[2].text(0, -1.7, 'атом: электронное облако (1 бит)\nвокруг протона (1836 бит)',
                 ha='center', fontsize=8)
    axes[2].set_title('Атомный масштаб (a ≈ 0.2 a₀)')
    axes[2].set_xlim(-2.5, 2.5)
    axes[2].set_ylim(-2.5, 2.5)
    axes[2].axis('equal')
    axes[2].axis('off')
    save(fig, OUT / 'scales.png')
    check('схема двух масштабов сохранена', True, 1)


# ----------------------------------------------- 5) тор и постоянная Маделунга (схема)

def torus():
    print('== 5. Схема: тор, периодические образы, Маделунг ==', flush=True)
    fig, ax = new_fig(9, 8)
    L = 3.0
    # образы источника (9 ячеек)
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            cx, cy = ix * L, iy * L
            col = 'crimson' if ix == 0 and iy == 0 else 'tab:gray'
            ax.plot(cx, cy, 'o', color=col, ms=7)
    # фундаментальная ячейка
    ax.add_patch(plt.Rectangle((-L / 2, -L / 2), L, L, fill=False, color='k', lw=1.2))
    # световые конусы
    for rr in (0.8, 1.6):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, color='tab:blue',
                                lw=1, ls='--' if rr == 1.6 else '-'))
    ax.annotate('фронт поля (c·t)', xy=(0.62, 0.62), xytext=(1.55, 1.3),
                arrowprops=dict(arrowstyle='->', color='tab:blue'), fontsize=8)
    ax.annotate('образы источника\n(ещё не дошли)', xy=(L, 0), xytext=(1.3, -1.6),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=8)
    ax.text(-L / 2, -L / 2 - 0.5, 'периодическая ячейка (тор)', fontsize=8, ha='center')
    ax.text(0, 2.4, 'G_per(0) = G(0) − ξ/(N·a),  ξ = 2.8373 (Маделунг)',
            ha='center', fontsize=9)
    ax.set_xlim(-2.2, 4.6)
    ax.set_ylim(-2.4, 2.8)
    ax.axis('equal')
    ax.axis('off')
    save(fig, OUT / 'torus.png')
    check('схема тора сохранена', True, 1)


# ------------------------------------------------------ 6) 6π⁵ (схема)

def six_pi5():
    print('== 6. Схема: 6π⁵ ≈ 6·17·18 ==', flush=True)
    fig, (ax1, ax2) = new_fig(12, 5.4, 1, 2)
    # колесо: 6 лучей × 17 колец
    for rr in range(1, 18):
        ax1.add_patch(plt.Circle((0, 0), rr / 18.0, fill=False,
                                 color='tab:blue', lw=0.5, alpha=0.7))
    for a6 in range(6):
        th = a6 * np.pi / 3
        ax1.plot([0, np.cos(th)], [0, np.sin(th)], color='tab:blue', lw=1.4)
    ax1.text(0, -1.25, '1836 = 6·17·18 = 3·(4·9·17)\n(6 лучей, 17 колец, 18 ячеек/луч)',
             ha='center', fontsize=9)
    ax1.set_title('Геометрия числа 1836')
    ax1.axis('equal')
    ax1.axis('off')
    # числовая ось
    vals = [1836.0, 1836.1181, 1836.1527]
    labels = ['1836 = 6·17·18', '6π⁵ = 1836.118', 'm_p/m_e = 1836.1527']
    for v, lab in zip(vals, labels):
        ax2.axvline(v, color='crimson' if v == 1836.0 else 'tab:blue', lw=1.2)
        ax2.text(v, 0.9 if v != 1836.0 else 0.75, lab, rotation=90, fontsize=8,
                 va='bottom', ha='center')
    ax2.text(1836.135, 0.4, '0.035 m_e', fontsize=8, ha='center')
    ax2.text(1836.005, 0.4, 'π⁵ ≈ 17·18 (6·10⁻⁵)', fontsize=8, ha='center', rotation=90)
    ax2.set_xlim(1835.9, 1836.3)
    ax2.set_ylim(0, 1)
    ax2.set_title('Формула Ленца: 6π⁵ ≈ 1836')
    ax2.set_yticks([])
    save(fig, OUT / 'six_pi5.png')
    check('схема 6π⁵ сохранена', True, 1)


def main():
    print('=== ВИЗУАЛЬНЫЙ СЛОВАРЬ ПОНЯТИЙ ===', flush=True)
    ca_rule()
    imagtime_collapse()
    checkerboard_walk()
    scales()
    torus()
    six_pi5()
    report['checks'] = checks
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding='utf-8')
    nfail = sum(1 for c in checks if not c['ok'])
    print(f'\nВсего проверок: {len(checks)}, провалено: {nfail}', flush=True)
    return nfail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
