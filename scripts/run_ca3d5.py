# -*- coding: utf-8 -*-
"""Часть 5: заряженный битовый газ — кулоновское поле, доменная сетка, блобы."""
from __future__ import annotations
import json, sys, time, zlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_coulomb import (density, poisson, run, seed_dipole, seed_neutral,
                              seed_positive, seed_checker)
from src.ca3d_render import render_voxels, slice_mosaic
from src.ca3d_shape import shape_metrics

OUT = ROOT / 'results' / 'ca3d5'
OUT.mkdir(parents=True, exist_ok=True)
L, N = 97, 16_000_000
R = {'runs': [], 'checks': []}


def check(name, ok, value):
    R['checks'].append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + f'   ->   {value}', flush=True)


def run_case(seed, T, beta=1.0, K=None, exchange=False, tag=None):
    p, n, phi, hist = run(L, T, N, seed, beta, K, exchange)
    d = np.log10(1 + p + n).astype(np.float32)
    sign = np.sign(p - n)
    walls = np.zeros((L, L, L), bool)
    for ax in range(3):
        for sh in (1, -1):
            walls |= sign != np.roll(sign, sh, axis=ax)
    m = shape_metrics((p + n) > 0)
    mw = shape_metrics(walls)
    rec = {'tag': tag or f'{seed}-T{T}', 'seed': seed, 'T': T, 'beta': beta,
           'K': K, 'exchange': exchange,
           'P': int(p.sum()), 'N': int(n.sum()), 'total': int(p.sum() + n.sum()),
           'cells': int(((p + n) > 0).sum()),
           'occ': float(((p + n) > 0).mean()),
           'max_log': float(d.max()), 'blob': m['blob'], 'network': m['network'],
           'wall_frac': float(walls.mean()), 'wall_network': mw['network'],
           'wall_clusters': mw['n_clusters'],
           'compression': float(len(zlib.compress(np.round(d * 10).astype(np.uint8).tobytes(), 6)) / d.size),
           'image': None, 'slices': None}
    R['runs'].append(rec)
    print(f"  {rec['tag']:12s} P={rec['P']} N={rec['N']} cells={rec['cells']} "
          f"blob={rec['blob']:.2f} net={rec['network']:.2f} walls={rec['wall_frac']:.3f} "
          f"wallnet={rec['wall_network']:.2f}", flush=True)
    return p, n, phi, d, walls, rec


def main():
    t0 = time.time()
    # 1) Доменная сетка: шахматный заряд
    for T in (6, 10, 16, 24):
        p, n, phi, d, walls, rec = run_case('checker', T, tag=f'checker-T{T}')
        if T in (10, 16):
            img = render_voxels(d ** 0.5, size=640, canvas_L=L,
                                front_radius=min(T + 3, L / 2), front_time=T,
                                inset=True, cmap='inferno', value_norm=3,
                                legend_label='log10 бит',
                                label=f'QUADRUPOLE-GERM: T={T}')
            fn = f'p5_checker_T{T}.png'
            Image.fromarray(img).save(OUT / fn)
            rec['image'] = fn
            if T == 10:
                signT = np.sign(p - n)
                slice_mosaic(signT.astype(np.int8), OUT / 'p5_checker_slices.png',
                             n=6, cmap='coolwarm',
                             title='Зарядовая сетка: срезы (+1/−1), T=10',
                             vmin=-1, vmax=1)
                rec['slices'] = 'p5_checker_slices.png'
    # 2) Блобы: диполь и квадруполь
    for seed, T in [('dipole', 16), ('quadrupole', 16)]:
        p, n, phi, d, walls, rec = run_case(seed, T, tag=f'{seed}-T{T}')
        img = render_voxels(d ** 0.5, size=640, canvas_L=L,
                            front_radius=min(T + 3, L / 2), front_time=T,
                            inset=True, cmap='inferno', value_norm=3,
                            legend_label='log10 бит', label=f'{seed.upper()}: T={T}')
        fn = f'p5_{seed}_T{T}.png'
        Image.fromarray(img).save(OUT / fn)
        rec['image'] = fn
    # 3) Кулоновское поле одного заряда: φ ~ 1/r, F ~ 1/r²
    p, n = seed_positive(L, N)
    rho = (p.astype(np.float32) - n.astype(np.float32)) / N
    phi = poisson(rho)
    c = (L - 1) / 2
    rr = np.sqrt((np.indices(phi.shape)[0] - c) ** 2 +
                 (np.indices(phi.shape)[1] - c) ** 2 +
                 (np.indices(phi.shape)[2] - c) ** 2)
    bins = np.arange(2, 25)
    prof = []
    for r0, r1 in zip(bins[:-1], bins[1:]):
        m = (rr >= r0) & (rr < r1)
        prof.append(float(phi[m].mean()))
    prof = np.array(prof)
    rmid = 0.5 * (bins[:-1] + bins[1:])
    sel = (rmid >= 3) & (rmid <= 20) & (np.abs(prof) > 1e-12)
    slope_phi = float(np.polyfit(np.log(rmid[sel]), np.log(np.abs(prof[sel])), 1)[0])
    # сила = -dφ/dr
    force = -np.gradient(prof, rmid)
    sel2 = (rmid >= 3) & (rmid <= 20) & (np.abs(force) > 1e-12)
    slope_f = float(np.polyfit(np.log(rmid[sel2]), np.log(np.abs(force[sel2])), 1)[0])
    print(f'  φ slope={slope_phi:.2f}  F slope={slope_f:.2f}', flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].loglog(rmid, np.abs(prof), 'o-', color='#3182ce')
    ax[0].loglog(rmid[sel], np.exp(np.polyval(np.polyfit(np.log(rmid[sel]), np.log(np.abs(prof[sel])), 1), np.log(rmid[sel]))),
                 'k--', label=f'наклон {slope_phi:.2f}')
    ax[0].set_xlabel('r, ячеек'); ax[0].set_ylabel('|φ|'); ax[0].set_title('Потенциал: φ ~ 1/r')
    ax[0].legend()
    ax[1].loglog(rmid, np.abs(force), 's-', color='#dd6b20')
    ax[1].loglog(rmid[sel2], np.exp(np.polyval(np.polyfit(np.log(rmid[sel2]), np.log(np.abs(force[sel2])), 1), np.log(rmid[sel2]))),
                 'k--', label=f'наклон {slope_f:.2f}')
    ax[1].set_xlabel('r, ячеек'); ax[1].set_ylabel('|F|'); ax[1].set_title('Сила: F ~ 1/r²')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / 'p5_coulomb.png', dpi=150)
    plt.close(fig)
    R['coulomb'] = {'phi_slope': slope_phi, 'force_slope': slope_f,
                    'r': rmid.tolist(), 'phi': prof.tolist(), 'force': force.tolist()}

    # проверки
    total_ok = all(r['total'] == N for r in R['runs'])
    check('во всех прогонах P+N = 16 000 000', total_ok,
          sorted({r['total'] for r in R['runs']}))
    check('потенциал ~ 1/r (наклон −1 ± 0.2)', abs(slope_phi + 1) < 0.2, round(slope_phi, 2))
    check('сила ~ 1/r² (наклон −2 ± 0.35)', abs(slope_f + 2) < 0.35, round(slope_f, 2))
    check('доменная сетка существует (wall network > 0.4)',
          max(r['wall_network'] for r in R['runs']) > 0.4,
          round(max(r['wall_network'] for r in R['runs']), 2))
    check('есть блобы', max(r['blob'] for r in R['runs']) > 0.8,
          round(max(r['blob'] for r in R['runs']), 2))
    check('картинки созданы',
          all((OUT / r['image']).exists() for r in R['runs'] if r['image']), True)
    R['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                 'python': sys.version.split()[0], 'numpy': np.__version__,
                 'run_s': round(time.time() - t0, 1)}
    (OUT / 'report.json').write_text(json.dumps(R, ensure_ascii=False, indent=1),
                                     encoding='utf-8')
    print(f"== report.json: {sum(c['ok'] for c in R['checks'])}/{len(R['checks'])} PASS, "
          f"{time.time()-t0:.1f} с ==", flush=True)


if __name__ == '__main__':
    main()
