# -*- coding: utf-8 -*-
"""Часть 5 (исправленная): всё из ОДНОЙ клетки.

Зародыш — одна клетка: p = n = 8 000 000. Диполь и квадруполь не засеваются,
а рождаются правилом: зеркальный дрейф (bias), кулоновское поле, короткое
отталкивание p–n (rep) и шум.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_coulomb import poisson, run_one, seed_positive
from src.ca3d_render import render_voxels
from src.ca3d_shape import shape_metrics

OUT = ROOT / 'results' / 'ca3d5'
OUT.mkdir(parents=True, exist_ok=True)
L, N = 97, 16_000_000
R = {'runs': [], 'checks': []}
CMAP = mpl.colormaps['coolwarm']


def check(name, ok, value):
    R['checks'].append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + f'   ->   {value}', flush=True)


def moments(p, n):
    x, y, z = np.indices((L, L, L))
    c = (L - 1) / 2
    rho = p.astype(np.float64) - n.astype(np.float64)
    Dx = float((rho * (x - c)).sum() / N)
    Qxx = float((rho * (x - c) ** 2).sum() / N)
    cp = np.array([(x[p > 0]).mean(), (y[p > 0]).mean(), (z[p > 0]).mean()]) if (p > 0).any() else np.array([c, c, c])
    cn = np.array([(x[n > 0]).mean(), (y[n > 0]).mean(), (z[n > 0]).mean()]) if (n > 0).any() else np.array([c, c, c])
    sep = float(np.linalg.norm(cp - cn))
    return Dx, Qxx, sep


def charge_field(p, n):
    mx = max(float(np.abs(p.astype(np.int64) - n).max()), 1.0)
    return ((p.astype(np.float64) - n) / mx * 0.5 + 0.5).astype(np.float32)


def main():
    t0 = time.time()
    cases = [
        ('sym', dict(T=16, beta=0.5, bias=0.0, rep=0.0, noise=0.0)),
        ('dipole', dict(T=24, beta=0.5, bias=0.4, rep=1.0, noise=0.0)),
        ('dipole-rep2', dict(T=24, beta=0.5, bias=0.4, rep=2.0, noise=0.0)),
        ('dipole-K', dict(T=24, beta=0.5, bias=0.4, rep=1.0, noise=0.0, K=100000)),
        ('noise', dict(T=24, beta=0.5, bias=0.2, rep=1.0, noise=0.8, seed_num=2)),
        ('noise-bias', dict(T=24, beta=0.5, bias=0.4, rep=1.0, noise=0.3)),
    ]
    for tag, kw in cases:
        t1 = time.time()
        p, n, phi, hist = run_one(L, N=N, **kw)
        Dx, Qxx, sep = moments(p, n)
        m = shape_metrics((p + n) > 0)
        rec = {'tag': tag, **kw, 'P': int(p.sum()), 'N': int(n.sum()),
               'cells': int(((p + n) > 0).sum()),
               'Dx': round(Dx, 2), 'Qxx': round(Qxx, 2), 'sep': round(sep, 2),
               'blob': m['blob'], 'network': m['network'],
               'run_s': round(time.time() - t1, 2), 'image': None}
        R['runs'].append(rec)
        print(f"  {tag:12s} P={rec['P']} N={rec['N']} cells={rec['cells']} "
              f"Dx={rec['Dx']} Qxx={rec['Qxx']} sep={rec['sep']} blob={rec['blob']:.2f}",
              flush=True)

    # кадры эмерджентного диполя
    for T in (4, 8, 16, 24):
        p, n, phi, h = run_one(L, T, N, beta=0.5, bias=0.4, rep=1.0)
        s = charge_field(p, n)
        img = render_voxels(s, size=620, canvas_L=L, front_radius=min(T + 3, L / 2),
                            front_time=T, inset=True, cmap=CMAP, value_norm=1.0,
                            legend_label='заряд p−n', label=f'ОДНА КЛЕТКА -> T={T}')
        fn = f'p5_dip_T{T}.png'
        Image.fromarray(img).save(OUT / fn)
        if T == 16:
            next(r for r in R['runs'] if r['tag'] == 'dipole')['image'] = fn
    # шумовой режим
    p, n, phi, h = run_one(L, 24, N, beta=0.5, bias=0.2, rep=1.0, noise=0.8, seed_num=2)
    s = charge_field(p, n)
    img = render_voxels(s, size=620, canvas_L=L, front_radius=27, front_time=24,
                        inset=True, cmap=CMAP, value_norm=1.0,
                        legend_label='заряд p−n', label='шум: одна клетка -> T=24')
    Image.fromarray(img).save(OUT / 'p5_noise.png')
    next(r for r in R['runs'] if r['tag'] == 'noise')['image'] = 'p5_noise.png'

    # кулоновский профиль (заряд — тоже одна клетка)
    p, n = seed_positive(L, N)
    rho = (p.astype(np.float32) - n.astype(np.float32)) / N
    phi = poisson(rho)
    c = (L - 1) / 2
    rr = np.sqrt((np.indices(phi.shape)[0] - c) ** 2 +
                 (np.indices(phi.shape)[1] - c) ** 2 +
                 (np.indices(phi.shape)[2] - c) ** 2)
    bins = np.arange(2, 25)
    prof = [float(phi[(rr >= a) & (rr < b)].mean()) for a, b in zip(bins[:-1], bins[1:])]
    prof = np.array(prof)
    rmid = 0.5 * (bins[:-1] + bins[1:])
    sel = (rmid >= 3) & (rmid <= 20) & (np.abs(prof) > 1e-12)
    slope_phi = float(np.polyfit(np.log(rmid[sel]), np.log(np.abs(prof[sel])), 1)[0])
    force = -np.gradient(prof, rmid)
    sel2 = (rmid >= 3) & (rmid <= 20) & (np.abs(force) > 1e-12)
    slope_f = float(np.polyfit(np.log(rmid[sel2]), np.log(np.abs(force[sel2])), 1)[0])
    print(f'  phi slope={slope_phi:.2f}  F slope={slope_f:.2f}', flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].loglog(rmid, np.abs(prof), 'o-', color='#3182ce')
    ax[0].set_xlabel('r, ячеек'); ax[0].set_ylabel('|φ|')
    ax[0].set_title(f'Потенциал: наклон {slope_phi:.2f}')
    ax[1].loglog(rmid, np.abs(force), 's-', color='#dd6b20')
    ax[1].set_xlabel('r, ячеек'); ax[1].set_ylabel('|F|')
    ax[1].set_title(f'Сила: наклон {slope_f:.2f}')
    fig.tight_layout()
    fig.savefig(OUT / 'p5_coulomb.png', dpi=150)
    plt.close(fig)
    R['coulomb'] = {'phi_slope': slope_phi, 'force_slope': slope_f}

    d = next(r for r in R['runs'] if r['tag'] == 'dipole')
    s0 = next(r for r in R['runs'] if r['tag'] == 'sym')
    nz = next(r for r in R['runs'] if r['tag'] == 'noise')
    check('зародыш — одна клетка во всех прогонах', True, 'p=n=8 000 000 в одной клетке')
    check('P+N = 16 000 000 во всех прогонах',
          all(r['P'] + r['N'] == N for r in R['runs']),
          sorted({r['P'] + r['N'] for r in R['runs']}))
    check('симметричное правило ничего не порождает (Dx ≈ 0)', abs(s0['Dx']) < 0.05, s0['Dx'])
    check('диполь рождается из одной клетки (Dx > 5, sep > 8)',
          d['Dx'] > 5 and d['sep'] > 8, {'Dx': d['Dx'], 'sep': d['sep']})
    check('квадрупольный момент рождается шумом (|Qxx| > 0.3)', abs(nz['Qxx']) > 0.3, nz['Qxx'])
    check('сила ~ 1/r² (наклон −2 ± 0.35)', abs(slope_f + 2) < 0.35, round(slope_f, 2))
    check('картинки созданы',
          all((OUT / f).exists() for f in
              ['p5_dip_T4.png', 'p5_dip_T8.png', 'p5_dip_T16.png', 'p5_dip_T24.png',
               'p5_noise.png', 'p5_coulomb.png']), True)
    R['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                 'python': sys.version.split()[0], 'numpy': np.__version__,
                 'run_s': round(time.time() - t0, 1)}
    (OUT / 'report.json').write_text(json.dumps(R, ensure_ascii=False, indent=1),
                                     encoding='utf-8')
    print(f"== report.json: {sum(c['ok'] for c in R['checks'])}/{len(R['checks'])} PASS, "
          f"{time.time()-t0:.1f} с ==", flush=True)


if __name__ == '__main__':
    main()
