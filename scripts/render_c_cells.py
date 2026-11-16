# -*- coding: utf-8 -*-
"""Зум активных клеток C-прогона: 3D-воксели и срезы. Без пересчёта CA."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_render import render_voxels, slice_mosaic

OUT = ROOT / 'results' / 'ca3d5_c'
data = (OUT / 'state.bin').read_bytes()
count = int(np.frombuffer(data[:8], np.uint64)[0])
rec = np.frombuffer(data[8:], np.int32).reshape(-1, 5)
coords = rec[:, :3].astype(np.int64)
tot = rec[:, 3].astype(np.int64) + rec[:, 4].astype(np.int64)

# самый плотный участок по грубой гистограмме
mn, mx = coords.min(0), coords.max(0)
bin_size = 32
shape = np.maximum(((mx - mn) // bin_size) + 1, 1)
idx = (coords - mn) // bin_size
hist = np.zeros(tuple(shape), np.int64)
np.add.at(hist, (idx[:, 0], idx[:, 1], idx[:, 2]), tot)
best = np.unravel_index(np.argmax(hist), hist.shape)
center = mn + (np.array(best) + 0.5) * bin_size

for S in (96, 128, 160):
    lo = (center - S // 2).astype(int)
    m = np.all((coords >= lo) & (coords < lo + S), axis=1)
    n = int(m.sum())
    print('S', S, 'cells', n)
    if 2000 <= n <= 120000:
        break

lo = (center - S // 2).astype(int)
m = np.all((coords >= lo) & (coords < lo + S), axis=1)
q = (coords[m] - lo).astype(int)
g = np.zeros((S, S, S), np.uint8)
vmax = max(float(tot[m].max()), 1)
g[q[:, 0], q[:, 1], q[:, 2]] = np.clip(
    np.log10(1 + tot[m]) / np.log10(1 + vmax) * 255, 1, 255).astype(np.uint8)

img = render_voxels(g, size=900, canvas_L=S, cell=0.82, max_points=200000,
                    grid_step=max(4, S // 12), cmap='inferno', value_norm=255,
                    cutaway=False, legend_label='log10 бит/ячейка',
                    label=f'Зум активных клеток: {S}³, {n:,} клеток'.replace(',', ' '))
Image.fromarray(img).save(OUT / 'p5_c_cells.png')
slice_mosaic(g, str(OUT / 'p5_c_cells_slices.png'), n=6, cmap='inferno',
             title=f'Активные клетки: срезы {S}³ (log10 бит)', vmin=0, vmax=255)
print('saved cells zoom S', S, 'n', n, 'center', center)
