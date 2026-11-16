# -*- coding: utf-8 -*-
"""Картинка структуры C-прогона, увеличенная на весь фронт. Без пересчёта CA."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_render import render_voxels

OUT = ROOT / 'results' / 'ca3d5_c'
data = (OUT / 'state.bin').read_bytes()
count = int(np.frombuffer(data[:8], np.uint64)[0])
rec = np.frombuffer(data[8:], np.int32).reshape(-1, 5)
coords = rec[:, :3].astype(np.int64)
tot = rec[:, 3].astype(np.int64) + rec[:, 4].astype(np.int64)
front = float(np.sqrt((coords.astype(float) ** 2).sum(1)).max())
S = int(2 * front) + 20
if S % 2 == 0:
    S += 1
c = S // 2
print('count', count, 'front', round(front, 2), 'S', S)
g = np.zeros((S, S, S), np.uint8)
q = np.clip(coords + c, 0, S - 1)
vmax = max(float(tot.max()), 1)
g[q[:, 0], q[:, 1], q[:, 2]] = np.clip(
    np.log10(1 + tot) / np.log10(1 + vmax) * 255, 1, 255).astype(np.uint8)
img = render_voxels(g, size=900, canvas_L=S, cmap='inferno', value_norm=255,
                    cutaway=False,
                    front_radius=front, front_time=front,
                    front_label='фронт структуры r',
                    legend_label='log10 бит/ячейка',
                    label=f'L03 C, T=10 000: весь фронт r={front:.0f}')
Image.fromarray(img).save(OUT / 'p5_c_front.png')
print('saved', OUT / 'p5_c_front.png')
