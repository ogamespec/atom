# -*- coding: utf-8 -*-
"""Изометрический рендер 3D-конфигураций клеточного автомата.

* объёмный рендер (emission-absorption) с палитрой Inferno;
* точечный рендер для разреженных фракталов;
* вырезание октанта (cutaway), чтобы видеть внутренность;
* мозаика срезов и композитные «слайды» с сеткой и масштабом.
"""
from __future__ import annotations

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from scipy.ndimage import affine_transform, gaussian_filter

matplotlib.use('Agg')

CMAP = 'inferno'


def rot_matrix(azim: float = 35.0, elev: float = 28.0) -> np.ndarray:
    a, e = np.radians(azim), np.radians(elev)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.stack([right, up, fwd])


def project_volume(g: np.ndarray, azim: float = 35, elev: float = 28, pad: int = 6):
    g = g.astype(np.float32)
    L = np.array(g.shape)
    R = rot_matrix(azim, elev)
    corners = np.array([[i, j, k] for i in (0, L[0]) for j in (0, L[1]) for k in (0, L[2])], float)
    pc = (corners - L / 2) @ R.T
    mn, mx = pc.min(0), pc.max(0)
    out = np.ceil(mx - mn).astype(int) + 2 * pad
    off = L / 2 + R.T @ (mn - pad)
    vol = affine_transform(g, R.T, offset=off, output_shape=tuple(out), order=1,
                           mode='constant', cval=0)
    return np.clip(vol, 0, None), mn


def composite(vol: np.ndarray, tau: float = 0.05) -> np.ndarray:
    d = vol * tau
    trans = np.exp(-np.cumsum(d, axis=0) + d)
    return (d * trans).sum(0)


def colorize(img: np.ndarray, gamma: float = 0.7, percent: float = 99.8,
             blur: float = 0.0, cmap: str = CMAP, gain: float = 1.0):
    if blur:
        img = gaussian_filter(img.astype(np.float32), blur)
    ref = np.percentile(img[img > 0], percent) if np.any(img > 0) else 1.0
    x = np.clip(gain * img / (ref + 1e-12), 0, 1) ** gamma
    cm = cmap if callable(cmap) else matplotlib.colormaps[cmap]
    return (cm(x)[..., :3] * 255).astype(np.uint8), x


def render_volume(g: np.ndarray, tau: float = 0.05, gamma: float = 0.7,
                  blur: float = 0.7, azim: float = 35, elev: float = 28,
                  cutaway: bool = False, cmap: str = CMAP) -> np.ndarray:
    g = g.astype(np.float32)
    if cutaway:
        c = g.shape[0] // 2
        g = g.copy()
        g[c:, c:, c:] = 0
    vol, _ = project_volume(g, azim, elev)
    img = composite(vol, tau)
    rgb, _ = colorize(img, gamma=gamma, blur=blur, cmap=cmap)
    return rgb


def render_points(g: np.ndarray, size: int = 512, blur: float = 1.2,
                  azim: float = 35, elev: float = 28, gamma: float = 0.7,
                  cmap: str = CMAP, gain: float = 1.0) -> np.ndarray:
    """Точечный рендер: каждая занятая ячейка — светящаяся точка с глубиной."""
    idx = np.argwhere(g > 0).astype(float)
    L = np.array(g.shape, float)
    if len(idx) == 0:
        return np.zeros((size, size, 3), np.uint8)
    R = rot_matrix(azim, elev)
    pc = (idx - L / 2) @ R.T
    mn, mx = pc.min(0), pc.max(0)
    span = max(mx[0] - mn[0], mx[1] - mn[1], 1.0)
    scale = (size - 80) / span
    u = (pc[:, 0] - (mn[0] + mx[0]) / 2) * scale + size / 2
    v = (mx[1] - pc[:, 1]) * scale + size / 2 - (mx[1] - mn[1]) * scale / 2
    depth = pc[:, 2]
    dmin, dmax = depth.min(), depth.max()
    dn = (depth - dmin) / max(dmax - dmin, 1e-9)
    # накопление с весом по глубине (ближние ярче)
    H, _, _ = np.histogram2d(v, u, bins=(size, size),
                             range=[[0, size], [0, size]],
                             weights=0.35 + 0.65 * (1 - dn))
    H = gaussian_filter(H, blur)
    ref = np.percentile(H[H > 0], 99.5) if np.any(H > 0) else 1.0
    x = np.clip(gain * H / (ref + 1e-9), 0, 1) ** gamma
    return (matplotlib.colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)


def _box_points(lo, hi, azim, elev):
    R = rot_matrix(azim, elev)
    center = (np.array(lo, float) + np.array(hi, float)) / 2
    L = np.array(hi, float) - np.array(lo, float)
    corners = np.array([[i, j, k] for i in (0, L[0]) for j in (0, L[1]) for k in (0, L[2])], float)
    pc = (corners + np.array(lo, float) - center) @ R.T
    return pc


def draw_box(img: np.ndarray, lo, hi, canvas_L: int, azim: float = 35, elev: float = 28,
             color: tuple = (90, 180, 240), width: int = 2, ticks: int = 0,
             label: str | None = None) -> np.ndarray:
    """Рисует каркас куба [lo,hi] в геометрии project_volume для куба canvas_L."""
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    W, H = im.size
    try:
        font = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), 15)
    except Exception:
        font = ImageFont.load_default()
    R = rot_matrix(azim, elev)
    Lv = np.array([canvas_L, canvas_L, canvas_L], float)
    corners_all = np.array([[i, j, k] for i in (0, Lv[0]) for j in (0, Lv[1]) for k in (0, Lv[2])], float)
    pc_all = (corners_all - Lv / 2) @ R.T
    mn, mx = pc_all.min(0), pc_all.max(0)
    pad = 6
    out = np.ceil(mx - mn).astype(int) + 2 * pad
    sx, sy = W / out[0], H / out[1]
    pc = _box_points(lo, hi, azim, elev)
    pts = np.stack([(pc[:, 0] - mn[0] + pad) * sx, (mx[1] - pc[:, 1] + pad) * sy], 1)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        dr.line([tuple(pts[a]), tuple(pts[b])], fill=color, width=width)
    for e0, e1 in [(0, 1), (0, 2), (0, 4)]:
        for i in range(1, ticks + 1):
            f = i / (ticks + 1)
            p = pts[e0] * (1 - f) + pts[e1] * f
            dr.line([(p[0] - 3, p[1] - 3), (p[0] + 3, p[1] + 3)], fill=color, width=1)
    if label:
        dr.text((10, 10), label, fill=(225, 235, 245), font=font)
    return np.array(im)


def upscale(img: np.ndarray, size: int = 512) -> np.ndarray:
    im = Image.fromarray(img)
    return np.array(im.resize((size, size), Image.LANCZOS))


def frame(g: np.ndarray, L: int, t: int, size: int = 512, cutaway: bool = False,
          sparse: bool = False, azim: float = 35, elev: float = 28,
          canvas_L: int | None = None, cmap: str = CMAP,
          scale_label: bool = True, tau: float | None = None) -> np.ndarray:
    """Кадр: куб активной области L внутри холста canvas_L (для инфляции)."""
    canvas_L = canvas_L or L
    g = np.asarray(g, dtype=np.float32)
    if g.shape[0] != canvas_L:
        c = canvas_L // 2
        r = g.shape[0] // 2
        pad = np.zeros((canvas_L,) * 3, np.float32)
        pad[c - r:c + r + 1, c - r:c + r + 1, c - r:c + r + 1] = g
        g = pad
    if sparse:
        rgb = render_points(g, size=size, azim=azim, elev=elev, cmap=cmap)
    else:
        if cutaway and L == canvas_L:
            c = L // 2
            g = g.copy(); g[c:, c:, c:] = 0
        if tau is None:
            occ = float(np.count_nonzero(g) / g.size)
            tau = 0.012 if occ > 0.05 else (0.05 if occ > 0.004 else 0.15)
        rgb = render_volume(g, tau=tau, cutaway=False, azim=azim, elev=elev, cmap=cmap)
        rgb = upscale(rgb, size)
    if scale_label:
        lab = f'L = {L}   t = {t}'
        if canvas_L > L:
            lab += f'   (холст {canvas_L})'
    else:
        lab = None
    rgb = draw_box(rgb, (0, 0, 0), (canvas_L,) * 3, canvas_L, azim, elev,
                   color=(55, 95, 130), width=2, ticks=4, label=lab)
    if canvas_L > L:
        off = (canvas_L - L) // 2
        rgb = draw_box(rgb, (off,) * 3, (off + L,) * 3, canvas_L, azim, elev,
                       color=(120, 210, 255), width=2, ticks=0)
    return rgb


def slice_mosaic(g: np.ndarray, path: str, n: int = 6, cmap=CMAP,
                 title: str | None = None, vmin: float | None = None,
                 vmax: float | None = None) -> None:
    """Мозаика n разных срезов по z (2x3, 2x2 или 1x3)."""
    L = g.shape[0]
    picks = np.linspace(0, L - 1, n).astype(int)
    cols = 3 if n > 4 else (2 if n > 2 else n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    axes = np.atleast_2d(axes)
    vmax = vmax if vmax is not None else (float(g.max()) if g.size else 1.0)
    vmin = vmin if vmin is not None else 0.0
    for ax, z in zip(axes.ravel(), picks):
        ax.imshow(g[z], cmap=cmap, origin='lower', interpolation='nearest',
                  vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'z={z}', fontsize=8)
    for ax in axes.ravel()[len(picks):]:
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _camera(canvas_L: int, azim: float = 38.0, elev: float = 24.0,
            dist: float = 2.05, size: int = 640):
    """Камера перспективной проекции. Возвращает eye, fwd, right, up, f."""
    c = (canvas_L - 1) / 2.0
    a, e = np.radians(azim), np.radians(elev)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    eye = np.array([c, c, c]) + d * dist * canvas_L
    fwd = np.array([c, c, c]) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = 1.30 * size
    return eye, fwd, right, up, f


def _project(pts: np.ndarray, eye, fwd, right, up, f, size: int):
    q = pts - eye
    z = q @ fwd
    x = q @ right
    y = q @ up
    z = np.where(z <= 1e-6, 1e-6, z)
    sx = size / 2 + f * x / z
    sy = size / 2 - f * y / z
    return sx, sy, z


def _sphere_lines(center, R, n_lat=5, n_lon=6, seg=48):
    """Полилинии каркасной сферы радиуса R вокруг center (мировые координаты)."""
    cx, cy, cz = center
    lines = []
    for i in range(1, n_lat + 1):
        lat = -np.pi / 2 + np.pi * i / (n_lat + 1)
        z = cz + R * np.sin(lat)
        r = R * np.cos(lat)
        th = np.linspace(0, 2 * np.pi, seg)
        lines.append(np.stack([cx + r * np.cos(th), cy + r * np.sin(th),
                               np.full_like(th, z)], 1))
    for j in range(n_lon):
        lon = np.pi * j / n_lon
        lat = np.linspace(-np.pi / 2, np.pi / 2, seg)
        x = cx + R * np.cos(lat) * np.cos(lon)
        y = cy + R * np.cos(lat) * np.sin(lon)
        z = cz + R * np.sin(lat)
        lines.append(np.stack([x, y, z], 1))
    return lines


def _draw_front(dr, center, R, eye, fwd, right, up, f, size, color=(90, 195, 255, 120),
                width=1):
    for line in _sphere_lines(center, R):
        sx, sy, _ = _project(line, eye, fwd, right, up, f, size)
        pts = [(float(a), float(b)) for a, b in zip(sx, sy)]
        dr.line(pts, fill=color, width=width)


def _draw_legend(dr, size, cmap, vmax, label, font):
    """Вертикальная цветовая шкала в правом нижнем углу кадра."""
    x0, y0, w, h = size - 52, size - 168, 14, 104
    steps = 80
    for i in range(steps):
        col = (cmap if callable(cmap) else matplotlib.colormaps[cmap])(1 - i / (steps - 1))
        rgb = tuple(int(255 * c) for c in col[:3])
        dr.line([(x0, y0 + i * h / steps), (x0 + w, y0 + i * h / steps)],
                fill=rgb + (255,), width=2)
    dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(180, 200, 220, 220), width=1)
    dr.text((x0 - 2, y0 - 17), label, fill=(210, 225, 240), font=font)
    mx = vmax if vmax else 1
    dr.text((x0 + w + 4, y0 - 2), f'{mx:g}', fill=(210, 225, 240), font=font)
    dr.text((x0 + w + 4, y0 + h - 10), '0', fill=(210, 225, 240), font=font)


def render_voxels(g: np.ndarray, size: int = 640, canvas_L: int | None = None,
                  azim: float = 38.0, elev: float = 24.0,
                  max_points: int = 60000, cell: float = 0.68,
                  cutaway: bool = False, region: str = 'full', grid_step: int | None = None,
                  cmap: str = CMAP, label: str | None = None,
                  value_norm: float | None = None, front_radius: float | None = None,
                  front_time: float | None = None, legend_label: str | None = None,
                  inset: bool = False) -> np.ndarray:
    """Перспективный рендер занятых ячеек как отдельных вокселей.

    В отличие от объёмного рендера, здесь видна решётка: каждая ячейка —
    кубик с зазором, а на стенках куба нарисована сетка. Для плотных
    конфигураций точки прореживаются, чтобы зазор сохранялся.
    """
    g = np.asarray(g)
    L = g.shape[0]
    canvas_L = canvas_L or L
    img = Image.new('RGB', (size, size), (5, 8, 13))
    dr = ImageDraw.Draw(img, 'RGBA')
    try:
        font = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), 15)
        font_s = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), 12)
    except Exception:
        font, font_s = ImageFont.load_default(), ImageFont.load_default()

    eye, fwd, right, up, f = _camera(canvas_L, azim, elev, size=size)
    c = (canvas_L - 1) / 2.0

    # ---- сетка на трёх задних стенках куба ----
    step = grid_step or max(4, canvas_L // 8)
    grid_col = (38, 74, 104, 190)
    edge_col = (95, 170, 220, 255)

    def line3(p0, p1, fill, width=1):
        sx, sy, _ = _project(np.array([p0, p1], float), eye, fwd, right, up, f, size)
        dr.line([(sx[0], sy[0]), (sx[1], sy[1])], fill=fill, width=width)

    # пол z=0 и две дальней стенки x=0, y=0
    for k in range(0, canvas_L + 1, step):
        line3((0, k, 0), (canvas_L - 1, k, 0), grid_col)
        line3((k, 0, 0), (k, canvas_L - 1, 0), grid_col)
        line3((0, k, 0), (0, k, canvas_L - 1), grid_col)
        line3((0, 0, k), (0, canvas_L - 1, k), grid_col)
        line3((0, 0, k), (canvas_L - 1, 0, k), grid_col)
        line3((k, 0, 0), (k, 0, canvas_L - 1), grid_col)
    # внешний каркас
    for p0, p1 in [((0, 0, 0), (canvas_L - 1, 0, 0)), ((0, 0, 0), (0, canvas_L - 1, 0)),
                   ((0, 0, 0), (0, 0, canvas_L - 1)), ((canvas_L - 1, 0, 0), (canvas_L - 1, canvas_L - 1, 0)),
                   ((canvas_L - 1, 0, 0), (canvas_L - 1, 0, canvas_L - 1)),
                   ((0, canvas_L - 1, 0), (canvas_L - 1, canvas_L - 1, 0)),
                   ((0, canvas_L - 1, 0), (0, canvas_L - 1, canvas_L - 1)),
                   ((0, 0, canvas_L - 1), (canvas_L - 1, 0, canvas_L - 1)),
                   ((0, 0, canvas_L - 1), (0, canvas_L - 1, canvas_L - 1)),
                   ((canvas_L - 1, canvas_L - 1, 0), (canvas_L - 1, canvas_L - 1, canvas_L - 1)),
                   ((canvas_L - 1, 0, canvas_L - 1), (canvas_L - 1, canvas_L - 1, canvas_L - 1)),
                   ((0, canvas_L - 1, canvas_L - 1), (canvas_L - 1, canvas_L - 1, canvas_L - 1))]:
        line3(p0, p1, edge_col, 2)

    # ---- фронт скорости света: каркасная сфера r = c*t ----
    if front_radius and front_radius > 0:
        R = min(float(front_radius), max(canvas_L / 2 - 0.5, 1.0))
        _draw_front(dr, (c, c, c), R, eye, fwd, right, up, f, size)

    # ---- активный куб (если холст больше) ----
    inner = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    dni = ImageDraw.Draw(inner, 'RGBA')
    if canvas_L > L:
        off = (canvas_L - L) // 2
        for p0, p1 in [((off, off, off), (off + L - 1, off, off)), ((off, off, off), (off, off + L - 1, off)),
                       ((off, off, off), (off, off, off + L - 1)),
                       ((off + L - 1, off, off), (off + L - 1, off + L - 1, off)),
                       ((off + L - 1, off, off), (off + L - 1, off, off + L - 1)),
                       ((off, off + L - 1, off), (off + L - 1, off + L - 1, off)),
                       ((off, off + L - 1, off), (off, off + L - 1, off + L - 1)),
                       ((off, off, off + L - 1), (off + L - 1, off, off + L - 1)),
                       ((off, off, off + L - 1), (off, off + L - 1, off + L - 1)),
                       ((off + L - 1, off + L - 1, off), (off + L - 1, off + L - 1, off + L - 1)),
                       ((off + L - 1, off, off + L - 1), (off + L - 1, off + L - 1, off + L - 1)),
                       ((off, off + L - 1, off + L - 1), (off + L - 1, off + L - 1, off + L - 1))]:
            sx, sy, _ = _project(np.array([p0, p1], float), eye, fwd, right, up, f, size)
            dni.line([(sx[0], sy[0]), (sx[1], sy[1])], fill=(120, 210, 255, 220), width=2)

    # ---- воксели ----
    occ = np.argwhere(g > 0).astype(np.float64)
    n_total = len(occ)
    half = L / 2 + 0.5
    if region == 'octant':
        m = (occ[:, 0] < half) & (occ[:, 1] < half) & (occ[:, 2] < half)
        occ = occ[m]
    elif cutaway and canvas_L == L:
        m = ~((occ[:, 0] >= half) & (occ[:, 1] >= half) & (occ[:, 2] >= half))
        occ = occ[m]
    if len(occ) == 0:
        img.paste(inner, (0, 0), inner)
        if label:
            dr.text((14, 12), label, fill=(230, 240, 250), font=font)
        return np.array(img)
    if canvas_L > L:
        off = (canvas_L - L) // 2
        occ = occ + off
    vals = g[tuple((occ - (off if canvas_L > L else 0)).astype(int).T)].astype(np.float32)
    norm = value_norm or float(g.max() or 1)
    t = np.clip(vals / norm, 0, 1)

    if len(occ) > max_points:
        k = int(np.ceil(len(occ) / max_points))
        occ = occ[::k]
        t = t[::k]

    sx, sy, z = _project(occ + 0.5, eye, fwd, right, up, f, size)
    order = np.argsort(-z)
    sx, sy, z, t = sx[order], sy[order], z[order], t[order]
    zmin, zmax = z.min(), z.max()
    cm = cmap if callable(cmap) else matplotlib.colormaps[cmap]
    base = cm(t)[:, :3]
    # глубина: дальние темнее
    depth = 1.0 - 0.55 * (z - zmin) / max(zmax - zmin, 1e-9)
    depth = 0.45 + 0.55 * (1 - (z - zmin) / max(zmax - zmin, 1e-9))
    pix = (base * depth[:, None] * 255).astype(np.int32)
    sz = np.clip(f * cell / z, 1.9, 13.0)
    for x, y, s, col in zip(sx, sy, sz, pix):
        r = int(s)
        if r <= 1:
            dr.point((x, y), fill=tuple(col) + (255,))
        else:
            dr.rectangle([x - r / 2, y - r / 2, x + r / 2, y + r / 2],
                         fill=tuple(col) + (235,))
    img.paste(inner, (0, 0), inner)
    # ---- врезка: увеличенный центр (область внутри фронта) ----
    if inset and front_radius and front_radius > 3 and canvas_L > 2 * front_radius + 8:
        R = int(front_radius) + 2
        cc = canvas_L // 2
        lo = max(cc - R, 0)
        hi = min(cc + R + 1, canvas_L)
        crop = g[lo:hi, lo:hi, lo:hi]
        isz = max(120, size // 3)
        sub = render_voxels(crop, size=isz, canvas_L=crop.shape[0], cell=0.8,
                            grid_step=4, value_norm=value_norm,
                            cmap=cmap, legend_label=None, label=None,
                            front_radius=None, front_time=None, inset=False)
        px, py = size - isz - 10, 10
        img.paste(Image.fromarray(sub), (px, py))
        dr.rectangle([px - 1, py - 1, px + isz, py + isz],
                     outline=(120, 200, 255, 220), width=2)
        dr.text((px + 4, py + isz + 4), 'центр (внутри фронта)',
                fill=(160, 205, 240), font=font_s)
    if legend_label:
        _draw_legend(dr, size, cmap, value_norm or float(g.max() or 1), legend_label,
                     font_s)
    # масштабная линейка: 10 ячеек у нижнего переднего ребра
    p0 = np.array([canvas_L * 0.10, canvas_L * 0.10, 0.0])
    p1 = p0 + np.array([min(10, canvas_L // 4), 0, 0])
    sx0, sy0, _ = _project(np.array([p0]), eye, fwd, right, up, f, size)
    sx1, sy1, _ = _project(np.array([p1]), eye, fwd, right, up, f, size)
    dr.line([(sx0[0], sy0[0]), (sx1[0], sy1[0])], fill=(230, 240, 250, 230), width=2)
    dr.text((float(sx0[0]), float(sy0[0]) + 6), f'{int(p1[0]-p0[0])} ячеек',
            fill=(180, 200, 220), font=font_s)
    if label:
        dr.text((14, 12), label, fill=(230, 240, 250), font=font)
        if front_time and front_time > 0:
            fr = f'фронт c·t = {front_time:.0f}'
            if front_time >= canvas_L / 2:
                fr += '  (дошёл до стенки)'
            dr.text((14, 54), fr, fill=(120, 195, 245), font=font_s)
        shown = len(occ)
        extra = f' (показано {shown:,})'.replace(',', ' ') if shown != n_total else ''
        dr.text((14, 34), f'ячеек: {n_total:,}'.replace(',', ' ') + extra,
                fill=(150, 172, 198), font=font_s)
    return np.array(img)
