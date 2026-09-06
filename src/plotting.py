# -*- coding: utf-8 -*-
"""Общие помощники для графиков (matplotlib, backend Agg)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 100,
})


def new_fig(w=10.0, h=6.0, nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return fig, axes


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
