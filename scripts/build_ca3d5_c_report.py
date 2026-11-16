# -*- coding: utf-8 -*-
"""Сборка отчёта C-прогона L03 T=20000."""
from __future__ import annotations
import html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ca3d5_c'
D = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
TPL = (ROOT / 'report' / 'ca3d5_c_template.html').read_text(encoding='utf-8')
IMG = '../results/ca3d5_c'


def e(x):
    return html.escape(str(x))


def metrics():
    rows = [('N бит', D['N']), ('P', D['P']), ('N-бит', D['Nbits']),
            ('активных клеток', D['count']), ('фронт структуры', D['front']),
            ('blob', D['blob']), ('network', D['network']),
            ('кластеров', D['clusters']), ('L', D['L']), ('R', D['space_radius']),
            ('R / c·t', D['ratio'])]
    body = ''.join(f'<tr><td>{e(a)}</td><td class="num">{e(b)}</td></tr>' for a, b in rows)
    return f'<table><thead><tr><th>параметр</th><th class="num">значение</th></tr></thead><tbody>{body}</tbody></table>'


def checkpoints():
    lines = (OUT / 'run.log').read_text(encoding='utf-8', errors='replace').splitlines()
    rows = []
    for ln in lines:
        if ln.startswith('cp t='):
            parts = dict(x.split('=') for x in ln.split()[1:])
            rows.append(f'<tr><td class="num">{parts["t"]}</td>'
                        f'<td class="num">{int(parts["active"]):,}</td>'.replace(',', ' ') +
                        f'<td class="num">{int(parts["bits"]):,}</td>'.replace(',', ' ') +
                        f'<td class="num">{parts["front"]}</td></tr>')
    return ('<table><thead><tr><th class="num">T</th><th class="num">активных</th>'
            '<th class="num">бит</th><th class="num">фронт</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def checks():
    rows = ''.join(f'<tr><td>{e(c["name"])}</td>'
                   f'<td><span class="tag {"ok" if c["ok"] else "no"}">'
                   f'{"PASS" if c["ok"] else "FAIL"}</span></td>'
                   f'<td class="num small">{e(c["value"])}</td></tr>' for c in D['checks'])
    return ('<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead>'
            '<tbody>' + rows + '</tbody></table>')


def gallery():
    caps = {'p5_c_full.png': 'Полный куб L=60 001: фронт c·t=20 000 и сфера R=30 000.',
            'p5_c_zoom.png': 'Центр: структура L03 после 20 000 тактов (16 млн бит).'}
    return ''.join(f'<figure><img loading="lazy" src="{IMG}/{fn}" alt="{e(cap)}">'
                   f'<figcaption><b>{e(fn)}.</b> {cap}</figcaption></figure>'
                   for fn, cap in caps.items() if (OUT / fn).exists())


def main():
    out = TPL
    out = out.replace('{{METRICS}}', metrics())
    out = out.replace('{{CHECKPOINTS}}', checkpoints())
    out = out.replace('{{CHECKS}}', checks())
    out = out.replace('{{GALLERY}}', gallery())
    out = out.replace('{{N_CHECKS}}', str(len(D['checks'])))
    out = out.replace('{{COUNT}}', f'{D["count"]:,}'.replace(',', ' '))
    out = out.replace('{{FRONT}}', f'{D["front"]:.1f}')
    out = out.replace('{{BLOB}}', f'{D["blob"]:.2f}')
    out = out.replace('{{NET}}', f'{D["network"]:.2f}')
    out = out.replace('{{GENERATED}}', e(D['meta']['generated']))
    left = sorted(set(re.findall(r'\{\{(\w+)\}\}', out)))
    if left:
        print('не заполнены:', left)
    (ROOT / 'report' / 'ca3d5_c.html').write_text(out, encoding='utf-8')
    print(f'написано report/ca3d5_c.html ({len(out)/1024:.0f} КБ)')


if __name__ == '__main__':
    main()
