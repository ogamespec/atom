# -*- coding: utf-8 -*-
"""Сборка HTML части 5 (исправленной): всё из одной клетки."""
from __future__ import annotations
import html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ca3d5'
D = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
TPL = (ROOT / 'report' / 'ca3d5_template.html').read_text(encoding='utf-8')
IMG = '../results/ca3d5'


def e(x):
    return html.escape(str(x))


def runs_table():
    rows = []
    for r in D['runs']:
        rows.append(f'<tr><td><code>{e(r["tag"])}</code></td>'
                    f'<td class="num">{r["bias"]}</td><td class="num">{r["rep"]}</td>'
                    f'<td class="num">{r["noise"]}</td><td class="num">{e(r.get("K"))}</td>'
                    f'<td class="num">{r["cells"]:,}</td>'.replace(',', ' ') +
                    f'<td class="num">{r["Dx"]}</td><td class="num">{r["Qxx"]}</td>'
                    f'<td class="num">{r["sep"]}</td><td class="num">{r["blob"]:.2f}</td></tr>')
    head = ('<table><thead><tr><th>прогон</th><th class="num">bias</th><th class="num">rep</th>'
            '<th class="num">noise</th><th class="num">K</th><th class="num">клеток</th>'
            '<th class="num">Dx</th><th class="num">Qxx</th><th class="num">sep</th>'
            '<th class="num">blob</th></tr></thead><tbody>')
    return head + ''.join(rows) + '</tbody></table>'


def checks_table():
    rows = []
    for c in D['checks']:
        rows.append(f'<tr><td>{e(c["name"])}</td>'
                    f'<td><span class="tag {"ok" if c["ok"] else "no"}">'
                    f'{"PASS" if c["ok"] else "FAIL"}</span></td>'
                    f'<td class="num small">{e(c["value"])}</td></tr>')
    return ('<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def gallery():
    caps = [
        ('p5_dip_T4.png', 'T=4: клетка начала раскалываться; появляются два облака.'),
        ('p5_dip_T8.png', 'T=8: p и n разошлись, кулоновское поле стягивает их.'),
        ('p5_dip_T16.png', 'T=16: связанная пара — эмерджентный диполь.'),
        ('p5_dip_T24.png', 'T=24: диполь стабилен, Dx ≈ 7, расстояние ≈ 12 клеток.'),
        ('p5_noise.png', 'Шум из одной клетки: пёстрый мультипольный узор (Qxx ≈ 0.84).'),
    ]
    cards = []
    for fn, cap in caps:
        if (OUT / fn).exists():
            cards.append(f'<figure><img loading="lazy" src="{IMG}/{fn}" alt="{e(cap)}">'
                         f'<figcaption><b>Рис.</b> {cap}</figcaption></figure>')
    return ''.join(cards)


def main():
    c = D['coulomb']
    out = TPL
    out = out.replace('{{F_SLOPE}}', f'{c["force_slope"]:.2f}'.replace('-', '−'))
    out = out.replace('{{PHI_SLOPE}}', f'{c["phi_slope"]:.2f}'.replace('-', '−'))
    out = out.replace('{{RUNS_TABLE}}', runs_table())
    out = out.replace('{{GALLERY}}', gallery())
    out = out.replace('{{CHECKS}}', checks_table())
    out = out.replace('{{N_CHECKS}}', str(len(D['checks'])))
    out = out.replace('{{GENERATED}}', e(D['meta']['generated']))
    out = out.replace('{{PYTHON}}', e(D['meta']['python']))
    left = sorted(set(re.findall(r'\{\{(\w+)\}\}', out)))
    if left:
        print('не заполнены:', left)
    (ROOT / 'report' / 'ca3d5.html').write_text(out, encoding='utf-8')
    print(f'написано report/ca3d5.html ({len(out)/1024:.0f} КБ)')


if __name__ == '__main__':
    main()
