# -*- coding: utf-8 -*-
"""Сборка HTML части 5: кулоновское правило."""
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
        rows.append(f'<tr><td><code>{e(r["tag"])}</code></td><td class="num">{r["P"]:,}</td>'.replace(',', ' ') +
                    f'<td class="num">{r["N"]:,}</td>'.replace(',', ' ') +
                    f'<td class="num">{r["cells"]:,}</td>'.replace(',', ' ') +
                    f'<td class="num">{r["blob"]:.2f}</td><td class="num">{r["network"]:.2f}</td>'
                    f'<td class="num">{r["wall_frac"]*100:.2f}%</td>'
                    f'<td class="num">{r["wall_network"]:.2f}</td></tr>')
    head = ('<table><thead><tr><th>прогон</th><th class="num">P</th><th class="num">N</th>'
            '<th class="num">клеток</th><th class="num">blob</th><th class="num">network</th>'
            '<th class="num">стенки</th><th class="num">wall network</th></tr></thead><tbody>')
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
    caps = {
        'p5_checker_T10.png': 'Зародыш-шахматка, T=10: блоб с полосатой доменной сеткой.',
        'p5_checker_T16.png': 'T=16: сетка укрупняется, блоб сжимается.',
        'p5_checker_slices.png': 'Срезы знака заряда: чередующиеся +/− полосы (доменная сетка).',
        'p5_dipole_T16.png': 'Диполь +/−: притяжение стягивает заряды в блоб.',
        'p5_quadrupole_T16.png': 'Квадруполь: несколько зарядов дают более сложный блоб.',
        'p5_coulomb.png': 'Потенциал и сила: F ~ 1/r², φ эффективно ~ r^−1.4.',
    }
    cards = []
    for fn, cap in caps.items():
        if (OUT / fn).exists():
            cards.append(f'<figure><img loading="lazy" src="{IMG}/{fn}" alt="{e(cap)}">'
                         f'<figcaption><b>Рис. {len(cards)+2}.</b> {cap}</figcaption></figure>')
    return ''.join(cards)


def main():
    c = D['coulomb']
    out = TPL
    out = out.replace('{{F_SLOPE}}', f'{c["force_slope"]:.2f}'.replace('-', '−'))
    out = out.replace('{{PHI_SLOPE}}', f'{c["phi_slope"]:.2f}'.replace('-', '−'))
    phi_ok = next(x for x in D['checks'] if 'потенциал' in x['name'])
    out = out.replace('{{PHI_STATUS}}', 'PASS' if phi_ok['ok'] else 'FAIL')
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
