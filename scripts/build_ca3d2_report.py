# -*- coding: utf-8 -*-
"""Сборка HTML второй части: 100 правил, сети и блобы."""
from __future__ import annotations
import html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ca3d2'
SCREEN = json.loads((OUT / 'screening.json').read_text(encoding='utf-8'))
DEEP = json.loads((OUT / 'deep.json').read_text(encoding='utf-8'))
TPL = (ROOT / 'report' / 'ca3d2_template.html').read_text(encoding='utf-8')
IMG = '../results/ca3d2'


def e(x):
    return html.escape(str(x))


def num(x, nd=3):
    if x is None:
        return '—'
    if isinstance(x, float):
        if abs(x) < 1e-3 and x != 0:
            return f'{x:.2e}'
        return f'{x:.{nd}f}'
    return str(x)


def top_table():
    rows = []
    for r in DEEP['selected']:
        m = r.get('metrics', {})
        net = m.get('network', 1.0 if r['family'] == 'bond' else 0)
        blob = m.get('blob', 0)
        if r['family'] == 'bond':
            key = f"узлов {r['metrics']['nodes']}, рёбер {r['metrics']['edges']}"
            occ = '—'
        elif r['family'] == 'potts':
            key = f"стенки {100*r['wall_frac']:.1f}% фрагмента"
            occ = f"{100*m.get('occupancy',0):.1f}%"
        elif r['family'] == 'tumor':
            key = f"некроз {100*r.get('necrotic',0):.1f}%"
            occ = f"{100*m.get('occupancy',0):.1f}%"
        elif r['family'] == 'eden':
            key = f"compactness {m.get('compactness',0):.2f}"
            occ = f"{100*m.get('occupancy',0):.1f}%"
        else:
            key = f"ветвления {m.get('branches',0)}"
            occ = f"{100*m.get('occupancy',0):.2f}%"
        rows.append(
            f'<tr><td><code>{e(r["sid"])}</code></td><td>{e(r["family"])}</td>'
            f'<td>{e(r["title"])}</td><td class="num">{occ}</td>'
            f'<td class="num">{net:.2f}</td><td class="num">{blob:.2f}</td>'
            f'<td class="small">{e(key)}</td></tr>')
    head = ('<table><thead><tr><th>ID</th><th>семейство</th><th>правило</th>'
            '<th class="num">занятость</th><th class="num">network</th>'
            '<th class="num">blob</th><th>ключевая метрика</th></tr></thead><tbody>')
    return head + ''.join(rows) + '</tbody></table>'


def all_table():
    rows = []
    for r in sorted(SCREEN['rules'], key=lambda x: (x['family'], x['sid'])):
        rows.append(
            f'<tr><td><code>{e(r["sid"])}</code></td><td>{e(r["family"])}</td>'
            f'<td>{e(r["title"])}</td><td class="num">{100*r["occ"]:.2f}%</td>'
            f'<td class="num">{r["network"]:.2f}</td><td class="num">{r["blob"]:.2f}</td>'
            f'<td>{e(r["verdict"])}</td></tr>')
    head = ('<table><thead><tr><th>ID</th><th>семейство</th><th>правило</th>'
            '<th class="num">занятость</th><th class="num">network</th>'
            '<th class="num">blob</th><th>вердикт</th></tr></thead><tbody>')
    return head + ''.join(rows) + '</tbody></table>'


def checks_table():
    allc = SCREEN.get('checks', []) + DEEP.get('checks', [])
    rows = []
    for c in allc:
        rows.append(f'<tr><td>{e(c["name"])}</td>'
                    f'<td><span class="tag {"ok" if c["ok"] else "no"}">'
                    f'{"PASS" if c["ok"] else "FAIL"}</span></td>'
                    f'<td class="num small">{e(c["value"])}</td></tr>')
    return ('<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def fig(path, cap, num_):
    return (f'<figure><img loading="lazy" src="{IMG}/{path}" alt="{e(cap)}">'
            f'<figcaption><b>Рис. {num_}.</b> {cap}</figcaption></figure>')


def networks():
    cards = [
        fig('n_invasion.png', 'INVASION: масса кластера 15 000 клеток в кубе 161³. '
            'Голубая сфера — фронт r = c·t = 50; справа сверху — увеличенный центр.', '4.1'),
        fig('n_invasion_skel.png', 'INVASION: скелет того же кластера — 11 345 узлов, '
            '17 852 ребра. Это и есть сетеподобная структура: ветви, развилки, обходы.', '4.2'),
        fig('n_invasion_slices.png', 'INVASION: срезы кластера. Видно, что ветви объёмные, '
            'а не плоские.', '4.3'),
        fig('n_bond.png', 'BOND: граф связей, выращенный из центра (770 узлов, 1094 ребра). '
            'Дополнительный пример сети, нарисованной рёбрами.', '4.4'),
    ]
    return '<div class="gallery">' + ''.join(cards) + '</div>'


def foam():
    cards = [
        fig('n_foam.png', 'POTTS: центральный фрагмент пены 41³ в кубе 161³. '
            'Зелёные точки — стенки между доменами; сфера — фронт r = c·t = 40.', '4.5'),
        fig('n_foam_slices.png', 'POTTS: шесть срезов фрагмента. Цвет — фаза домена, '
            'чёрное — стенки. Видно клеточное (пенообразное) строение.', '4.6'),
    ]
    return '<div class="gallery">' + ''.join(cards) + '</div>'


def blobs():
    cards = [
        fig('b_tumor.png', 'TUMOR: округлый блоб с тёмным некротическим ядром '
            '(8 571 клеток). Фронт r = c·t = 40.', '5.1'),
        fig('b_tumor_slices.png', 'TUMOR: срезы. Оранжевое — живые клетки, тёмное — некроз; '
            'ядро голодает, внешний слой растёт.', '5.2'),
        fig('b_eden.png', 'EDEN: компактный огранённый блоб (48 174 клетки), '
            'выращенный из центрального зародыша. Фронт r = c·t = 30.', '5.3'),
    ]
    return '<div class="gallery">' + ''.join(cards) + '</div>'


def main():
    out = TPL
    f = SCREEN['funnel']
    out = out.replace('{{N_RULES}}', str(f['total']))
    out = out.replace('{{N_ALIVE}}', str(f['alive']))
    out = out.replace('{{N_STRUCT}}', str(f['structured']))
    out = out.replace('{{N_SEL}}', str(len(DEEP['selected'])))
    out = out.replace('{{N_CHECKS}}', str(len(SCREEN.get('checks', [])) + len(DEEP.get('checks', []))))
    out = out.replace('{{FUNNEL}}', fig('funnel.png', 'Воронка отсева: 100 правил → 82 живых → '
                       f'{f["structured"]} структурных → {len(DEEP["selected"])} в отчёте.', '3.1'))
    out = out.replace('{{SCATTER}}', fig('morph_scatter.png', 'Каждое правило — точка '
                       '(network, blob); цвет — семейство. Белые кружки — отобранные.', '3.2'))
    out = out.replace('{{TOP_TABLE}}', top_table())
    out = out.replace('{{ALL_TABLE}}', all_table())
    out = out.replace('{{CHECKS}}', checks_table())
    out = out.replace('{{NET_GALLERY}}', networks())
    out = out.replace('{{FOAM_GALLERY}}', foam())
    out = out.replace('{{BLOB_GALLERY}}', blobs())
    out = out.replace('{{GENERATED}}', e(SCREEN['meta']['generated']))
    out = out.replace('{{PYTHON}}', e(SCREEN['meta']['python']))
    left = sorted(set(re.findall(r'\{\{(\w+)\}\}', out)))
    if left:
        print('не заполнены:', left)
    (ROOT / 'report' / 'ca3d2.html').write_text(out, encoding='utf-8')
    print(f'написано report/ca3d2.html ({len(out)/1024:.0f} КБ)')


if __name__ == '__main__':
    main()
