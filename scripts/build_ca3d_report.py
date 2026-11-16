# -*- coding: utf-8 -*-
"""Сборка HTML-отчёта issue #1 из шаблона и results/ca3d/report.json."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ca3d'
REPORT = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
TPL = (ROOT / 'report' / 'ca3d_template.html').read_text(encoding='utf-8')

IMG = '../results/ca3d'


def e(x) -> str:
    return html.escape(str(x))


def num(x, nd=3) -> str:
    if x is None:
        return '—'
    if isinstance(x, float):
        if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
            return f'{x:.2e}'
        return f'{x:.{nd}f}'
    return str(x)


def sup10(v: float) -> str:
    """10^v с нормальным форматированием показателя."""
    if v >= 1e9:
        mant, ex = f'{v:.2e}'.split('e')
        return f'10<sup>{mant}&middot;10<sup>{int(ex)}</sup></sup>'
    if v >= 1e6:
        return f'10<sup>{v:,.0f}</sup>'.replace(',', ' ')
    return f'10<sup>{v:,.1f}</sup>'.replace(',', ' ')


def pct(x: float) -> str:
    return f'{100*x:.1f}%'


# ---------------------------------------------------------------- таблицы
def counts_table(c: dict) -> str:
    g = c['general_binary']
    sym = c['symmetry']
    rows = [
        ('Общее бинарное правило, 27 ячеек (центр + 26 соседей)',
         f'2<sup>2<sup>27</sup></sup> &asymp; {sup10(g["log10"])}',
         f'{g["digits"]:,} десятичных цифр'.replace(',', ' ')),
        ('Инвариантное относительно 48 вращений/отражений куба',
         f'2<sup>{sym["config_orbits_48"]:,}</sup> &asymp; {sup10(sym["invariant_rules_48_log10"])}'.replace(',', ' '),
         'правило одинаково на каждой орбите конфигураций'),
        ('Инвариантное относительно 24 собственных вращений',
         f'2<sup>{sym["config_orbits_24"]:,}</sup> &asymp; {sup10(sym["invariant_rules_24_log10"])}'.replace(',', ' '),
         'более слабая симметрия → больше правил'),
        ('Классы эквивалентности правил (48)',
         f'&asymp; {sup10(sym["rule_orbits_48_log10"])}',
         'лемма Бёрнсайда по таблицам; фактор всего 48'),
        ('B/S-правила 3D-Жизни (26 соседей)',
         f'2<sup>54</sup> = {2**54:,}'.replace(',', ' '),
         'B и S — независимые подмножества {0..26}'),
        ('Одноуровневое тоталистическое f(k)',
         f'2<sup>27</sup> = {2**27:,}'.replace(',', ' '),
         'зависит только от числа соседей'),
        ('Интервальные правила (a,b/c,d)',
         f'{int(c["bays_intervals"]["value"]):,}'.replace(',', ' '),
         'класс, в котором Bays искал 3D-Жизнь'),
        ('B = 8 бит, 27 ячеек',
         f'256<sup>256<sup>27</sup></sup> = 2<sup>8&middot;2<sup>216</sup></sup>',
         f'&asymp; 10<sup>2.5&middot;10<sup>65</sup></sup> — число цифр не помещается в память'),
        ('B = 8 бит, 6 соседей + центр',
         f'2<sup>8&middot;2<sup>56</sup></sup>',
         '&asymp; 10<sup>1.7&middot;10<sup>17</sup></sup>'),
        ('Бинарное, 6 соседей + центр',
         f'2<sup>128</sup> = {2**128:,}'.replace(',', ' '),
         'строгая причинность c = 1'),
        ('Обратимые блочные (Марголус), блок 2&times;2&times;2',
         f'256! &asymp; {sup10(c["block_bijections_binary"]["log10"])}',
         'биекции 256 состояний блока'),
        ('Линейные бинарные правила, 27 коэффициентов',
         f'2<sup>27</sup> = {2**27:,}'.replace(',', ' '),
         'из них обратимы на бесконечной решётке только 27 сдвигов'),
    ]
    body = ''.join(
        f'<tr><td>{a}</td><td class="num">{b}</td><td class="small muted">{d}</td></tr>'
        for a, b, d in rows)
    return ('<table><thead><tr><th>Уровень описания</th><th class="num">Число правил</th>'
            '<th>Комментарий</th></tr></thead><tbody>' + body + '</tbody></table>')


def seed_table(s: dict) -> str:
    parts = {k.replace('&ge;', '≥'): v for k, v in (s.get('by_rule') or s['categories']).items()}
    meaning = {
        'смерть': 'зародыш исчезает за 1–2 такта',
        'застыло': 'одна клетка выживает, но не растёт',
        'рост': 'рождается оболочка 26 клеток и далее растёт',
        'взрыв/игниция': 'вспыхивает весь вакуум (0 ∈ B)',
        'игниция вакуума (B=0)': 'вспыхивает весь вакуум: все пустые клетки рождаются',
        'оболочка 26 клеток (B=1)': 'первый такт рождает 26 соседей точки; далее — рост/застывание',
        'одна застывшая клетка (B≥2, S=0)': 'клетка выживает с нулём соседей и больше ничего не происходит',
        'смерть (остальные)': 'нет ни рождения при 1 соседе, ни выживания при 0',
    }
    body = ''.join(
        f'<tr><td>{e(k)}</td><td class="num">{v}</td><td class="small muted">{meaning.get(k,"")}</td></tr>'
        for k, v in parts.items())
    body += (f'<tr><td><b>всего</b></td><td class="num"><b>{s["n_rules"]}</b></td>'
             f'<td class="small muted">B = {{b}}, S = {{s}}, b,s &isin; 0..26</td></tr>')
    return ('<table><thead><tr><th>Исход</th><th class="num">Правил</th><th>Что происходит</th>'
            '</tr></thead><tbody>' + body + '</tbody></table>')


def sweep_table(s: dict) -> str:
    rows = []
    seen = set()
    uniq = []
    pool = sorted(s.get('results') or s['top'], key=lambda x: -x['richness'])
    groups = {}
    for r in pool:
        key = (r['op'], round(r['occupancy'], 3), round(r['richness'], 2))
        groups.setdefault(key, []).append(r)
    for i, (key, members) in enumerate(sorted(groups.items(), key=lambda kv: -kv[1][0]['richness'])[:10], 1):
        r = members[0]
        rows.append(
            f'<tr><td class="num">{i}</td><td><code>{e(r["name"])}</code></td>'
            f'<td class="num">{r["occupancy"]:.4f}</td><td class="num">{r["richness"]:.2f}</td>'
            f'<td class="num">{r["activity"]:.4f}</td><td class="num">{num(r["box_dim"],2)}</td>'
            f'<td class="num">{int(r["n_clusters"])}</td><td class="num">{len(members)}</td>'
            f'<td>{e(r["class"])}</td></tr>')
    head = ('<table><thead><tr><th class="num">#</th><th>Представитель</th><th class="num">занятость</th>'
            '<th class="num">богатство</th><th class="num">активность</th><th class="num">D</th>'
            '<th class="num">кластеров</th><th class="num">вариаций</th><th>класс</th></tr></thead><tbody>')
    return head + ''.join(rows) + '</tbody></table>'


def deep_table(d: dict) -> str:
    rows = []
    for name, rec in d.items():
        m = rec['metrics'][-1]
        rows.append(
            f'<tr><td><code>{e(name)}</code><br><span class="small muted">{e(rec["description"])}</span></td>'
            f'<td class="num">{m["occupancy"]:.4f}</td><td class="num">{m["entropy_norm"]:.3f}</td>'
            f'<td class="num">{m["compression"]:.3f}</td><td class="num">{num(m["box_dim"],2)}</td>'
            f'<td class="num">{int(m.get("n_clusters",0))}</td><td class="num">{m["activity"]:.4f}</td>'
            f'<td class="num">{m["bits_total"]:,}</td>'.replace(',', ' ') +
            f'<td class="num">{m["richness"]:.2f}</td><td>{e(m["class"])}</td></tr>')
    head = ('<table><thead><tr><th>Правило</th><th class="num">занятость</th><th class="num">H/H<sub>max</sub></th>'
            '<th class="num">сжатие</th><th class="num">D</th><th class="num">кластеров</th>'
            '<th class="num">активность</th><th class="num">биты</th><th class="num">богатство</th>'
            '<th>класс</th></tr></thead><tbody>')
    return head + ''.join(rows) + '</tbody></table>'


def inflation_table(inf: dict) -> str:
    ref = inf.get('exp', {}).get('state_hash')
    rows = []
    for kind, rec in inf.items():
        same = '—' if kind == 'exp' else ('да' if rec['state_hash'] == ref else 'нет')
        rows.append(
            f'<tr><td><code>{e(kind)}</code></td><td class="num">{rec["t_end"]:.1f}</td>'
            f'<td class="num">{rec["v0"]:.2f}c</td><td class="num">{rec["L_final"]}</td>'
            f'<td class="num">{rec["final_occ"]:.4f}</td>'
            f'<td>{"<span class=\"tag ok\">да</span>" if same=="да" else ("<span class=\"tag no\">нет</span>" if same=="нет" else "—")}</td></tr>')
    return ('<table><thead><tr><th>Закон</th><th class="num">t<sub>end</sub></th>'
            '<th class="num">v(0)</th><th class="num">L<sub>final</sub></th>'
            '<th class="num">занятость</th><th>состояние = exp</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def checks_table(ch: list) -> str:
    rows = []
    for c in ch:
        val = c['value']
        if isinstance(val, dict):
            val = ', '.join(f'{k}={v}' for k, v in val.items())
        rows.append(f'<tr><td>{e(c["name"])}</td>'
                    f'<td><span class="tag {"ok" if c["ok"] else "no"}">{"PASS" if c["ok"] else "FAIL"}</span></td>'
                    f'<td class="num small">{e(val)}</td></tr>')
    return ('<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')



def family_table() -> str:
    rows = [
        ('SPROUT', '3 направления (упоряд.), операция, поворот',
         '6&middot;5&middot;4 &middot; 3 &middot; 2 = 720', '720 (все)'),
        ('SPROUT-sym', 'все 6 направлений, парные AND', 'до 2<sup>6</sup> масок', '3'),
        ('HELIX', 'каноническая хиральная формула', '1', '1'),
        ('TOPPLE', 'порог k и подмножество 6 соседей', '25 &middot; 2<sup>6</sup> = 1600', '1'),
        ('CYCLIC', 'S = 3..8, 6 или 26 соседей', '12', '1'),
        ('TWIN', 'базовое правило F', 'столько же, сколько F', '1'),
        ('3D-Жизнь B/S', 'B, S &sub; {0..26}', '2<sup>54</sup>', '729 (B={b}, S={s})'),
    ]
    body = ''.join(f'<tr><td><b>{a}</b></td><td>{b}</td><td class="num">{c}</td>'
                   f'<td class="num">{d}</td></tr>' for a, b, c, d in rows)
    return ('<table><thead><tr><th>Семейство</th><th>Параметры</th>'
            '<th class="num">Возможных</th><th class="num">Проверено</th></tr></thead>'
            '<tbody>' + body + '</tbody></table>')

# ---------------------------------------------------------------- фигуры
def fig(path: str, caption: str, num_: str) -> str:
    return (f'<figure><img loading="lazy" src="{IMG}/{path}" alt="{e(caption)}">'
            f'<figcaption><b>Рис. {num_}.</b> {caption}</figcaption></figure>')


def gallery(d: dict) -> str:
    order = ['helix', 'sprout', 'sprout_or', 'topple', 'cyclic4', 'twin']
    cards = []
    for name in order:
        if name not in d:
            continue
        m = d[name]['metrics'][-1]
        occ_full = m.get('full_occupancy', m['occupancy'])
        cards.append(
            f'<figure><img loading="lazy" src="{IMG}/d_{name}_final.png" alt="{name}">'
            f'<figcaption><b>{name}</b> — {e(d[name]["description"])}. '
            f'Заполнение {100*occ_full:.3f}% полного куба, богатство {m["richness"]:.2f}. '
            f'<a href="{IMG}/d_{name}_rotate.gif">вращение</a> · '
            f'<a href="{IMG}/d_{name}.gif">рост</a></figcaption></figure>')
    return '<div class="gallery">' + ''.join(cards) + '</div>'


def viewers(d: dict) -> str:
    out = []
    for name in ('helix', 'sprout', 'cyclic4'):
        if name not in d:
            continue
        times = d[name]['frames']
        m = d[name]['metrics'][-1]
        cap = {'helix': 'HELIX: интерференционная решётка после замыкания тора',
               'sprout': 'SPROUT: ветвящийся рост из плотного зародыша',
               'cyclic4': 'CYCLIC4: фазовая волна внутри причинного конуса'}[name]
        out.append(
            f'<div class="viewer" data-base="{IMG}/frames/{name}" data-times="{",".join(map(str,times))}">'
            f'<img src="{IMG}/frames/{name}_{times[-1]:03d}.png" alt="{name}">'
            f'<div class="vrow"><button type="button">▶</button>'
            f'<input type="range" min="0" max="{len(times)-1}" value="{len(times)-1}">'
            f'<span class="tlabel">t = {times[-1]}</span></div>'
            f'<div class="small muted" style="padding:2px 4px 4px">{cap}. '
            f'L<sub>max</sub> = {d[name]["L_max"]}, кадров: {len(times)}.</div></div>')
    return ''.join(out)


def deep_figs(d: dict) -> str:
    cards = []
    for name in ('helix', 'sprout', 'cyclic4', 'topple'):
        if name not in d:
            continue
        cards.append(
            f'<figure><img loading="lazy" src="{IMG}/d_{name}_scales.png" alt="scales {name}">'
            f'<figcaption><b>{name}: масштабы.</b> Одна и та же структура, вырезанная '
            f'центральными кубами 1, 1/2, 1/4 и 1/8 стороны.</figcaption></figure>')
    cards.append(
        f'<figure><img loading="lazy" src="{IMG}/d_cyclic4_slices.png" alt="slices cyclic">'
        f'<figcaption><b>CYCLIC4: срезы.</b> 36 последовательных z-срезов; видно, что '
        f'«туман» на самом деле состоит из фазовых доменов.</figcaption></figure>')
    return '<div class="gallery">' + ''.join(cards) + '</div>'


def replace(tpl: str, key: str, val: str) -> str:
    return tpl.replace('{{' + key + '}}', val)


def main() -> None:
    counts = REPORT['counts']
    seed = REPORT.get('seed', {})
    sweep = REPORT.get('sweep', {})
    deep = REPORT['deep']
    inf = REPORT['inflation']
    checks = REPORT['checks']
    n_tested = seed.get('n_rules', 0) + sweep.get('n_rules', 0)

    out = TPL
    out = replace(out, 'COUNTS_TABLE', counts_table(counts))
    out = replace(out, 'FAMILY_TABLE', family_table())
    out = replace(out, 'SEED_TABLE', seed_table(seed) if seed else '<p class="muted">нет данных</p>')
    out = replace(out, 'SWEEP_TABLE', sweep_table(sweep) if sweep else '<p class="muted">нет данных</p>')
    out = replace(out, 'DEEP_TABLE', deep_table(deep))
    out = replace(out, 'INFLATION_TABLE', inflation_table(inf))
    out = replace(out, 'CHECKS_TABLE', checks_table(checks))
    out = replace(out, 'SEED_FIG', fig('f2_seed_map.png',
                 'Карта исходов для точечного зародыша: 729 правил B/S. '
                 'Ненулевой рост дают только B = 0 (игниция) и B = 1 (оболочка).', '6.1')
                 if (OUT / 'f2_seed_map.png').exists() else '')
    out = replace(out, 'SWEEP_FIG', fig('f3_sweep.png',
                 'Свип 720 правил SPROUT: занятость против индекса богатства. '
                 'Цвет — операция; подписи — лучшие пять правил.', '6.2')
                 if (OUT / 'f3_sweep.png').exists() else '')
    out = replace(out, 'SERIES_FIG', fig('f4_series.png',
                 'Временные ряды: занятость, богатство, число кластеров и суммарные биты '
                 'для шести глубоких прогонов.', '6.3'))
    out = replace(out, 'INFLATION_FIG', fig('f5_inflation.png',
                 'Слева — рост куба L(t) для пяти законов; справа — радиус фронта '
                 'структуры против светового конуса c·t.', '6.4'))
    out = replace(out, 'LIGHTCONE_FIG', fig('f9_lightcone.png',
                 'Радиус фронта структуры во времени. Все глубокие правила 6-соседние, '
                 'поэтому предел — c·t; для 26-соседних он был бы &radic;3·t.', '6.5'))
    out = replace(out, 'GALLERY', gallery(deep))
    out = replace(out, 'VIEWERS', viewers(deep))
    out = replace(out, 'DEEP_FIGS', deep_figs(deep))
    rot = OUT / 'd_helix_rotate.gif'
    closed = REPORT.get('closed')
    out = replace(out, 'CLOSED_FIG', (
        fig('d_closed_helix.png',
            'Закрытый контрольный куб: L = 97, T = 160. Фронт дошёл до стенки, '
            'куб замкнулся в тор — и HELIX дал плотную интерференционную решётку. '
            'В открытом режиме (321&sup3;) такой картины нет.', '6.3.1')
        if closed and (OUT / 'd_closed_helix.png').exists() else ''))
    out = replace(out, 'FRONT_FIG', (
        fig('f_front.png',
            'Фронт скорости света r = c·t и развитие автомата в центре. '
            'Четыре кадра CYCLIC4: t = 8, 16, 32, 64. Куб — текущий объём '
            'пространства; сфера — граница причинно связанной области; '
            'при t = 64 она ещё далека от стенок; фазовая волна заполняет причинный конус изнутри.', '6.5')
        if (OUT / 'f_front.png').exists() else ''))
    out = replace(out, 'ROTATE_FIG', (
        f'<figure><img loading="lazy" src="{IMG}/d_helix_rotate.gif" alt="вращение HELIX">'
        f'<figcaption><b>Вращение камеры.</b> 24 кадра по азимуту: одна и та же '
        f'конфигурация HELIX (L = 97, t = 160) с разных сторон. Видно, что это '
        f'объёмная решётка, а не плоская картинка.</figcaption></figure>')
        if rot.exists() else '')
    modes = 0
    if sweep.get('results'):
        modes = len({(r['op'], round(r['occupancy'], 3), round(r['richness'], 2))
                     for r in sweep['results']})
    out = replace(out, 'SWEEP_MODES', str(modes))
    zfigs = []
    for name in ('helix', 'sprout', 'cyclic4', 'topple'):
        f = OUT / f'd_{name}_zoom.png'
        if f.exists():
            zfigs.append(f'<figure><img loading="lazy" src="{IMG}/d_{name}_zoom.png" alt="zoom {name}">'
                         f'<figcaption><b>{name}, микроскоп 24&times;24&times;24.</b> '
                         f'Каждый кубик — одна ячейка; видны зазоры решётки и цвет по '
                         f'плотности информации.</figcaption></figure>')
    out = replace(out, 'ZOOM_GALLERY', '<div class="gallery">' + ''.join(zfigs) + '</div>')

    gen = counts['general_binary']['log10']
    out = replace(out, 'GEN_LOG', f'2<sup>2<sup>27</sup></sup> &asymp; 10<sup>{gen:,.0f}</sup>'.replace(',', ' '))
    out = replace(out, 'N_TESTED', f'{n_tested:,}'.replace(',', ' '))
    out = replace(out, 'N_DEEP', str(len(deep)))
    out = replace(out, 'T_MAX', str(max(d['T'] for d in deep.values())))
    out = replace(out, 'N_CHECKS', str(len(checks)))
    s = seed
    if s:
        cats = s['categories']
        parts = {k.replace('&ge;', '≥'): v for k, v in (s.get('by_rule') or s['categories']).items()}
        def pick(*keys):
            for k in keys:
                if k in parts:
                    return parts[k]
            return 0
        out = replace(out, 'SEED_DEAD', str(pick('смерть (остальные)', 'смерть')))
        out = replace(out, 'SEED_FROZEN', str(pick('одна застывшая клетка (B≥2, S=0)',
                                                   'одна застывшая клетка (B&ge;2, S=0)', 'застыло')))
        out = replace(out, 'SEED_GROWTH', str(pick('оболочка 26 клеток (B=1)', 'рост')))
        out = replace(out, 'SEED_IGNITE', str(pick('игниция вакуума (B=0)', 'взрыв/игниция')))
    out = replace(out, 'GENERATED', e(REPORT['meta']['generated']))
    out = replace(out, 'PYTHON', e(REPORT['meta']['python']))
    out = replace(out, 'NUMPY', e(REPORT['meta']['numpy']))

    missing = [k for k in ('{{',) if k in out]
    if '{{' in out:
        import re
        left = sorted(set(re.findall(r'\{\{(\w+)\}\}', out)))
        print('не заполнены:', left)
    (ROOT / 'report' / 'ca3d.html').write_text(out, encoding='utf-8')
    print(f'написано report/ca3d.html ({len(out)/1024:.0f} КБ)')


if __name__ == '__main__':
    main()
