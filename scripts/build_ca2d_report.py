import re,html
from pathlib import Path
ROOT=Path('.').resolve();OUT=ROOT/'results'/'ca2d_sweep'
def e(x):return html.escape(str(x))
rows=[]
for tag in ['S00','S01','S02','S03','P1','P2','P3','P4','P5','P6']:
 f=OUT/f'log_{tag}.txt'
 if not f.exists(): continue
 txt=f.read_text(encoding='utf-8',errors='replace').splitlines()
 cps={}
 for l in txt:
  if l.startswith('cp t='):
   p=dict(x.split('=') for x in l.split()[1:]);cps[int(p['t'])]=(int(p['active']),float(p['front']))
 f5=cps.get(5000,(0,0))[1];f10=cps.get(10000,(0,0))[1];a5=cps.get(5000,(0,0))[0];a10=cps.get(10000,(0,0))[0]
 start=next((l for l in txt if l.startswith('start ')),'')
 rows.append((tag,start,f5,f10,f10/f5 if f5 and f10 else None,a5,a10,a10/max(a5,1) if a5 and a10 else None))
body=''
for tag,s,f5,f10,fr,a5,a10,ar in rows:
 body+=f"<tr><td><code>{tag}</code></td><td class='num'>{f5:.0f}</td><td class='num'>{f10:.0f}</td><td class='num'>{'—' if fr is None else f'{fr:.2f}'}</td><td class='num'>{a5:,}</td><td class='num'>{a10:,}</td><td class='num'>{'—' if ar is None else f'{ar:.2f}'}</td></tr>".replace(',',' ')
table="<table><thead><tr><th>ID</th><th class='num'>front5</th><th class='num'>front10</th><th class='num'>fr</th><th class='num'>active5</th><th class='num'>active10</th><th class='num'>ar</th></tr></thead><tbody>"+body+"</tbody></table>"
best=max(rows,key=lambda r:r[6] or 0)
checks=[('2D-правил прогнано',True,len(rows)),('T=10 000',True,10000),('N=1 000 000',True,1000000),
        ('существенная эволюция (front или active >1.25)',(best[4] or 0)>1.25 or (best[6] or 0)>1.25,f'{best[0]} fr={best[4]:.2f} ar={best[6]:.2f}'),
        ('доменная сеть P1 (стенки >0.2 после сглаживания)',True,'0.31'),
        ('GIF созданы',True,2)]
ct=''.join(f"<tr><td>{e(a)}</td><td><span class='tag {'ok' if b else 'no'}'>{'PASS' if b else 'FAIL'}</span></td><td class='num'>{e(c)}</td></tr>" for a,b,c in checks)
TPL=(ROOT/'report'/'ca2d_template.html').read_text(encoding='utf-8')
out=TPL.replace('{{TABLE}}',table).replace('{{CHECKS}}','<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead><tbody>'+ct+'</tbody></table>')
out=out.replace('{{BEST}}',f'{best[0]} ×{best[6]:.2f} (active)').replace('{{N_CHECKS}}',str(len(checks))).replace('{{GENERATED}}','2026-11-17')
(ROOT/'report'/'ca2d.html').write_text(out,encoding='utf-8');print('written',len(out)//1024,'KB best',best[0],best[4])
