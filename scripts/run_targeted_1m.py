import json,subprocess,time,sys
from pathlib import Path
import numpy as np
ROOT=Path('.').resolve(); OUT=ROOT/'results'/'ca3d5_sweep'
BIN=ROOT/'native'/'ca3d_sweep'
T=10000;N=1_000_000
cfgs=[
 ('E1',0,0.0,5.0,0.0,0.0,100,0.5),
 ('E2',1,0.0,5.0,0.0,0.0,100,0.5),
 ('E3',0,-2.0,-2.0,0.0,0.5,50,0.5),
 ('E4',0,-2.0,2.0,0.0,0.5,50,0.5),
 ('E5',0,0.0,2.0,0.5,0.8,100,0.8),
 ('E6',0,0.0,0.0,0.5,0.0,100,0.5),
 ('E7',0,0.0,0.0,0.0,0.8,100,0.5),
]
for tag,mode,a,r,b,no,mob,f in cfgs:
 t0=time.time(); st=OUT/f'state_{tag}.bin'; log=OUT/f'log_{tag}.txt'
 r_=subprocess.run([str(BIN),str(T),str(N),str(f),str(mode),str(a),str(r),str(b),str(no),str(mob),str(st)],capture_output=True,text=True,timeout=1200)
 log.write_text(r_.stdout+r_.stderr,encoding='utf-8')
 cps={}
 for ln in (r_.stdout+r_.stderr).splitlines():
  if ln.startswith('cp t='):
   p=dict(x.split('=') for x in ln.split()[1:]);cps[int(p['t'])]=(int(p['active']),float(p['front']))
 print(tag, 'time',round(time.time()-t0,1),'cps',cps,flush=True)
