import subprocess,time,json
from pathlib import Path
ROOT=Path('.').resolve();OUT=ROOT/'results'/'ca3d5_sweep';BIN=ROOT/'native'/'ca3d_nonrecip'
cfgs=[
 ('N1',1,2.0,0.0,-2.0,0.0,0.0,0.3,50,0.8),
 ('N2',1,2.0,0.0,-1.0,1.0,0.0,0.3,50,0.8),
 ('N3',0,1.0,1.0,-2.0,0.0,0.3,0.5,50,0.8),
 ('N4',0,2.0,-1.0,-2.0,0.0,0.0,0.5,50,0.8),
 ('N5',0,0.0,2.0,0.0,-2.0,0.0,0.3,50,0.8),
 ('N6',1,2.0,0.0,-2.0,0.0,0.3,0.8,50,0.8),
]
R={'runs':[]}
for tag,mode,ap,rp,an,rn,bias,noise,mob,frac in cfgs:
 t0=time.time();st=OUT/f'state_{tag}.bin';log=OUT/f'log_{tag}.txt'
 r=subprocess.run([str(BIN),'10000','1000000',str(frac),str(mode),str(ap),str(rp),str(an),str(rn),str(bias),str(noise),str(mob),str(st)],capture_output=True,text=True,timeout=1800)
 log.write_text(r.stdout+r.stderr,encoding='utf-8');cps={}
 for ln in (r.stdout+r.stderr).splitlines():
  if ln.startswith('cp t='):
   p=dict(x.split('=') for x in ln.split()[1:]);cps[int(p['t'])]=(int(p['active']),float(p['front']))
 fr=cps.get(10000,(0,0))[1]/max(cps.get(5000,(0,1))[1],1e-9)
 ar=cps.get(10000,(0,0))[0]/max(cps.get(5000,(0,1))[0],1)
 print(tag,'time',round(time.time()-t0,1),'cps',cps,'fr',round(fr,2),'ar',round(ar,2),flush=True)
 R['runs'].append({'tag':tag,'cps':cps,'front_ratio':round(fr,3),'active_ratio':round(ar,3)})
(OUT/'nonrecip.json').write_text(json.dumps(R,ensure_ascii=False,indent=1),encoding='utf-8')
