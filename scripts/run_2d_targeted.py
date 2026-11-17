import subprocess,time,json
from pathlib import Path
ROOT=Path('.').resolve();OUT=ROOT/'results'/'ca2d_sweep';BIN=ROOT/'native'/'ca2d'
cfgs=[('P1',0,-2,-2,-2,-2,0,0.8,50,0.5),
      ('P2',0,-2,0,-2,0,0,0.8,50,0.5),
      ('P3',0,2,0,-2,0,0,0.8,50,0.5),
      ('P4',0,0,2,0,2,0,0.8,50,0.5),
      ('P5',0,-1,-1,1,1,0,0.5,50,0.5),
      ('P6',1,1,1,1,1,0.2,0.5,20,0.5)]
for tag,mode,ap,rp,an,rn,bias,noise,mob,frac in cfgs:
 t0=time.time();st=OUT/f'state_{tag}.bin';log=OUT/f'log_{tag}.txt'
 r=subprocess.run([str(BIN),'10000','1000000',str(frac),str(mode),str(ap),str(rp),str(an),str(rn),str(bias),str(noise),str(mob),str(st)],capture_output=True,text=True,timeout=900)
 log.write_text(r.stdout+r.stderr,encoding='utf-8');cps={}
 for ln in (r.stdout+r.stderr).splitlines():
  if ln.startswith('cp t='):
   p=dict(x.split('=') for x in ln.split()[1:]);cps[int(p['t'])]=(int(p['active']),float(p['front']))
 print(tag,'time',round(time.time()-t0,1),'cps',cps,flush=True)
