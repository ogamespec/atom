import numpy as np, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib as mpl
OUT=Path('results/ca2d_gif'); T=20000; A=4.0; canvas=640; font=ImageFont.load_default()
def comoving_frame(p, box=256, r=170):
 d=Path(p).read_bytes();cnt=int(np.frombuffer(d[:8],np.uint64)[0]);rec=np.frombuffer(d[8:],np.int32).reshape(-1,4)
 c=rec[:,:2].astype(float);p_=rec[:,2].astype(np.int64);n_=rec[:,3].astype(np.int64)
 ch=np.zeros((box,box));q=np.clip(((c+r)/(2*r)*box).astype(int),0,box-1);ch[q[:,0],q[:,1]]=p_-n_
 sm=ndimage.uniform_filter(ch,3);vmax=max(np.abs(sm).max(),1);t=np.clip(0.5+0.5*sm/vmax,0,1)
 rgb=(mpl.colormaps['coolwarm'](t)[:,:,:3]*255).astype(np.uint8);rgb[ch==0]=0
 return Image.fromarray(rgb)
def pulse(a, com):
 im=Image.new('RGB',(canvas,canvas),(5,8,13)); dr=ImageDraw.Draw(im)
 # пространство: круг радиуса a*80
 R=int(80*a); cc=canvas//2
 dr.ellipse([cc-R,cc-R,cc+R,cc+R],outline=(90,180,230),width=2)
 # структура: com 256 -> 160*a
 sz=max(2,int(160*a)); st=com.resize((sz,sz),Image.NEAREST)
 im.paste(st,(cc-sz//2,cc-sz//2))
 dr.text((10,10),f'a = {a:.2f}',fill=(235,242,250),font=font)
 return im
frames=[]; mont=[]
for k,t in enumerate(range(1000,T+1,1000)):
 a=A*math.sin(math.pi*t/T)
 com=comoving_frame(OUT/f'p1_pulse.bin.f{t:07d}.bin')
 im=pulse(a,com); frames.append(im)
 if t in (2000,5000,10000,15000,19000,20000): mont.append(im)
# финальная точка
pt=Image.new('RGB',(canvas,canvas),(5,8,13)); d=ImageDraw.Draw(pt); d.ellipse([canvas//2-3,canvas//2-3,canvas//2+3,canvas//2+3],fill=(255,255,255)); d.text((10,10),'a = 0.00',fill=(235,242,250),font=font)
frames.append(pt)
frames[0].save(OUT/'p1_pulse.gif',save_all=True,append_images=frames[1:],duration=350,loop=0)
m=Image.new('RGB',(canvas*3,canvas*2),(0,0,0))
for i,im in enumerate(mont): m.paste(im,((i%3)*canvas,(i//3)*canvas))
m.save(OUT/'p1_pulse_montage.png')
print('saved',len(frames))
