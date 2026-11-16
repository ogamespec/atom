#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define L 60001
#define OFF 30000
#define CAP (1u<<25)
#define LCAP (1u<<22)
#define NTH 8
#define LMAX 4000000u
#define MAXACT 20000000u
#define N_BITS 16000000LL
typedef struct { uint64_t key; uint32_t p,n,epoch; } Entry;
static Entry *cur,*nxt,*lt[NTH];
static uint32_t *ll[NTH],*act,*nact;
static uint32_t lep[NTH];
static size_t lc[NTH],nact_n;
static uint64_t epoch=1;
static inline uint64_t mix64(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;return x;}
static inline uint64_t key_of(int x,int y,int z){return ((uint64_t)(x+OFF)<<32)|((uint64_t)(y+OFF)<<16)|(uint64_t)(z+OFF);}
static inline int dec_x(uint64_t k){return (int)((k>>32)&0xFFFF)-OFF;}
static inline int dec_y(uint64_t k){return (int)((k>>16)&0xFFFF)-OFF;}
static inline int dec_z(uint64_t k){return (int)(k&0xFFFF)-OFF;}
static inline int lookup(Entry*t,uint64_t cap,uint32_t ep,uint64_t key,uint32_t*p,uint32_t*n){
 uint64_t h=mix64(key)&(cap-1); while(t[h].epoch==ep){if(t[h].key==key){*p=t[h].p;*n=t[h].n;return 1;}h=(h+1)&(cap-1);}return 0;}
static inline void add(Entry*t,uint64_t cap,uint32_t ep,uint64_t key,int64_t p,int64_t n,uint32_t*list,size_t*ln,size_t maxn){
 uint64_t h=mix64(key)&(cap-1); while(t[h].epoch==ep&&t[h].key!=key)h=(h+1)&(cap-1);
 if(t[h].epoch!=ep){t[h].epoch=ep;t[h].key=key;t[h].p=0;t[h].n=0;if(*ln>=maxn){fprintf(stderr,"list overflow %zu\n",maxn);exit(2);}list[(*ln)++]=h;}
 t[h].p+=(uint32_t)p;t[h].n+=(uint32_t)n;}
static inline int rnd3(uint64_t key,uint32_t t,int d){uint64_t h=key^((uint64_t)t*0x9E3779B97F4A7C15ULL)^((uint64_t)(d+1)*0xBF58476D1CE4E5B9ULL);h^=h>>30;h*=0x94D049BB133111EBULL;h^=h>>27;h*=0x9E3779B97F4A7C15ULL;return (int)(h%3)-1;}
static void checkpoint(const char*tag,uint32_t t){uint64_t bits=0;double front=0;
 for(size_t i=0;i<nact_n;i++){Entry*e=&cur[act[i]];bits+=(uint64_t)e->p+e->n;int x=dec_x(e->key),y=dec_y(e->key),z=dec_z(e->key);double r=sqrt((double)x*x+(double)y*y+(double)z*z);if(r>front)front=r;}
 printf("%s t=%u active=%zu bits=%llu front=%.2f\n",tag,t,nact_n,(unsigned long long)bits,front);fflush(stdout);}
int main(int argc,char**argv){
 uint32_t T=argc>1?(uint32_t)atoi(argv[1]):20000;double frac=argc>2?atof(argv[2]):0.8;const char*out=argc>3?argv[3]:"state.bin";
 cur=calloc(CAP,sizeof(Entry));nxt=calloc(CAP,sizeof(Entry));act=malloc((size_t)MAXACT*4);nact=malloc((size_t)MAXACT*4);
 for(int i=0;i<NTH;i++){lt[i]=calloc(LCAP,sizeof(Entry));ll[i]=malloc((size_t)LMAX*4);}
 if(!cur||!nxt||!act||!nact)return 1;
 int64_t P=llround(N_BITS*frac),Nn=N_BITS-P;uint64_t k0=key_of(0,0,0),h=mix64(k0)&(CAP-1);
 cur[h].epoch=1;cur[h].key=k0;cur[h].p=(uint32_t)P;cur[h].n=(uint32_t)Nn;act[0]=(uint32_t)h;nact_n=1;
 printf("start T=%u N=%lld frac=%.3f threads=%d\n",T,(long long)N_BITS,frac,omp_get_max_threads());fflush(stdout);
 const int B[6]={1,-1,1,-1,1,-1};const int dx[6]={1,-1,0,0,0,0},dy[6]={0,0,1,-1,0,0},dz[6]={0,0,0,0,1,-1};
 time_t t0=time(NULL);
 for(uint32_t t=1;t<=T;t++){
  uint32_t nep=(uint32_t)(epoch+1);size_t nn=0;
  #pragma omp parallel num_threads(NTH)
  {
   int tid=omp_get_thread_num();uint32_t le=nep;lep[tid]=le;size_t ln=0;
   #pragma omp for schedule(static)
   for(size_t ai=0;ai<nact_n;ai++){
    Entry*e=&cur[act[ai]];int x=dec_x(e->key),y=dec_y(e->key),z=dec_z(e->key);int64_t p=e->p,n=e->n;
    int64_t wp[7],wn[7],swp=0,swn=0;
    for(int d=0;d<6;d++){
     uint32_t pn=0,nnv=0;lookup(cur,CAP,(uint32_t)epoch,key_of(x+dx[d],y+dy[d],z+dz[d]),&pn,&nnv);int r=rnd3(e->key,t,d);
     int64_t wpp=1000+(3000*((int64_t)nnv-pn))/N_BITS+(3000*(p-(int64_t)pn))/N_BITS+400*B[d]+500*r;
     int64_t wnn=1000-(3000*((int64_t)nnv-pn))/N_BITS+(3000*(n-(int64_t)nnv))/N_BITS-400*B[d]+500*r;
     if(wpp<1)wpp=1;if(wnn<1)wnn=1;wp[d]=wpp;wn[d]=wnn;swp+=wpp;swn+=wnn;}
    wp[6]=1000;wn[6]=1000;swp+=1000;swn+=1000;
    int64_t sp=0,sn=0,shp[6],shn[6];
    for(int d=0;d<6;d++){shp[d]=(p*wp[d])/swp;shn[d]=(n*wn[d])/swn;shp[d]=shp[d]*20/100;shn[d]=shn[d]*20/100;sp+=shp[d];sn+=shn[d];}
    int64_t kp=p-sp,kn=n-sn;
    if(kp||kn)add(lt[tid],LCAP,le,e->key,kp,kn,ll[tid],&ln,LMAX);
    for(int d=0;d<6;d++)if(shp[d]||shn[d])add(lt[tid],LCAP,le,key_of(x+dx[d],y+dy[d],z+dz[d]),shp[d],shn[d],ll[tid],&ln,LMAX);
   }
   lc[tid]=ln;
  }
  for(int tid=0;tid<NTH;tid++)for(size_t i=0;i<lc[tid];i++){Entry*e=&lt[tid][ll[tid][i]];add(nxt,CAP,nep,e->key,e->p,e->n,nact,&nn,MAXACT);}
  Entry*tmp=cur;cur=nxt;nxt=tmp;uint32_t*ta=act;act=nact;nact=ta;nact_n=nn;epoch=nep;
  if(t%1000==0){checkpoint("cp",t);fprintf(stderr,"elapsed %ld s\n",(long)(time(NULL)-t0));}
 }
 checkpoint("final",T);
 FILE*f=fopen(out,"wb");if(!f)return 1;uint64_t cnt=nact_n;fwrite(&cnt,8,1,f);
 for(size_t i=0;i<nact_n;i++){Entry*e=&cur[act[i]];int32_t x=dec_x(e->key),y=dec_y(e->key),z=dec_z(e->key);uint32_t p=e->p,n=e->n;fwrite(&x,4,1,f);fwrite(&y,4,1,f);fwrite(&z,4,1,f);fwrite(&p,4,1,f);fwrite(&n,4,1,f);}
 fclose(f);printf("wrote %s count=%llu\n",out,(unsigned long long)cnt);return 0;}
