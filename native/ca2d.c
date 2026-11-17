#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define L 20001
#define OFF 10000
#define CAP (1u<<21)
#define LCAP (1u<<18)
#define NTH 8
#define LMAX 500000u
#define MAXACT 1000000u
typedef struct { uint32_t key; uint32_t p,n,epoch; } Entry;
static Entry *cur,*nxt,*lt[NTH];
static uint32_t *ll[NTH],*act,*nact;
static size_t lc[NTH],nact_n;
static uint32_t epoch=1;
static int64_t NB; static double AP,RP,AN,RN,BIAS,NOISE; static int MOB,MODE;
static inline uint32_t mix32(uint32_t x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
static inline uint32_t key_of(int x,int y){return ((uint32_t)(x+OFF)<<16)|(uint32_t)(y+OFF);}
static inline int dec_x(uint32_t k){return (int)((k>>16)&0xFFFF)-OFF;}
static inline int dec_y(uint32_t k){return (int)(k&0xFFFF)-OFF;}
static inline int lookup(Entry*t,uint32_t ep,uint32_t key,uint32_t*p,uint32_t*n){uint32_t h=mix32(key)&(CAP-1);while(t[h].epoch==ep){if(t[h].key==key){*p=t[h].p;*n=t[h].n;return 1;}h=(h+1)&(CAP-1);}return 0;}
static inline void add(Entry*t,uint32_t cap,uint32_t ep,uint32_t key,int64_t p,int64_t n,uint32_t*list,size_t*ln,size_t maxn){uint32_t h=mix32(key)&(cap-1);while(t[h].epoch==ep&&t[h].key!=key)h=(h+1)&(cap-1);if(t[h].epoch!=ep){t[h].epoch=ep;t[h].key=key;t[h].p=0;t[h].n=0;if(*ln>=maxn){fprintf(stderr,"overflow\n");exit(2);}list[(*ln)++]=h;}t[h].p+=(uint32_t)p;t[h].n+=(uint32_t)n;}
static inline int rnd3(uint32_t key,uint32_t t,int d){uint32_t h=key^(t*0x9E3779B9U)^((uint32_t)(d+1)*0x85EBCA6BU);h^=h>>16;h*=0x7feb352dU;h^=h>>15;return (int)(h%3)-1;}
static void checkpoint(const char*tag,uint32_t t){uint64_t bits=0;double front=0;for(size_t i=0;i<nact_n;i++){Entry*e=&cur[act[i]];bits+=(uint64_t)e->p+e->n;int x=dec_x(e->key),y=dec_y(e->key);double r=sqrt((double)x*x+(double)y*y);if(r>front)front=r;}printf("%s t=%u active=%zu bits=%llu front=%.2f\n",tag,t,nact_n,(unsigned long long)bits,front);fflush(stdout);}
int main(int argc,char**argv){
 uint32_t T=atoi(argv[1]);NB=atoll(argv[2]);double frac=atof(argv[3]);MODE=atoi(argv[4]);AP=atof(argv[5]);RP=atof(argv[6]);AN=atof(argv[7]);RN=atof(argv[8]);BIAS=atof(argv[9]);NOISE=atof(argv[10]);MOB=atoi(argv[11]);const char*out=argv[12];
 cur=calloc(CAP,sizeof(Entry));nxt=calloc(CAP,sizeof(Entry));act=malloc((size_t)MAXACT*4);nact=malloc((size_t)MAXACT*4);
 for(int i=0;i<NTH;i++){lt[i]=calloc(LCAP,sizeof(Entry));ll[i]=malloc((size_t)LMAX*4);}
 int64_t P=llround(NB*frac),Nn=NB-P;uint32_t k0=key_of(0,0),h=mix32(k0)&(CAP-1);cur[h].epoch=1;cur[h].key=k0;cur[h].p=P;cur[h].n=Nn;act[0]=h;nact_n=1;
 printf("start T=%u N=%lld frac=%.3f mode=%d AP=%.2f RP=%.2f AN=%.2f RN=%.2f bias=%.2f noise=%.2f mob=%d\n",T,(long long)NB,frac,MODE,AP,RP,AN,RN,BIAS,NOISE,MOB);fflush(stdout);
 const int Bx[4]={1,-1,0,0},Bd[4]={1,-1,1,-1};const int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
 for(uint32_t t=1;t<=T;t++){
  uint32_t nep=epoch+1;size_t nn=0;
  #pragma omp parallel num_threads(NTH)
  {
   int tid=omp_get_thread_num();size_t ln=0;
   #pragma omp for schedule(static)
   for(size_t ai=0;ai<nact_n;ai++){
    Entry*e=&cur[act[ai]];int x=dec_x(e->key),y=dec_y(e->key);int64_t p=e->p,n=e->n;int64_t wp[5],wn[5],swp=0,swn=0;
    for(int d=0;d<4;d++){
     uint32_t pn=0,nnv=0;lookup(cur,epoch,key_of(x+dx[d],y+dy[d]),&pn,&nnv);int rp=rnd3(e->key,t,d);int rn=rnd3(e->key,t,d+8);int bd=MODE?Bd[d]:Bx[d];
     int64_t wpp=1000+(int64_t)(3000.0*AP*((int64_t)nnv-pn)/NB)+(int64_t)(3000.0*RP*(p-(int64_t)pn)/NB)+(int64_t)(2000.0*BIAS*bd)+(int64_t)(1000.0*NOISE*rp);
     int64_t wnn=1000-(int64_t)(3000.0*AN*((int64_t)nnv-pn)/NB)+(int64_t)(3000.0*RN*(n-(int64_t)nnv)/NB)-(int64_t)(2000.0*BIAS*bd)+(int64_t)(1000.0*NOISE*rn);
     if(wpp<1)wpp=1;if(wnn<1)wnn=1;wp[d]=wpp;wn[d]=wnn;swp+=wpp;swn+=wnn;}
    wp[4]=1000;wn[4]=1000;swp+=1000;swn+=1000;
    int64_t sp=0,sn=0,shp[4],shn[4];
    for(int d=0;d<4;d++){shp[d]=(p*wp[d])/swp;shn[d]=(n*wn[d])/swn;shp[d]=shp[d]*MOB/100;shn[d]=shn[d]*MOB/100;sp+=shp[d];sn+=shn[d];}
    int64_t kp=p-sp,kn=n-sn;
    if(kp||kn)add(lt[tid],LCAP,nep,e->key,kp,kn,ll[tid],&ln,LMAX);
    for(int d=0;d<4;d++)if(shp[d]||shn[d])add(lt[tid],LCAP,nep,key_of(x+dx[d],y+dy[d]),shp[d],shn[d],ll[tid],&ln,LMAX);
   }
   lc[tid]=ln;
  }
  for(int tid=0;tid<NTH;tid++)for(size_t i=0;i<lc[tid];i++){Entry*e=&lt[tid][ll[tid][i]];add(nxt,CAP,nep,e->key,e->p,e->n,nact,&nn,MAXACT);}
  Entry*tmp=cur;cur=nxt;nxt=tmp;uint32_t*ta=act;act=nact;nact=ta;nact_n=nn;epoch=nep;
  if(t%2500==0){checkpoint("cp",t);}
 }
 checkpoint("final",T);
 FILE*f=fopen(out,"wb");uint64_t cnt=nact_n;fwrite(&cnt,8,1,f);
 for(size_t i=0;i<nact_n;i++){Entry*e=&cur[act[i]];int32_t x=dec_x(e->key),y=dec_y(e->key);uint32_t p=e->p,n=e->n;fwrite(&x,4,1,f);fwrite(&y,4,1,f);fwrite(&p,4,1,f);fwrite(&n,4,1,f);}
 fclose(f);printf("wrote %s count=%llu\n",out,(unsigned long long)cnt);return 0;}
