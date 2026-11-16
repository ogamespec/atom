// Sparse local cellular automaton, C version.
// State: hash table (x,y,z)->(p,n). Only active cells processed.
// Rule L03: attr=1, rep=1, bias=0.2 diag, noise=0.5, mob=20%.
// Space: coordinates in [-30000,30000], L=60001.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define L 60001
#define OFF 30000
#define CAP (1u<<26)          // 67M slots
#define MAXACT (20000000u)    // 20M active cells max
#define N_BITS 16000000LL

typedef struct { uint64_t key; uint32_t p, n, epoch; } Entry;

static Entry *cur, *nxt;
static uint32_t *act, *nact;
static size_t nact_n, nact_cap = MAXACT;
static uint64_t epoch = 1;

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}
static inline uint64_t key_of(int x, int y, int z) {
    return ((uint64_t)(x + OFF) << 32) | ((uint64_t)(y + OFF) << 16) | (uint64_t)(z + OFF);
}
static inline int dec_x(uint64_t k){ return (int)((k>>32)&0xFFFF) - OFF; }
static inline int dec_y(uint64_t k){ return (int)((k>>16)&0xFFFF) - OFF; }
static inline int dec_z(uint64_t k){ return (int)(k & 0xFFFF) - OFF; }

static inline int lookup(Entry *tab, uint32_t ep, uint64_t key, uint32_t *p, uint32_t *n) {
    uint64_t h = mix64(key) & (CAP - 1);
    while (tab[h].epoch == ep) {
        if (tab[h].key == key) { *p = tab[h].p; *n = tab[h].n; return 1; }
        h = (h + 1) & (CAP - 1);
    }
    return 0;
}
static inline void add(Entry *tab, uint32_t ep, uint64_t key,
                       int64_t p, int64_t n,
                       uint32_t *list, size_t *ln) {
    uint64_t h = mix64(key) & (CAP - 1);
    while (tab[h].epoch == ep && tab[h].key != key) h = (h + 1) & (CAP - 1);
    if (tab[h].epoch != ep) {
        tab[h].epoch = ep; tab[h].key = key; tab[h].p = 0; tab[h].n = 0;
        if (*ln >= MAXACT) { fprintf(stderr, "active overflow\n"); exit(2); }
        list[(*ln)++] = (uint32_t)h;
    }
    tab[h].p += (uint32_t)p; tab[h].n += (uint32_t)n;
}

static inline int rnd3(uint64_t key, uint32_t t, int d) {
    uint64_t h = key ^ ((uint64_t)t * 0x9E3779B97F4A7C15ULL)
               ^ ((uint64_t)(d + 1) * 0xBF58476D1CE4E5B9ULL);
    h ^= h >> 30; h *= 0x94D049BB133111EBULL; h ^= h >> 27; h *= 0x9E3779B97F4A7C15ULL;
    return (int)(h % 3) - 1;
}

static void checkpoint(const char *tag, uint32_t t) {
    uint64_t bits = 0; double front = 0;
    for (size_t i = 0; i < nact_n; i++) {
        Entry *e = &cur[act[i]];
        bits += (uint64_t)e->p + e->n;
        int x = dec_x(e->key), y = dec_y(e->key), z = dec_z(e->key);
        double r = sqrt((double)x*x + (double)y*y + (double)z*z);
        if (r > front) front = r;
    }
    printf("%s t=%u active=%zu bits=%llu front=%.2f\n", tag, t, nact_n,
           (unsigned long long)bits, front);
    fflush(stdout);
}

int main(int argc, char **argv) {
    uint32_t T = argc > 1 ? (uint32_t)atoi(argv[1]) : 20000;
    double frac = argc > 2 ? atof(argv[2]) : 0.8;
    const char *out = argc > 3 ? argv[3] : "state.bin";
    cur = (Entry*)calloc(CAP, sizeof(Entry));
    nxt = (Entry*)calloc(CAP, sizeof(Entry));
    act = (uint32_t*)malloc((size_t)MAXACT * sizeof(uint32_t));
    nact = (uint32_t*)malloc((size_t)MAXACT * sizeof(uint32_t));
    if (!cur || !nxt || !act || !nact) { fprintf(stderr, "alloc failed\n"); return 1; }
    int64_t P = (int64_t)llround(N_BITS * frac);
    int64_t Nn = N_BITS - P;
    uint64_t k0 = key_of(0, 0, 0);
    uint64_t h = mix64(k0) & (CAP - 1);
    cur[h].epoch = 1; cur[h].key = k0; cur[h].p = (uint32_t)P; cur[h].n = (uint32_t)Nn;
    act[0] = (uint32_t)h; nact_n = 1;
    printf("start T=%u N=%lld frac=%.3f P=%lld\n", T, (long long)N_BITS, frac, (long long)P);
    fflush(stdout);
    const int B[6] = {1, -1, 1, -1, 1, -1};
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};
    time_t t0 = time(NULL);
    for (uint32_t t = 1; t <= T; t++) {
        uint32_t nep = (uint32_t)(epoch + 1);
        size_t nn = 0;
        for (size_t ai = 0; ai < nact_n; ai++) {
            Entry *e = &cur[act[ai]];
            int x = dec_x(e->key), y = dec_y(e->key), z = dec_z(e->key);
            int64_t p = e->p, n = e->n;
            int64_t wp[7], wn[7];
            int64_t swp = 0, swn = 0;
            for (int d = 0; d < 6; d++) {
                int qx = x + dx[d], qy = y + dy[d], qz = z + dz[d];
                uint32_t pn = 0, nnv = 0;
                lookup(cur, (uint32_t)epoch, key_of(qx, qy, qz), &pn, &nnv);
                int r = rnd3(e->key, t, d);
                int64_t wpp = 1000
                    + (3000 * ((int64_t)nnv - pn)) / N_BITS
                    + (3000 * (p - (int64_t)pn)) / N_BITS
                    + 400 * B[d] + 500 * r;
                int64_t wnn = 1000
                    - (3000 * ((int64_t)nnv - pn)) / N_BITS
                    + (3000 * (n - (int64_t)nnv)) / N_BITS
                    - 400 * B[d] + 500 * r;
                if (wpp < 1) wpp = 1;
                if (wnn < 1) wnn = 1;
                wp[d] = wpp; wn[d] = wnn;
                swp += wpp; swn += wnn;
            }
            wp[6] = 1000; wn[6] = 1000;
            swp += 1000; swn += 1000;
            int64_t sp = 0, sn = 0;
            int64_t shp[6], shn[6];
            for (int d = 0; d < 6; d++) {
                shp[d] = (p * wp[d]) / swp;
                shn[d] = (n * wn[d]) / swn;
                shp[d] = shp[d] * 20 / 100;
                shn[d] = shn[d] * 20 / 100;
                sp += shp[d]; sn += shn[d];
            }
            int64_t kp = p - sp, kn = n - sn;
            if (kp || kn) add(nxt, nep, e->key, kp, kn, nact, &nn);
            for (int d = 0; d < 6; d++) {
                if (shp[d] || shn[d]) {
                    uint64_t qk = key_of(x + dx[d], y + dy[d], z + dz[d]);
                    add(nxt, nep, qk, shp[d], shn[d], nact, &nn);
                }
            }
        }
        Entry *tmp = cur; cur = nxt; nxt = tmp;
        uint32_t *ta = act; act = nact; nact = ta;
        nact_n = nn; epoch = nep;
        if (t % 1000 == 0) {
            checkpoint("cp", t);
            fprintf(stderr, "elapsed %ld s\n", (long)(time(NULL)-t0));
        }
    }
    checkpoint("final", T);
    // dump
    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }
    uint64_t cnt = nact_n; fwrite(&cnt, sizeof(cnt), 1, f);
    for (size_t i = 0; i < nact_n; i++) {
        Entry *e = &cur[act[i]];
        int32_t x = dec_x(e->key), y = dec_y(e->key), z = dec_z(e->key);
        uint32_t p = e->p, n = e->n;
        fwrite(&x, 4, 1, f); fwrite(&y, 4, 1, f); fwrite(&z, 4, 1, f);
        fwrite(&p, 4, 1, f); fwrite(&n, 4, 1, f);
    }
    fclose(f);
    printf("wrote %s count=%llu\n", out, (unsigned long long)cnt);
    return 0;
}
