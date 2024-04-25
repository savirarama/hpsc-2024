#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], fxt[N], fyt[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    fxt[i] = fyt[i] = 0;
  }
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
      float ai = i;
      float bj = j;
      __m512 avec = _mm512_set1_ps(ai);
      __m512 bvec = _mm512_set1_ps(bj);
      __m512 mvec = _mm512_load_ps(m);
      __mmask16 mask = _mm512_cmpeq_ps_mask(avec, bvec);
      __m512 zeros = _mm512_set1_ps(0);
      __m512 xivec = _mm512_set1_ps(x[i]);
      __m512 xjvec = _mm512_set1_ps(x[j]);
      __m512 yivec = _mm512_set1_ps(y[i]);
      __m512 yjvec = _mm512_set1_ps(y[j]);
      __m512 mjvec = _mm512_set1_ps(m[j]);
      
      //Calculate rx = x[i] - x[j]
      __m512 rx = _mm512_sub_ps(xivec,xjvec);

      //Calculate ry = y[i] - y[j]
      __m512 ry = _mm512_sub_ps(yivec,yjvec);
      //Calculate (rx * rx + ry * ry)
      __m512 rxsq = _mm512_mul_ps(rx,rx);
      __m512 rysq = _mm512_mul_ps(ry,ry);
      __m512 rsq = _mm512_add_ps(rxsq,rysq);
    
     //Calculate reciprocal std::sqrt(rx * rx + ry * ry)
      __m512 r_recp = _mm512_rsqrt14_ps(rsq);
      //Calculate (r * r * r)^-1
      __m512 r_recp_cb = _mm512_mul_ps(_mm512_mul_ps(r_recp,r_recp),r_recp); 
      //Calculate rx * m[j] / (r * r * r) with mask
      __m512 fxtvec = _mm512_mask_blend_ps(mask,_mm512_mul_ps(_mm512_mul_ps(rx,mjvec),r_recp_cb),zeros);
      //Calculate ry * m[j] / (r * r * r) with mask
      __m512 fytvec = _mm512_mask_blend_ps(mask,_mm512_mul_ps(_mm512_mul_ps(ry,mjvec),r_recp_cb),zeros);
      _mm512_store_ps(fxt,fxtvec);
      _mm512_store_ps(fyt,fytvec);
      fx[i]-=fxt[j];
      fy[i]-=fyt[j];
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);

  }
}
