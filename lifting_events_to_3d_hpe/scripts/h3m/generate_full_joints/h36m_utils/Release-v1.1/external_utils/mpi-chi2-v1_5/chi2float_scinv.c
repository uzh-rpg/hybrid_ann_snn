// fast chi-squared distance function in x86 compiler intrinsics
// (C) 2007-2008 Christoph Lampert <christoph.lampert@gmail.com>

#include <stdio.h>
#include <values.h>	// for FLT_MIN
/* We calculate calculate chi2=(a-b)**2/(a+b+FLT_MIN) to avoid division-by-zero:
   If a+b != 0, then (a+b+FLT_MIN)==(a+b) and nothing changed.
   If a+b == 0, then the numerator is 0 as well, and we don't divide by 0. 
*/


/* Using SSE compiler intrinsics can have a huge speedup effect: 
   8x for float and 3.5x for double on Intel Core2.
   You have to compile with the right CPU setting, e.g. gcc -march=k8 or -march=nocona */
#ifdef __SSE__
#include <xmmintrin.h> // for float
#endif

/* OpenMP allows to achieve almost linear speedup on multiCore CPUs: use gcc-4.2 -fopenmp */
#ifdef _OPENMP
#include <omp.h>
#endif


static inline float chi2_baseline_float(const int n, const float* x, const float* y) {
    float result = 0.f;
    int i;
    for (i=0; i<n; i++) {
        const float num = x[i]*y[i];
        const float denom = 1./(x[i]+y[i]+FLT_MIN);
        result += 2*num*denom;
    }
    return result;
}

/* use compiler intrinsics for 4x parallel processing */
/*
static inline float chi2_intrinsic_float(int n, const float* x, const float* y) {
    float result=0;
    const __m128 eps = _mm_set1_ps(FLT_MIN);
    const __m128 zero = _mm_setzero_ps();
    __m128 chi2 = _mm_setzero_ps();
    
    for (; n>3; n-=4) {
        const __m128 a = _mm_loadu_ps(x);
        const __m128 b = _mm_loadu_ps(y);
        const __m128 a_plus_eps = _mm_add_ps(a,eps);
        const __m128 a_plus_b_plus_eps = _mm_add_ps(a_plus_eps,b);
        const __m128 a_minus_b = _mm_sub_ps(a,b);
        const __m128 a_minus_b_sq = _mm_mul_ps(a_minus_b, a_minus_b);
        const __m128 prod = _mm_div_ps(a_minus_b_sq, a_plus_b_plus_eps);
        chi2 = _mm_add_ps(chi2, prod);
	x+=4;
	y+=4;
    }
    const __m128 shuffle1 = _mm_shuffle_ps(chi2, chi2, _MM_SHUFFLE(1,0,3,2));
    const __m128 sum1 = _mm_add_ps(chi2, shuffle1);
    const __m128 shuffle2 = _mm_shuffle_ps(sum1, sum1, _MM_SHUFFLE(2,3,0,1));
    const __m128 sum2 = _mm_add_ps(sum1, shuffle2);
// with SSE3, we could use hadd_ps, but the difference is negligible 

    _mm_store_ss(&result,sum2);
    _mm_empty();
    
    if (n)
        result += chi2_baseline_float(n, x, y);	// remaining 1-3 entries
    return result;
}
*/
/* calculate the chi2-distance between two vectors/histograms */
float chi2_float(const int dim, const float* const x, const float* const y) {
    float (*chi2_float)(const int, const float*, const float*) = chi2_baseline_float;
/*#ifdef __SSE__
    chi2_float = chi2_intrinsic_float;
#endif*/
    return chi2_float(dim, x, y);
}

/* calculate the chi2-distance matrix between a sets of vectors/histograms. */
float chi2sym_distance_float(const int dim, const int nx, const float* const x, 
                             float* const K) {
    float (*chi2_float)(const int, const float*, const float*) = chi2_baseline_float;
/*#ifdef __SSE__
    chi2_float = chi2_intrinsic_float;
#endif*/
	
    float sumK=0.f;
#pragma omp parallel
    {
        int i,j;
#pragma omp for reduction (+:sumK) schedule (dynamic,2)
        for (i=0;i<nx;i++) {
    	    K[i*nx+i]=0.;
            for (j=0;j<i;j++) {
	    	const float chi2 = (*chi2_float)(dim, &x[i*dim], &x[j*dim]); 
                K[i*nx+j] = chi2;
                K[j*nx+i] = chi2;
	        	sumK += 2*chi2;
            }
	    }
    }
    return sumK/((float)(nx*nx)); 
}

/* calculate the chi2-distance matrix between two sets of vectors/histograms. */
float chi2_distance_float(const int dim, const int nx, const float* const x, 
                          const int ny, const float* const y, float* const K) {
    float (*chi2_float)(const int, const float*, const float*) = chi2_baseline_float;
/*#ifdef __SSE__
    chi2_float = chi2_intrinsic_float;
#endif
	*/
    float sumK=0.f;
#pragma omp parallel
    {
        int i,j;
#pragma omp for reduction (+:sumK) schedule (dynamic,2)
        for (i=0;i<nx;i++) {
            for (j=0;j<ny;j++) {
	    	float chi2 = (*chi2_float)(dim, &x[i*dim], &y[j*dim]); 
                K[i*ny+j] = chi2;
        		sumK += chi2;
            }
	    }
    }
    return sumK/((float)(nx*ny)); 
}


#ifdef __MAIN__

#include <stdlib.h>
#include <malloc.h>
#include <time.h>
int main()
{
    const int dim=3000;
    const int n1=1000;
    const int n2=2000;
    int i,j;

/* test calculating a kernel with float entries */
    float *data1 = (float*)memalign(16,dim*n1*sizeof(float));
    float *data2 = (float*)memalign(16,dim*n2*sizeof(float));
    float *K = (float*)malloc(n1*n2*sizeof(float));
    if ((!data1) || (!data2) || (!K)) {
       free(data1);
       free(data2);
       free(K);
       return 1;
    }

    const clock_t before_init=clock();
    for (i=0;i<n1*dim;i++)
    	data1[i]=1./(float)(i+1.);
    for (i=0;i<n2*dim;i++)
    	data2[i]=1./(float)(i+1.);
    const clock_t after_init=clock();
    printf("init time: %8.4f\n",(after_init-before_init)*1./CLOCKS_PER_SEC);

    const clock_t before_chi2=clock();
    const float mean_K = chi2_distance_float(dim, n1, data1, n2, data2, K);
    const clock_t after_chi2=clock();
    printf("chi2 time: %8.4f\n",(after_chi2-before_chi2)*1./CLOCKS_PER_SEC);

    printf("result: %e\n",mean_K);

    free(data1);
    free(data2);
    free(K);
    
    return 0;
}

#endif
