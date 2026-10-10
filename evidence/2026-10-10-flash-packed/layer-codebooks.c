#include <stdint.h>
#include <stdio.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
static uint32_t pair16(int a,int b){return (a&65535u)|((uint32_t)(b&65535)<<16);}
int main(void){
	puts("static __device__ __align__(16) uint32_t packed_iq2_codes[65536] = {");
	for(int q=0;q<65536;q++){uint64_t grid=iq2xs_grid[q&511];int sign=ksigns_iq2xs[q>>9];uint32_t v=0;for(int j=0;j<8;j++){int a=(grid>>(8*j))&255;if(a!=8&&a!=25&&a!=43)return 1;int index=(a==8?0:a==25?1:2)|((sign>>j&1)<<2);v|=(uint32_t)index<<(4*j);}printf("\t0x%08x,\n",v);}
	puts("};\nstatic __device__ __align__(16) uint2 packed_iq3[4096] = {");
	for(int q=0;q<4096;q++){uint32_t grid=iq3xxs_grid[q&255],v[2];int sign=q>>8;for(int j=0;j<2;j++){int a=(grid>>(16*j))&255,b=(grid>>(16*j+8))&255;if(sign&(1<<(2*j)))a=-a;if(sign&(2<<(2*j)))b=-b;v[j]=pair16(a,b);}printf("\t{0x%08x,0x%08x},\n",v[0],v[1]);}
	puts("};");return 0;
}
