#include <stdint.h>
#include <stdio.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
static uint32_t pair16(int a,int b){return (a&65535u)|((uint32_t)(b&65535)<<16);}
int main(void){
	puts("static __device__ __align__(16) uint4 packed_iq2[65536] = {");
	for(int q=0;q<65536;q++){uint64_t grid=iq2xs_grid[q&511];int sign=ksigns_iq2xs[q>>9];uint32_t v[4];for(int j=0;j<4;j++){int a=(grid>>(16*j))&255,b=(grid>>(16*j+8))&255;if(sign&(1<<(2*j)))a=-a;if(sign&(2<<(2*j)))b=-b;v[j]=pair16(a,b);}printf("\t{0x%08x,0x%08x,0x%08x,0x%08x},\n",v[0],v[1],v[2],v[3]);}
	puts("};\nstatic __device__ __align__(16) uint2 packed_iq3[4096] = {");
	for(int q=0;q<4096;q++){uint32_t grid=iq3xxs_grid[q&255],v[2];int sign=q>>8;for(int j=0;j<2;j++){int a=(grid>>(16*j))&255,b=(grid>>(16*j+8))&255;if(sign&(1<<(2*j)))a=-a;if(sign&(2<<(2*j)))b=-b;v[j]=pair16(a,b);}printf("\t{0x%08x,0x%08x},\n",v[0],v[1]);}
	puts("};\nstatic __device__ __align__(16) uint2 packed_iq4[65536] = {");
	for(int q=0;q<65536;q++){int a=q&255,b=q>>8;printf("\t{0x%08x,0x%08x},\n",pair16(kvalues_iq4nl[a&15],kvalues_iq4nl[b&15]),pair16(kvalues_iq4nl[a>>4],kvalues_iq4nl[b>>4]));}
	puts("};");return 0;
}
