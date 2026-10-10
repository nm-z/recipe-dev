#include <stdint.h>
#include <stdio.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
typedef struct { uint32_t x,y; } uint2;
typedef struct { uint32_t x,y,z,w; } uint4;
#define __device__
#define __align__(value)
#include "codebooks.inc"
static uint32_t pair16(int a,int b){return (a&65535u)|((uint32_t)(b&65535)<<16);}
int main(void){
	unsigned mismatch=0,iq2_cases=0,iq3_cases=0,index_cases=0;
	for(unsigned code=0;code<65536;code++){
		uint64_t grid=iq2xs_grid[code&511];unsigned sign=ksigns_iq2xs[code>>9];uint4 entry=packed_iq2[code];uint32_t values[4]={entry.x,entry.y,entry.z,entry.w};
		for(int p=0;p<4;p++){int a=(grid>>(16*p))&255,b=(grid>>(16*p+8))&255;if(sign&(1u<<(2*p)))a=-a;if(sign&(2u<<(2*p)))b=-b;
			mismatch+=values[p]!=pair16(a,b);iq2_cases++;}
	}
	for(unsigned s=0;s<128;s++)for(unsigned g=0;g<256;g++)for(int half=0;half<2;half++){
		unsigned sign=ksigns_iq2xs[s],code=g+(((sign>>(4*half))&15)<<8);uint32_t grid=iq3xxs_grid[g];uint2 entry=packed_iq3[code];
		for(int p=0;p<2;p++){int a=(grid>>(16*p))&255,b=(grid>>(16*p+8))&255;if(sign&(1u<<(4*half+2*p)))a=-a;if(sign&(2u<<(4*half+2*p)))b=-b;
			mismatch+=(p?entry.y:entry.x)!=pair16(a,b);iq3_cases++;}
	}
	for(unsigned s=0;s<128;s++)for(unsigned grids=0;grids<65536;grids++){
		unsigned sign=ksigns_iq2xs[s],offset=((sign&15)<<8)|((sign>>4)<<24);
		unsigned codes=(grids&255)|((grids&65280)<<8)|offset,lo=codes&65535,hi=codes>>16;
		mismatch+=lo!=((grids&255)|((sign&15)<<8));mismatch+=hi!=((grids>>8)|((sign>>4)<<8));
		mismatch+=8192+lo*8+8>40960;mismatch+=8192+hi*8+8>40960;index_cases+=2;
	}
	printf("iq2_pair_checks=%u iq3_pair_checks=%u iq3_index_checks=%u mismatches=%u\n",iq2_cases,iq3_cases,index_cases,mismatch);
	return mismatch?1:0;
}
