// Packed Flash-Next matvec probes. Weight storage stays in its GGUF layout.
#include <cuda_fp16.h>
#include <stdint.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
using u32 = uint32_t;
static __device__ __forceinline__ int mad2(u32 w,u32 x,int a) {
	asm("{ .reg .b16 wl,wh,xl,xh; mov.b32 {wl,wh}, %1; mov.b32 {xl,xh}, %2; mad.wide.s16 %0,wl,xl,%0; mad.wide.s16 %0,wh,xh,%0; }" : "+r"(a) : "r"(w),"r"(x));
	return a;
}
static __device__ __forceinline__ u32 pair(int a,int b){return (a&65535u)|(u32(b&65535)<<16);}
static __device__ __forceinline__ u32 sign_pair(u32 q){u32 v;asm("prmt.b32 %0,%1,0,0x9180;":"=r"(v):"r"(q));return v;}
static __device__ __forceinline__ float magic(int q){return __uint_as_float(0x4b000000u+u32(q+128))-8388736.0f;}
static __device__ __forceinline__ float half_at(const uint8_t *p){return __half2float(__ushort_as_half(__ldg((const uint16_t*)p)));}
// 32-value activation groups, adjacent signed int16 pairs. Group padding avoids shared-memory bank conflicts.
extern "C" __global__ void pack_x(const float *x,u32 *p,float2 *ds,int k,int n) {
	int b=blockIdx.x*8+(threadIdx.x>>5),l=threadIdx.x&31,c=blockIdx.y;
	if(b>=k/32||c>=n)return;
	float v=x[c*k+b*32+l],a=fabsf(v);
	#pragma unroll
	for(int d=16;d;d>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
	float scale=a/127;int q=a==0?0:int(roundf(v/scale)),s=q;
	#pragma unroll
	for(int d=16;d;d>>=1)s+=__shfl_xor_sync(0xffffffff,s,d);
	int partner=__shfl_xor_sync(0xffffffff,q,1);
	if(!(l&1))p[(c*k/32+b)*16+l/2]=pair(q,partner);
	if(l==0){float dh=__half2float(__float2half(scale));ds[c*k/32+b]=make_float2(dh,dh*s);}
}
template<int T>struct Format;
#define FMT(T,B,S) template<>struct Format<T>{static constexpr int block=B,bytes=S;};
FMT(12,256,144) FMT(13,256,176) FMT(14,256,210) FMT(8,32,34) FMT(17,256,74) FMT(18,256,98) FMT(20,32,18)
static __device__ __forceinline__ void scale_min(int g,const uint8_t *s,int &sc,int &mn){
	if(g<4){sc=s[g]&63;mn=s[g+4]&63;}else{sc=(s[g+4]&15)|((s[g-4]>>6)<<4);mn=(s[g+4]>>4)|((s[g]>>6)<<4);}
}
// Decode a group once, retain packed signed int16 operands across all columns, and scale after the integer sums.
// Each lane decodes eight adjacent values. Four lanes share one activation scale.
template<int T>static __device__ __forceinline__ void decode8(const uint8_t *row,int b,int sub,u32 (&w)[4],float &d,float &sc,float &mn){
	const uint8_t *p=row+(size_t)(b/(Format<T>::block/32))*Format<T>::bytes;
	int g=b%(Format<T>::block/32);sc=1;mn=0;
	if constexpr(T==12||T==13){
		d=half_at(p);int s,m;scale_min(g,p+4,s,m);sc=s;mn=half_at(p+2)*m;
		const uint8_t *qs=p+(T==12?16:48)+(g/2)*32+sub*8;
		uint2 v=__ldg((const uint2*)qs);u32 words[2]={v.x,v.y};
		u32 high[2]={0,0};
		if constexpr(T==13){uint2 h=__ldg((const uint2*)(p+16+sub*8));high[0]=h.x;high[1]=h.y;}
		#pragma unroll
		for(int j=0;j<2;j++){
			u32 q=((words[j]>>(4*(g&1)))&0x0f0f0f0f)|(((high[j]>>g)&0x01010101)<<4);
			w[2*j]=__byte_perm(q,0,0x4140);w[2*j+1]=__byte_perm(q,0,0x4342);
		}
	}else if constexpr(T==14){
		d=half_at(p+208)*0.25f;sc=((const int8_t*)p)[192+2*g+sub/2];
		int region=g/4,part=g%4;const uint8_t *ql=p+region*64+(part&1)*32+sub*8,*qh=p+128+region*32+sub*8;
		#pragma unroll
		for(int j=0;j<4;j++){uint16_t a=*(const uint16_t*)(ql+2*j),h=*(const uint16_t*)(qh+2*j);u32 q=((a>>(4*(part/2)))&0x0f0f)|(((h>>(2*part))&0x0303)<<4);q=(q^0x2020)<<2;w[j]=sign_pair(q);}
	}else if constexpr(T==8){
		d=half_at(p);
		#pragma unroll
		for(int j=0;j<4;j++){uint16_t q=*(const uint16_t*)(p+2+sub*8+2*j);w[j]=sign_pair(q);}
	}else if constexpr(T==20){
		d=half_at(p);int shift=(sub/2)*4;
		#pragma unroll
		for(int j=0;j<4;j++){uint16_t q=*(const uint16_t*)(p+2+(sub&1)*8+2*j);w[j]=pair(kvalues_iq4nl[(q>>shift)&15],kvalues_iq4nl[(q>>(shift+8))&15]);}
	}else if constexpr(T==17){
		d=half_at(p)*0.125f;sc=1+2*((p[66+g]>>(4*(sub/2)))&15);
		uint16_t q=*(const uint16_t*)(p+2+8*g+2*sub);uint64_t grid=__ldg(iq2xs_grid+(q&511)),sign=__ldg(ksigns64+(q>>9));
		#pragma unroll
		for(int j=0;j<4;j++){u32 v=u32(grid>>(32*(j/2))),s=u32(sign>>(32*(j/2)));u32 a=__byte_perm(v,0,(j&1)?0x4342:0x4140),mask=__byte_perm(s,0,(j&1)?0x3322:0x1100);w[j]=(a^mask)+(mask&0x00010001);}
	}else if constexpr(T==18){
		d=half_at(p)*0.25f;u32 a=u32(*(const uint16_t*)(p+66+4*g))|(u32(*(const uint16_t*)(p+68+4*g))<<16);sc=1+2*(a>>28);
		uint64_t sign=__ldg(ksigns64+((a>>(7*sub))&127));u32 grids[2]={__ldg(iq3xxs_grid+p[2+8*g+2*sub]),__ldg(iq3xxs_grid+p[3+8*g+2*sub])};
		#pragma unroll
		for(int j=0;j<4;j++){u32 v=grids[j/2],s=u32(sign>>(32*(j/2)));u32 a=__byte_perm(v,0,(j&1)?0x4342:0x4140),mask=__byte_perm(s,0,(j&1)?0x3322:0x1100);w[j]=(a^mask)+(mask&0x00010001);}
	}
}
#include "codebooks.inc"
static __device__ __forceinline__ uint4 load_shared4(const u32 *p){uint4 v;u32 a=__cvta_generic_to_shared(p);asm("ld.shared.v4.b32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(a));return v;}
static __device__ __forceinline__ void store_shared4(u32 *p,uint4 v){u32 a=__cvta_generic_to_shared(p);asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(a),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w));}
static __device__ __forceinline__ float2 load_shared2(const float2 *p){float2 v;u32 a=__cvta_generic_to_shared(p);asm("ld.shared.v2.f32 {%0,%1},[%2];":"=f"(v.x),"=f"(v.y):"r"(a));return v;}
static __device__ __forceinline__ void store_shared2(float2 *p,float2 v){u32 a=__cvta_generic_to_shared(p);asm volatile("st.shared.v2.f32 [%0],{%1,%2};"::"r"(a),"f"(v.x),"f"(v.y));}
// Preserve the measured prefetch/shared-tile mapping for formats with 32-value activation groups.
static __device__ __forceinline__ void load32(const uint8_t *p,u32 (&v)[8]){
	u32 shift=(uintptr_t(p)&2)*8;const u32 *q=(const u32*)(uintptr_t(p)&~uintptr_t(3));u32 raw[8];
	#pragma unroll
	for(int j=0;j<8;j++)raw[j]=__ldg(q+j);
	u32 tail=shift?*(const uint16_t*)(p+30):0;
	#pragma unroll
	for(int j=0;j<7;j++)v[j]=__funnelshift_r(raw[j],raw[j+1],shift);
	v[7]=__funnelshift_r(raw[7],tail,shift);
}
static __device__ __forceinline__ int lut_shared(const int *table,int index){int v;u32 address=__cvta_generic_to_shared(table+index);asm("ld.shared.s32 %0,[%1];":"=r"(v):"r"(address));return v;}
static __device__ __forceinline__ u32 times17(u32 a){u32 v;asm("{.reg .b16 lo,hi;mov.b32 {lo,hi},%1;mul.wide.u16 %0,lo,17;}":"=r"(v):"r"(a));return v;}
static __device__ __forceinline__ u32 iq2_pair(u32 index){u32 i=times17(index&0x77)&0x0f0f,selector=times17(i)|0x8080,v;asm("prmt.b32 %0,%1,%2,%3;":"=r"(v):"r"(0x002b1908u),"r"(0x00d5e7f8u),"r"(selector));return v;}
static __device__ __forceinline__ uint2 iq3_shared(const u32 *scratch,int code){uint2 v;u32 a=__cvta_generic_to_shared(scratch)+8192+code*8;asm("ld.shared.v2.b32 {%0,%1},[%2];":"=r"(v.x),"=r"(v.y):"r"(a));return v;}
static __device__ __forceinline__ uint2 shared_pair2(const u32 *p){uint2 v;u32 a=__cvta_generic_to_shared(p);asm("ld.shared.v2.b32 {%0,%1},[%2];":"=r"(v.x),"=r"(v.y):"r"(a));return v;}
static __device__ __forceinline__ void store_shared_pair2(u32 *p,uint2 v){u32 a=__cvta_generic_to_shared(p);asm volatile("st.shared.v2.b32 [%0],{%1,%2};"::"r"(a),"r"(v.x),"r"(v.y));}
static __device__ __forceinline__ void signed_bytes4(u32 v,u32 &a,u32 &b){asm("prmt.b32 %0,%2,0,0x9180;prmt.b32 %1,%2,0,0xb3a2;":"=&r"(a),"=&r"(b):"r"(v));}
template<int T>static __device__ __forceinline__ void prepare_iq_tables(u32 *scratch){
	constexpr int BASE=6144,SIGNS=T==17?7168:6400;
	if constexpr(T==17){for(int i=threadIdx.x;i<512;i+=blockDim.x){uint64_t g=__ldg(iq2xs_grid+i);store_shared_pair2(scratch+BASE+2*i,make_uint2(u32(g),u32(g>>32)));}}
	else{for(int i=threadIdx.x;i<256;i+=blockDim.x)scratch[BASE+i]=__ldg(iq3xxs_grid+i);}
	for(int i=threadIdx.x;i<128;i+=blockDim.x){uint64_t g=__ldg(ksigns64+i);store_shared_pair2(scratch+SIGNS+2*i,make_uint2(u32(g),u32(g>>32)));}
	__syncthreads();
}
template<int T,bool DICT>static __device__ __forceinline__ void decode_stream8(const uint8_t *row,int b,int sub,u32 (&w)[4],float &d,float &sc,const int *table,const u32 *scratch,int warp_lut){
	const uint8_t *p=row+(size_t)(b/(Format<T>::block/32))*Format<T>::bytes;int g=b%(Format<T>::block/32);
	if constexpr(DICT&&T==17){
		d=half_at(p)*0.125f;sc=1+2*((p[66+g]>>(4*(sub/2)))&15);uint16_t q=__ldg((const uint16_t*)(p+2+8*g+2*sub));uint2 v=shared_pair2(scratch+6144+2*(q&511)),z=shared_pair2(scratch+7168+2*(q>>9));u32 a=(v.x^z.x)+(z.x&0x01010101),b=(v.y^z.y)+(z.y&0x01010101);signed_bytes4(a,w[0],w[1]);signed_bytes4(b,w[2],w[3]);
	}else if constexpr(DICT&&T==18){
		d=half_at(p)*0.25f;u32 a=u32(*(const uint16_t*)(p+66+4*g))|(u32(*(const uint16_t*)(p+68+4*g))<<16);sc=1+2*(a>>28);uint2 z=shared_pair2(scratch+6400+2*((a>>(7*sub))&127));u32 v0=scratch[6144+__ldg(p+2+8*g+2*sub)],v1=scratch[6144+__ldg(p+3+8*g+2*sub)];v0=(v0^z.x)+(z.x&0x01010101);v1=(v1^z.y)+(z.y&0x01010101);signed_bytes4(v0,w[0],w[1]);signed_bytes4(v1,w[2],w[3]);
	}else if constexpr(T==20){
		d=half_at(p);sc=1;
		#pragma unroll
		for(int j=0;j<4;j++){u32 q=__ldg((const uint16_t*)(p+2+(sub&1)*8+2*j));int i0=(q>>(4*(sub/2)))&15,i1=(q>>(8+4*(sub/2)))&15;int a,b;if constexpr(DICT){a=__shfl_sync(0xffffffff,warp_lut,i0);b=__shfl_sync(0xffffffff,warp_lut,i1);}else{a=lut_shared(table,i0);b=lut_shared(table,i1);}w[j]=__byte_perm(a,b,0x5410);}
	}else{float mn;decode8<T>(row,b,sub,w,d,sc,mn);}
}
template<int T,int N,bool DICT>static __device__ __forceinline__ void decode32(const uint8_t *row,int b,u32 (&w)[16],float &d,float &s0,float &s1,float &mn,const int *table,const u32 *scratch,int warp_lut){
	const uint8_t *p=row+(size_t)(b/(Format<T>::block/32))*Format<T>::bytes;int g=b%(Format<T>::block/32);mn=0;
	if constexpr(DICT&&T==17){
		d=half_at(p)*0.125f;s0=1+2*(p[66+g]&15);s1=1+2*(p[66+g]>>4);
		#pragma unroll
		for(int j=0;j<4;j++){uint16_t q=__ldg((const uint16_t*)(p+2+8*g+2*j));u32 v=__ldg(packed_iq2_codes+q);w[4*j]=iq2_pair(v);w[4*j+1]=iq2_pair(v>>8);w[4*j+2]=iq2_pair(v>>16);w[4*j+3]=iq2_pair(v>>24);}
	}else if constexpr(DICT&&T==18){
		d=half_at(p)*0.25f;u32 a=u32(*(const uint16_t*)(p+66+4*g))|(u32(*(const uint16_t*)(p+68+4*g))<<16);s0=s1=1+2*(a>>28);
		#pragma unroll
		for(int j=0;j<4;j++){u32 sign=__ldg(ksigns_iq2xs+((a>>(7*j))&127));int c0=__ldg(p+2+8*g+2*j)+((sign&15)<<8),c1=__ldg(p+3+8*g+2*j)+((sign>>4)<<8);uint2 v0,v1;if constexpr(N<=2){v0=iq3_shared(scratch,c0);v1=iq3_shared(scratch,c1);}else{v0=__ldg(packed_iq3+c0);v1=__ldg(packed_iq3+c1);}w[4*j]=v0.x;w[4*j+1]=v0.y;w[4*j+2]=v1.x;w[4*j+3]=v1.y;}
	}else if constexpr(DICT&&T==20){
		d=half_at(p);s0=s1=1;
		#pragma unroll
		for(int j=0;j<8;j++){u32 q=__ldg((const uint16_t*)(p+2+2*j));int a=__shfl_sync(0xffffffff,warp_lut,q&15),b=__shfl_sync(0xffffffff,warp_lut,(q>>8)&15),c=__shfl_sync(0xffffffff,warp_lut,(q>>4)&15),e=__shfl_sync(0xffffffff,warp_lut,q>>12);w[j]=__byte_perm(a,b,0x5410);w[j+8]=__byte_perm(c,e,0x5410);}
	}else if constexpr(T==8){
		d=half_at(p);s0=s1=1;u32 v[8];load32(p+2,v);
		#pragma unroll
		for(int j=0;j<8;j++){w[2*j]=sign_pair(v[j]&0xffff);w[2*j+1]=sign_pair(v[j]>>16);}
	}else if constexpr(T==14){
		d=half_at(p+208)*0.25f;s0=((const int8_t*)p)[192+2*g];s1=((const int8_t*)p)[193+2*g];
		int region=g/4,part=g%4;u32 ql[8],qh[8];load32(p+region*64+(part&1)*32,ql);load32(p+128+region*32,qh);
		#pragma unroll
		for(int j=0;j<8;j++){u32 q=((ql[j]>>(4*(part/2)))&0x0f0f0f0f)|(((qh[j]>>(2*part))&0x03030303)<<4);q=(q^0x20202020)<<2;w[2*j]=sign_pair(q&0xffff);w[2*j+1]=sign_pair(q>>16);}
	}else if constexpr(T==20){
		d=half_at(p);s0=s1=1;
		#pragma unroll
		for(int j=0;j<8;j++){u32 q=*(const uint16_t*)(p+2+2*j);w[j]=__byte_perm(lut_shared(table,q&15),lut_shared(table,(q>>8)&15),0x5410);w[j+8]=__byte_perm(lut_shared(table,(q>>4)&15),lut_shared(table,q>>12),0x5410);}
	}else{
	#pragma unroll
	for(int sub=0;sub<4;sub++){
		u32 v[4];float sc;decode8<T>(row,b,sub,v,d,sc,mn);
		#pragma unroll
		for(int j=0;j<4;j++)w[sub*4+j]=v[j];
		if(sub==0)s0=sc;if(sub==2)s1=sc;
	}
	}
}
template<int T,int N,int NW,int LW,bool MAGIC,bool DICT>static __device__ __forceinline__ void group32(const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,int row_base,int active,u32 position_mask,u32 cta_count,u32 *scratch,int warp_lut){
	constexpr int NT=NW*32,TU=LW,YS=20,NY=(TU*4+NT-1)/NT;
	u32 (*ys)[2][TU*YS]=(u32 (*)[2][TU*YS])scratch;
	float2 (*dss)[2][TU]=(float2 (*)[2][TU])(scratch+N*2*TU*YS);
	int *table=(int*)((float2*)dss+N*2*TU);
	int tid=threadIdx.x,l=tid&(LW-1),row=row_base+(tid/LW)*cta_count,nb=k/32,nt=(nb+TU-1)/TU;
	int source[N];
	#pragma unroll
	for(int c=0;c<N;c++){u32 mask=position_mask;for(int j=0;j<c;j++)mask&=mask-1;source[c]=c>=active?-1:position_mask?__ffs(mask)-1:c;}
	if constexpr(T==20&&!DICT){if(tid<16)table[tid]=kvalues_iq4nl[tid];__syncthreads();}
	const uint8_t *rp=W+(size_t)(row<m?row:0)*(k/Format<T>::block)*Format<T>::bytes;
	uint4 yr[N][NY];float2 dr[N];
	#pragma unroll
	for(int c=0;c<N;c++){
		#pragma unroll
		for(int i=0;i<NY;i++){int ii=tid+i*NT;yr[c][i]=make_uint4(0,0,0,0);if(source[c]>=0&&ii<TU*4&&ii<nb*4)yr[c][i]=__ldg((const uint4*)x+source[c]*nb*4+ii);if(ii<TU*4)store_shared4(&ys[c][0][(ii/4)*YS+(ii%4)*4],yr[c][i]);}
		dr[c]=make_float2(0,0);if(source[c]>=0&&tid<TU&&tid<nb)dr[c]=__ldg(ds+source[c]*nb+tid);if(tid<TU)store_shared2(&dss[c][0][tid],dr[c]);
	}
	__syncthreads();float acc[N]={};
	for(int tile=0;tile<nt;tile++){
		int buf=tile&1;
		if(tile+1<nt){
			#pragma unroll
			for(int c=0;c<N;c++){
				#pragma unroll
				for(int i=0;i<NY;i++){int ii=tid+i*NT;yr[c][i]=make_uint4(0,0,0,0);if(source[c]>=0&&ii<TU*4&&(tile+1)*TU*4+ii<nb*4)yr[c][i]=__ldg((const uint4*)x+source[c]*nb*4+(tile+1)*TU*4+ii);}
				dr[c]=make_float2(0,0);if(source[c]>=0&&tid<TU&&(tile+1)*TU+tid<nb)dr[c]=__ldg(ds+source[c]*nb+(tile+1)*TU+tid);
			}
		}
		u32 w[16];float d,s0,s1,mn;int b=tile*TU+l;
		if(row<m||(T==20&&DICT))decode32<T,N,DICT>(rp,b<nb?b:0,w,d,s0,s1,mn,table,scratch,warp_lut);
		#pragma unroll
		for(int c=0;c<N;c++){
			if(c>=active||row>=m)continue;
			const u32 *xp=&ys[c][buf][l*YS];float z0,z1;
			if constexpr(!MAGIC){int a0=0,a1=0;
				#pragma unroll
				for(int h=0;h<2;h++){uint4 a=load_shared4(xp+4*h),b=load_shared4(xp+8+4*h);u32 x0[4]={a.x,a.y,a.z,a.w},x1[4]={b.x,b.y,b.z,b.w};
					#pragma unroll
					for(int j=0;j<4;j++){a0=mad2(w[h*4+j],x0[j],a0);a1=mad2(w[h*4+j+8],x1[j],a1);}}
				z0=a0;z1=a1;
			}else{float a0=0,a1=0,b0=0,b1=0;
				#pragma unroll
				for(int h=0;h<2;h++){uint4 a=load_shared4(xp+4*h),b=load_shared4(xp+8+4*h);u32 x0[4]={a.x,a.y,a.z,a.w},x1[4]={b.x,b.y,b.z,b.w};
					#pragma unroll
					for(int j=0;j<4;j++){int v=h*4+j;a0=fmaf(magic((int16_t)w[v]),float((int16_t)x0[j]),a0);b0=fmaf(magic((int16_t)(w[v]>>16)),float((int16_t)(x0[j]>>16)),b0);a1=fmaf(magic((int16_t)w[v+8]),float((int16_t)x1[j]),a1);b1=fmaf(magic((int16_t)(w[v+8]>>16)),float((int16_t)(x1[j]>>16)),b1);}}
				z0=a0+b0;z1=a1+b1;
			}
			float2 scale=load_shared2(&dss[c][buf][l]);acc[c]+=fmaf(d*scale.x,fmaf(s0,z0,s1*z1),-mn*scale.y);
		}
		if(tile+1<nt){
			#pragma unroll
			for(int c=0;c<N;c++){
				#pragma unroll
				for(int i=0;i<NY;i++){int ii=tid+i*NT;if(ii<TU*4)store_shared4(&ys[c][buf^1][(ii/4)*YS+(ii%4)*4],yr[c][i]);}
				if(tid<TU)store_shared2(&dss[c][buf^1][tid],dr[c]);
			}
		}
		__syncthreads();
	}
	#pragma unroll
	for(int c=0;c<N;c++){
		#pragma unroll
		for(int j=LW/2;j;j>>=1)acc[c]+=__shfl_xor_sync(0xffffffff,acc[c],j);
		if(l==0&&row<m&&c<active)out[c*m+row]=acc[c];
	}
}
#define G(T,N,W,L,K) extern "C" __device__ __noinline__ void packed_g_##T##_##N##_##W##_##K##_##L(const uint8_t *a,const u32 *b,const float2 *c,float *d,int k,int m,int active,u32 mask,u32 cta_index,u32 cta_count,u32 *scratch){int warp_lut=0;if constexpr(T==20&&K==2)warp_lut=__ldg(kvalues_iq4nl+(threadIdx.x&15));if constexpr(T==18&&K==2&&N<=2){for(int i=threadIdx.x;i<2048;i+=blockDim.x)store_shared4(scratch+2048+i*4,__ldg((const uint4*)packed_iq3+i));__syncthreads();}for(int base=cta_index;base<m;base+=cta_count*(W*32/L))group32<T,N,W,L,(K==1),(K==2)>(a,b,c,d,k,m,base,active,mask,cta_count,scratch,warp_lut);}
#define GK(T,N,W,L) G(T,N,W,L,0) G(T,N,W,L,1)
#define GW(T,N,L) GK(T,N,2,L) GK(T,N,4,L) GK(T,N,8,L) GK(T,N,16,L)
#define GN(T,N) GW(T,N,8) GW(T,N,16)
#define GT(T) GN(T,1) GN(T,2) GN(T,4) GN(T,8)
#define WIDEG(T,N) GK(T,N,32,8) GK(T,N,32,16)
#define WIDET(T) WIDEG(T,1) WIDEG(T,2)
#define DW(T,N,L) G(T,N,2,L,2) G(T,N,4,L,2) G(T,N,8,L,2) G(T,N,16,L,2)
#define DN(T,N) DW(T,N,8) DW(T,N,16)
#define DT(T) DN(T,1) DN(T,2) DN(T,4) DN(T,8) G(T,1,32,8,2) G(T,1,32,16,2) G(T,2,32,8,2) G(T,2,32,16,2)
static __device__ __forceinline__ void decode_scales(u32 S0, u32 S1, u32 S2, int g, int& scA, int& scB, int& mA, int& mB) {
  const int a = (g & 1) * 16;
  const u32 x0 = (S0 >> a) & 0xffffu, x1 = (S1 >> a) & 0xffffu, x2 = (S2 >> a) & 0xffffu;
  u32 sp, mp;
  if (g < 2) { sp = x0 & 0x3f3fu; mp = x1 & 0x3f3fu; }
  else { sp = (x2 & 0x0f0fu) | ((x0 & 0xc0c0u) >> 2); mp = ((x2 >> 4) & 0x0f0fu) | ((x1 & 0xc0c0u) >> 2); }
  scA = sp & 0xff; scB = sp >> 8; mA = mp & 0xff; mB = mp >> 8;
}

static __device__ __forceinline__ float finish_unit(u32 dd, int dotA, int dotB, int scA, int scB, int mA, int mB, float2 dA, float2 dB) {
  const float d = __half2float(__ushort_as_half((unsigned short)(dd & 0xffff)));
  const float dmin = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
  const float pd = fmaf((float)scA * dA.x, (float)dotA, (float)scB * dB.x * (float)dotB);
  const float pm = fmaf((float)mA, dA.y, (float)mB * dB.y);
  return fmaf(d, pd, -dmin * pm);
}

static __device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Measured #1091 mapping, with real-shape tail handling and column reuse.
template<int T,int NW,int N,int LW>static __device__ __forceinline__ void k_width(const uint8_t *W,const u32 *y16,const float2 *dsf,float *out,int k,int m,int row_base,int active,u32 position_mask,u32 cta_count,u32 *scratch){
	constexpr int TU=LW,YS16=36,NT=NW*32,NY=(TU*8+NT-1)/NT,TSB=TU/4,BYTES=T==12?144:176,QS=T==12?16:48;
	u32 (*ys)[2][TU*YS16]=(u32 (*)[2][TU*YS16])scratch;
	float2 (*dss)[2][TU*2]=(float2 (*)[2][TU*2])(scratch+N*2*TU*YS16);
	int tid=threadIdx.x,lane=tid&(LW-1),row=row_base+(tid/LW)*cta_count,nsb=k/256,nt=(nsb+TSB-1)/TSB;
	int source[N];
	#pragma unroll
	for(int c=0;c<N;c++){u32 mask=position_mask;for(int j=0;j<c;j++)mask&=mask-1;source[c]=c>=active?-1:position_mask?__ffs(mask)-1:c;}
	const uint8_t *rowp=W+(size_t)(row<m?row:0)*nsb*BYTES;
	int sbl=lane>>2,g=lane&3;
	uint4 yr[N][NY];float2 dr[N];
	#pragma unroll
	for(int c=0;c<N;c++){
		#pragma unroll
		for(int i=0;i<NY;i++){int ii=tid+i*NT;yr[c][i]=make_uint4(0,0,0,0);if(source[c]>=0&&ii<TU*8&&ii<k/8)yr[c][i]=((const uint4*)y16)[source[c]*k/8+ii];}
		dr[c]=make_float2(0,0);if(source[c]>=0&&tid<TU*2&&tid<k/32)dr[c]=dsf[source[c]*k/32+tid];
		#pragma unroll
		for(int i=0;i<NY;i++){int ii=tid+i*NT;if(ii<TU*8)store_shared4(&ys[c][0][(ii>>3)*YS16+(ii&7)*4],yr[c][i]);}
		if(tid<TU*2)store_shared2(&dss[c][0][tid],dr[c]);
	}
	__syncthreads();float acc[N]={};
	for(int t=0;t<nt;t++){
		int buf=t&1;
		if(t+1<nt){
			#pragma unroll
			for(int c=0;c<N;c++){
				#pragma unroll
				for(int i=0;i<NY;i++){int ii=tid+i*NT;yr[c][i]=make_uint4(0,0,0,0);if(source[c]>=0&&ii<TU*8&&(t+1)*TU*8+ii<k/8)yr[c][i]=((const uint4*)y16)[source[c]*k/8+(t+1)*TU*8+ii];}
				dr[c]=make_float2(0,0);if(source[c]>=0&&tid<TU*2&&(t+1)*TU*2+tid<k/32)dr[c]=dsf[source[c]*k/32+(t+1)*TU*2+tid];
			}
		}
		int sb=t*TSB+sbl;const uint8_t *bp=rowp+(sb<nsb?sb:0)*BYTES;
		uint4 q0=__ldg((const uint4*)(bp+QS+32*g)),q1=__ldg((const uint4*)(bp+QS+16+32*g)),h0,h1;
		if constexpr(T==13){h0=__ldg((const uint4*)(bp+16));h1=__ldg((const uint4*)(bp+32));}
		u32 S0=__ldg((const u32*)(bp+4)),S1=__ldg((const u32*)(bp+8)),S2=__ldg((const u32*)(bp+12)),dd=__ldg((const u32*)bp);
		u32 w[8]={q0.x,q0.y,q0.z,q0.w,q1.x,q1.y,q1.z,q1.w};int scA,scB,mA,mB;decode_scales(S0,S1,S2,g,scA,scB,mA,mB);
		u32 wh[8];if constexpr(T==13){wh[0]=h0.x;wh[1]=h0.y;wh[2]=h0.z;wh[3]=h0.w;wh[4]=h1.x;wh[5]=h1.y;wh[6]=h1.z;wh[7]=h1.w;}
		#pragma unroll
		for(int c=0;c<N;c++){
			if(c>=active||row>=m)continue;
			const u32 *yu=&ys[c][buf][lane*YS16];int aA0=0,aA1=0,aB0=0,aB1=0;
			#pragma unroll
			for(int h=0;h<4;h++){
				uint4 ya=load_shared4(yu+4*h),yb=load_shared4(yu+16+4*h);
				#pragma unroll
				for(int j=0;j<2;j++){
					u32 ww=w[2*h+j],v=ww>>4;
					if constexpr(T==13){ww=(ww&0x0f0f0f0f)|(((wh[2*h+j]>>(2*g))&0x01010101)<<4);v=(v&0x0f0f0f0f)|(((wh[2*h+j]>>(2*g+1))&0x01010101)<<4);}
					constexpr u32 MASK=T==12?15:31;
					u32 a0=(ww&MASK)|((ww&(MASK<<8))<<8),a1=((ww>>16)&MASK)|((ww>>8)&(MASK<<16));
					u32 b0=(v&MASK)|((v&(MASK<<8))<<8),b1=((v>>16)&MASK)|((v>>8)&(MASK<<16));
					aA0=mad2(a0,j?ya.z:ya.x,aA0);aA1=mad2(a1,j?ya.w:ya.y,aA1);
					aB0=mad2(b0,j?yb.z:yb.x,aB0);aB1=mad2(b1,j?yb.w:yb.y,aB1);
				}
			}
			acc[c]+=finish_unit(dd,aA0+aA1,aB0+aB1,scA,scB,mA,mB,load_shared2(&dss[c][buf][sbl*8+2*g]),load_shared2(&dss[c][buf][sbl*8+2*g+1]));
		}
		if(t+1<nt){
			#pragma unroll
			for(int c=0;c<N;c++){
				#pragma unroll
				for(int i=0;i<NY;i++){int ii=tid+i*NT;if(ii<TU*8)store_shared4(&ys[c][buf^1][(ii>>3)*YS16+(ii&7)*4],yr[c][i]);}
				if(tid<TU*2)store_shared2(&dss[c][buf^1][tid],dr[c]);
			}
		}
		__syncthreads();
	}
	#pragma unroll
	for(int c=0;c<N;c++){
		#pragma unroll
		for(int j=LW/2;j;j>>=1)acc[c]+=__shfl_xor_sync(0xffffffff,acc[c],j);
		if(lane==0&&row<m&&c<active)out[c*m+row]=acc[c];
	}
}
#define WIDTH(T,N,W,L) extern "C" __device__ __noinline__ void packed_h_##T##_##N##_##W##_##L(const uint8_t *a,const u32 *b,const float2 *c,float *d,int k,int m,int active,u32 mask,u32 cta_index,u32 cta_count,u32 *scratch){for(int base=cta_index;base<m;base+=cta_count*(W*32/L))k_width<T,W,N,L>(a,b,c,d,k,m,base,active,mask,cta_count,scratch);}
#define WIDTHW(T,N,L) WIDTH(T,N,2,L) WIDTH(T,N,4,L) WIDTH(T,N,8,L) WIDTH(T,N,16,L)
#define WIDTHN(T,N) WIDTHW(T,N,8) WIDTHW(T,N,16)
#define WIDTHT(T) WIDTHN(T,1) WIDTHN(T,2) WIDTHN(T,4) WIDTHN(T,8)
extern "C" __device__ __noinline__ void packed_prepare(const float *x,u32 *p,float2 *ds,int k,int capacity,int active,u32 cta_index,u32 cta_count){
	if(cta_count==0||cta_index>=cta_count)return;
	int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=blockDim.x>>5;
	for(int c=0;c<capacity;c++)for(int b=cta_index*warps+warp;b<k/32;b+=cta_count*warps){
		float v=c<active?x[c*k+b*32+lane]:0,a=fabsf(v);
		#pragma unroll
		for(int j=16;j;j>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,j));
		float d=a/127;int q=a==0?0:int(roundf(v/d)),sum=q;
		#pragma unroll
		for(int j=16;j;j>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,j);
		int other=__shfl_xor_sync(0xffffffff,q,1);
		if(!(lane&1))p[(c*k/32+b)*16+lane/2]=pair(q,other);
		if(lane==0){float dh=__half2float(__float2half(d));ds[c*k/32+b]=make_float2(dh,dh*sum);}
	}
}
extern "C" __global__ void packed_prepare_probe(const float *x,u32 *p,float2 *ds,int k,int n){packed_prepare(x,p,ds,k,n,n,blockIdx.x,gridDim.x);}
static __device__ __forceinline__ bool valid_packed(int type,int capacity,int active,int kind,int lanes,u32 mask,int k,int m,u32 cta_index,u32 cta_count,u32 *scratch){
	return active>=0&&active<=capacity&&kind>=0&&kind<=2&&(kind!=2||type==17||type==18||type==20)&&(lanes==8||lanes==16||(type==20&&lanes==4))&&k>0&&m>0&&!(uintptr_t(scratch)&15)&&cta_count>0&&cta_count<=gridDim.x&&cta_index<cta_count&&
		(type==8||type==12||type==13||type==14||type==17||type==18||type==20)&&!(k%(type==8||type==20?32:256))&&
		(capacity==1||capacity==2||capacity==4||capacity==8)&&(!mask||__popc(mask)==active)&&blockDim.y==1&&blockDim.z==1;
}
// One shared-input walk produces SiLU(gate) * up in compact expert columns.
template<int T,int N,int NW,int LW>static __device__ __forceinline__ void fused_pair(const uint8_t *W,const uint8_t *U,const u32 *x,const float2 *ds,float *product,int k,int m,int active,u32 mask,u32 index,u32 count,u32 *scratch){
	constexpr int NT=NW*32,TU=LW,YS=20,NY=(TU*4+NT-1)/NT;
	u32 (*ys)[2][TU*YS]=(u32 (*)[2][TU*YS])scratch;float2 (*dss)[2][TU]=(float2 (*)[2][TU])(scratch+N*2*TU*YS);
	int tid=threadIdx.x,l=tid&(LW-1),nb=k/32,nt=(nb+TU-1)/TU,source[N];
	prepare_iq_tables<T>(scratch);
	#pragma unroll
	for(int c=0;c<N;c++){u32 bits=mask;for(int j=0;j<c;j++)bits&=bits-1;source[c]=c>=active?-1:mask?__ffs(bits)-1:c;}
	for(int base=index;base<m;base+=count*(NT/LW)){
		int row=base+(tid/LW)*count;const uint8_t *rp=W+size_t(row<m?row:0)*(k/Format<T>::block)*Format<T>::bytes,*up=U+size_t(row<m?row:0)*(k/Format<T>::block)*Format<T>::bytes;
		uint4 yr[N][NY];float2 dr[N];
		#pragma unroll
		for(int c=0;c<N;c++){
			#pragma unroll
			for(int i=0;i<NY;i++){int ii=tid+i*NT;yr[c][i]=make_uint4(0,0,0,0);if(source[c]>=0&&ii<TU*4&&ii<nb*4)yr[c][i]=__ldg((const uint4*)x+source[c]*nb*4+ii);if(ii<TU*4)store_shared4(&ys[c][0][(ii/4)*YS+(ii%4)*4],yr[c][i]);}
			dr[c]=make_float2(0,0);if(source[c]>=0&&tid<TU&&tid<nb)dr[c]=__ldg(ds+source[c]*nb+tid);if(tid<TU)store_shared2(&dss[c][0][tid],dr[c]);
		}
		__syncthreads();float gate[N]={},upper[N]={};
		for(int tile=0;tile<nt;tile++){
			int buf=tile&1;
			if(tile+1<nt){
				#pragma unroll
				for(int c=0;c<N;c++){
					#pragma unroll
					for(int i=0;i<NY;i++){int ii=tid+i*NT;yr[c][i]=make_uint4(0,0,0,0);if(source[c]>=0&&ii<TU*4&&(tile+1)*TU*4+ii<nb*4)yr[c][i]=__ldg((const uint4*)x+source[c]*nb*4+(tile+1)*TU*4+ii);}
					dr[c]=make_float2(0,0);if(source[c]>=0&&tid<TU&&(tile+1)*TU+tid<nb)dr[c]=__ldg(ds+source[c]*nb+(tile+1)*TU+tid);
				}
			}
			int b=tile*TU+l;int gi0[N]={},gi1[N]={},ui0[N]={},ui1[N]={};float dg=0,du=0,sg0=0,sg1=0,su0=0,su1=0;
			#pragma unroll
			for(int sub=0;sub<4;sub++){
				u32 wg[4],wu[4];float sg,su;
				decode_stream8<T,true>(rp,b<nb?b:0,sub,wg,dg,sg,nullptr,scratch,0);decode_stream8<T,true>(up,b<nb?b:0,sub,wu,du,su,nullptr,scratch,0);
				if(sub==0){sg0=sg;su0=su;}if(sub==2){sg1=sg;su1=su;}
				#pragma unroll
				for(int c=0;c<N;c++){uint4 v=load_shared4(&ys[c][buf][l*YS+4*sub]);u32 xv[4]={v.x,v.y,v.z,v.w};int ag=0,au=0;
					#pragma unroll
					for(int j=0;j<4;j++){ag=mad2(wg[j],xv[j],ag);au=mad2(wu[j],xv[j],au);}if constexpr(T==18){gi0[c]+=ag;ui0[c]+=au;}else{if(sub<2){gi0[c]+=ag;ui0[c]+=au;}else{gi1[c]+=ag;ui1[c]+=au;}}}
			}
			#pragma unroll
			for(int c=0;c<N;c++){float q=load_shared2(&dss[c][buf][l]).x;if constexpr(T==18){gate[c]=fmaf(dg*sg0*q,float(gi0[c]),gate[c]);upper[c]=fmaf(du*su0*q,float(ui0[c]),upper[c]);}else{gate[c]=fmaf(dg*q,fmaf(sg0,float(gi0[c]),sg1*float(gi1[c])),gate[c]);upper[c]=fmaf(du*q,fmaf(su0,float(ui0[c]),su1*float(ui1[c])),upper[c]);}}
			if(tile+1<nt){
				#pragma unroll
				for(int c=0;c<N;c++){
					#pragma unroll
					for(int i=0;i<NY;i++){int ii=tid+i*NT;if(ii<TU*4)store_shared4(&ys[c][buf^1][(ii/4)*YS+(ii%4)*4],yr[c][i]);}
					if(tid<TU)store_shared2(&dss[c][buf^1][tid],dr[c]);
				}
			}
			__syncthreads();
		}
		#pragma unroll
		for(int c=0;c<N;c++){
			#pragma unroll
			for(int j=LW/2;j;j>>=1){gate[c]+=__shfl_xor_sync(0xffffffff,gate[c],j);upper[c]+=__shfl_xor_sync(0xffffffff,upper[c],j);}
			if(l==0&&row<m&&c<active)product[c*m+row]=(gate[c]/(1.0f+expf(-gate[c])))*upper[c];
		}
		__syncthreads();
	}
}
#define FUSED(T,N,W,L) extern "C" __device__ __noinline__ u32 packed_gate_up_##T##_##N##_##W##_##L(const uint8_t *g,const uint8_t *u,const u32 *x,const float2 *ds,float *out,int active,u32 mask,u32 index,u32 count){extern __shared__ __align__(16) u32 scratch[];if(active<0||active>N||!count||index>=count||(mask&&__popc(mask)!=active))return 0;if(active==0)return 1;fused_pair<T,N,W,L>(g,u,x,ds,out,2560,640,active,mask,index,count,scratch);return 1;}
#define FUSEDN(T,N) FUSED(T,N,8,8) FUSED(T,N,8,16) FUSED(T,N,16,8) FUSED(T,N,16,16)
FUSEDN(17,1) FUSEDN(17,2) FUSEDN(18,1) FUSEDN(18,2)
G(17,1,8,8,0)
G(17,1,8,16,0)
G(17,1,16,8,0)
G(17,1,16,16,0)
G(17,2,8,8,0)
G(17,2,8,16,0)
G(17,2,16,8,0)
G(17,2,16,16,0)
G(18,1,8,8,2)
G(18,1,8,16,2)
G(18,1,16,8,2)
G(18,1,16,16,2)
G(18,2,8,8,2)
G(18,2,8,16,2)
G(18,2,16,8,2)
G(18,2,16,16,2)
G(20,1,8,8,2)
G(20,1,8,16,2)
G(20,1,16,8,2)
G(20,1,16,16,2)
G(20,2,8,8,2)
G(20,2,8,16,2)
G(20,2,16,8,2)
G(20,2,16,16,2)
extern "C" __device__ __noinline__ u32 packed_matvec(int type,int capacity,int active,int kind,int lanes,u32 position_mask,const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,u32 cta_index,u32 cta_count,u32 *scratch){
	if(!valid_packed(type,capacity,active,kind,lanes,position_mask,k,m,cta_index,cta_count,scratch))return 0;
	if(active==0)return 1;
	if(type==17&&capacity==1&&blockDim.x==256&&kind==0&&lanes==8){packed_g_17_1_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==17&&capacity==1&&blockDim.x==256&&kind==0&&lanes==16){packed_g_17_1_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==17&&capacity==1&&blockDim.x==512&&kind==0&&lanes==8){packed_g_17_1_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==17&&capacity==1&&blockDim.x==512&&kind==0&&lanes==16){packed_g_17_1_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==17&&capacity==2&&blockDim.x==256&&kind==0&&lanes==8){packed_g_17_2_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==17&&capacity==2&&blockDim.x==256&&kind==0&&lanes==16){packed_g_17_2_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==17&&capacity==2&&blockDim.x==512&&kind==0&&lanes==8){packed_g_17_2_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==17&&capacity==2&&blockDim.x==512&&kind==0&&lanes==16){packed_g_17_2_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==18&&capacity==1&&blockDim.x==256&&kind==2&&lanes==8){packed_g_18_1_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==18&&capacity==1&&blockDim.x==256&&kind==2&&lanes==16){packed_g_18_1_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==18&&capacity==1&&blockDim.x==512&&kind==2&&lanes==8){packed_g_18_1_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==18&&capacity==1&&blockDim.x==512&&kind==2&&lanes==16){packed_g_18_1_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==18&&capacity==2&&blockDim.x==256&&kind==2&&lanes==8){packed_g_18_2_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==18&&capacity==2&&blockDim.x==256&&kind==2&&lanes==16){packed_g_18_2_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==18&&capacity==2&&blockDim.x==512&&kind==2&&lanes==8){packed_g_18_2_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==18&&capacity==2&&blockDim.x==512&&kind==2&&lanes==16){packed_g_18_2_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==20&&capacity==1&&blockDim.x==256&&kind==2&&lanes==8){packed_g_20_1_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==20&&capacity==1&&blockDim.x==256&&kind==2&&lanes==16){packed_g_20_1_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==20&&capacity==1&&blockDim.x==512&&kind==2&&lanes==8){packed_g_20_1_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==20&&capacity==1&&blockDim.x==512&&kind==2&&lanes==16){packed_g_20_1_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==20&&capacity==2&&blockDim.x==256&&kind==2&&lanes==8){packed_g_20_2_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==20&&capacity==2&&blockDim.x==256&&kind==2&&lanes==16){packed_g_20_2_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==20&&capacity==2&&blockDim.x==512&&kind==2&&lanes==8){packed_g_20_2_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	if(type==20&&capacity==2&&blockDim.x==512&&kind==2&&lanes==16){packed_g_20_2_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
	return 0;
}
extern "C" __device__ __noinline__ u32 packed_matvec_wide(int type,int capacity,int active,int kind,int lanes,u32 mask,const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,u32 index,u32 count,u32 *scratch){
	if(blockDim.x!=1024||capacity>2)return 0;
	return packed_matvec(type,capacity,active,kind,lanes,mask,W,x,ds,out,k,m,index,count,scratch);
}
extern "C" __global__ void __launch_bounds__(512,1) packed_probe(const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,int type,int capacity,int active,int kind,int lanes,u32 mask,u32 *status,int experts){
	extern __shared__ __align__(16) u32 scratch[];
	if(experts<1||experts>gridDim.x){if(threadIdx.x==0)status[blockIdx.x]=0;return;}
	u32 expert=blockIdx.x*experts/gridDim.x,begin=(expert*gridDim.x+experts-1)/experts,end=((expert+1)*gridDim.x+experts-1)/experts;
	size_t bytes=size_t(k)*m*(type==8?34:type==12?144:type==13?176:type==14?210:type==17?74:type==18?98:18)/(type==8||type==20?32:256);
	W+=expert*bytes;out+=size_t(expert)*m*capacity;
	u32 ok=packed_matvec(type,capacity,active,kind,lanes,mask,W,x,ds,out,k,m,blockIdx.x-begin,end-begin,scratch);
	if(threadIdx.x==0)status[blockIdx.x]=ok;
}
extern "C" __global__ void __launch_bounds__(1024,1) packed_probe_wide(const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,int type,int capacity,int active,int kind,int lanes,u32 mask,u32 *status,int experts){
	extern __shared__ __align__(16) u32 scratch[];
	if(experts<1||experts>gridDim.x){if(threadIdx.x==0)status[blockIdx.x]=0;return;}
	u32 expert=blockIdx.x*experts/gridDim.x,begin=(expert*gridDim.x+experts-1)/experts,end=((expert+1)*gridDim.x+experts-1)/experts;
	size_t bytes=size_t(k)*m*(type==8?34:type==12?144:type==13?176:type==14?210:type==17?74:type==18?98:18)/(type==8||type==20?32:256);
	W+=expert*bytes;out+=size_t(expert)*m*capacity;
	u32 ok=packed_matvec_wide(type,capacity,active,kind,lanes,mask,W,x,ds,out,k,m,blockIdx.x-begin,end-begin,scratch);
	if(threadIdx.x==0)status[blockIdx.x]=ok;
}
static __device__ __forceinline__ void local_probe_barrier(volatile u32 *flags,u32 epoch,u32 begin,u32 end){
	__syncthreads();__threadfence();__syncthreads();
	if(threadIdx.x==0){flags[blockIdx.x]=epoch;for(u32 b=begin;b<end;b++)while(flags[b]<epoch){asm volatile("":::"memory");}}
	__syncthreads();
}
#define LAYER_BODY(T,N,W,L,IS_FUSED,PREFIX) extern "C" __global__ void __launch_bounds__(W*32,1) PREFIX##_##T##_##N##_##W##_##L(const uint8_t *gate,const uint8_t *upper,const uint8_t *down,const float *input,u32 *px,float2 *ds,float *g,float *u,float *product,u32 *py,float2 *ys,float *out,u32 *flags,u32 *status,unsigned long long *trace,int experts,u32 epoch){ \
	extern __shared__ __align__(16) u32 scratch[]; \
	u32 bid=blockIdx.x,expert=bid*experts/gridDim.x,begin=(expert*gridDim.x+experts-1)/experts,end=((expert+1)*gridDim.x+experts-1)/experts,rank=bid-begin,count=end-begin; \
	int k=2560,m=640;size_t gb=size_t(k)*m*Format<T>::bytes/Format<T>::block,db=size_t(k)*m*18/32; \
	float *p=product+expert*m*N,*gp=g+expert*m*N,*up=u+expert*m*N;u32 *yp=py+expert*m*N/2;float2 *yd=ys+expert*m*N/32; \
	unsigned long long t0=clock64();packed_prepare(input,px,ds,k,N,N,bid,gridDim.x);local_probe_barrier(flags,epoch,0,gridDim.x);unsigned long long t1=clock64(); \
	u32 pair_ok=1;if(IS_FUSED){pair_ok=packed_gate_up_##T##_##N##_##W##_##L(gate+expert*gb,upper+expert*gb,px,ds,p,N,0,rank,count);} \
	else{constexpr int kind=T==17?0:2;packed_matvec(T,N,N,kind,L,0,gate+expert*gb,px,ds,gp,k,m,rank,count,scratch);packed_matvec(T,N,N,kind,L,0,upper+expert*gb,px,ds,up,k,m,rank,count,scratch); \
		local_probe_barrier(flags,epoch+1,begin,end); \
		for(u32 i=rank*blockDim.x+threadIdx.x;i<m*N;i+=count*blockDim.x)p[i]=(gp[i]/(1.0f+expf(-gp[i])))*up[i];} \
	local_probe_barrier(flags,epoch+2,begin,end);unsigned long long t2=clock64();packed_prepare(p,yp,yd,m,N,N,rank,count);local_probe_barrier(flags,epoch+3,begin,end);unsigned long long t3=clock64(); \
	u32 ok=packed_matvec(20,N,N,2,8,0,down+expert*db,yp,yd,out+expert*k*N,m,k,rank,count,scratch);unsigned long long t4=clock64(); \
	if(threadIdx.x==0){status[bid]=ok&pair_ok;trace[bid*4]=t1-t0;trace[bid*4+1]=t2-t1;trace[bid*4+2]=t3-t2;trace[bid*4+3]=t4-t3;} \
}
#define LAYER(T,N,W,L) LAYER_BODY(T,N,W,L,false,plain_layer) LAYER_BODY(T,N,W,L,true,fused_layer)
LAYER(17,1,8,8) LAYER(17,1,8,16) LAYER(17,1,16,8) LAYER(17,1,16,16)
LAYER(17,2,8,8) LAYER(17,2,8,16) LAYER(17,2,16,8) LAYER(17,2,16,16)
LAYER(18,1,8,8) LAYER(18,1,8,16) LAYER(18,1,16,8) LAYER(18,1,16,16)
LAYER(18,2,8,8) LAYER(18,2,8,16) LAYER(18,2,16,8) LAYER(18,2,16,16)
#define PAIR_PROBE(T,N,W,L) extern "C" __global__ void __launch_bounds__(W*32,1) pair_probe_##T##_##N##_##W##_##L(const uint8_t *g,const uint8_t *u,const u32 *x,const float2 *ds,float *out,int active,u32 mask,u32 index,u32 count,u32 *status){u32 ok=packed_gate_up_##T##_##N##_##W##_##L(g,u,x,ds,out,active,mask,index,count);if(threadIdx.x==0)status[blockIdx.x]=ok;}
PAIR_PROBE(17,1,8,8) PAIR_PROBE(17,1,8,16) PAIR_PROBE(17,1,16,8) PAIR_PROBE(17,1,16,16)
PAIR_PROBE(17,2,8,8) PAIR_PROBE(17,2,8,16) PAIR_PROBE(17,2,16,8) PAIR_PROBE(17,2,16,16)
PAIR_PROBE(18,1,8,8) PAIR_PROBE(18,1,8,16) PAIR_PROBE(18,1,16,8) PAIR_PROBE(18,1,16,16)
PAIR_PROBE(18,2,8,8) PAIR_PROBE(18,2,8,16) PAIR_PROBE(18,2,16,8) PAIR_PROBE(18,2,16,16)
