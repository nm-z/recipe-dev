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
static __device__ __forceinline__ float half_at(const uint8_t *p){return __half2float(__ushort_as_half(*(const uint16_t*)p));}
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
template<int T,bool DICT>static __device__ __forceinline__ void decode32(const uint8_t *row,int b,u32 (&w)[16],float &d,float &s0,float &s1,float &mn,const int *table){
	const uint8_t *p=row+(size_t)(b/(Format<T>::block/32))*Format<T>::bytes;int g=b%(Format<T>::block/32);mn=0;
	if constexpr(DICT&&T==17){
		d=half_at(p)*0.125f;s0=1+2*(p[66+g]&15);s1=1+2*(p[66+g]>>4);
		#pragma unroll
		for(int j=0;j<4;j++){uint16_t q=*(const uint16_t*)(p+2+8*g+2*j);uint4 v=__ldg(packed_iq2+q);w[4*j]=v.x;w[4*j+1]=v.y;w[4*j+2]=v.z;w[4*j+3]=v.w;}
	}else if constexpr(DICT&&T==18){
		d=half_at(p)*0.25f;u32 a=u32(*(const uint16_t*)(p+66+4*g))|(u32(*(const uint16_t*)(p+68+4*g))<<16);s0=s1=1+2*(a>>28);
		#pragma unroll
		for(int j=0;j<4;j++){u32 sign=ksigns_iq2xs[(a>>(7*j))&127];uint2 v0=__ldg(packed_iq3+p[2+8*g+2*j]+((sign&15)<<8)),v1=__ldg(packed_iq3+p[3+8*g+2*j]+((sign>>4)<<8));w[4*j]=v0.x;w[4*j+1]=v0.y;w[4*j+2]=v1.x;w[4*j+3]=v1.y;}
	}else if constexpr(DICT&&T==20){
		d=half_at(p);s0=s1=1;
		#pragma unroll
		for(int j=0;j<8;j++){uint16_t q=*(const uint16_t*)(p+2+2*j);uint2 v=__ldg(packed_iq4+q);w[j]=v.x;w[j+8]=v.y;}
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
template<int T,int N,int NW,int LW,bool MAGIC,bool DICT>static __device__ __forceinline__ void group32(const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,int row_base,int active,u32 position_mask,u32 cta_count,u32 *scratch){
	constexpr int NT=NW*32,TU=LW,YS=20,NY=(TU*4+NT-1)/NT;
	u32 (*ys)[2][TU*YS]=(u32 (*)[2][TU*YS])scratch;
	float2 (*dss)[2][TU]=(float2 (*)[2][TU])(scratch+N*2*TU*YS);
	int *table=(int*)((float2*)dss+N*2*TU);
	int tid=threadIdx.x,l=tid&(LW-1),row=row_base+(tid/LW)*cta_count,nb=k/32,nt=(nb+TU-1)/TU;
	int source[N];
	#pragma unroll
	for(int c=0;c<N;c++){u32 mask=position_mask;for(int j=0;j<c;j++)mask&=mask-1;source[c]=c>=active?-1:position_mask?__ffs(mask)-1:c;}
	if constexpr(T==20){if(tid<16)table[tid]=kvalues_iq4nl[tid];__syncthreads();}
	const uint8_t *rp=W+(size_t)(row<m?row:0)*(k/Format<T>::block)*Format<T>::bytes;
	uint4 yr[N][NY];float2 dr[N];
	#pragma unroll
	for(int c=0;c<N;c++){
		#pragma unroll
		for(int i=0;i<NY;i++){int ii=tid+i*NT;yr[c][i]=make_uint4(0,0,0,0);if(source[c]>=0&&ii<TU*4&&ii<nb*4)yr[c][i]=__ldg((const uint4*)x+source[c]*nb*4+ii);if(ii<TU*4)store_shared4(&ys[c][0][(ii/4)*YS+(ii%4)*4],yr[c][i]);}
		dr[c]=make_float2(0,0);if(source[c]>=0&&tid<TU&&tid<nb)dr[c]=__ldg(ds+source[c]*nb+tid);if(tid<TU)store_shared2(&dss[c][0][tid],dr[c]);
	}
	u32 w[16];float d,s0,s1,mn;if(row<m)decode32<T,DICT>(rp,l<nb?l:0,w,d,s0,s1,mn,table);
	__syncthreads();float acc[N]={};
	for(int tile=0;tile<nt;tile++){
		int buf=tile&1;u32 next[16];float nd,ns0,ns1,nmn;
		if(tile+1<nt){
			int b=(tile+1)*TU+l;if(row<m)decode32<T,DICT>(rp,b<nb?b:0,next,nd,ns0,ns1,nmn,table);
			#pragma unroll
			for(int c=0;c<N;c++){
				#pragma unroll
				for(int i=0;i<NY;i++){int ii=tid+i*NT;yr[c][i]=make_uint4(0,0,0,0);if(source[c]>=0&&ii<TU*4&&(tile+1)*TU*4+ii<nb*4)yr[c][i]=__ldg((const uint4*)x+source[c]*nb*4+(tile+1)*TU*4+ii);}
				dr[c]=make_float2(0,0);if(source[c]>=0&&tid<TU&&(tile+1)*TU+tid<nb)dr[c]=__ldg(ds+source[c]*nb+(tile+1)*TU+tid);
			}
		}
		#pragma unroll
		for(int c=0;c<N;c++){
			if(c>=active||row>=m)continue;
			const u32 *xp=&ys[c][buf][l*YS];uint4 a=load_shared4(xp),b=load_shared4(xp+4),e=load_shared4(xp+8),f=load_shared4(xp+12);
			u32 xv[16]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w,e.x,e.y,e.z,e.w,f.x,f.y,f.z,f.w};float z0,z1;
			if constexpr(!MAGIC){int a0=0,a1=0;
				#pragma unroll
				for(int j=0;j<8;j++){a0=mad2(w[j],xv[j],a0);a1=mad2(w[j+8],xv[j+8],a1);}z0=a0;z1=a1;
			}else{float a0=0,a1=0,b0=0,b1=0;
				#pragma unroll
				for(int j=0;j<8;j++){u32 x0=xv[j],x1=xv[j+8];a0=fmaf(magic((int16_t)w[j]),float((int16_t)x0),a0);b0=fmaf(magic((int16_t)(w[j]>>16)),float((int16_t)(x0>>16)),b0);a1=fmaf(magic((int16_t)w[j+8]),float((int16_t)x1),a1);b1=fmaf(magic((int16_t)(w[j+8]>>16)),float((int16_t)(x1>>16)),b1);}z0=a0+b0;z1=a1+b1;
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
			if(row<m){
				#pragma unroll
				for(int j=0;j<16;j++)w[j]=next[j];d=nd;s0=ns0;s1=ns1;mn=nmn;
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
template<int T,int N,bool DICT>static __device__ __forceinline__ void decode32_fast(const uint8_t *row,int b,u32 (&w)[16],float &d,float &s0,float &s1,float &mn,const int *,const u32 *,int warp_lut){
	const uint8_t *p=row+size_t(b)*18;d=half_at(p);s0=s1=1;mn=0;
	#pragma unroll
	for(int j=0;j<8;j++){u32 q=__ldg((const uint16_t*)(p+2+2*j));int a=__shfl_sync(0xffffffff,warp_lut,q&15),b=__shfl_sync(0xffffffff,warp_lut,(q>>8)&15),c=__shfl_sync(0xffffffff,warp_lut,(q>>4)&15),e=__shfl_sync(0xffffffff,warp_lut,q>>12);w[j]=__byte_perm(a,b,0x5410);w[j+8]=__byte_perm(c,e,0x5410);}
}
template<int T,int N,int NW,int LW,bool MAGIC,bool DICT>static __device__ __forceinline__ void group32_fast(const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,int row_base,int active,u32 position_mask,u32 cta_count,u32 *scratch,int warp_lut){
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
		if(row<m||(T==20&&DICT))decode32_fast<T,N,DICT>(rp,b<nb?b:0,w,d,s0,s1,mn,table,scratch,warp_lut);
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
#define G(T,N,W,L,K) extern "C" __device__ __noinline__ void packed_g_##T##_##N##_##W##_##K##_##L(const uint8_t *a,const u32 *b,const float2 *c,float *d,int k,int m,int active,u32 mask,u32 cta_index,u32 cta_count,u32 *scratch){if constexpr(T==20&&K==2&&N<=2&&L==8&&(W==8||W==16)){int lut=__ldg(kvalues_iq4nl+(threadIdx.x&15));for(int base=cta_index;base<m;base+=cta_count*(W*32/L))group32_fast<T,N,W,L,false,true>(a,b,c,d,k,m,base,active,mask,cta_count,scratch,lut);return;}for(int base=cta_index;base<m;base+=cta_count*(W*32/L))group32<T,N,W,L,(K==1),(K==2)>(a,b,c,d,k,m,base,active,mask,cta_count,scratch);}
#define GK(T,N,W,L) G(T,N,W,L,0) G(T,N,W,L,1)
#define GW(T,N,L) GK(T,N,2,L) GK(T,N,4,L) GK(T,N,8,L) GK(T,N,16,L)
#define GN(T,N) GW(T,N,8) GW(T,N,16)
#define GT(T) GN(T,1) GN(T,2) GN(T,4) GN(T,8)
GT(8) GT(12) GT(13) GT(14) GT(17) GT(18) GT(20)
#define WIDEG(T,N) GK(T,N,32,8) GK(T,N,32,16)
#define WIDET(T) WIDEG(T,1) WIDEG(T,2)
WIDET(8) WIDET(12) WIDET(13) WIDET(14) WIDET(17) WIDET(18) WIDET(20)
#define DW(T,N,L) G(T,N,2,L,2) G(T,N,4,L,2) G(T,N,8,L,2) G(T,N,16,L,2)
#define DN(T,N) DW(T,N,8) DW(T,N,16)
#define DT(T) DN(T,1) DN(T,2) DN(T,4) DN(T,8) G(T,1,32,8,2) G(T,1,32,16,2) G(T,2,32,8,2) G(T,2,32,16,2)
DT(17) DT(18) DT(20)
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
	const uint8_t *bp=rowp+(sbl<nsb?sbl:0)*BYTES;
	uint4 q0=*(const uint4*)(bp+QS+32*g),q1=*(const uint4*)(bp+QS+16+32*g),h0,h1;
	if constexpr(T==13){h0=*(const uint4*)(bp+16);h1=*(const uint4*)(bp+32);}
	u32 S0=*(const u32*)(bp+4),S1=*(const u32*)(bp+8),S2=*(const u32*)(bp+12),dd=*(const u32*)bp;
	__syncthreads();float acc[N]={};
	for(int t=0;t<nt;t++){
		int buf=t&1;
		uint4 n0,n1,nh0,nh1;u32 nS0,nS1,nS2,ndd;
		if(t+1<nt){
			int sb=(t+1)*TSB+sbl;const uint8_t *np=rowp+(sb<nsb?sb:0)*BYTES;
			n0=*(const uint4*)(np+QS+32*g);n1=*(const uint4*)(np+QS+16+32*g);nS0=*(const u32*)(np+4);nS1=*(const u32*)(np+8);nS2=*(const u32*)(np+12);ndd=*(const u32*)np;
			if constexpr(T==13){nh0=*(const uint4*)(np+16);nh1=*(const uint4*)(np+32);}
			#pragma unroll
			for(int c=0;c<N;c++){
				#pragma unroll
				for(int i=0;i<NY;i++){int ii=tid+i*NT;yr[c][i]=make_uint4(0,0,0,0);if(source[c]>=0&&ii<TU*8&&(t+1)*TU*8+ii<k/8)yr[c][i]=((const uint4*)y16)[source[c]*k/8+(t+1)*TU*8+ii];}
				dr[c]=make_float2(0,0);if(source[c]>=0&&tid<TU*2&&(t+1)*TU*2+tid<k/32)dr[c]=dsf[source[c]*k/32+(t+1)*TU*2+tid];
			}
		}
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
			q0=n0;q1=n1;S0=nS0;S1=nS1;S2=nS2;dd=ndd;
			if constexpr(T==13){h0=nh0;h1=nh1;}
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
WIDTHT(12) WIDTHT(13)
WIDTH(12,1,32,8) WIDTH(12,1,32,16) WIDTH(12,2,32,8) WIDTH(12,2,32,16)
WIDTH(13,1,32,8) WIDTH(13,1,32,16) WIDTH(13,2,32,8) WIDTH(13,2,32,16)
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
	return active>=0&&active<=capacity&&kind>=0&&kind<=2&&(kind!=2||type==17||type==18||type==20)&&(lanes==8||lanes==16)&&k>0&&m>0&&!(uintptr_t(scratch)&15)&&cta_count>0&&cta_count<=gridDim.x&&cta_index<cta_count&&
		(type==8||type==12||type==13||type==14||type==17||type==18||type==20)&&!(k%(type==8||type==20?32:256))&&
		(capacity==1||capacity==2||capacity==4||capacity==8)&&(!mask||__popc(mask)==active)&&blockDim.y==1&&blockDim.z==1;
}
static __device__ __forceinline__ u32 fused_shared_word(u32 a){u32 v;asm("ld.shared.b32 %0,[%1];":"=r"(v):"r"(a));return v;}
template<int T,int N>static __device__ __forceinline__ void prepare_stream_pairs(u32 *scratch){
	if constexpr(T==17){}
	else{
		constexpr int SIG=N==1?1792:3840;
		for(int i=threadIdx.x;i<128;i+=blockDim.x){u32 sign=__ldg(ksigns_iq2xs+i);scratch[SIG+i]=((sign&15)<<8)|((sign>>4)<<24);}
		if constexpr(N==1){for(int i=threadIdx.x;i<2048;i+=blockDim.x)store_shared4(scratch+2048+i*4,__ldg((const uint4*)packed_iq3+i));}
	}
	__syncthreads();
}
static __device__ __forceinline__ void iq3_group(const uint8_t *row,u32 block,u32 (&grids)[4],u32 &sign,float &d){
	const uint8_t *p=row+(block/8)*98;u32 g=block%8,a,b;
	asm("ld.global.u16 %0,[%2];ld.global.u16 %1,[%2+2];":"=r"(a),"=r"(b):"l"(p+66+4*g));sign=a|(b<<16);d=half_at(p);
	asm("ld.global.nc.u16 %0,[%4];ld.global.nc.u16 %1,[%4+2];ld.global.nc.u16 %2,[%4+4];ld.global.nc.u16 %3,[%4+6];":"=r"(grids[0]),"=r"(grids[1]),"=r"(grids[2]),"=r"(grids[3]):"l"(p+2+8*g));
}
template<int T,int N>static __device__ __forceinline__ void decode_stream_pairs(const uint8_t *row,u32 block,u32 sub,u32 (&words)[4],float &d,float &scale,const u32 *scratch,const u32 (&group_grids)[4],u32 group_sign){
	const uint8_t *p=row+(block/8)*Format<T>::bytes;u32 g=block%8,base=__cvta_generic_to_shared(scratch);
	if constexpr(T==17){
		d=half_at(p)*0.125f;scale=1+2*((p[66+g]>>(4*(sub/2)))&15);
		u32 q=__ldg((const uint16_t*)(p+2+8*g+2*sub));uint4 values=__ldg(packed_iq2+q);
		words[0]=values.x;words[1]=values.y;words[2]=values.z;words[3]=values.w;
	}else{
		constexpr int SIG=N==1?1792:3840;
		u32 a=group_sign;scale=fmaf(float(a>>28),0.5f,0.25f);
		u32 offsets=fused_shared_word(base+SIG*4+(((a>>(7*sub))&127)<<2)),grids=group_grids[sub];
		u32 codes=__byte_perm(grids,0,0x4140)|offsets,c0=codes&65535,c1=codes>>16;uint2 v0,v1;
		if constexpr(N==1){
			u32 table=base+8192;
			asm("{.reg .b32 a,b;bfi.b32 a,%4,0,3,12;shr.u32 b,%4,13;add.u32 a,a,%5;add.u32 b,b,%5;ld.shared.v2.b32 {%0,%1},[a];ld.shared.v2.b32 {%2,%3},[b];}":"=r"(v0.x),"=r"(v0.y),"=r"(v1.x),"=r"(v1.y):"r"(codes),"r"(table));
		}
		else{v0=__ldg(packed_iq3+c0);v1=__ldg(packed_iq3+c1);}
		words[0]=v0.x;words[1]=v0.y;words[2]=v1.x;words[3]=v1.y;
	}
}
// One shared-input walk produces SiLU(gate) * up in compact expert columns.
template<int T,int N,int NW,int LW>static __device__ __forceinline__ void fused_pair(const uint8_t *W,const uint8_t *U,const u32 *x,const float2 *ds,float *product,int k,int m,int active,u32 mask,u32 index,u32 count,u32 *scratch){
	constexpr int NT=NW*32,NB=80,YS=20;
	u32 (*ys)[NB*YS]=(u32 (*)[NB*YS])scratch;float2 (*dss)[NB]=(float2 (*)[NB])(scratch+N*NB*YS);
	int tid=threadIdx.x,l=tid&(LW-1),source[N];prepare_stream_pairs<T,N>(scratch);
	#pragma unroll
	for(int c=0;c<N;c++){u32 bits=mask;for(int j=0;j<c;j++)bits&=bits-1;source[c]=c>=active?-1:mask?__ffs(bits)-1:c;}
	#pragma unroll
	for(int c=0;c<N;c++){
		for(int j=tid;j<NB*4;j+=NT){uint4 v=make_uint4(0,0,0,0);if(source[c]>=0)v=__ldg((const uint4*)x+source[c]*NB*4+j);store_shared4(&ys[c][(j/4)*YS+(j%4)*4],v);}
		for(int b=tid;b<NB;b+=NT){float2 v=make_float2(0,0);if(source[c]>=0)v=__ldg(ds+source[c]*NB+b);store_shared2(&dss[c][b],v);}
	}
	__syncthreads();
	for(int base=index;base<m;base+=count*(NT/LW)){
		int row=base+(tid/LW)*count;const uint8_t *rp=W+u32(row<m?row:0)*u32((k/Format<T>::block)*Format<T>::bytes),*up=U+u32(row<m?row:0)*u32((k/Format<T>::block)*Format<T>::bytes);float gate[N]={},upper[N]={};
		#pragma unroll (T==18&&N==1?2:1)
		for(u32 b=u32(l);b<NB;b+=LW){
			int gi0[N]={},gi1[N]={},ui0[N]={},ui1[N]={};float dg=0,du=0,sg0=0,sg1=0,su0=0,su1=0;
			u32 gg[4]={},ug[4]={},gs=0,us=0;if constexpr(T==18){iq3_group(rp,b,gg,gs,dg);iq3_group(up,b,ug,us,du);}
			#pragma unroll
			for(int sub=0;sub<4;sub++){
				u32 wg[4],wu[4];float sg,su;decode_stream_pairs<T,N>(rp,b,sub,wg,dg,sg,scratch,gg,gs);decode_stream_pairs<T,N>(up,b,sub,wu,du,su,scratch,ug,us);
				if(sub==0){sg0=sg;su0=su;}if(sub==2){sg1=sg;su1=su;}
				#pragma unroll
				for(int c=0;c<N;c++){uint4 v=load_shared4(&ys[c][b*YS+4*sub]);u32 xv[4]={v.x,v.y,v.z,v.w};int ag=0,au=0;
					#pragma unroll
					for(int j=0;j<4;j++){if constexpr(T==18){gi0[c]=mad2(wg[j],xv[j],gi0[c]);ui0[c]=mad2(wu[j],xv[j],ui0[c]);}else{ag=mad2(wg[j],xv[j],ag);au=mad2(wu[j],xv[j],au);}}if constexpr(T!=18){if(sub<2){gi0[c]+=ag;ui0[c]+=au;}else{gi1[c]+=ag;ui1[c]+=au;}}}
			}
			#pragma unroll
			for(int c=0;c<N;c++){float q=load_shared2(&dss[c][b]).x;if constexpr(T==18){gate[c]=fmaf(dg*sg0*q,float(gi0[c]),gate[c]);upper[c]=fmaf(du*su0*q,float(ui0[c]),upper[c]);}else{gate[c]=fmaf(dg*q,fmaf(sg0,float(gi0[c]),sg1*float(gi1[c])),gate[c]);upper[c]=fmaf(du*q,fmaf(su0,float(ui0[c]),su1*float(ui1[c])),upper[c]);}}
		}
		#pragma unroll
		for(int c=0;c<N;c++){
			#pragma unroll
			for(int j=LW/2;j;j>>=1){gate[c]+=__shfl_xor_sync(0xffffffff,gate[c],j);upper[c]+=__shfl_xor_sync(0xffffffff,upper[c],j);}
			if(l==0&&row<m&&c<active)product[c*m+row]=(gate[c]/(1.0f+expf(-gate[c])))*upper[c];
		}
	}
}
#define FUSED(T,N,W,L) extern "C" __device__ __noinline__ u32 packed_gate_up_##T##_##N##_##W##_##L(const uint8_t *g,const uint8_t *u,const u32 *x,const float2 *ds,float *out,int active,u32 mask,u32 index,u32 count){extern __shared__ __align__(16) u32 scratch[];if(active<0||active>N||!count||index>=count||(mask&&__popc(mask)!=active))return 0;if(active==0)return 1;fused_pair<T,N,W,L>(g,u,x,ds,out,2560,640,active,mask,index,count,scratch);return 1;}
#define FUSEDN(T,N) FUSED(T,N,8,8) FUSED(T,N,8,16) FUSED(T,N,16,8) FUSED(T,N,16,16)
FUSEDN(17,1) FUSEDN(17,2) FUSEDN(18,1) FUSEDN(18,2)
extern "C" __device__ __noinline__ u32 packed_gate_up(int type,int capacity,int active,int lanes,u32 mask,const uint8_t *gate,const uint8_t *up,const u32 *x,const float2 *ds,float *product,int k,int m,u32 rank,u32 count){
	if(k!=2560||m!=640)return 0;
	if(blockDim.x==256&&lanes==16){
		if(type==17&&capacity==1)return packed_gate_up_17_1_8_16(gate,up,x,ds,product,active,mask,rank,count);
		if(type==17&&capacity==2)return packed_gate_up_17_2_8_16(gate,up,x,ds,product,active,mask,rank,count);
		if(type==18&&capacity==1)return packed_gate_up_18_1_8_16(gate,up,x,ds,product,active,mask,rank,count);
		if(type==18&&capacity==2)return packed_gate_up_18_2_8_16(gate,up,x,ds,product,active,mask,rank,count);
	}
	if(blockDim.x==512){
		if(type==17&&capacity==1&&lanes==8)return packed_gate_up_17_1_16_8(gate,up,x,ds,product,active,mask,rank,count);
		if(type==17&&capacity==1&&lanes==16)return packed_gate_up_17_1_16_16(gate,up,x,ds,product,active,mask,rank,count);
		if(type==17&&capacity==2&&lanes==8)return packed_gate_up_17_2_16_8(gate,up,x,ds,product,active,mask,rank,count);
		if(type==17&&capacity==2&&lanes==16)return packed_gate_up_17_2_16_16(gate,up,x,ds,product,active,mask,rank,count);
		if(type==18&&capacity==1&&lanes==8)return packed_gate_up_18_1_16_8(gate,up,x,ds,product,active,mask,rank,count);
		if(type==18&&capacity==1&&lanes==16)return packed_gate_up_18_1_16_16(gate,up,x,ds,product,active,mask,rank,count);
		if(type==18&&capacity==2&&lanes==8)return packed_gate_up_18_2_16_8(gate,up,x,ds,product,active,mask,rank,count);
		if(type==18&&capacity==2&&lanes==16)return packed_gate_up_18_2_16_16(gate,up,x,ds,product,active,mask,rank,count);
	}
	return 0;
}
extern "C" __device__ __noinline__ u32 packed_matvec_wide(int type,int capacity,int active,int kind,int lanes,u32 position_mask,const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,u32 cta_index,u32 cta_count,u32 *scratch){
	if(!valid_packed(type,capacity,active,kind,lanes,position_mask,k,m,cta_index,cta_count,scratch))return 0;
	if(blockDim.x!=1024||capacity>2)return 0;
	if(active==0)return 1;
	if(type==8){
		if(capacity==1){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_8_1_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_1_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_1_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_1_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_8_2_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_2_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_2_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_2_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==12){
		if(capacity==1){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_h_12_1_32_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_1_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_1_32_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_1_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_h_12_2_32_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_2_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_2_32_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_2_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==13){
		if(capacity==1){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_h_13_1_32_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_1_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_1_32_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_1_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_h_13_2_32_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_2_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_2_32_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_2_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==14){
		if(capacity==1){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_14_1_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_1_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_1_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_1_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_14_2_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_2_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_2_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_2_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==17){
		if(capacity==1){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_17_1_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_1_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_1_32_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_1_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_1_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_1_32_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_17_2_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_2_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_2_32_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_2_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_2_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_2_32_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==18){
		if(capacity==1){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_18_1_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_1_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_1_32_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_1_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_1_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_1_32_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_18_2_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_2_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_2_32_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_2_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_2_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_2_32_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==20){
		if(capacity==1){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_20_1_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_1_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_1_32_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_1_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_1_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_1_32_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==1024){
				if(lanes==8&&kind==0){packed_g_20_2_32_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_2_32_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_2_32_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_2_32_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_2_32_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_2_32_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	return 0;
}
extern "C" __device__ __noinline__ u32 packed_matvec(int type,int capacity,int active,int kind,int lanes,u32 position_mask,const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,u32 cta_index,u32 cta_count,u32 *scratch){
	if(!valid_packed(type,capacity,active,kind,lanes,position_mask,k,m,cta_index,cta_count,scratch))return 0;
	if(blockDim.x==1024)return packed_matvec_wide(type,capacity,active,kind,lanes,position_mask,W,x,ds,out,k,m,cta_index,cta_count,scratch);
	if(blockDim.x!=64&&blockDim.x!=128&&blockDim.x!=256&&blockDim.x!=512)return 0;
	if(active==0)return 1;
	if(type==8){
		if(capacity==1){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_8_1_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_1_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_1_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_1_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_8_1_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_1_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_1_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_1_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_8_1_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_1_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_1_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_1_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_8_1_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_1_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_1_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_1_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_8_2_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_2_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_2_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_2_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_8_2_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_2_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_2_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_2_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_8_2_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_2_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_2_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_2_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_8_2_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_2_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_2_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_2_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==4){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_8_4_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_4_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_4_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_4_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_8_4_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_4_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_4_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_4_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_8_4_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_4_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_4_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_4_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_8_4_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_4_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_4_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_4_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==8){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_8_8_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_8_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_8_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_8_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_8_8_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_8_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_8_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_8_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_8_8_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_8_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_8_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_8_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_8_8_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_8_8_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_8_8_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_8_8_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==12){
		if(capacity==1){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_h_12_1_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_1_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_1_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_1_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_h_12_1_4_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_1_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_1_4_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_1_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_h_12_1_8_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_1_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_1_8_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_1_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_h_12_1_16_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_1_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_1_16_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_1_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_h_12_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_2_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_2_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_h_12_2_4_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_2_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_2_4_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_2_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_h_12_2_8_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_2_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_2_8_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_2_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_h_12_2_16_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_2_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_2_16_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_2_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==4){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_h_12_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_4_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_4_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_h_12_4_4_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_4_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_4_4_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_4_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_h_12_4_8_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_4_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_4_8_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_4_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_h_12_4_16_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_4_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_4_16_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_4_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==8){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_h_12_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_8_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_8_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_h_12_8_4_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_8_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_8_4_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_8_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_h_12_8_8_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_8_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_8_8_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_8_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_h_12_8_16_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_12_8_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_12_8_16_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_12_8_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==13){
		if(capacity==1){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_h_13_1_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_1_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_1_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_1_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_h_13_1_4_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_1_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_1_4_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_1_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_h_13_1_8_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_1_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_1_8_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_1_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_h_13_1_16_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_1_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_1_16_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_1_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_h_13_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_2_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_2_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_h_13_2_4_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_2_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_2_4_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_2_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_h_13_2_8_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_2_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_2_8_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_2_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_h_13_2_16_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_2_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_2_16_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_2_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==4){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_h_13_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_4_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_4_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_h_13_4_4_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_4_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_4_4_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_4_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_h_13_4_8_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_4_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_4_8_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_4_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_h_13_4_16_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_4_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_4_16_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_4_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==8){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_h_13_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_8_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_8_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_h_13_8_4_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_8_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_8_4_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_8_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_h_13_8_8_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_8_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_8_8_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_8_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_h_13_8_16_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_13_8_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_h_13_8_16_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_13_8_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==14){
		if(capacity==1){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_14_1_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_1_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_1_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_1_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_14_1_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_1_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_1_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_1_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_14_1_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_1_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_1_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_1_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_14_1_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_1_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_1_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_1_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_14_2_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_2_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_2_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_2_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_14_2_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_2_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_2_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_2_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_14_2_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_2_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_2_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_2_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_14_2_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_2_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_2_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_2_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==4){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_14_4_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_4_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_4_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_4_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_14_4_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_4_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_4_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_4_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_14_4_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_4_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_4_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_4_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_14_4_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_4_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_4_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_4_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==8){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_14_8_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_8_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_8_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_8_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_14_8_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_8_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_8_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_8_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_14_8_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_8_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_8_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_8_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_14_8_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_14_8_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_14_8_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_14_8_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==17){
		if(capacity==1){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_17_1_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_1_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_1_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_1_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_1_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_1_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_17_1_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_1_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_1_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_1_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_1_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_1_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_17_1_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_1_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_1_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_1_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_1_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_1_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_17_1_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_1_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_1_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_1_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_1_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_1_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_17_2_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_2_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_2_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_2_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_2_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_2_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_17_2_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_2_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_2_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_2_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_2_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_2_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_17_2_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_2_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_2_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_2_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_2_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_2_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_17_2_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_2_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_2_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_2_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_2_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_2_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==4){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_17_4_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_4_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_4_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_4_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_4_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_4_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_17_4_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_4_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_4_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_4_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_4_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_4_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_17_4_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_4_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_4_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_4_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_4_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_4_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_17_4_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_4_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_4_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_4_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_4_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_4_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==8){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_17_8_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_8_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_8_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_8_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_8_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_8_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_17_8_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_8_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_8_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_8_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_8_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_8_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_17_8_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_8_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_8_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_8_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_8_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_8_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_17_8_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_17_8_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_17_8_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_17_8_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_17_8_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_17_8_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==18){
		if(capacity==1){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_18_1_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_1_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_1_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_1_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_1_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_1_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_18_1_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_1_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_1_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_1_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_1_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_1_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_18_1_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_1_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_1_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_1_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_1_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_1_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_18_1_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_1_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_1_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_1_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_1_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_1_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_18_2_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_2_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_2_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_2_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_2_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_2_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_18_2_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_2_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_2_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_2_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_2_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_2_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_18_2_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_2_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_2_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_2_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_2_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_2_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_18_2_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_2_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_2_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_2_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_2_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_2_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==4){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_18_4_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_4_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_4_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_4_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_4_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_4_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_18_4_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_4_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_4_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_4_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_4_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_4_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_18_4_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_4_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_4_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_4_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_4_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_4_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_18_4_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_4_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_4_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_4_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_4_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_4_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==8){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_18_8_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_8_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_8_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_8_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_8_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_8_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_18_8_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_8_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_8_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_8_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_8_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_8_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_18_8_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_8_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_8_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_8_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_8_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_8_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_18_8_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_18_8_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_18_8_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_18_8_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_18_8_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_18_8_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	if(type==20){
		if(capacity==1){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_20_1_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_1_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_1_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_1_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_1_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_1_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_20_1_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_1_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_1_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_1_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_1_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_1_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_20_1_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_1_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_1_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_1_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_1_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_1_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_20_1_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_1_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_1_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_1_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_1_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_1_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==2){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_20_2_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_2_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_2_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_2_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_2_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_2_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_20_2_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_2_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_2_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_2_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_2_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_2_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_20_2_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_2_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_2_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_2_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_2_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_2_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_20_2_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_2_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_2_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_2_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_2_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_2_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==4){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_20_4_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_4_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_4_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_4_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_4_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_4_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_20_4_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_4_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_4_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_4_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_4_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_4_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_20_4_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_4_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_4_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_4_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_4_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_4_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_20_4_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_4_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_4_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_4_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_4_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_4_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
		if(capacity==8){
			if(blockDim.x==64){
				if(lanes==8&&kind==0){packed_g_20_8_2_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_8_2_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_8_2_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_8_2_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_8_2_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_8_2_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==128){
				if(lanes==8&&kind==0){packed_g_20_8_4_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_8_4_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_8_4_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_8_4_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_8_4_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_8_4_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==256){
				if(lanes==8&&kind==0){packed_g_20_8_8_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_8_8_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_8_8_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_8_8_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_8_8_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_8_8_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
			if(blockDim.x==512){
				if(lanes==8&&kind==0){packed_g_20_8_16_0_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==1){packed_g_20_8_16_1_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==8&&kind==2){packed_g_20_8_16_2_8(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==0){packed_g_20_8_16_0_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==1){packed_g_20_8_16_1_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
				if(lanes==16&&kind==2){packed_g_20_8_16_2_16(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}
			}
		}
	}
	return 0;
}
extern "C" __global__ void __launch_bounds__(512,1) packed_probe(const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,int type,int capacity,int active,int kind,int lanes,u32 mask,u32 *status,int experts){
	extern __shared__ __align__(16) u32 scratch[];
	if(experts>1){if(blockIdx.x>=experts){if(threadIdx.x==0)status[blockIdx.x]=1;return;}size_t bytes=size_t(k)*m*(type==8?34:type==12?144:type==13?176:type==14?210:type==17?74:type==18?98:18)/(type==8||type==20?32:256);W+=blockIdx.x*bytes;out+=size_t(blockIdx.x)*m*capacity;}
	u32 ok=packed_matvec(type,capacity,active,kind,lanes,mask,W,x,ds,out,k,m,experts>1?0:blockIdx.x,experts>1?1:gridDim.x,scratch);
	if(threadIdx.x==0)status[blockIdx.x]=ok;
}
extern "C" __global__ void __launch_bounds__(1024,1) packed_probe_wide(const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,int type,int capacity,int active,int kind,int lanes,u32 mask,u32 *status,int experts){
	extern __shared__ __align__(16) u32 scratch[];
	if(experts>1){if(blockIdx.x>=experts){if(threadIdx.x==0)status[blockIdx.x]=1;return;}size_t bytes=size_t(k)*m*(type==8?34:type==12?144:type==13?176:type==14?210:type==17?74:type==18?98:18)/(type==8||type==20?32:256);W+=blockIdx.x*bytes;out+=size_t(blockIdx.x)*m*capacity;}
	u32 ok=packed_matvec_wide(type,capacity,active,kind,lanes,mask,W,x,ds,out,k,m,experts>1?0:blockIdx.x,experts>1?1:gridDim.x,scratch);
	if(threadIdx.x==0)status[blockIdx.x]=ok;
}
#define PAIR_PROBE(T,N,W,L) extern "C" __global__ void __launch_bounds__(W*32,1) pair_probe_##T##_##N##_##W##_##L(const uint8_t *g,const uint8_t *u,const u32 *x,const float2 *ds,float *out,int active,u32 mask,u32 index,u32 count,u32 *status){u32 ok=packed_gate_up_##T##_##N##_##W##_##L(g,u,x,ds,out,active,mask,index,count);if(threadIdx.x==0)status[blockIdx.x]=ok;}
PAIR_PROBE(17,1,8,8) PAIR_PROBE(17,1,8,16) PAIR_PROBE(17,1,16,8) PAIR_PROBE(17,1,16,16)
PAIR_PROBE(17,2,8,8) PAIR_PROBE(17,2,8,16) PAIR_PROBE(17,2,16,8) PAIR_PROBE(17,2,16,16)
PAIR_PROBE(18,1,8,8) PAIR_PROBE(18,1,8,16) PAIR_PROBE(18,1,16,8) PAIR_PROBE(18,1,16,16)
PAIR_PROBE(18,2,8,8) PAIR_PROBE(18,2,8,16) PAIR_PROBE(18,2,16,8) PAIR_PROBE(18,2,16,16)
extern "C" __global__ void __launch_bounds__(512,1) packed_gate_up_probe(const uint8_t *gate,const uint8_t *up,const u32 *x,const float2 *ds,float *product,int type,int capacity,int active,int lanes,u32 mask,int k,int m,u32 *status){u32 ok=packed_gate_up(type,capacity,active,lanes,mask,gate,up,x,ds,product,k,m,blockIdx.x,gridDim.x);if(threadIdx.x==0)status[blockIdx.x]=ok;}
