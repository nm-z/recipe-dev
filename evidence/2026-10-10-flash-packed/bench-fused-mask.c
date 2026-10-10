// Probe the same stored GGUF bytes against stock ggml CUDA and an independent CPU dequantization sum.
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-quants.h"
#define CHK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char *s;cuGetErrorString(r,&s);fprintf(stderr,"%s: %s\n",#x,s);exit(1);}}while(0)
static uint64_t rng=88172645463325252ull;
static uint32_t rnd(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng>>16;}
static float half_round(float v){return (float)(_Float16)v;}
static void deq(int t,const void *p,float *o,int k){
	switch(t){case 8:dequantize_row_q8_0(p,o,k);break;case 12:dequantize_row_q4_K(p,o,k);break;case 13:dequantize_row_q5_K(p,o,k);break;case 14:dequantize_row_q6_K(p,o,k);break;case 17:dequantize_row_iq2_xs(p,o,k);break;case 18:dequantize_row_iq3_xxs(p,o,k);break;case 20:dequantize_row_iq4_nl(p,o,k);break;default:exit(1);}
}
static double error(const float *a,const float *b,int n){double mx=0,e=0;for(int i=0;i<n;i++){if(!isfinite(a[i])||!isfinite(b[i]))return INFINITY;mx=fmax(mx,fabs(b[i]));e=fmax(e,fabs((double)a[i]-b[i]));}return mx?e/mx:e;}
int main(int argc,char **argv){
	const char *uuid="GPU-ccb8b3a3-f45d-8962-215b-c2b140e4bb28";if(argc!=4||!getenv("CUDA_VISIBLE_DEVICES")||strcmp(getenv("CUDA_VISIBLE_DEVICES"),uuid))return 1;
	CHK(cuInit(0));int count;CHK(cuDeviceGetCount(&count));if(count!=1)return 1;CUdevice dev;CHK(cuDeviceGet(&dev,0));CUuuid id;CHK(cuDeviceGetUuid(&id,dev));const unsigned char expected[16]={0xcc,0xb8,0xb3,0xa3,0xf4,0x5d,0x89,0x62,0x21,0x5b,0xc2,0xb1,0x40,0xe4,0xbb,0x28};if(memcmp(id.bytes,expected,16))return 1;CUcontext ctx;CHK(cuCtxCreate(&ctx,0,dev));CUmodule mod;CHK(cuModuleLoad(&mod,argv[1]));
	int k=2560,m=640;float x[3*2560],xq[3*2560],ds[3*80*2],refs[3*640],row[2560];uint32_t pairs[3*1280];
	for(int i=0;i<3*k;i++)x[i]=((rnd()&65535)/32768.0f-1)*(i%67==0?8:1);
	for(int b=0;b<3*k/32;b++){float mx=0;int q[32],sum=0;for(int j=0;j<32;j++)mx=fmaxf(mx,fabsf(x[b*32+j]));float d=mx/127,dh=half_round(d);for(int j=0;j<32;j++){q[j]=mx?roundf(x[b*32+j]/d):0;sum+=q[j];xq[b*32+j]=dh*q[j];}for(int j=0;j<16;j++)pairs[b*16+j]=(q[2*j]&65535u)|(((uint32_t)(q[2*j+1]&65535u))<<16);ds[2*b]=dh;ds[2*b+1]=dh*sum;}
	CUdeviceptr px,sd,out,status;CHK(cuMemAlloc(&px,sizeof(pairs)));CHK(cuMemAlloc(&sd,sizeof(ds)));CHK(cuMemAlloc(&out,2*m*4));CHK(cuMemAlloc(&status,4));CHK(cuMemcpyHtoD(px,pairs,sizeof(pairs)));CHK(cuMemcpyHtoD(sd,ds,sizeof(ds)));
	printf("type,capacity,warps,row_lanes,active,mask,receipt,padding,cpu_error,correct\n");int fail=0,cases=0;
	for(int t=17;t<=18;t++){
		size_t bytes=ggml_row_size(t,k)*m;void *hg=malloc(bytes),*hu=malloc(bytes);char path[1024];int layer=t==17?0:2;snprintf(path,sizeof(path),"%s/l%d-gate-e3.bin",argv[3],layer);FILE *f=fopen(path,"rb");if(!f||fread(hg,1,bytes,f)!=bytes)return 1;fclose(f);snprintf(path,sizeof(path),"%s/l%d-up-e3.bin",argv[3],layer);f=fopen(path,"rb");if(!f||fread(hu,1,bytes,f)!=bytes)return 1;fclose(f);
		for(int r=0;r<m;r++){double g[3]={},u[3]={};deq(t,(const uint8_t*)hg+r*ggml_row_size(t,k),row,k);for(int c=0;c<3;c++)for(int j=0;j<k;j++)g[c]+=(double)row[j]*xq[c*k+j];deq(t,(const uint8_t*)hu+r*ggml_row_size(t,k),row,k);for(int c=0;c<3;c++)for(int j=0;j<k;j++)u[c]+=(double)row[j]*xq[c*k+j];for(int c=0;c<3;c++){float a=g[c],b=u[c];refs[c*m+r]=(a/(1+expf(-a)))*b;}}
		CUdeviceptr dg,du;CHK(cuMemAlloc(&dg,bytes));CHK(cuMemAlloc(&du,bytes));CHK(cuMemcpyHtoD(dg,hg,bytes));CHK(cuMemcpyHtoD(du,hu,bytes));FILE *cfg=fopen(argv[2],"r");if(!cfg)return 1;char line[1024];int seen[3][17][17]={};
		while(fgets(line,sizeof(line),cfg)){int ct,cm,ck,n,e,nw,l;if(sscanf(line,"%d%d%d%d%d%d%d",&ct,&cm,&ck,&n,&e,&nw,&l)!=7||ct!=t||seen[n][nw][l]++)continue;
			char name[64];snprintf(name,sizeof(name),"pair_probe_%d_%d_%d_%d",t,n,nw,l);CUfunction fn;CHK(cuModuleGetFunction(&fn,mod,name));
			for(int probe_case=0;probe_case<5;probe_case++){if(probe_case==3&&n==1)continue;int active=probe_case==0?0:probe_case==2?n:probe_case==3?2:1;uint32_t mask=probe_case==1?4:probe_case==3||probe_case==4?5:0,index=0,group=1;void *args[]={&dg,&du,&px,&sd,&out,&active,&mask,&index,&group,&status};CHK(cuMemsetD32(out,0x7fc12345,n*m));CHK(cuMemsetD32(status,0,1));CHK(cuLaunchKernel(fn,1,1,1,nw*32,1,1,40960,0,args,0));CHK(cuCtxSynchronize());float got[2*640],gold[2*640];uint32_t receipt;CHK(cuMemcpyDtoH(got,out,n*m*4));CHK(cuMemcpyDtoH(&receipt,status,4));int valid=probe_case!=4,padding=1;
				uint32_t bits=mask;for(int c=0;c<active&&valid;c++){int source=mask?__builtin_ctz(bits):c;bits&=bits-1;memcpy(gold+c*m,refs+source*m,m*4);}for(int i=valid?active*m:0;i<n*m;i++){uint32_t v;memcpy(&v,got+i,4);if(v!=0x7fc12345)padding=0;}double er=valid&&active?error(got,gold,active*m):0;int ok=receipt==(uint32_t)valid&&padding&&er<=3e-6;if(!ok)fail=1;cases++;printf("%d,%d,%d,%d,%d,%u,%u,%d,%.9g,%d\n",t,n,nw,l,active,mask,receipt,padding,er,ok);fflush(stdout);
			}
		}fclose(cfg);CHK(cuMemFree(dg));CHK(cuMemFree(du));free(hg);free(hu);
	}
	fprintf(stderr,"cases=%d failed=%d\n",cases,fail);CHK(cuCtxDestroy(ctx));return fail?2:0;
}
