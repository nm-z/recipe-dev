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
static void quantize(const float *src,float *dst,int k,int n){
	for(int b=0;b<k*n/32;b++){float mx=0;for(int j=0;j<32;j++)mx=fmaxf(mx,fabsf(src[b*32+j]));float d=mx/127,dh=half_round(d);for(int j=0;j<32;j++)dst[b*32+j]=mx?dh*roundf(src[b*32+j]/d):0;}
}
static void *read_weights(const char *path,size_t bytes){void *p=malloc(bytes);FILE *f=fopen(path,"rb");if(!p||!f||fread(p,1,bytes,f)!=bytes)exit(1);fclose(f);return p;}
int main(int argc,char **argv){
	const char *uuid="GPU-ccb8b3a3-f45d-8962-215b-c2b140e4bb28";if(argc!=7||!getenv("CUDA_VISIBLE_DEVICES")||strcmp(getenv("CUDA_VISIBLE_DEVICES"),uuid))return 1;
	int t=atoi(argv[1]),experts=atoi(argv[2]),n=atoi(argv[3]),k=2560,m=640;if((t!=17&&t!=18)||experts<1||experts>3||(n!=1&&n!=2))return 1;
	CHK(cuInit(0));int devices;CHK(cuDeviceGetCount(&devices));if(devices!=1)return 1;CUdevice dev;CHK(cuDeviceGet(&dev,0));CUuuid id;CHK(cuDeviceGetUuid(&id,dev));const unsigned char expected[16]={0xcc,0xb8,0xb3,0xa3,0xf4,0x5d,0x89,0x62,0x21,0x5b,0xc2,0xb1,0x40,0xe4,0xbb,0x28};if(memcmp(id.bytes,expected,16))return 1;
	ggml_backend_t be=ggml_backend_cuda_init(0);if(!be)return 1;CUcontext ctx;CHK(cuDevicePrimaryCtxRetain(&ctx,dev));CHK(cuCtxSetCurrent(ctx));CUmodule mod;CHK(cuModuleLoad(&mod,argv[6]));int sms;CHK(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev));if(sms!=16)return 1;
	size_t gb=ggml_row_size(t,k)*m*experts,db=ggml_row_size(20,m)*k*experts,bundle=2*gb+db;void *hg=read_weights(argv[4],gb),*hu=read_weights(argv[5],gb);char down_path[1024];snprintf(down_path,sizeof(down_path),"%.*sdown-e3.bin",(int)(strstr(argv[4],"gate-e3.bin")-argv[4]),argv[4]);void *hd=read_weights(down_path,db);
	int copies=(32*1024*1024+bundle-1)/bundle,iters=100;CUdeviceptr dg,du,dd;CHK(cuMemAlloc(&dg,gb*copies));CHK(cuMemAlloc(&du,gb*copies));CHK(cuMemAlloc(&dd,db*copies));for(int i=0;i<copies;i++){CHK(cuMemcpyHtoD(dg+i*gb,hg,gb));CHK(cuMemcpyHtoD(du+i*gb,hu,gb));CHK(cuMemcpyHtoD(dd+i*db,hd,db));}
	float *hx=malloc(4*k*n),*xq=malloc(4*k*n),*cpu_product=malloc(4*m*experts*n),*pq=malloc(4*m*experts*n),*row=malloc(4*k),*ref=calloc(k*experts*n,4),*stock=malloc(4*k*experts*n),*candidate=malloc(4*k*experts*n),*product=malloc(4*m*experts*n);
	for(int i=0;i<k*n;i++){hx[i]=((rnd()&65535)/32768.0f-1)*(i%67==0?8:1);}quantize(hx,xq,k,n);
	for(int e=0;e<experts;e++)for(int r=0;r<m;r++){double ag[2]={},au[2]={};deq(t,(const uint8_t*)hg+(e*m+r)*ggml_row_size(t,k),row,k);for(int c=0;c<n;c++)for(int j=0;j<k;j++)ag[c]+=(double)row[j]*xq[c*k+j];deq(t,(const uint8_t*)hu+(e*m+r)*ggml_row_size(t,k),row,k);for(int c=0;c<n;c++)for(int j=0;j<k;j++)au[c]+=(double)row[j]*xq[c*k+j];for(int c=0;c<n;c++){float g=ag[c],u=au[c];cpu_product[(e*n+c)*m+r]=(g/(1+expf(-g)))*u;}}
	quantize(cpu_product,pq,m,experts*n);int nr=64;
	for(int e=0;e<experts;e++)for(int s=0;s<nr;s++){int r=s*k/nr;deq(20,(const uint8_t*)hd+(e*k+r)*ggml_row_size(20,m),row,m);for(int c=0;c<n;c++){double a=0;for(int j=0;j<m;j++)a+=(double)row[j]*pq[(e*n+c)*m+j];ref[(e*n+c)*k+r]=a;}}
	struct ggml_init_params ip={.mem_size=8*1024*1024,.no_alloc=true};struct ggml_context *gc=ggml_init(ip);struct ggml_tensor *wg=ggml_new_tensor_3d(gc,t,k,m,experts),*wu=ggml_new_tensor_3d(gc,t,k,m,experts),*wd=ggml_new_tensor_3d(gc,20,m,k,experts),*x=ggml_new_tensor_3d(gc,GGML_TYPE_F32,k,1,n),*ids=ggml_new_tensor_2d(gc,GGML_TYPE_I32,experts,n),*g=ggml_mul_mat_id(gc,wg,x,ids),*u=ggml_mul_mat_id(gc,wu,x,ids),*p=ggml_mul(gc,ggml_silu(gc,g),u),*o=ggml_mul_mat_id(gc,wd,p,ids);struct ggml_cgraph *graph=ggml_new_graph(gc);ggml_build_forward_expand(graph,o);ggml_backend_buffer_t buffer=ggml_backend_alloc_ctx_tensors(gc,be);ggml_backend_tensor_set(wg,hg,0,gb);ggml_backend_tensor_set(wu,hu,0,gb);ggml_backend_tensor_set(wd,hd,0,db);ggml_backend_tensor_set(x,hx,0,4*k*n);int32_t picks[6];for(int c=0;c<n;c++)for(int e=0;e<experts;e++)picks[c*experts+e]=e;ggml_backend_tensor_set(ids,picks,0,4*experts*n);ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));if(!ggml_gallocr_alloc_graph(ga,graph))return 1;ggml_backend_graph_compute(be,graph);ggml_backend_synchronize(be);ggml_backend_tensor_get(o,stock,0,4*k*experts*n);
	CUevent start,end;CHK(cuEventCreate(&start,0));CHK(cuEventCreate(&end,0));void *wg_orig=wg->data,*wu_orig=wu->data,*wd_orig=wd->data;
	for(int j=0;j<10;j++){wg->data=(void*)(uintptr_t)(dg+(j%copies)*gb);wu->data=(void*)(uintptr_t)(du+(j%copies)*gb);wd->data=(void*)(uintptr_t)(dd+(j%copies)*db);ggml_backend_graph_compute(be,graph);}ggml_backend_synchronize(be);CHK(cuEventRecord(start,0));for(int j=0;j<iters;j++){wg->data=(void*)(uintptr_t)(dg+(j%copies)*gb);wu->data=(void*)(uintptr_t)(du+(j%copies)*gb);wd->data=(void*)(uintptr_t)(dd+(j%copies)*db);ggml_backend_graph_compute(be,graph);}ggml_backend_synchronize(be);CHK(cuEventRecord(end,0));CHK(cuEventSynchronize(end));float ms;CHK(cuEventElapsedTime(&ms,start,end));double stock_us=1000.0*ms/iters;wg->data=wg_orig;wu->data=wu_orig;wd->data=wd_orig;
	CUdeviceptr dx,px,ds,gp,up,prod,py,ys,out,flags,status,trace;CHK(cuMemAlloc(&dx,4*k*n));CHK(cuMemAlloc(&px,2*k*n));CHK(cuMemAlloc(&ds,8*k*n/32));CHK(cuMemAlloc(&gp,4*m*experts*n));CHK(cuMemAlloc(&up,4*m*experts*n));CHK(cuMemAlloc(&prod,4*m*experts*n));CHK(cuMemAlloc(&py,2*m*experts*n));CHK(cuMemAlloc(&ys,8*m*experts*n/32));CHK(cuMemAlloc(&out,4*k*experts*n));CHK(cuMemAlloc(&flags,4*sms));CHK(cuMemAlloc(&status,4*sms));CHK(cuMemAlloc(&trace,8*sms*4));CHK(cuMemcpyHtoD(dx,hx,4*k*n));CHK(cuMemsetD32(flags,0,sms));uint32_t epoch=1;int fail=0;
	printf("type,experts,columns,warps,row_lanes,fused,layer_us,stock_layer_us,product_cpu_error,down_cpu_error,stock_error,receipts,prepare_max_cycles,pair_max_cycles,product_prepare_max_cycles,down_max_cycles\n");
	for(int nw=8;nw<=16;nw*=2)for(int lanes=8;lanes<=16;lanes*=2)for(int fused=0;fused<=1;fused++){
		char name[64];snprintf(name,sizeof(name),"%s_%d_%d_%d_%d",fused?"fused_layer":"plain_layer",t,n,nw,lanes);CUfunction fn;CHK(cuModuleGetFunction(&fn,mod,name));CUdeviceptr gw=dg,uw=du,dw=dd;void *args[]={&gw,&uw,&dw,&dx,&px,&ds,&gp,&up,&prod,&py,&ys,&out,&flags,&status,&trace,&experts,&epoch};CHK(cuLaunchKernel(fn,sms,1,1,nw*32,1,1,40960,0,args,0));epoch+=4;CHK(cuCtxSynchronize());CHK(cuMemcpyDtoH(candidate,out,4*k*experts*n));CHK(cuMemcpyDtoH(product,prod,4*m*experts*n));uint32_t receipts[16];CHK(cuMemcpyDtoH(receipts,status,4*sms));int receipt_ok=1;for(int b=0;b<sms;b++)if(receipts[b]!=1)receipt_ok=0;
		double pe=error(product,cpu_product,m*experts*n),ce=0,mx=0;for(int e=0;e<experts;e++)for(int c=0;c<n;c++)for(int s=0;s<nr;s++){int i=(e*n+c)*k+s*k/nr;ce=fmax(ce,fabs((double)candidate[i]-ref[i]));mx=fmax(mx,fabs(ref[i]));}ce=mx?ce/mx:ce;float *ordered=malloc(4*k*experts*n);for(int e=0;e<experts;e++)for(int c=0;c<n;c++)memcpy(ordered+(c*experts+e)*k,candidate+(e*n+c)*k,4*k);double se=error(ordered,stock,k*experts*n);free(ordered);if(!receipt_ok||pe>3e-6||ce>1e-3)fail=1;
		for(int j=0;j<10;j++){gw=dg+(j%copies)*gb;uw=du+(j%copies)*gb;dw=dd+(j%copies)*db;CHK(cuLaunchKernel(fn,sms,1,1,nw*32,1,1,40960,0,args,0));epoch+=4;}CHK(cuCtxSynchronize());CHK(cuEventRecord(start,0));for(int j=0;j<iters;j++){gw=dg+(j%copies)*gb;uw=du+(j%copies)*gb;dw=dd+(j%copies)*db;CHK(cuLaunchKernel(fn,sms,1,1,nw*32,1,1,40960,0,args,0));epoch+=4;}CHK(cuEventRecord(end,0));CHK(cuEventSynchronize(end));CHK(cuEventElapsedTime(&ms,start,end));unsigned long long ticks[64],maxima[4]={};CHK(cuMemcpyDtoH(ticks,trace,sizeof(ticks)));for(int b=0;b<16;b++)for(int j=0;j<4;j++)if(ticks[4*b+j]>maxima[j])maxima[j]=ticks[4*b+j];
		printf("%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.9g,%.9g,%.9g,%d,%llu,%llu,%llu,%llu\n",t,experts,n,nw,lanes,fused,1000.0*ms/iters,stock_us,pe,ce,se,receipt_ok,maxima[0],maxima[1],maxima[2],maxima[3]);fflush(stdout);
	}
	ggml_gallocr_free(ga);ggml_backend_buffer_free(buffer);ggml_free(gc);CUdeviceptr allocations[]={dg,du,dd,dx,px,ds,gp,up,prod,py,ys,out,flags,status,trace};for(unsigned i=0;i<sizeof(allocations)/sizeof(allocations[0]);i++){CHK(cuMemFree(allocations[i]));}CHK(cuEventDestroy(start));CHK(cuEventDestroy(end));CHK(cuModuleUnload(mod));ggml_backend_free(be);CHK(cuDevicePrimaryCtxRelease(dev));return fail?2:0;
}
