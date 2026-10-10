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
static double sampled_error(const float *a,const float *b,int m,int n,int nr){double mx=0,e=0;for(int c=0;c<n;c++)for(int s=0;s<nr;s++){int i=c*m+(int)((int64_t)s*m/nr);if(!isfinite(a[i])||!isfinite(b[i]))return INFINITY;mx=fmax(mx,fabs(b[i]));e=fmax(e,fabs((double)a[i]-b[i]));}return mx?e/mx:e;}
int main(int ac,char **av){
	if(ac!=5){fprintf(stderr,"usage: bench type rows columns sample.bin\n");return 1;}
	const char *uuid="GPU-ccb8b3a3-f45d-8962-215b-c2b140e4bb28";
	if(!getenv("CUDA_VISIBLE_DEVICES")||strcmp(getenv("CUDA_VISIBLE_DEVICES"),uuid)){fprintf(stderr,"CUDA_VISIBLE_DEVICES must select the authorized die UUID\n");return 1;}
	int t=atoi(av[1]),m=atoi(av[2]),k=atoi(av[3]);if(m<=0||k<=0||k%ggml_blck_size(t))return 1;
	int helper=1,experts=getenv("EXPERTS")?atoi(getenv("EXPERTS")):1;if(experts<1||experts>16)return 1;
	CHK(cuInit(0));int count;CHK(cuDeviceGetCount(&count));if(count!=1)return 1;CUdevice dev;CHK(cuDeviceGet(&dev,0));CUuuid id;CHK(cuDeviceGetUuid(&id,dev));
	const unsigned char expected[16]={0xcc,0xb8,0xb3,0xa3,0xf4,0x5d,0x89,0x62,0x21,0x5b,0xc2,0xb1,0x40,0xe4,0xbb,0x28};if(memcmp(id.bytes,expected,16))return 1;
	ggml_backend_t be=ggml_backend_cuda_init(0);if(!be)return 1;
	CUcontext ctx;CHK(cuDevicePrimaryCtxRetain(&ctx,dev));CHK(cuCtxSetCurrent(ctx));CUmodule mod;CHK(cuModuleLoad(&mod,getenv("PACKED_CUBIN")?getenv("PACKED_CUBIN"):"packed.cubin"));CUfunction prep;CHK(cuModuleGetFunction(&prep,mod,helper?"packed_prepare_probe":"pack_x"));
	int sms;CHK(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev));CUdeviceptr receipts;CHK(cuMemAlloc(&receipts,4*sms));
	size_t rowbytes=ggml_row_size(t,k),bytes=rowbytes*m*experts;int total_m=m*experts;void *hw=malloc(bytes);if(!hw)return 1;
	FILE *file=fopen(av[4],"rb");if(!file||fread(hw,1,bytes,file)!=bytes||fgetc(file)!=EOF){fprintf(stderr,"sample size mismatch\n");return 1;}fclose(file);
	int iters=getenv("NITER")?atoi(getenv("NITER")):100;if(iters<10)return 1;
	// Rotate weight copies totaling at least 32 MiB. Small expert matrices must not masquerade as DRAM throughput from L2 residency.
	int copies=(32*1024*1024+bytes-1)/bytes;if(copies<1)copies=1;CUdeviceptr dw;CHK(cuMemAlloc(&dw,bytes*copies));for(int i=0;i<copies;i++)CHK(cuMemcpyHtoD(dw+i*bytes,hw,bytes));
	CUevent start,end;CHK(cuEventCreate(&start,0));CHK(cuEventCreate(&end,0));
	printf("type,m,k,batch,path,warps,weight_GBs,kernel_plus_prep_GBs,stock_GBs,stock_error,cpu_error,copies,correct,stock_float_noise,stock_cpu_error,cpu_rows,experts\n");fflush(stdout);
	int fail=0;
	for(int n=1;n<=8;n*=2){
		if(getenv("BATCH_ONLY")&&n!=atoi(getenv("BATCH_ONLY")))continue;
		uint32_t position_mask=getenv("POSITION_MASK")?strtoul(getenv("POSITION_MASK"),0,0):0;
		int active=getenv("ACTIVE_COLUMNS")?atoi(getenv("ACTIVE_COLUMNS")):position_mask?__builtin_popcount(position_mask):n;if(active<0||active>n)continue;
		int reject=getenv("EXPECT_REJECT")?atoi(getenv("EXPECT_REJECT")):0;
		int source_columns=position_mask?32-__builtin_clz(position_mask):n;
		float *hx=malloc(4*k*source_columns),*sourceq=malloc(4*k*source_columns),*gather=calloc(k*n,4),*xq=calloc(k*n,4),*stock=malloc(4*total_m*n),*candidate=malloc(4*total_m*n),*raw_candidate=malloc(4*total_m*n),*ref=malloc(4*total_m*n),*row=malloc(4*k);
		for(int c=0;c<source_columns;c++)for(int j=0;j<k;j++){float v=((rnd()&0xffff)/32768.0f-1);hx[c*k+j]=v*(j%67==0?8:1);}
		for(int b=0;b<k*source_columns/32;b++){float mx=0;for(int j=0;j<32;j++)mx=fmaxf(mx,fabsf(hx[b*32+j]));float d=mx/127,dh=half_round(d);for(int j=0;j<32;j++)sourceq[b*32+j]=mx?dh*roundf(hx[b*32+j]/d):0;}
		uint32_t mask=position_mask;for(int c=0;c<active;c++){int source=position_mask?__builtin_ctz(mask):c;mask&=mask-1;memcpy(gather+c*k,hx+source*k,4*k);memcpy(xq+c*k,sourceq+source*k,4*k);}
		// Check all outputs against stock CUDA and spaced rows against the independent f64 sum.
		int nr=getenv("CPU_ROWS")?atoi(getenv("CPU_ROWS")):64;if(nr<=0||nr>m)nr=m;nr*=experts;
		for(int s=0;s<nr;s++){int r=(int)((int64_t)s*total_m/nr);deq(t,(const uint8_t*)hw+r*rowbytes,row,k);for(int c=0;c<n;c++){double sum=0;for(int j=0;j<k;j++)sum+=(double)row[j]*xq[c*k+j];ref[c*total_m+r]=sum;}}
		struct ggml_init_params ip={.mem_size=4*1024*1024,.no_alloc=true};struct ggml_context *gc=ggml_init(ip);
		struct ggml_tensor *w=experts>1?ggml_new_tensor_3d(gc,t,k,m,experts):ggml_new_tensor_2d(gc,t,k,m),*x=experts>1?ggml_new_tensor_3d(gc,GGML_TYPE_F32,k,1,n):ggml_new_tensor_2d(gc,GGML_TYPE_F32,k,n),*ids=experts>1?ggml_new_tensor_2d(gc,GGML_TYPE_I32,experts,n):NULL,*o=experts>1?ggml_mul_mat_id(gc,w,x,ids):ggml_mul_mat(gc,w,x);struct ggml_cgraph *graph=ggml_new_graph(gc);ggml_build_forward_expand(graph,o);
		ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(gc,be);ggml_backend_tensor_set(w,hw,0,bytes);ggml_backend_tensor_set(x,gather,0,4*k*n);
		if(ids){int32_t *hi=malloc(4*experts*n);for(int c=0;c<n;c++)for(int e=0;e<experts;e++)hi[c*experts+e]=e;ggml_backend_tensor_set(ids,hi,0,4*experts*n);free(hi);}
		ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));if(!ggml_gallocr_alloc_graph(ga,graph))return 1;
		ggml_backend_graph_compute(be,graph);ggml_backend_synchronize(be);ggml_backend_tensor_get(o,stock,0,4*total_m*n);
		// Stock timing uses the public graph, including activation preparation. Reference bytes remain identical.
		void *stock_weights=w->data;
		for(int j=0;j<10;j++){w->data=(void*)(uintptr_t)(dw+(j%copies)*bytes);ggml_backend_graph_compute(be,graph);}
		ggml_backend_synchronize(be);
		CHK(cuEventRecord(start,0));for(int j=0;j<iters;j++){w->data=(void*)(uintptr_t)(dw+(j%copies)*bytes);ggml_backend_graph_compute(be,graph);}ggml_backend_synchronize(be);CHK(cuEventRecord(end,0));CHK(cuEventSynchronize(end));float ms;CHK(cuEventElapsedTime(&ms,start,end));double stock_bw=bytes*iters/(ms*1e6);w->data=stock_weights;
		CUdeviceptr dx,px,ds,out;CHK(cuMemAlloc(&dx,4*k*source_columns));CHK(cuMemAlloc(&px,2*k*source_columns));CHK(cuMemAlloc(&ds,8*k*source_columns/32));CHK(cuMemAlloc(&out,4*total_m*n));CHK(cuMemcpyHtoD(dx,hx,4*k*source_columns));
		void *pa[]={&dx,&px,&ds,&k,&source_columns};CHK(cuLaunchKernel(prep,(k/32+7)/8,helper?1:source_columns,1,256,1,1,0,0,pa,0));
		const char *prefix="r";
		int nwmin=getenv("WARPS_MIN")?atoi(getenv("WARPS_MIN")):4;
		int row_lanes=getenv("ROW_LANES")?atoi(getenv("ROW_LANES")):16;
		int nwmax=getenv("WARPS_MAX")?atoi(getenv("WARPS_MAX")):32,selected_kind=-1;
		if(getenv("CONFIG_TSV")){
			FILE *cfg=fopen(getenv("CONFIG_TSV"),"r");if(!cfg)return 1;char line[1024];int found=0;
			while(fgets(line,sizeof(line),cfg)){int ct,cm,ck,cn,ce,cg,cw,cl;if(sscanf(line,"%d%d%d%d%d%d%d%d",&ct,&cm,&ck,&cn,&ce,&cg,&cw,&cl)==8&&ct==t&&cm==m&&ck==k&&cn==n&&ce==experts){selected_kind=cg;nwmin=nwmax=cw;row_lanes=cl;found=1;break;}}
			fclose(cfg);if(!found){fprintf(stderr,"missing measured configuration type=%d m=%d k=%d batch=%d experts=%d\n",t,m,k,n,experts);return 1;}
		}
		for(int kind=0;kind<3;kind++)for(int nw=nwmin;nw<=nwmax;nw*=2){
			if(selected_kind>=0&&kind!=selected_kind)continue;
			if(kind==2&&t!=17&&t!=18&&t!=20)continue;
			if(nw==32&&n>2)continue;
			if((*prefix=='f'||*prefix=='h')&&kind)continue;
			char name[64],path[64];snprintf(name,sizeof(name),nw==32?"packed_probe_wide":"packed_probe");snprintf(path,sizeof(path),"%s%d_%s",prefix,row_lanes,kind==1?"magic_fadd":kind==2?"xmad_dict":"xmad");CUfunction fn;CHK(cuModuleGetFunction(&fn,mod,name));CUdeviceptr wp=dw;void *args[]={&wp,&px,&ds,&out,&k,&m,&t,&n,&active,&kind,&row_lanes,&position_mask,&receipts,&experts};int grid=sms,shared=40960;
			CHK(cuMemsetD32(out,0x7fc12345,total_m*n));CHK(cuMemsetD32(receipts,0,sms));
			CHK(cuLaunchKernel(fn,grid,1,1,nw*32,1,1,shared,0,args,0));CHK(cuCtxSynchronize());CHK(cuMemcpyDtoH(raw_candidate,out,4*total_m*n));
			for(int e=0;e<experts;e++)for(int c=0;c<n;c++)memcpy(candidate+(c*experts+e)*m,raw_candidate+(e*n+c)*m,4*m);
			int receipt_ok=1,padding_ok=1;uint32_t seen[sms];CHK(cuMemcpyDtoH(seen,receipts,4*sms));for(int j=0;j<sms;j++)if(seen[j]!=(reject&&(experts==1||j<experts)?0u:1u))receipt_ok=0;for(int j=reject?0:active*total_m;j<n*total_m;j++){uint32_t bits;memcpy(&bits,candidate+j,4);if(bits!=0x7fc12345)padding_ok=0;}
			double es=reject?0:error(candidate,stock,total_m*active),er=reject?0:sampled_error(candidate,ref,total_m,active,nr),sr=sampled_error(stock,ref,total_m,active,nr);int correct=er<=3e-6&&receipt_ok&&padding_ok,stock_noise=es<=3e-6;if(!correct)fail=1;
			if(helper)fprintf(stderr,"helper type=%d batch=%d active=%d mask=%u warps=%d lanes=%d kind=%d receipts=%d padding=%d reject=%d\n",t,n,active,position_mask,nw,row_lanes,kind,receipt_ok,padding_ok,reject);
			if(correct&&active==n&&!reject){
				for(int j=0;j<10;j++){wp=dw+(j%copies)*bytes;CHK(cuLaunchKernel(fn,grid,1,1,nw*32,1,1,shared,0,args,0));}CHK(cuCtxSynchronize());
				CHK(cuEventRecord(start,0));for(int j=0;j<iters;j++){wp=dw+(j%copies)*bytes;CHK(cuLaunchKernel(fn,grid,1,1,nw*32,1,1,shared,0,args,0));}CHK(cuEventRecord(end,0));CHK(cuEventSynchronize(end));CHK(cuEventElapsedTime(&ms,start,end));double bw=bytes*iters/(ms*1e6);
				CHK(cuEventRecord(start,0));for(int j=0;j<iters;j++){wp=dw+(j%copies)*bytes;CHK(cuLaunchKernel(prep,(k/32+7)/8,helper?1:source_columns,1,256,1,1,0,0,pa,0));CHK(cuLaunchKernel(fn,grid,1,1,nw*32,1,1,shared,0,args,0));}CHK(cuEventRecord(end,0));CHK(cuEventSynchronize(end));CHK(cuEventElapsedTime(&ms,start,end));double pair_bw=bytes*iters/(ms*1e6);
				printf("%d,%d,%d,%d,%s,%d,%.3f,%.3f,%.3f,%.9g,%.9g,%d,1,%d,%.9g,%d,%d\n",t,m,k,n,path,nw,bw,pair_bw,stock_bw,es,er,copies,stock_noise,sr,nr,experts);
			}else printf("%d,%d,%d,%d,%s,%d,0,0,%.3f,%.9g,%.9g,%d,%d,%d,%.9g,%d,%d\n",t,m,k,n,path,nw,stock_bw,es,er,copies,correct,stock_noise,sr,nr,experts);
			fflush(stdout);
		}
		cuMemFree(dx);cuMemFree(px);cuMemFree(ds);cuMemFree(out);ggml_gallocr_free(ga);ggml_backend_buffer_free(buf);ggml_free(gc);free(hx);free(gather);free(sourceq);free(xq);free(stock);free(candidate);free(raw_candidate);free(ref);free(row);
	}
	cuMemFree(receipts);cuMemFree(dw);free(hw);cuEventDestroy(start);cuEventDestroy(end);cuModuleUnload(mod);ggml_backend_free(be);cuDevicePrimaryCtxRelease(dev);return fail?2:0;
}
