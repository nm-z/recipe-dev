#!/bin/bash
set -euo pipefail
source=$1
winners=$2
output=$3
scratch=$4
awk -F'\t' 'NR>1{family=($1==12||$1==13)&&$6==0?"h":"g";name=family=="h"?sprintf("packed_h_%s_%s_%s_%s",$1,$4,$7,$8):sprintf("packed_g_%s_%s_%s_%s_%s",$1,$4,$7,$6,$8);if(!seen[name]++)print $1,$4,$7,$8,$6,family,name}' "$winners" > "$scratch"
awk '/^extern "C" __device__ __noinline__ u32 packed_matvec_wide/{exit} /^(GT|WIDET|DT|WIDTHT|WIDTH|GW|DW|GK|G)\([0-9]/{next} {print}' "$source" > "$output"
while read -r type cap warps lanes kind family symbol; do
	if [[ $family == h ]]; then printf 'WIDTH(%s,%s,%s,%s)\n' "$type" "$cap" "$warps" "$lanes" >> "$output"; else printf 'G(%s,%s,%s,%s,%s)\n' "$type" "$cap" "$warps" "$lanes" "$kind" >> "$output"; fi
done < "$scratch"
cat >> "$output" <<'PART'
extern "C" __device__ __noinline__ u32 packed_matvec(int type,int capacity,int active,int kind,int lanes,u32 position_mask,const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,u32 cta_index,u32 cta_count,u32 *scratch){
	if(!valid_packed(type,capacity,active,kind,lanes,position_mask,k,m,cta_index,cta_count,scratch))return 0;
	if(active==0)return 1;
PART
while read -r type cap warps lanes kind family symbol; do
	printf '\tif(type==%s&&capacity==%s&&blockDim.x==%s&&kind==%s&&lanes==%s){%s(W,x,ds,out,k,m,active,position_mask,cta_index,cta_count,scratch);return 1;}\n' "$type" "$cap" "$((warps*32))" "$kind" "$lanes" "$symbol" >> "$output"
done < "$scratch"
cat >> "$output" <<'PART'
	return 0;
}
extern "C" __device__ __noinline__ u32 packed_matvec_wide(int type,int capacity,int active,int kind,int lanes,u32 mask,const uint8_t *W,const u32 *x,const float2 *ds,float *out,int k,int m,u32 index,u32 count,u32 *scratch){
	if(blockDim.x!=1024||capacity>2)return 0;
	return packed_matvec(type,capacity,active,kind,lanes,mask,W,x,ds,out,k,m,index,count,scratch);
}
PART
