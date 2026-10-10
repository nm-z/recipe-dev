#include <stdint.h>
#include <stdio.h>
static unsigned pop(unsigned x){return __builtin_popcount(x);}
int main(void){
	unsigned bad=0,layouts=0,masks=0,rows=0;
	for(unsigned n=1;n<=8;n*=2)for(unsigned lanes=8;lanes<=16;lanes*=2){
		unsigned inputs=n*lanes*20*4,scales=n*lanes*8,table=inputs+scales;
		bad+=table+64>12288;bad+=12288+n*4>16384;bad+=16384+256*n*4>24576;bad+=24576+256*16*4>40960;layouts++;
		for(unsigned mask=0;mask<256;mask++)if(pop(mask)<=n){
			unsigned list[8],count=0;for(unsigned bit=0;bit<8;bit++)if(mask&(1u<<bit))list[count++]=bit;
			unsigned bits=mask;for(unsigned c=0;c<count;c++){unsigned source=__builtin_ffs(bits)-1;bad+=source!=list[c];bits&=bits-1;masks++;}
		}
		for(unsigned total=1;total<=16;total++)for(unsigned count=1;count<=total;count++){
			unsigned seen[2560]={};for(unsigned rank=0;rank<count;rank++)for(unsigned base=rank;base<2560;base+=count*(256/lanes))
				for(unsigned thread=0;thread<256;thread+=lanes){unsigned row=base+(thread/lanes)*count;if(row<2560)seen[row]++;}
			for(unsigned row=0;row<2560;row++){bad+=seen[row]!=1;rows++;}
		}
	}
	printf("layouts=%u mask_positions=%u row_partitions=%u mismatches=%u\n",layouts,masks,rows,bad);return bad?1:0;
}
