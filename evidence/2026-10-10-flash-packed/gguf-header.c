#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static FILE *f;
static void rd(void *p,size_t n) { if(fread(p,1,n,f)!=n){fprintf(stderr,"truncated header\n");exit(1);} }
static uint32_t u32(void){uint32_t x;rd(&x,4);return x;}
static uint64_t u64(void){uint64_t x;rd(&x,8);return x;}
static char *str(void){uint64_t n=u64();if(n>20000000)exit(1);char *s=calloc(n+1,1);rd(s,n);return s;}
static void skip(uint32_t t){static const int sizes[]={1,1,2,2,4,4,4,1,0,0,8,8,8};if(t>12)exit(1);if(t==8){char *s=str();free(s);}else if(t==9){uint32_t a=u32();uint64_t n=u64();if(n>10000000)exit(1);while(n--)skip(a);}else {char b[8];rd(b,sizes[t]);}}
int main(int ac,char **av){puts("shard\tname\ttype\tk\tm\texperts\tndim\toffset");for(int a=1;a<ac;a++){f=fopen(av[a],"rb");if(!f)return 1;uint32_t magic=u32(),v=u32(),alignment=32;if(magic!=0x46554747||v!=3)return 1;uint64_t nt=u64(),nk=u64();for(uint64_t i=0;i<nk;i++){char *s=str();uint32_t t=u32();if((t==4||t==5)&&(!strcmp(s,"general.alignment")||strstr(s,"expert")||strstr(s,"block_count"))){uint32_t x=u32();fprintf(stderr,"metadata %s=%u\n",s,x);if(!strcmp(s,"general.alignment"))alignment=x;}else skip(t);free(s);}for(uint64_t i=0;i<nt;i++){char *s=str();uint32_t nd=u32();uint64_t d[4]={1,1,1,1};if(nd>4)exit(1);for(uint32_t j=0;j<nd;j++)d[j]=u64();uint32_t t=u32();uint64_t o=u64();printf("%d\t%s\t%u\t%lu\t%lu\t%lu\t%u\t%lu\n",a,s,t,d[0],d[1],d[2]*d[3],nd,o);free(s);}long end=ftell(f);fprintf(stderr,"shard %d: %lu tensors, tensor table ends at %ld, data starts at %lu\n",a,nt,end,(end+alignment-1)/alignment*alignment);fclose(f);}return 0;}
