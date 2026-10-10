#!/bin/bash
# usage: sassloop.sh kernel  -> opcode histogram of the main k-tile loop (region of the back-branch)
k=$1
/opt/cuda/bin/cuobjdump -sass -fun $k kern.cubin | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's,/\* 0x[0-9a-f]+ \*/,,;s/^ +//' > /tmp/sl.$k
gawk -v K=$k '
{ a[NR]=strtonum("0x" substr($1,3,4)); line[NR]=$0 }
END{
 for(i=1;i<=NR;i++){ if (line[i] ~ /BRA 0x/){ t=strtonum(gensub(/.*BRA (0x[0-9a-f]+).*/,"\\1","g",line[i])); if (t<a[i] && a[i]-t > hi-lo) {lo=t; hi=a[i]} } }
 n=0; for(i=1;i<=NR;i++) if(a[i]>=lo && a[i]<=hi){ n++; split(line[i],f," "); o=f[2]; if (o ~ /^@/) o=f[3]; if (o=="{") o=f[3]; sub(/\..*/,"",o); h[o]++ }
 printf "%s loop 0x%x-0x%x: %d instr\n", K, lo, hi, n; for (o in h) printf "  %-8s %d\n", o, h[o] }' /tmp/sl.$k
