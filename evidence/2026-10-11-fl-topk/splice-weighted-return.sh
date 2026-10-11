#!/bin/bash
set -euo pipefail
if (( $# != 4 )); then
	printf 'Usage: splice-weighted-return.sh SOURCE_PTX EXPECTED_SHA256 BARRIER_PTX OUTPUT_PTX\n' >&2
	exit 2
fi
return_source=$1
return_expected=$2
return_library=$3
return_output=$4
return_actual=$(sha256sum "$return_source")
[[ ${return_actual%% *} == "$return_expected" ]] || { printf 'Resident PTX hash differs.\n' >&2; exit 1; }
[[ $return_output != "$return_source" ]] || { printf 'Output must differ from resident PTX.\n' >&2; exit 1; }
awk '
/^\.address_size 64/ {
	print;
	print ".func (.param .u32 result) flash_weighted_return(.param .u64 records,.param .u64 coefficients,.param .u64 expert_outputs,.param .u64 response,.param .u64 slots,.param .u64 remote_sequence,.param .u32 sequence,.param .u64 deadline,.param .u32 top_k,.param .u32 rows,.param .u32 positions,.param .u32 packet_rows);";
	declarations++; next;
}
/^worker_reduce:/ {
	if(reductions++) {print "Repeated expert reduce body." > "/dev/stderr"; exit 1}
	print;
	print "\tmov.u64 a21,%globaltimer; @p0 st.global.u64 [a2+16],a21;";
	print "\t{ .param .u64 ids,coef,in,out,sl,fl,dl; .param .u32 seq,top,rows,cols,stride,ok;";
	print "\tst.param.u64 [ids],a5; st.param.u64 [coef],a6; st.param.u64 [in],a11; st.param.u64 [out],a16; st.param.u64 [sl],a14; st.param.u64 [fl],a17; st.param.u64 [dl],a22;";
	print "\tst.param.u32 [seq],r0; st.param.u32 [top],r5; st.param.u32 [rows],r2; st.param.u32 [cols],r1; mov.u32 r13,4096; st.param.u32 [stride],r13;";
	print "\tcall.uni (ok),flash_weighted_return,(ids,coef,in,out,sl,fl,seq,dl,top,rows,cols,stride); ld.param.u32 r13,[ok]; }";
	print "\tsetp.ne.u32 p1,r13,1; @p1 mov.u32 r12,r13;";
	skip=1; next;
}
skip && /^worker_done:/ {skip=0; completions++}
skip {
	if(/call.uni \(ok\),split_local_combine,/) combines++;
	if(/call.uni \(ok\),split_return,/) returns++;
	if(/mad.lo.u32 r22,r0,64,62;/) fences++;
	next;
}
{print}
END {
	if(skip || declarations!=1 || reductions!=1 || completions!=1 || combines!=1 || returns!=1 || fences!=1) {
		print "Unsupported resident expert combine/return pattern." > "/dev/stderr"; exit 1;
	}
}
' "$return_source" > "$return_output.part"
awk '
/^\/\/ BEGIN FLASH WEIGHTED RETURN/ {copy=1; starts++}
copy {print}
/^\/\/ END FLASH WEIGHTED RETURN/ {copy=0; ends++}
END {if(starts!=1 || ends!=1 || copy) exit 1}
' "$return_library" >> "$return_output.part"
mv -- "$return_output.part" "$return_output"
sha256sum "$return_source" "$return_library" "$return_output"
