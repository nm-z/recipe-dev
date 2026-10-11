#!/bin/bash
set -euo pipefail
if (( $# != 4 )); then
	printf 'Usage: splice-router.sh SOURCE_PTX EXPECTED_SHA256 BARRIER_PTX OUTPUT_PTX\n' >&2
	exit 2
fi
router_source=$1
router_expected=$2
router_library=$3
router_output=$4
router_actual=$(sha256sum "$router_source")
[[ ${router_actual%% *} == "$router_expected" ]] || { printf 'Resident PTX hash differs.\n' >&2; exit 1; }
[[ $router_output != "$router_source" ]] || { printf 'Output must differ from resident PTX.\n' >&2; exit 1; }
awk '
function emit_function(    i,line,target,anchor) {
	if (!topk) {printf "%s",body; return}
	if (body !~ /mov.b64[ \t]+%rd16, contraction_tile_f32;/) {print "Unsupported TopK tile register." > "/dev/stderr"; exit 1}
	n=split(body,lines,"\n"); anchor=0;
	for(i=1;i<n;i++) {
		line=lines[i];
		if(line ~ /@%p60 bra[ \t]+\$L__BB[0-9]+_42;/) {
			target=line; sub(/.*bra[ \t]+/,"",target); sub(/;.*/,"",target);
			print "\t{ .param .u64 router_tile; .param .u32 router_ok; .reg .u32 router_rc; .reg .pred router_fast;";
			print "\tst.param.u64 [router_tile], %rd16; call.uni (router_ok), flash_router_top10, (router_tile);";
			print "\tld.param.u32 router_rc, [router_ok]; setp.ne.u32 router_fast, router_rc, 0; @router_fast bra "target"; }";
			anchor++;
		}
		print line;
	}
	if(anchor!=1) {print "Unsupported TopK selection pattern." > "/dev/stderr"; exit 1}
	patched++;
}
/^\.address_size 64/ {
	print;
	print ".func (.param .u32 result) flash_router_top10(.param .u64 tile0);";
	declarations++; next;
}
/^\.func recipe_operation_n[0-9]+\(/ {capture=1; body=""; topk=0}
capture {
	body=body $0 "\n";
	if($0 ~ /topk_forward_shared_body.exit.i/) topk=1;
	if($0 ~ /^}/) {emit_function(); capture=0}
	next;
}
{print}
END {
	if(capture || declarations!=1 || patched!=48) {
		print "Expected one header and 48 complete TopK replacements; observed "declarations", "patched > "/dev/stderr"; exit 1;
	}
}
' "$router_source" > "$router_output.part"
awk '
/^\/\/ BEGIN FLASH ROUTER TOP10/ {copy=1; starts++}
copy {print}
/^\/\/ END FLASH ROUTER TOP10/ {copy=0; ends++}
END {if(starts!=1 || ends!=1 || copy) exit 1}
' "$router_library" >> "$router_output.part"
mv -- "$router_output.part" "$router_output"
sha256sum "$router_source" "$router_library" "$router_output"
