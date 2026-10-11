/^\.func /&&/packed_(g_|h_|gate_up_)/{capture=1;depth=0;opened=0}
capture{
	line=$0;gsub(/packed_gate_up_/,"packed_worker_gate_up_",line);gsub(/packed_g_/,"packed_worker_g_",line);gsub(/packed_h_/,"packed_worker_h_",line);print line
	copy=$0;opens=gsub(/\{/,"",copy);copy=$0;closes=gsub(/\}/,"",copy);if(opens)opened=1;depth+=opens-closes
	if(opened&&!depth){capture=0;count++}
}
END{if(count!=105){print "unexpected worker helper count "count >"/dev/stderr";exit 1}}
