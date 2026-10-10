function name_of(line, value) { value=line;sub(/.* packed_gate_up_/,"packed_gate_up_",value);sub(/\(.*/,"",value);return value }
FNR==NR {
	if($0 ~ /^\.func .* packed_gate_up_(17|18)_/) { name=name_of($0);capture=1;depth=0;opened=0;text="" }
	if(capture) {
		line=$0;gsub(/stream_iq2_pairs/,"packed_iq2",line);text=text line "\n"
		copy=line;opens=gsub(/\{/,"",copy);copy=line;closes=gsub(/\}/,"",copy)
		if(opens)opened=1;depth+=opens-closes
		if(opened&&depth==0) { bodies[name]=text;capture=0 }
	}
	next
}
{
	if($0 ~ /^\.func .* packed_gate_up_(17|18)_/) { name=name_of($0);if(!(name in bodies)){print "missing fused body " name > "/dev/stderr";exit 1}printf "%s",bodies[name];skip=1;depth=0;opened=0;replaced++ }
	if(skip) {
		copy=$0;opens=gsub(/\{/,"",copy);copy=$0;closes=gsub(/\}/,"",copy)
		if(opens)opened=1;depth+=opens-closes
		if(opened&&depth==0)skip=0
		next
	}
	print
}
END {print "replaced_fused_functions="replaced > "/dev/stderr"}
