function emit(line){if(line!="")print line}
/^\.func .*packed_[gh]_/ {
	name=$0;sub(/.*packed_/,"packed_",name);sub(/\(.*/,"",name);header=1;pending=""
}
header {
	if(index($0,name "_param_10")){sub(/,$/,"",pending);emit(pending);pending="";next}
	if($0==")"){emit(pending);print;header=0;next}
	emit(pending);pending=$0;next
}
{
	if($0~/ld.param.u64/&&index($0,name "_param_10")){
		line=$0;sub(/ld.param.u64/,"cvta.shared.u64",line);sub(/\[.*\]/,"scratch",line);print line;next
	}
	print
}
