FNR==NR {
	if ($1==".global") { name=$5;sub(/\[.*/,"",name);globals[name]=1;next }
	for(name in globals)if($0~("(^|[^[:alnum:]_])"name"([^[:alnum:]_]|$)"))used[name]=1;
	next;
}
$1==".global" {name=$5;sub(/\[.*/,"",name);if(!used[name])next}
{print}
