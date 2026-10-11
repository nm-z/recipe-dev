# Keep device functions and tables; remove standalone probe entries.
/^\.address_size/ { print; print "// BEGIN PACKED HELPERS"; next }
/^\.visible \.entry/ { dropping=1; depth=0; opened=0 }
{
	if (dropping) {
		line=$0; opens=gsub(/\{/,"{",line); closes=gsub(/\}/,"}",line);
		if (opens) opened=1;
		depth+=opens-closes;
		if (opened && depth==0) dropping=0;
		next;
	}
	sub(/[ \t]+$/, "");
	if ($0 == "") { blanks++; next }
	while (blanks > 0) { print ""; blanks-- }
	print;
}
