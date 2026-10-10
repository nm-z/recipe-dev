match($0,/\/\*([0-9a-f]+)\*\//,pc) {
	address=strtonum("0x" pc[1]); instructions[address]=$0
	if(match($0,/BRA[[:space:]]+(0x[0-9a-f]+)/,branch)) {
		target=strtonum(branch[1]); if(target<address) { first[++loops]=target; last[loops]=address }
	}
}
END {
	for(loop=1;loop<=loops;loop++) {
		count=0;for(address in instructions)if((address+0)>=first[loop]&&(address+0)<=last[loop]&&instructions[address]!~/NOP/)count++
		printf "0x%x..0x%x instructions=%d\n",first[loop],last[loop],count
	}
}
