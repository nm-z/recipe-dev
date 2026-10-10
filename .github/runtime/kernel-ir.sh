#!/bin/bash
# Assembles, call-checks and compiles every kernel module build.rs generates, for each target and precision.
# The runtime appends a few definitions per model; each is declared from its first call site, and only when recipe.rs defines it.
# Usage: kernel-ir.sh <build out directory> <recipe.rs> <Cargo.toml>
set -u
out=$1
recipe=$2
manifest=$3
work=$(mktemp -d)
fail=0

value() { sed -n "s/^$1 *= *\([0-9]*\).*/\1/p" "$manifest" | head -1; }
register_m=$(value contraction-register-m)
register_n=$(value contraction-register-n)
fragment_k=$(value contraction-fragment-k)
chunk_k=$(value contraction-chunk-k)

# Opaque pointers let a call with the wrong argument count or types pass llvm-as; a device then faults on it.
cat > "$work/calls.rb" <<'RUBY'
text = File.read(ARGV[0]).lines.map { |l| l.sub(/;.*$/, "") }.join
ATTRIBUTES = /\b(nocapture|readonly|writeonly|readnone|noalias|nonnull|nounwind|inreg|zeroext|signext|immarg|align \d+|dereferenceable\(\d+\)|noundef)\b/
def split_top(text)
	parts, depth, current = [], 0, +""
	text.each_char do |c|
		depth += 1 if "(<{[".include?(c)
		depth -= 1 if ")>}]".include?(c)
		if c == "," && depth == 0
			parts << current
			current = +""
		else
			current << c
		end
	end
	parts << current unless current.strip.empty? && parts.empty?
	parts.map { |p| p.strip.gsub(/\s+/, " ") }
end
signatures = {}
text.scan(/^(?:define|declare) ([^@\n]*?)@([\w.$]+)\(((?:[^()]|\([^()]*\))*)\)/m) do |ret, name, params|
	next if params.include?("...")
	signatures[name] = [ret.gsub(/\b(internal|dso_local|private|linkonce_odr|noundef|hidden|nounwind)\b/, "").gsub(/\s+/, " ").strip,
		split_top(params).map { |p| p.gsub(ATTRIBUTES, "").gsub(/\s+/, " ").strip.sub(/ %[\w.$]+\z/, "") }]
end
problems = 0
text.scan(/\b(?:call|invoke) ((?:[\w.]+ )*?)((?:<[^>]*>|\{[^}]*\}|[\w.]+(?: addrspace\(\d+\))?\*?)) @([\w.$]+)\(((?:[^()]|\([^()]*\))*)\)/m) do |_, ret, name, args|
	signature = signatures[name] or next
	given = split_top(args).map { |a| a.gsub(ATTRIBUTES, "").gsub(/\s+/, " ").strip.sub(/ (%[\w.$]+|-?[\w.+\-]+|true|false|null|poison|undef|zeroinitializer)\z/, "") }
	if given.size != signature[1].size
		puts "#{name}: call passes #{given.size} arguments, definition takes #{signature[1].size}"
		problems += 1
		next
	end
	given.zip(signature[1]).each_with_index do |(g, e), i|
		next if g == e
		puts "#{name}: argument #{i + 1} is #{g}, definition takes #{e}"
		problems += 1
	end
	if ret.strip != signature[0]
		puts "#{name}: call returns #{ret.strip}, definition returns #{signature[0]}"
		problems += 1
	end
end
exit(problems.zero? ? 0 : 1)
RUBY

count=0
for file in "$out"/recipe-*.ll; do
	name=$(basename "$file" .ll)
	case $name in recipe-cpu*) body='#1';; *) body='#3';; esac
	sed -e "s/RECIPE_WORKGROUP_SIZE/256/g;s/RECIPE_REGISTER_COUNT/$((register_m * register_n))/g;s/RECIPE_REGISTER_M/$register_m/g;s/RECIPE_REGISTER_N/$register_n/g;s/RECIPE_FRAGMENT_K/$fragment_k/g;s/RECIPE_CHUNK_K/$chunk_k/g;s/RECIPE_CHUNK_VALUES/$chunk_k/g;s/RECIPE_CHUNK_BIAS_VALUES/$register_n/g;s/RECIPE_SCRATCH_ROW_MASK/1023/g;s/RECIPE_SCRATCH_ROW_CLEAR/-1024/g;s/RECIPE_GRADIENT_SCRATCH_BASE/0/g" -e "s/RECIPE_CONTRACTION_BODY/$body/g" "$file" > "$work/module.ll"
	declared=" "
	while :; do
		message=$(llvm-as "$work/module.ll" -o "$work/module.bc" 2>&1) && break
		symbol=$(printf '%s\n' "$message" | sed -n "s/.*use of undefined value '@\([^']*\)'.*/\1/p" | head -1)
		case $declared in *" $symbol "*) symbol=;; esac
		if [ -z "$symbol" ] || ! grep -F "@$symbol(" "$recipe" | grep -q define; then
			printf '== %s\n%s\n' "$name" "$(printf '%s\n' "$message" | head -3)"
			fail=1
			continue 2
		fi
		call=$(grep -m1 -o "call [^@]*@$symbol([^)]*)" "$work/module.ll")
		ret=$(printf '%s' "$call" | sed "s/^call \(.*\) @.*/\1/")
		args=$(printf '%s' "$call" | sed "s/.*@$symbol(\(.*\))/\1/" | awk -F', ' '{ for (i = 1; i <= NF; i++) { n = split($i, w, " "); t = ""; for (j = 1; j < n; j++) t = t (j > 1 ? " " : "") w[j]; printf "%s%s", (i > 1 ? ", " : ""), t } }')
		printf 'declare %s @%s(%s)\n' "$ret" "$symbol" "$args" >> "$work/module.ll"
		declared="$declared$symbol "
	done
	if ! message=$(ruby "$work/calls.rb" "$work/module.ll"); then
		printf '== %s (calls)\n%s\n' "$name" "$(printf '%s\n' "$message" | head -5)"
		fail=1
	fi
	case $name in
		recipe-amd-gfx12*) target="-mcpu=gfx1200";;
		recipe-amd-gfx11*) target="-mcpu=gfx1100";;
		recipe-amd*) target="-mcpu=gfx900";;
		recipe-nvidia*) target="-mcpu=sm_70";;
		*) target=;;
	esac
	if ! message=$(llc -O1 $target "$work/module.bc" -filetype=null 2>&1); then
		printf '== %s (llc)\n%s\n' "$name" "$(printf '%s\n' "$message" | head -4)"
		fail=1
	fi
	count=$((count + 1))
done
echo "kernel modules checked: $count"
[ "$count" -gt 0 ] || fail=1
exit $fail
