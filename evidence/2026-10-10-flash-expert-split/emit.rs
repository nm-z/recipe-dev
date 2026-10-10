fn main() {
	let entries = std::fs::read_to_string("/home/nate/codex/cx-fl-split-build/probe-entries.ptx").unwrap();
	let source = format!(".version 7.4\n.target sm_52\n.address_size 64\n{}\n{}\n{}", recipe::p2p_functions(), recipe::expert_split_functions(), entries);
	std::fs::write("/home/nate/codex/cx-fl-split-build/probe.ptx", source).unwrap();
}
