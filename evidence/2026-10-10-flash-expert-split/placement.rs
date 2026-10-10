use recipe::{ExpertDieBudget, ExpertSplitPlan, Gguf};
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
	let root = PathBuf::from("/mnt/sentry-nfs/flash-next-q2");
	let files: Vec<_> = (1..=3).map(|index| root.join(format!("Qwen3.8-Flash-Next-UD-Q2_K_XL-{index:05}-of-00003.gguf"))).collect();
	let target = Gguf::tensor_headers(&files)?;
	let head = Gguf::tensor_headers(&[root.join("mtp-header.partial")])?;
	assert_eq!(target.len(), 1224);
	let expert_bytes: usize = target.iter().filter(|tensor| tensor.name.ends_with("_exps.weight")).map(|tensor| tensor.bytes).sum();
	assert_eq!(expert_bytes, 46_084_915_200);
	let main_bytes: usize = target.iter().chain(&head).filter(|tensor| !matches!(tensor.name.as_str(), "token_embd.weight" | "per_layer_token_embd.weight") && (!tensor.name.ends_with("_exps.weight") || tensor.name.starts_with("blk.48."))).map(|tensor| tensor.bytes).sum();
	assert_eq!(main_bytes, 6_987_059_712);
	println!("Real headers: {} target tensors, {} head tensors, {} expert bytes, {} main-die bytes including the head's own Q8 output.", target.len(), head.len(), expert_bytes, main_bytes);
	let capacity = std::fs::read_to_string("/home/nate/codex/cx-fl-split-build/vram.csv")?;
	let mut live = [ExpertDieBudget { free_bytes: 0, reserve_bytes: 256 << 20 }; 6];
	let mut total = live;
	for line in capacity.lines() {
		let values: Vec<usize> = line.split(',').map(|word| word.trim().parse().unwrap()).collect();
		let die = values[0];
		if die >= 6 { continue; }
		live[die].free_bytes = values[1] << 20;
		total[die].free_bytes = values[2] << 20;
	}
	// Existing manifest's conservative nv0 usable cap stays below the bad band.
	live[0].free_bytes = live[0].free_bytes.min(6_186_598_400);
	total[0].free_bytes = total[0].free_bytes.min(6_186_598_400);
	live[2].reserve_bytes = 512 << 20;
	total[2].reserve_bytes = 512 << 20;
	for (name, budgets) in [("Live free VRAM, nv0 cap, KV/scratch reserved", live), ("Physical capacity upper bound after service release, nv0 cap, reserves", total)] {
		let plan = ExpertSplitPlan::new(&target, &head, 2, budgets)?;
		assert!(!plan.fits());
		assert!(plan.placements.iter().all(|item| item.expert.is_none() || item.die != 2));
		assert_eq!(plan.experts.iter().sum::<usize>() + plan.unplaced_experts, 48 * 512);
		let assigned: usize = plan.placements.iter().filter(|item| item.expert.is_some()).map(|item| item.tensor.bytes).sum();
		assert_eq!(assigned + plan.unplaced_bytes, expert_bytes);
		println!("\n## {name}\n\n{}", plan.table());
	}
	let roomy = [ExpertDieBudget { free_bytes: 32 << 30, reserve_bytes: 512 << 20 }; 6];
	let plan = ExpertSplitPlan::new(&target, &head, 2, roomy)?;
	assert!(plan.fits());
	assert_eq!(plan.unplaced_experts, 0);
	let mut owners = std::collections::HashMap::new();
	let mut matrices = std::collections::HashMap::new();
	for item in &plan.placements {
		assert_eq!(item.offset % 256, 0);
		if let Some(expert) = item.expert {
			let layer: usize = item.tensor.name.split('.').nth(1).unwrap().parse().unwrap();
			let key = (layer, expert);
			assert_eq!(*owners.entry(key).or_insert(item.die), item.die);
			*matrices.entry(key).or_insert(0) += 1;
		}
	}
	assert_eq!(owners.len(), 48 * 512);
	assert!(matrices.values().all(|count| *count == 3));
	let picks = [10, 0, 17, 63, 99, 147, 231, 301, 379, 443, 511];
	assert_eq!(plan.selected_dies(0, &picks, 10)?.iter().map(|(_, slots)| slots.len()).sum::<usize>(), 10);
	assert!(plan.selected_dies(0, &[2, 3, 3], 2).is_err());
	assert!(plan.selected_dies(0, &[1, -1], 1).is_err());
	assert!(plan.selected_dies(0, &[1, 512], 1).is_err());
	assert!(ExpertSplitPlan::new(&target, &head, 6, roomy).is_err());
	assert!(ExpertSplitPlan::new(&target, &head, 7, roomy).is_err());
	println!("Planner checks PASS: byte conservation, 24,576 complete expert bundles, same-die matrices, alignment, selected-slot mapping, invalid selection rejection, and forbidden main dies. The roomy budget is a CPU fixture, not an Archy fit claim.");
	Ok(())
}
