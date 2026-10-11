use recipe::*;
fn main() {
	let started = std::time::Instant::now();
	let bundle = format!("/tmp/recipe-513-{}.ogdl", std::process::id());
	let data = recipe.data("data/image/class_subfolders").target("target");
	let model = recipe.model().lstm(8).cos().norm(batch).iq(2).m.gru(8).qi(8).0.loss(mae);
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(14625434013680).epochs(1).log(all).stop(0.0).int(1).save(&bundle).run(&model, &data);
	let trained = started.elapsed().as_secs_f64();
	let output = recipe.infer(&bundle, &[0.0; 3072]);
	println!("wall_total={:.3} train_call={trained:.3} epoch_seconds={:.4} final={} infer_finite={}", started.elapsed().as_secs_f64(), report.epoch_seconds(), report.final_loss(), output.iter().all(|v| v.is_finite()));
}
