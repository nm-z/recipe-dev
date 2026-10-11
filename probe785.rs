use recipe::*;
fn main() {
	let data = recipe.data("data/numeric/sample_subfolders").target("target");
	let epochs: usize = std::env::var("EPOCHS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
	let case = std::env::var("CASE").unwrap_or("svm".into());
	let model = recipe.model().layer(16);
	let model = match case.as_str() {
		"svm" => model.moe(1, [svm()]).layer(1).loss(mse),
		"svm-only" => model.moe(1, [svm()]).loss(mse),
		"res-svm" => model.res([svm()]).layer(1).loss(mse),
		"layer" => model.moe(1, [layer(16)]).layer(1).loss(mse),
		"knn" => model.moe(1, [knn(3)]).layer(1).loss(mse),
		"xgbst" => model.moe(1, [xgbst(2)]).layer(1).loss(mse),
		"bayes" => model.moe(1, [bayes()]).layer(1).loss(mse),
		"first-svm" => recipe.model().svm().loss(mse),
		"first-svm-layer" => recipe.model().svm().layer(1).loss(mse),
		_ => panic!("unknown case"),
	};
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(17).epochs(epochs).fp(32).run(&model, &data);
	let p = report.predictions();
	let (mn, mx) = p.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
	let ip = report.initial_predictions();
	let (imn, imx) = ip.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
	eprintln!("RESULT case={case} epochs={epochs} initial_loss={} final_loss={} initial_pred=[{imn:.4},{imx:.4}] final_pred=[{mn:.4},{mx:.4}] rows={}", report.initial_loss(), report.final_loss(), p.len());
}
