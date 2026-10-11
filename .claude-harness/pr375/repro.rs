use recipe::*;

fn main() {
	let bundle = std::env::var("REVIEW_BUNDLE").expect("REVIEW_BUNDLE is unset");
	let source = std::env::var("REVIEW_DATA").expect("REVIEW_DATA is unset");
	let data = recipe.data(source.as_str()).target("target");
	let model = recipe.model().layer(9).relu().layer(5).relu().layer(1).loss(mse);
	let report = recipe.train().optimizer(adamw).lr(0.01).seed(17).epochs(3).log(all).fp(32).save(&bundle).run(&model, &data);
	println!("initial {:016x}", report.initial_loss().to_bits());
	println!("final {:016x}", report.final_loss().to_bits());
	println!("predictions {}", report.predictions().len());
	for value in report.predictions().iter().take(4) {
		println!("prediction {:016x}", value.to_bits());
	}
	let inferred = recipe.infer(&bundle, &[0.5]);
	for value in &inferred {
		println!("infer {:016x}", value.to_bits());
	}
}
