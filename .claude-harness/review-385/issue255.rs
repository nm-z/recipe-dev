use recipe::*;

fn main() {
	let bundle = format!("{}/issue255-{}.ogdl", std::env::var("REVIEW_ARTIFACTS").unwrap(), std::process::id());
	let data = recipe.data("data/numeric/single_csv.csv")
		.target("target")
		.split(0.8);
	let model = recipe.model()
		.forest(2).leak().qi(4).k
		.xgbst().gelu().norm(batch).iq(4).xs
		.loss(rmse);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.0001)
		.seed(33684227594820)
		.epochs(1)
		.log(all)
		.fp(64)
		.save(&bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
	println!("final_loss {:.17e}", report.final_loss());
}
