use recipe::*;

fn main() {
	let bundle = std::env::var("REVIEW_BUNDLE").expect("REVIEW_BUNDLE");
	let data = recipe.data("data/numeric/single_csv.csv")
		.target("target");
	let model = recipe.model()
		.layer(8)
		.res([layer(8), layer(8)]).norm(batch).qi(4).k.m
		.cbst().selu().norm(batch).qi(6).k
		.loss(focal);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.001)
		.seed(65681931961096)
		.epochs(1)
		.log(all)
		.int(1)
		.save(&bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
}
