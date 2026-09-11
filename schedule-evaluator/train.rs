use recipe::*;

#[rustfmt::skip]
fn main() {
	let data = recipe.data(auto);
	let evaluator = recipe.model()
		.layer(16).tanh()
		.loss(mse);
	let proposal = recipe.model()
		.res([
			attn(32)
		])
		.layer(24).prelu()
		.loss(&evaluator);
	recipe.train()
		.rat(history, "./evaluate-c")
		.target(0.0)
		.lr(0.1)
		.epochs(10000000)
		.log(all)
		.run(&proposal, &data);
}
