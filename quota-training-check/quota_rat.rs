use recipe::*;

#[rustfmt::skip]
const TARGETS: [&str; 12] = [
	"gpt_5_6_sol_input_percent_per_million",
	"gpt_5_6_sol_output_percent_per_million",
	"gpt_5_6_luna_input_percent_per_million",
	"gpt_5_6_luna_output_percent_per_million",
	"gpt_5_6_terra_input_percent_per_million",
	"gpt_5_6_terra_output_percent_per_million",
	"gpt_6_astra_input_percent_per_million",
	"gpt_6_astra_output_percent_per_million",
	"cache_ttl_seconds",
	"cache_read_multiplier",
	"cache_write_additional_multiplier",
	"over_272k_input_multiplier",
];
#[rustfmt::skip]
fn main() {
	let data = recipe
		.data("corpus-clean.tsv")
		.split(0.5)
		.target(TARGETS);

	let evaluator = recipe.model()
		.layer(1)
		.loss(mse);

	let proposal = recipe.model()
		.layer(1)
		.loss(&evaluator);

	recipe.train()
		.rat("/home/nate/Desktop/recipe-dev/evaluate-quota")
		.fp(32)
		.lr(0.001)
		.stop(0.0001)
		.epochs(10)
		.log(all)
		.save("quota-model.ogdl")
		.run(&proposal, &data);

}
