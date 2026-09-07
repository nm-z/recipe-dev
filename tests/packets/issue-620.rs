use recipe::*;

/// The input width the saved bundle records, so the separate inference feeds
/// the model the shape it was trained on without restating it here.
fn input_width(bundle: &std::path::Path) -> usize {
	let text = std::fs::read_to_string(bundle).unwrap();
	let mut width = 0;
	for line in text.lines() {
		let mut fields = line.trim_start().split_whitespace();
		if fields.next() == Some("feature") {
			width += fields.next().unwrap().parse::<usize>().unwrap();
		}
	}
	assert!(width != 0);
	width
}

fn main() {
	let directory = std::path::Path::new("target/packets");
	std::fs::create_dir_all(directory).unwrap();
	let bundle = directory.join("issue-620.ogdl");
	if std::env::var_os("RECIPE_PACKET_INFER").is_some() {
		let before = std::fs::read(&bundle).unwrap();
		let output = recipe.infer(&bundle, &vec![0.0; input_width(&bundle)]);
		assert!(!output.is_empty());
		assert!(output.iter().all(|value| value.is_finite()));
		assert_eq!(std::fs::read(&bundle).unwrap(), before);
		println!("inference {output:?}; bundle unchanged");
		return;
	}
	let data = recipe.data("data/text/records_xml.xml")
		.target("target")
		.norm(z_score);
	let model = recipe.model()
		.cbst().qi(6).k
		.knn(3).elu().norm(batch).iq(3).xxs
		.loss(focal);
	let report = recipe.train()
		.optimizer(adamw)
		.lr(0.0001)
		.seed(31353528716383)
		.epochs(1)
		.log(all)
		.fp(8)
		.save(&bundle)
		.run(&model, &data);
	assert!(report.final_loss().is_finite());
	assert!(!report.predictions().is_empty());
	assert!(report.predictions().iter().all(|value| value.is_finite()));
	assert!(std::fs::metadata(&bundle).unwrap().len() > 0);
	println!("bundle {}", bundle.display());
}
