use recipe::*;
use std::time::Instant;
fn main() {
	let case = std::env::var("CASE").unwrap_or("509".into());
	let (path, split): (&str, f64) = match case.as_str() {
		"619" | "618" | "497" => ("data/temporal/series_sqlite.sqlite", 1.0),
		"520" => ("data/image/arrays_npz.npz", 1.0),
		"623" => ("data/image/archive_zip.zip", 1.0),
		"277" => ("data/categoric/sample_subfolders", 0.8),
		"622" | "510" => ("data/temporal/chronological_splits", 0.8),
		_ => ("data/temporal/chronological_splits", 1.0),
	};
	let data = recipe.data(path).target("target").split(split);
	let model = match case.as_str() {
		"509" => recipe.model().perc(8).relu().qi(4).k.s.layer(4).exp().norm(batch).iq(2).xxs.loss(ce),
		"621" => recipe.model().layer(8).res([layer(8), layer(8)]).selu().qi(8).0.layer(8).res([conv(8, 1), relu(), layer(8)]).exp().qi(5).k.loss(ce),
		"622" => recipe.model().layer(8).tanh().norm(batch).qi(2).k.kmeans(2).silu().norm(batch).qi(5).0.loss(ce),
		"633" => recipe.model().rnn(8).silu().norm(batch).iq(4).nl.layer(4).gelu().norm(batch).qi(3).k.l.loss(focal),
		"629" => recipe.model().conv(8, 1).tan().norm(batch).qi(5).k.m.layer(8).res([layer(8), layer(8)]).sigmoid().norm(batch).qi(3).k.l.loss(mae),
		"623" => recipe.model().conv(8, 1).leak().iq(3).m.xgbst(8).selu().iq(2).s.loss(mse),
		"619" => recipe.model().xgbst(8).silu().qi(4).0.attn(1).huber().qi(4).k.m.loss(bce),
		"618" => recipe.model().kmeans(2).ln().qi(8).1.layer(8).moe(1, [layer(8), layer(8)]).elu().norm(batch).qi(3).k.m.loss(rmse),
		"611" => recipe.model().attn(1).selu().norm(batch).iq(2).xxs.lgbm(8).norm(batch).qi(3).k.loss(ce),
		"520" => recipe.model().attn(1).exp().iq(4).xs.gru(8).elu().norm(batch).iq(2).xs.loss(bce),
		"518" => recipe.model().layer(8).res([conv(8, 1), relu(), layer(8)]).sigmoid().iq(3).xxs.layer(8).moe(1, [layer(8), layer(8)]).sigmoid().qi(3).k.loss(focal),
		"512" => recipe.model().layer(8).res([conv(8, 1), conv(8, 1)]).silu().qi(4).k.s.xgbst(2).silu().loss(rmse),
		"511" => recipe.model().layer(8).res([layer(8), layer(8)]).cos().norm(batch).qi(5).0.layer(8).res([conv(8, 1), conv(8, 1)]).norm(batch).iq(2).xxs.loss(huber),
		"510" => recipe.model().lgbm(8).prelu().norm(batch).qi(3).k.m.layer(4).log().norm(batch).qi(3).k.l.loss(rmse),
		"497" => recipe.model().xgbst(8).log().norm(batch).iq(4).nl.xgbst(8).log().norm(batch).iq(1).m.loss(rmse),
		"491" => recipe.model().cbst(8).silu().qi(5).k.m.layer(8).res([layer(8), layer(8)]).leak().qi(4).nf.loss(mse),
		"277" => recipe.model().lgbm(8).norm(batch).iq(2).xxs.conv(8, 1).tan().norm(batch).qi(4).1.loss(ce),
		_ => panic!("unknown case"),
	};
	let start = Instant::now();
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(17).epochs(1).fp(32).run(&model, &data);
	eprintln!("RESULT case={case} data={path} train_seconds={:.1} loss={}", start.elapsed().as_secs_f64(), report.final_loss());
}
