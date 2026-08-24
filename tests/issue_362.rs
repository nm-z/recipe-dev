use recipe::*;

#[test]
fn knn_teacher_labeling_completes() {
    let bundle = format!("/tmp/recipe-362-train-{}.ogdl", std::process::id());
    let data = recipe.data("data/ordinal/string_levels.csv")
        .target("target")
        .norm(z_score);
    let model = recipe.model()
        .knn(3).prelu().iq(3).m
        .kmeans(2).exp().qi(2).k
        .loss(mae);
    let report = recipe.train()
        .optimizer(adamw)
        .lr(0.0001)
        .seed(57143384586813)
        .epochs(1)
        .log(all)
        .stop(0.8)
        .int(8)
        .save(&bundle)
        .run(&model, &data);
    assert!(report.final_loss().is_finite());
    let _ = std::fs::remove_file(&bundle);
}

#[test]
fn knn_teacher_save_resume() {
    let bundle = format!("/tmp/recipe-362-resume-{}.ogdl", std::process::id());
    let data = recipe.data("data/ordinal/string_levels.csv")
        .target("target")
        .norm(z_score);
    let model = recipe.model()
        .knn(3).prelu().iq(3).m
        .kmeans(2).exp().qi(2).k
        .loss(mae);
    let report = recipe.train()
        .optimizer(adamw)
        .lr(0.0001)
        .seed(57143384586813)
        .epochs(1)
        .stop(0.8)
        .int(8)
        .save(&bundle)
        .run(&model, &data);
    assert!(report.final_loss().is_finite());
    let resumed = recipe.train()
        .optimizer(adamw)
        .lr(0.0001)
        .seed(57143384586813)
        .epochs(1)
        .stop(0.8)
        .int(8)
        .resume(&bundle)
        .save(&bundle)
        .run(&model, &data);
    assert!(resumed.final_loss().is_finite());
    let _ = std::fs::remove_file(&bundle);
}

#[test]
fn knn_teacher_infer() {
    let bundle = format!("/tmp/recipe-362-infer-{}.ogdl", std::process::id());
    let data = recipe.data("data/ordinal/string_levels.csv")
        .target("target")
        .norm(z_score);
    let model = recipe.model()
        .knn(3).prelu().iq(3).m
        .kmeans(2).exp().qi(2).k
        .loss(mae);
    recipe.train()
        .optimizer(adamw)
        .lr(0.0001)
        .seed(57143384586813)
        .epochs(1)
        .stop(0.8)
        .int(8)
        .save(&bundle)
        .run(&model, &data);
    let result = recipe.infer(&bundle, &[1.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(result.len(), 1);
    assert!(result[0].is_finite());
    let _ = std::fs::remove_file(&bundle);
}

#[test]
fn knn_k1_teacher() {
    let data = recipe.data("data/numeric/single_csv.csv")
        .target("target")
        .norm(z_score)
        .split(0.001);
    let model = recipe.model()
        .knn(1).layer(1)
        .loss(mse);
    let report = recipe.train()
        .seed(42)
        .epochs(1)
        .stop(0.8)
        .run(&model, &data);
    assert!(report.final_loss().is_finite());
}

#[test]
fn knn_standalone_no_teacher() {
    let bundle = format!("/tmp/recipe-362-standalone-{}.ogdl", std::process::id());
    let data = recipe.data("data/numeric/single_csv.csv")
        .target("target")
        .norm(z_score)
        .split(0.001);
    let model = recipe.model()
        .knn(1)
        .loss(mse);
    let report = recipe.train()
        .seed(42)
        .epochs(1)
        .stop(0.8)
        .save(&bundle)
        .run(&model, &data);
    assert!(report.final_loss().is_finite());
    let result = recipe.infer(&bundle, &[0.5]);
    assert_eq!(result.len(), 1);
    assert!(result[0].is_finite());
    let _ = std::fs::remove_file(&bundle);
}
