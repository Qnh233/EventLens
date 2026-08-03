from eventlens.config import load_settings


def test_load_settings_reads_single_central_config():
    settings = load_settings("configs/app.yaml")

    assert settings.app.name == "EventLens"
    assert settings.model.text.max_content_chars == 1200
    assert settings.cluster.threshold == 0.72
    assert settings.credibility.alert_thresholds.credibility_high == 0.75
    assert settings.evaluation.time_train_ratio == 0.8
    assert settings.paths.model_dir == "artifacts/models"
