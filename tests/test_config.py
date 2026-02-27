from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_config_resolves():
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="config")

    assert "analysis" in cfg
    assert "features" in cfg
    assert "samples" in cfg
    assert "merge" in cfg
    assert "paths" in cfg
    assert cfg.analysis.scope is not None
