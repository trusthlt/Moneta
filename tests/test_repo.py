from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def test_package_imports():
    import moneta  # noqa: F401


def test_config_paths_are_relative():
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    for key, value in config["paths"].items():
        assert not Path(value).is_absolute(), f"paths.{key} must be relative: {value}"


def test_released_data_is_present():
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    for key in ("data", "nace_descriptions", "osm_mapping"):
        assert (ROOT / config["paths"][key]).exists(), f"missing paths.{key}"


def test_models_have_path_and_revision():
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    for name, source in config["models"].items():
        assert source["path"] and source["revision"], name
