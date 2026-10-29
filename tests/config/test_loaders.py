# tests\config\test_loaders.py
from xai_metrics.config import ConfigController

def test_config_controller_loads_config_from_mapping():
    config = {
        "metrics": [
            {"name": "dummy", "params": {"value": 1.0}}
        ]
    }

    controller = ConfigController(config=config)

    assert controller.config == config
    assert controller.get_metrics_config() == config['metrics']


def test_config_controller_loads_config_from_yaml_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "metrics:\n"
            "  - name: dummy\n"
            "    params:\n"
            "      value: 3.0\n"
        ),
        encoding="utf-8"
    )

    controller = ConfigController(config=config_path)

    assert controller.get_metrics_config() == [
        {"name": "dummy", "params": {"value": 3.0}}
    ]