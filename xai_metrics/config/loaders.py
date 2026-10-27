# xai_metrics\config\loaders.py
from pathlib import Path
import pickle
import joblib
import torch
import yaml

from typing import Any, Mapping, Dict

def default_model_loader(model_path: str | Path) -> Any:
    """
    Load a model from disk.

    Files with ``.pkl`` or ``.pickle`` extensions are loaded with
    :mod:`pickle`. Files with ``.joblib`` or ``.jl`` extensions are loaded with
    :mod:`joblib`. Any other file extension is loaded with :func:`torch.load`
    using ``weights_only=False``.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to the saved model file.

    Returns
    -------
    Any
        Loaded model object.
    """
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as f:
            return pickle.load(f)

    if suffix in {".joblib", ".jl"}:
        return joblib.load(model_path)

    return torch.load(model_path, weights_only=False)


def load_config(config: Mapping[str, Any] | str | Path) -> Dict:
    """
    Load configuration data from a mapping or YAML file.

    Parameters
    ----------
    config : Mapping[str, Any], str or pathlib.Path
        Configuration mapping or path to a YAML configuration file. If the
        value is ``"config.yaml"``, the file is loaded from the directory
        containing this module.

    Returns
    -------
    Dict
        Loaded configuration dictionary. If the YAML file is empty, an
        empty dictionary is returned.
    """
    if not isinstance(config, Mapping) and config == "config.yaml":
        config_path = Path(__file__).resolve().parent / config
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
        
    if isinstance(config, Mapping):
        return dict(config)
    
    config_path = Path(config)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}