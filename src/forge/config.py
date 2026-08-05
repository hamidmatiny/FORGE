"""Hydra Compose API config loading.

FORGE's CLI entry point is Typer, not Hydra's own ``@hydra.main`` decorator —
the two frameworks both want to own argv parsing, so this module uses Hydra's
`Compose API <https://hydra.cc/docs/advanced/compose_api/>`_ instead: it loads
a YAML config from ``conf/`` and merges in explicit key=value overrides
supplied by the calling CLI command. Hydra still owns config composition and
override syntax; Typer still owns the command surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONF_DIR = Path(__file__).resolve().parent.parent.parent / "conf"


def load_config(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose a Hydra config from ``conf/<config_name>.yaml`` plus overrides.

    Args:
        config_name: File stem under ``conf/`` (without ``.yaml``).
        overrides: Hydra-style ``key=value`` override strings.
    """
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    if not isinstance(cfg, DictConfig):  # pragma: no cover - compose always returns DictConfig here
        raise TypeError(f"Expected a DictConfig from compose(), got {type(cfg)}")
    return cfg


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a composed config into a plain dict."""
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):  # pragma: no cover - defensive
        raise TypeError(f"Expected a dict from OmegaConf.to_container(), got {type(container)}")
    return {str(key): value for key, value in container.items()}
