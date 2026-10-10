"""Config drift guard: desktop.yaml and pi5-4gb.yaml must have the same schema."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from agent_core.config import Config, load

ROOT = Path(__file__).resolve().parent.parent


def _schema(obj: object) -> set[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()

    def walk(o: object, path: tuple[str, ...]) -> None:
        if not is_dataclass(o) or isinstance(o, type):
            return
        for f in fields(o):
            p = path + (f.name,)
            seen.add(p)
            walk(getattr(o, f.name), p)

    walk(obj, ())
    return seen


def test_both_configs_load() -> None:
    desktop = load(ROOT / "configs" / "desktop.yaml")
    pi = load(ROOT / "configs" / "pi5-4gb.yaml")
    assert isinstance(desktop, Config) and isinstance(pi, Config)


def test_schema_identical() -> None:
    desktop = load(ROOT / "configs" / "desktop.yaml")
    pi = load(ROOT / "configs" / "pi5-4gb.yaml")
    assert _schema(desktop) == _schema(pi), "configs drifted — add the key to BOTH yaml files"


def test_unknown_key_rejected() -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "configs" / "desktop.yaml").read_text(encoding="utf-8"))
    raw["nonexistent_key"] = 1
    with pytest.raises(ValueError):
        _rebuild(raw)


def _rebuild(raw: dict) -> Config:
    from agent_core import config as cfgmod

    return cfgmod._build(Config, raw, "config")


def test_model_paths_resolve() -> None:
    cfg = load(ROOT / "configs" / "desktop.yaml")
    for p in (cfg.turn.model, cfg.stt.model_dir, cfg.tts.voice, cfg.wake.dir):
        assert Path(p).is_absolute(), f"path not resolved: {p}"
    assert Path(cfg.stt.model_dir, "model.bin").is_file()
    assert Path(cfg.tts.voice).is_file()
    assert Path(cfg.turn.model).is_file()
    for name in cfg.wake.names:
        assert list(Path(cfg.wake.dir).glob(f"{name}*.onnx")), "wake word model missing"
