"""Config loading.

Strict on purpose: an unknown or missing key raises ConfigError. That keeps
configs/desktop.yaml and configs/pi5-4gb.yaml in sync (see tests/test_config_drift.py)
and turns silent default drift into a loud failure.
"""
from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

# Offline guards must be set before any model library is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ServerCfg:
    host: str
    port: int


@dataclass(frozen=True)
class AudioCfg:
    sample_rate: int
    frame_ms: int

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


@dataclass(frozen=True)
class VadCfg:
    threshold: float
    min_silence_ms: int
    speech_pad_ms: int
    min_speech_ms: int
    barge_sustain_ms: int      # continuous speech needed to interrupt a playing reply
    barge_prob: float          # per-frame VAD confidence needed to count toward barge


@dataclass(frozen=True)
class TurnCfg:
    model: str
    threshold: float
    force_after_ms: int


@dataclass(frozen=True)
class WakeCfg:
    dir: str
    names: list[str]
    threshold: float
    refractory_ms: int


@dataclass(frozen=True)
class SttCfg:
    model_dir: str
    compute_type: str
    language: str | None
    partial_every_ms: int


@dataclass(frozen=True)
class TtsCfg:
    voice: str
    gain: float


@dataclass(frozen=True)
class BrainCfg:
    type: str                       # 'stub' | 'groq' (dev bridge) | later 'llm'
    model: str = "llama-3.3-70b-versatile"
    api_key_env: str = "GROQ_API_KEY"


@dataclass(frozen=True)
class DormancyCfg:
    timeout_s: int


@dataclass(frozen=True)
class SessionCfg:
    log_dir: str
    save_utterance_audio: bool


@dataclass(frozen=True)
class StoreCfg:
    path: str


@dataclass(frozen=True)
class WeatherCfg:
    enabled: bool


@dataclass(frozen=True)
class SearchCfg:
    enabled: bool
    max_results: int


@dataclass(frozen=True)
class TelegramCfg:
    enabled: bool
    token_env: str
    chat_id_env: str


@dataclass(frozen=True)
class ToolsCfg:
    weather: WeatherCfg
    search: SearchCfg
    telegram: TelegramCfg


@dataclass(frozen=True)
class DiarizeCfg:
    segmentation: str
    embedding: str
    num_clusters: int
    threshold: float


@dataclass(frozen=True)
class MeetingCfg:
    diarize: DiarizeCfg


@dataclass(frozen=True)
class Config:
    server: ServerCfg
    audio: AudioCfg
    vad: VadCfg
    turn: TurnCfg
    wake: WakeCfg
    stt: SttCfg
    tts: TtsCfg
    brain: BrainCfg
    greeting: str
    dormancy: DormancyCfg
    session: SessionCfg
    store: StoreCfg
    tools: ToolsCfg
    meeting: MeetingCfg

    @property
    def root(self) -> Path:
        return ROOT


ROOT = Path(__file__).resolve().parent.parent

_REL_PATH_FIELDS = {
    ("turn", "model"), ("wake", "dir"), ("stt", "model_dir"), ("tts", "voice"),
    ("session", "log_dir"), ("store", "path"),
    ("meeting", "diarize", "segmentation"), ("meeting", "diarize", "embedding"),
}


def _build(t: Any, v: Any, where: str) -> Any:
    origin = get_origin(t)
    if origin in (Union, UnionType):
        args = [a for a in get_args(t) if a is not type(None)]
        if v is None:
            return None
        if len(args) == 1:
            return _build(args[0], v, where)
        raise ConfigError(f"{where}: unsupported optional type {t}")
    if is_dataclass(t):
        if not isinstance(v, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(v).__name__}")
        hints = get_type_hints(t)
        out: dict[str, Any] = {}
        for f in fields(t):
            if f.name in v:
                out[f.name] = _build(hints[f.name], v[f.name], f"{where}.{f.name}")
            elif f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError(f"{where}.{f.name} is missing")
        unknown = sorted(set(v) - {f.name for f in fields(t)})
        if unknown:
            raise ConfigError(f"{where}: unknown key(s) {unknown}")
        return t(**out)
    if t is bool:
        if not isinstance(v, bool):
            raise ConfigError(f"{where}: expected bool, got {v!r}")
        return v
    if t is int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(f"{where}: expected int, got {v!r}")
        return v
    if t is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{where}: expected float, got {v!r}")
        return float(v)
    if t is str:
        if not isinstance(v, str):
            raise ConfigError(f"{where}: expected str, got {v!r}")
        return v
    if get_origin(t) is list:
        (inner,) = get_args(t)
        if inner is not str or not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ConfigError(f"{where}: expected list of str, got {v!r}")
        return list(v)
    raise ConfigError(f"{where}: unsupported type {t}")


def _resolve(cfg: Config) -> Config:
    """Turn relative model/log paths into absolute paths anchored at the v2 root."""
    def fix(v: Any, where: tuple[str, ...]) -> Any:
        if is_dataclass(v) and not isinstance(v, type):
            kwargs = {f.name: fix(getattr(v, f.name), where + (f.name,)) for f in fields(v)}
            return type(v)(**kwargs)
        if where in _REL_PATH_FIELDS and isinstance(v, str) and not os.path.isabs(v):
            return str(ROOT / v)
        return v
    return fix(cfg, ())


def load(path: str | Path) -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return _resolve(_build(Config, raw, "config"))
