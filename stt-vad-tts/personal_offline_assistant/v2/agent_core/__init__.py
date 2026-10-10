"""agent_core — offline full-duplex speech-to-speech pipeline for v2.

Importing this package sets offline guards BEFORE any third-party model code
loads. Nothing here may touch the network at runtime.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OPENWAKEWORD_PROJECT", "home_assistant")  # placeholder; unused at runtime
