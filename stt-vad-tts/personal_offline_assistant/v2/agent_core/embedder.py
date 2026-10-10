"""Sentence embeddings: MiniLM-L6-v2 int8 ONNX (torch-free), for the memory layer."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_dir: str) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        path = Path(model_dir)
        if not (path / "model.onnx").is_file():
            raise FileNotFoundError(f"{path}/model.onnx missing — run tools/download_models.py")
        self.tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def embed(self, text: str) -> np.ndarray:
        """L2-normalised 384-d sentence vector."""
        enc = self.tok(text or " ", return_tensors="np", padding="max_length", truncation=True, max_length=64)
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in enc and any(i.name == "token_type_ids" for i in self.session.get_inputs()):
            feeds["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
        hidden = self.session.run(None, feeds)[0]  # [1, seq, 384]
        mask = feeds["attention_mask"][:, :, None].astype(np.float32)
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        vec = (summed / counts).reshape(-1)  # flat 384-d vector
        return vec / max(float(np.linalg.norm(vec)), 1e-9)
