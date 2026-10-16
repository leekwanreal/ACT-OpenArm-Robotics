#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test for ONNX Export & Runtime Inference Parity:
- Exports ACTPolicy to ONNX with Dynamic Batching
- Verifies companion metadata JSON is written
- Runs inference via ONNX Runtime
- Asserts numerical parity between PyTorch output and ONNX Runtime: |y_pt - y_ort| < 1e-4
"""

import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np
import onnxruntime as ort

from act_pipeline.config import ModelConfig
from act_pipeline.models.act_model import ACTPolicy
from act_pipeline.utils.onnx_export import export_act_to_onnx


def test_onnx_export_and_runtime_parity(dummy_dual_cam_rgbd, dummy_qpos, mock_norm_stats):
    """
    Test end-to-end ONNX export and verify that ONNX Runtime produces
    numerically equivalent predictions to PyTorch model (tolerance < 1e-4 rad).
    """
    cfg = ModelConfig(
        backbone_type="yolov8_nano",
        decoder_layers=7,
        num_cameras=2,
        d_model=512,
        chunk_size=50,
        action_dim=16,
    )
    model = ACTPolicy(cfg).eval()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = str(Path(tmp_dir) / "test_act_yolov8.onnx")
        meta_file = str(Path(tmp_dir) / "test_act_yolov8_metadata.json")
        
        # 1. Export to ONNX
        exported_path = export_act_to_onnx(
            model=model,
            output_path=onnx_file,
            config=cfg,
            stats=mock_norm_stats,
            opset_version=18,
            dynamic_batch=True,
            verify_with_ort=True,
        )
        assert Path(exported_path).exists(), "Exported .onnx file does not exist!"
        assert Path(meta_file).exists(), "Companion metadata .json does not exist!"
        
        # 2. PyTorch deterministic inference
        with torch.no_grad():
            py_pred = model.predict_action_chunk(dummy_dual_cam_rgbd, dummy_qpos).cpu().numpy()
            
        assert py_pred.shape == (2, 50, 16)
        
        # 3. ONNX Runtime inference
        session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
        input_names = [i.name for i in session.get_inputs()]
        assert "image" in input_names and "qpos" in input_names
        
        ort_inputs = {
            "image": dummy_dual_cam_rgbd.cpu().numpy(),
            "qpos": dummy_qpos.cpu().numpy(),
        }
        ort_pred = session.run(None, ort_inputs)[0]
        
        assert ort_pred.shape == (2, 50, 16)
        
        # 4. Parity Check: maximum absolute difference < 1e-4 radians
        max_diff = np.max(np.abs(py_pred - ort_pred))
        assert max_diff < 1e-4, f"Parity difference too large: {max_diff:.6f} rad (expected < 1e-4)"
