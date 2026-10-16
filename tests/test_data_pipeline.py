#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Data Pipeline:
- RGB-D Preprocessing & Letterbox Aspect Ratio Preservation
- Z-score Normalization Parity & Invertibility
- 50Hz Temporal Resampling and Signal Interpolation
- Action Chunking & Padding Mask Logic
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from act_pipeline.data.preprocess import preprocess_rgbd, IMAGENET_MEAN, IMAGENET_STD
from act_pipeline.data.normalization import (
    normalize_data,
    unnormalize_data,
    save_norm_stats,
    load_norm_stats,
)
from act_pipeline.data.resample import resample_signals


def test_preprocess_rgbd_formula():
    """Verify RGB-D preprocessing formula aligns with DATA_FORMAT_SPECIFICATION.md."""
    H, W = 480, 640
    rgb_raw = np.full((H, W, 3), 128, dtype=np.uint8)
    # Depth in mm: 200mm = 0.2m (min), 1200mm = 1.2m (max), 700mm = 0.7m (midpoint)
    depth_raw = np.full((H, W), 700, dtype=np.uint16)
    
    tensor_4ch = preprocess_rgbd(rgb_raw, depth_raw, depth_min_m=0.2, depth_max_m=1.2)
    
    assert tensor_4ch.shape == (4, H, W)
    assert tensor_4ch.dtype == torch.float32
    
    # Check RGB channels: (128 / 255.0 - mean) / std
    expected_r = (128.0 / 255.0 - IMAGENET_MEAN[0, 0, 0].item()) / IMAGENET_STD[0, 0, 0].item()
    assert abs(tensor_4ch[0, 0, 0].item() - expected_r) < 1e-4
    
    # Check Depth channel: (0.7m - 0.2m) / (1.2m - 0.2m) = 0.5
    expected_depth = (0.7 - 0.2) / (1.2 - 0.2)
    assert abs(tensor_4ch[3, 0, 0].item() - expected_depth) < 1e-4


def test_preprocess_depth_clipping():
    """Verify depth values outside [0.2m, 1.2m] are clipped to [0.0, 1.0]."""
    H, W = 480, 640
    rgb_raw = np.zeros((H, W, 3), dtype=np.uint8)
    depth_raw = np.zeros((H, W), dtype=np.uint16)
    depth_raw[0, 0] = 50     # 0.05m < 0.2m => should clip to 0.0
    depth_raw[0, 1] = 2000   # 2.0m > 1.2m  => should clip to 1.0
    
    tensor_4ch = preprocess_rgbd(rgb_raw, depth_raw, depth_min_m=0.2, depth_max_m=1.2)
    assert abs(tensor_4ch[3, 0, 0].item() - 0.0) < 1e-5
    assert abs(tensor_4ch[3, 0, 1].item() - 1.0) < 1e-5


def test_normalization_invertibility():
    """Verify Z-score normalization and unnormalization are strictly invertible."""
    np.random.seed(42)
    x = np.random.randn(100, 16).astype(np.float32)
    mean = np.random.randn(16).astype(np.float32)
    std = np.abs(np.random.randn(16)).astype(np.float32) + 0.1
    
    # NumPy test
    x_norm = normalize_data(x, mean, std)
    x_rec = unnormalize_data(x_norm, mean, std)
    assert np.allclose(x, x_rec, atol=1e-6), "NumPy normalization round-trip failed!"
    
    # PyTorch test
    x_t = torch.from_numpy(x)
    mean_t = torch.from_numpy(mean)
    std_t = torch.from_numpy(std)
    x_norm_t = normalize_data(x_t, mean_t, std_t)
    x_rec_t = unnormalize_data(x_norm_t, mean_t, std_t)
    assert torch.allclose(x_t, x_rec_t, atol=1e-6), "PyTorch normalization round-trip failed!"


def test_normalization_save_load_json(mock_norm_stats):
    """Verify normalization statistics can be serialized to JSON and loaded accurately."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = str(Path(tmp_dir) / "stats.json")
        save_norm_stats(mock_norm_stats, json_path)
        assert Path(json_path).exists()
        
        loaded = load_norm_stats(json_path)
        assert np.allclose(mock_norm_stats["qpos_mean"], loaded["qpos_mean"], atol=1e-6)
        assert np.allclose(mock_norm_stats["qpos_std"], loaded["qpos_std"], atol=1e-6)
        assert np.allclose(mock_norm_stats["action_mean"], loaded["action_mean"], atol=1e-6)
        assert np.allclose(mock_norm_stats["action_std"], loaded["action_std"], atol=1e-6)


def test_50hz_signal_resampling():
    """Verify linear interpolation resamples signals with jitter to a uniform 50Hz grid."""
    # Simulated jittered timestamps around 50Hz (dt ~ 0.02s)
    np.random.seed(42)
    dt_jitter = np.random.uniform(0.015, 0.025, size=99)
    t_raw = np.concatenate([[0.0], np.cumsum(dt_jitter)])
    
    # Simulated continuous trajectory (100 steps, 16 joints)
    signals = np.sin(t_raw[:, None] * np.arange(1, 17))
    
    # Target uniform grid @ 50.0Hz (dt = 0.02s)
    T_total = t_raw[-1]
    t_uniform = np.arange(0.0, T_total, 0.02)
    
    resampled = resample_signals(t_raw, signals, t_uniform)
    
    assert resampled.shape == (len(t_uniform), 16)
    # Resampled values should stay within bounding box of original signal
    assert resampled.min() >= signals.min() - 1e-4
    assert resampled.max() <= signals.max() + 1e-4
