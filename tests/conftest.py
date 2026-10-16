#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest Fixtures for OpenArm ACT Pipeline Unit & Integration Tests.
Provides dummy RGB-D tensors, joint state vectors, actions chunks, and mock metadata.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

# Ensure act_training root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def dummy_single_cam_rgbd():
    """Single camera RGB-D tensor: [Batch=2, Channels=4 (RGB+Depth), H=480, W=640]"""
    torch.manual_seed(42)
    return torch.randn(2, 4, 480, 640, dtype=torch.float32)


@pytest.fixture
def dummy_dual_cam_rgbd():
    """Dual camera RGB-D tensor: [Batch=2, Num_Cams=2 (Chest+Front), Channels=4, H=480, W=640]"""
    torch.manual_seed(42)
    return torch.randn(2, 2, 4, 480, 640, dtype=torch.float32)


@pytest.fixture
def dummy_qpos():
    """Current 16-DOF joint angles in radians: [Batch=2, 16]"""
    torch.manual_seed(42)
    return torch.randn(2, 16, dtype=torch.float32)


@pytest.fixture
def dummy_actions():
    """Future 50-step action trajectory: [Batch=2, Chunk=50, 16]"""
    torch.manual_seed(42)
    return torch.randn(2, 50, 16, dtype=torch.float32)


@pytest.fixture
def dummy_is_pad():
    """Padding mask indicating artificial padded timesteps at episode boundary: [Batch=2, Chunk=50]"""
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[:, -5:] = True  # Last 5 steps padded
    return mask


@pytest.fixture
def mock_norm_stats():
    """Mock mean and std stats for 16 motors"""
    np.random.seed(42)
    qpos_mean = np.zeros(16, dtype=np.float32)
    qpos_std = np.ones(16, dtype=np.float32) * 0.5
    action_mean = np.zeros(16, dtype=np.float32)
    action_std = np.ones(16, dtype=np.float32) * 0.5
    return {
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
