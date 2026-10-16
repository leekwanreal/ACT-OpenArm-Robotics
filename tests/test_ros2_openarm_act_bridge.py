#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ROS 2 openarm_act deployment bridge:
- Temporal Ensembling ActionChunker (exp(-k * age) weighted blending)
- TrajectorySmoother Kinematic Clamping (Velocity and Acceleration limits)
- Interop bridge between act_pipeline checkpoints and ROS 2 openarm_act node
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add openarm_act module to path
ACT_TRAINING_DIR = Path(__file__).resolve().parent.parent
OPENARM_ACT_SRC = ACT_TRAINING_DIR / "openarm_mc" / "src" / "openarm_act"
if str(OPENARM_ACT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENARM_ACT_SRC))

from openarm_act.chunking import ActionChunker
from openarm_act.smoothing import TrajectorySmoother, ARM_V_LIMIT, ARM_A_LIMIT


def test_action_chunker_temporal_ensembling():
    """Verify ActionChunker merges overlapping chunks with exponential weights."""
    chunker = ActionChunker(chunk_size=50, temporal_ensemble=True, ensemble_k=0.01)
    chunker.reset()
    
    # Chunk 1 at step 0: all values 1.0
    chunk1 = np.ones((50, 16), dtype=np.float32) * 1.0
    chunker.add(start_step=0, actions=chunk1)
    
    # Query step 0
    a0 = chunker.action(0)
    assert a0 is not None
    assert np.allclose(a0, 1.0, atol=1e-5)
    
    # Chunk 2 at step 5: all values 2.0
    chunk2 = np.ones((50, 16), dtype=np.float32) * 2.0
    chunker.add(start_step=5, actions=chunk2)
    
    # At step 5, both chunk 1 (age 5) and chunk 2 (age 0) cover it
    # Weight of chunk 2 is exp(-0.01*0)=1.0, weight of chunk 1 is exp(-0.01*5)=0.9512
    a5 = chunker.action(5)
    assert a5 is not None
    # Weighted average should be between 1.0 and 2.0, closer to 2.0
    assert 1.0 < a5[0] < 2.0
    assert a5[0] > 1.5, f"Newest prediction should have larger weight: {a5[0]}"


def test_trajectory_smoother_velocity_and_acceleration_limits():
    """
    Verify TrajectorySmoother enforces maximum velocity and acceleration limits
    on each motor, preventing sudden jumps that could damage gears.
    """
    dt = 0.02  # 50 Hz control loop
    smoother = TrajectorySmoother(vel_scale=1.0, dt=dt)
    
    initial_q = np.zeros(16, dtype=np.float32)
    smoother.reset(initial_q)
    
    # Sudden huge step jump: from 0.0 to 5.0 radians on all joints
    target_jump = np.ones(16, dtype=np.float32) * 5.0
    
    step1_q = smoother.step(target_jump)
    
    # Velocity on step 1: (step1_q - initial_q) / dt
    vel1 = (step1_q - initial_q) / dt
    
    # Acceleration on step 1: (vel1 - 0.0) / dt
    acc1 = vel1 / dt
    
    v_limits = np.concatenate([ARM_V_LIMIT, ARM_V_LIMIT])
    a_limits = np.concatenate([ARM_A_LIMIT, ARM_A_LIMIT])
    
    # Check that velocity does not exceed limit
    assert np.all(vel1 <= v_limits + 1e-4), f"Velocity exceeded limits: {vel1} > {v_limits}"
    # Check that acceleration does not exceed limit
    assert np.all(acc1 <= a_limits + 1e-4), f"Acceleration exceeded limits: {acc1} > {a_limits}"


def test_trajectory_smoother_idle():
    """Verify that when target equals current position, smoother stays stationary."""
    dt = 0.02
    smoother = TrajectorySmoother(vel_scale=1.0, dt=dt)
    initial_q = np.ones(16, dtype=np.float32) * 0.25
    smoother.reset(initial_q)
    
    out = smoother.step(initial_q)
    assert np.allclose(out, initial_q, atol=1e-6)
    assert np.allclose(smoother.v_cmd, 0.0, atol=1e-6)


def test_trajectory_smoother_progress():
    """Verify that stepping towards a target advances the commanded position smoothly."""
    dt = 0.02
    smoother = TrajectorySmoother(vel_scale=1.0, dt=dt)
    initial_q = np.zeros(16, dtype=np.float32)
    smoother.reset(initial_q)
    
    target = np.ones(16, dtype=np.float32) * 1.0
    step1 = smoother.step(target)
    step2 = smoother.step(target)
    
    # Commanded positions must increase strictly monotonically towards target
    assert np.all(step1 > initial_q), "Step 1 should move towards target"
    assert np.all(step2 > step1), "Step 2 should continue moving towards target"
    assert np.all(step2 <= target), "Step 2 should not overshoot immediately"
