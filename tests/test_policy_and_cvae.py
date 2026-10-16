#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for CVAE Encoder, 7-Layer Policy Decoder, and ACTPolicy Core:
- CVAE latent vector sampling (train vs deterministic eval)
- 7-Layer Transformer Policy Decoder (Stanford ALOHA standard)
- Dual camera RGB-D memory token fusion (2400 visual + 1 qpos + 1 z = 2402 tokens)
- Loss computation (L1 Reconstruction + KL Divergence with beta=10.0)
- Backward pass and gradient health (No NaNs or Infs)
"""

import pytest
import torch
from act_pipeline.config import ModelConfig
from act_pipeline.models.act_model import ACTPolicy
from act_pipeline.models.cvae import CVAEEncoder
from act_pipeline.models.policy import TransformerPolicyDecoder


def test_cvae_latent_sampling():
    """Verify CVAE encoder generates mu, logvar and z in R^32."""
    cvae = CVAEEncoder(d_model=512, latent_dim=32, qpos_dim=16, action_dim=16, num_layers=2)
    qpos = torch.randn(2, 16)
    actions = torch.randn(2, 50, 16)
    
    # Train mode: stochastic sampling
    cvae.train()
    z_token_train, mu, logvar = cvae(qpos=qpos, actions=actions)
    assert z_token_train.shape == (2, 1, 512)
    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)
    
    # Eval mode: deterministic z=0
    cvae.eval()
    with torch.no_grad():
        z_token_eval, mu_eval, logvar_eval = cvae(qpos=qpos, actions=None)
        assert z_token_eval.shape == (2, 1, 512)
        assert mu_eval is None
        assert logvar_eval is None


def test_7_layer_policy_decoder():
    """Verify 7-Layer Transformer Policy Decoder processes 2402 Memory tokens to produce [B, 50, 16]."""
    decoder = TransformerPolicyDecoder(
        d_model=512,
        nheads=8,
        dim_feedforward=2048,
        num_layers=7,
        chunk_size=50,
        action_dim=16,
    )
    # Memory: 2 cameras x 1200 tokens + 1 qpos token + 1 z token = 2402 tokens
    memory = torch.randn(2, 2402, 512)
    out = decoder(memory)
    assert out.shape == (2, 50, 16), f"Decoder output shape mismatch: {out.shape}"


def test_act_policy_forward_backward(dummy_dual_cam_rgbd, dummy_qpos, dummy_actions, dummy_is_pad):
    """
    Test full ACTPolicy forward and backward pass:
    - Input: dual camera RGB-D [2, 2, 4, 480, 640] + qpos [2, 16] + actions [2, 50, 16]
    - Loss: L1 + 10.0 * KL
    - Backward: Verify gradients flow to trainable layers and have no NaNs.
    """
    cfg = ModelConfig(
        backbone_type="yolov8_nano",
        decoder_layers=7,
        d_model=512,
        num_cameras=2,
        chunk_size=50,
        action_dim=16,
        qpos_dim=16,
    )
    model = ACTPolicy(cfg)
    model.train()
    
    # Forward pass
    pred_actions, mu, logvar = model(
        image=dummy_dual_cam_rgbd,
        qpos=dummy_qpos,
        actions=dummy_actions,
        is_pad=dummy_is_pad,
    )
    assert pred_actions.shape == (2, 50, 16), f"pred_actions shape mismatch: {pred_actions.shape}"
    assert mu is not None and logvar is not None
    
    # Loss computation returns dict {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
    loss_dict = model.compute_loss(
        pred_actions=pred_actions,
        actions=dummy_actions,
        is_pad=dummy_is_pad,
        mu=mu,
        logvar=logvar,
        kl_weight=10.0,
    )
    loss = loss_dict["loss"]
    assert not torch.isnan(loss), "Loss contains NaN"
    assert "recon_loss" in loss_dict and "kl_loss" in loss_dict
    assert loss_dict["recon_loss"] >= 0.0
    assert loss_dict["kl_loss"] >= 0.0
    
    # Backward pass
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient after backward pass"
            assert not torch.isnan(p.grad).any(), f"Gradient of {name} contains NaN"
            assert not torch.isinf(p.grad).any(), f"Gradient of {name} contains Inf"


def test_deterministic_eval_mode(dummy_dual_cam_rgbd, dummy_qpos):
    """Verify that in eval mode (z=0), policy produces deterministic identical trajectory."""
    cfg = ModelConfig(backbone_type="yolov8_nano", decoder_layers=7, num_cameras=2)
    model = ACTPolicy(cfg)
    model.eval()
    
    with torch.no_grad():
        out1, _, _ = model(dummy_dual_cam_rgbd, dummy_qpos)
        out2, _, _ = model(dummy_dual_cam_rgbd, dummy_qpos)
        
    assert torch.allclose(out1, out2, atol=1e-6), "Eval mode output is not deterministic!"
    assert out1.shape == (2, 50, 16)
