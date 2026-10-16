#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Vision Backbones in ACT:
- YOLOv8-Nano / YOLOv8-Small
- ResNet-18 / ResNet-34 / ResNet-50
- ConvNeXt-Nano
Verifies 4-channel RGB-D input adapter, weight initialization, selective freezing, and visual tokens dimensions.
"""

import pytest
import torch
import torch.nn as nn
from act_pipeline.models.backbone import (
    build_vision_backbone,
    SinusoidalPositionEmbedding2D,
    RGBDResNetBackbone,
    RGBDYOLOBackbone,
    RGBDConvNeXtBackbone,
)


def test_sinusoidal_position_embedding_2d():
    """Verify 2D Sinusoidal Positional Encoding generates expected shape."""
    d_model = 512
    pos_embed = SinusoidalPositionEmbedding2D(d_model=d_model)
    x = torch.zeros(2, 512, 30, 40)
    pos = pos_embed(x)
    assert pos.shape == (2, 512, 30, 40), f"Position embedding shape mismatch: {pos.shape}"


@pytest.mark.parametrize("b_type", ["yolov8_nano", "yolov8_small", "resnet18", "resnet34", "resnet50", "convnext_nano"])
def test_all_backbones_output_shape(b_type, dummy_single_cam_rgbd):
    """
    Verify all 6 backbones extract visual tokens with exact shape [B, N_tokens, d_model].
    For 480x640:
    - YOLOv8 & ResNet at Stride 16: H_feat=30, W_feat=40 => 1200 visual tokens per camera.
    - ConvNeXt-Nano at Stride 32: H_feat=15, W_feat=20 => 300 visual tokens per camera.
    """
    if b_type == "convnext_nano":
        try:
            import timm
        except ImportError:
            pytest.skip("Thư viện 'timm' chưa được cài đặt trong môi trường này")

    backbone = build_vision_backbone(
        backbone_type=b_type,
        d_model=512,
        freeze_backbone=True,
        in_channels=4,
        stride=16,
    )
    backbone.eval()
    with torch.no_grad():
        tokens = backbone(dummy_single_cam_rgbd)
    
    expected_tokens = 300 if b_type == "convnext_nano" else 1200
    assert tokens.shape == (2, expected_tokens, 512), f"Failed for backbone '{b_type}': shape was {tokens.shape}"


def test_depth_weight_initialization_resnet():
    """
    Verify 4-channel conv1_adapter initializes the 4th channel (Depth)
    as the exact arithmetic mean of the 3 RGB pre-trained channels.
    """
    backbone = build_vision_backbone(backbone_type="resnet18", d_model=512)
    assert hasattr(backbone, "conv1_adapter"), "ResNet must have conv1_adapter"
    w = backbone.conv1_adapter.weight.data  # shape: [out_channels, 4, kernel_h, kernel_w]
    w_rgb_mean = w[:, :3, :, :].mean(dim=1)
    w_depth = w[:, 3, :, :]
    assert torch.allclose(w_depth, w_rgb_mean, atol=1e-5), "Depth weights are not equal to RGB mean!"


def test_selective_freezing_resnet():
    """
    Verify selective freezing on ResNet:
    - Pretrained residual layers (layer1..layer4, bn1) have requires_grad == False
    - 4-channel stem adapter (conv1_adapter) and projection (img_proj) have requires_grad == True
    - Calling backbone.train() preserves eval() mode for frozen BatchNorm2d layers
    """
    backbone = build_vision_backbone(backbone_type="resnet18", d_model=512, freeze_backbone=True)
    
    # Check trainable parameters
    assert backbone.conv1_adapter.weight.requires_grad is True, "Stem adapter must be trainable"
    assert backbone.img_proj.weight.requires_grad is True, "1x1 projection must be trainable"
    
    # Check frozen parameters
    for p in backbone.layer1.parameters():
        assert p.requires_grad is False, "layer1 parameters must be frozen"
    for p in backbone.layer4.parameters():
        assert p.requires_grad is False, "layer4 parameters must be frozen"
        
    # Check eval mode persistence
    backbone.train(True)
    assert backbone.bn1.training is False, "bn1 must remain in eval mode"
    assert backbone.layer1.training is False, "layer1 must remain in eval mode"


def test_selective_freezing_yolov8():
    """
    Verify selective freezing on YOLOv8:
    - Feature backbone stages have requires_grad == False
    - Stem adapter and 1x1 projection have requires_grad == True
    """
    backbone = build_vision_backbone(backbone_type="yolov8_nano", d_model=512, freeze_backbone=True)
    
    # In YOLOv8, stem conv is inside backbone.stem
    assert backbone.stem.conv.weight.requires_grad is True, "YOLOv8 stem conv must be trainable"
    assert backbone.img_proj.weight.requires_grad is True, "YOLOv8 1x1 projection must be trainable"
    for p in backbone.stage1.parameters():
        assert p.requires_grad is False, "YOLOv8 backbone stage1 must be frozen"
    for p in backbone.stage2.parameters():
        assert p.requires_grad is False, "YOLOv8 backbone stage2 must be frozen"
