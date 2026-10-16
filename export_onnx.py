#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone CLI tool to export trained ACT Checkpoints to ONNX format.
Repo: OpenArm_ACT_Training

Usage:
    python export_onnx.py --checkpoint checkpoints/act_openarm/best_checkpoint.pth
    python export_onnx.py --checkpoint checkpoints/act_openarm/best_checkpoint.pth --output checkpoints/act_openarm/model.onnx
"""

import os
import sys
import argparse
import torch

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from act_pipeline.config import ModelConfig
from act_pipeline.models.act_model import ACTPolicy
from act_pipeline.data.normalization import load_norm_stats
from act_pipeline.utils.onnx_export import export_act_to_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Xuất mô hình ACT PyTorch (.pth) sang định dạng ONNX (.onnx)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/act_openarm/best_checkpoint.pth",
        help="Đường dẫn tới file checkpoint PyTorch (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file .onnx đích (mặc định thay .pth bằng .onnx)",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Đường dẫn file dataset_stats.pkl (mặc định tìm cùng thư mục với checkpoint)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="Phiên bản ONNX opset (mặc định 18)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Thiết bị thực hiện xuất mô hình (mặc định cpu để tương thích cao nhất)",
    )
    parser.add_argument(
        "--fixed_batch",
        action="store_true",
        help="Cố định kích thước batch = 1 thay vì dynamic batch (phù hợp một số chip AI biên)",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Bỏ qua bước kiểm tra xác thực số học với ONNX Runtime",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[X] LỖI: Không tìm thấy file checkpoint: {args.checkpoint}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        if args.checkpoint.endswith(".pth"):
            output_path = args.checkpoint[:-4] + ".onnx"
        else:
            output_path = args.checkpoint + ".onnx"

    print("=" * 80)
    print("   📦 TIỆN ÍCH XUẤT MÔ HÌNH ACT SANG ĐỊNH DẠNG ONNX (OPENARM BIMANUAL)")
    print("=" * 80)
    print(f"[*] Checkpoint đầu vào : {args.checkpoint}")
    print(f"[*] File ONNX đầu ra   : {output_path}")
    print(f"[*] Thiết bị xuất      : {args.device}")
    print(f"[*] Phiên bản Opset    : {args.opset}")
    print("=" * 80)

    # 1. Tải checkpoint
    checkpoint_data = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    # 2. Tải stats
    stats = checkpoint_data.get("stats")
    if stats is None:
        stats_path = args.stats
        if stats_path is None:
            stats_path = os.path.join(os.path.dirname(args.checkpoint), "dataset_stats.pkl")
        if os.path.exists(stats_path):
            stats = load_norm_stats(stats_path)
            print(f"[✓] Đã nạp thống kê chuẩn hóa từ: {stats_path}")
        else:
            print("[!] Cảnh báo: Không tìm thấy dataset_stats.pkl. File ONNX sẽ không chứa thông số chuẩn hóa.")

    # 3. Khởi tạo mô hình
    saved_cfg = checkpoint_data.get("config", {})
    bb_type = saved_cfg.get("backbone_type", "yolov8_nano")

    model_cfg = ModelConfig(
        in_channels=saved_cfg.get("in_channels", 4),
        num_cameras=saved_cfg.get("num_cameras", 2),
        camera_names=tuple(saved_cfg.get("camera_names", ("chest", "front"))),
        backbone_type=bb_type,
        freeze_backbone=True,
        yolo_stride=saved_cfg.get("yolo_stride", 16),
        img_height=saved_cfg.get("img_height", 480),
        img_width=saved_cfg.get("img_width", 640),
        d_model=saved_cfg.get("d_model", 512),
        nheads=saved_cfg.get("nheads", 8),
        dim_feedforward=saved_cfg.get("dim_feedforward", 2048),
        dropout=0.0,
        cvae_layers=saved_cfg.get("cvae_layers", 2),
        latent_dim=saved_cfg.get("latent_dim", 32),
        decoder_layers=saved_cfg.get("decoder_layers", 7),
        chunk_size=saved_cfg.get("chunk_size", 50),
        action_dim=saved_cfg.get("action_dim", 16),
        qpos_dim=saved_cfg.get("qpos_dim", 16),
        use_3d_features=saved_cfg.get("use_3d_features", False),
        point_grid_h=saved_cfg.get("point_grid_h", 15),
        point_grid_w=saved_cfg.get("point_grid_w", 20),
    )
    model = ACTPolicy(config=model_cfg).to(args.device)

    # Nạp state_dict
    state_dict = checkpoint_data["model_state_dict"] if "model_state_dict" in checkpoint_data else checkpoint_data
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("[✓] Đã nạp 100% trọng số từ checkpoint.")

    # 4. Xuất sang ONNX
    exported_file = export_act_to_onnx(
        model=model,
        output_path=output_path,
        config=model_cfg,
        stats=stats,
        opset_version=args.opset,
        dynamic_batch=not args.fixed_batch,
        verify_with_ort=not args.no_verify,
    )

    print("=" * 80)
    print(f"🎉 XUẤT MÔ HÌNH THÀNH CÔNG: {exported_file}")
    print(f"   Metadata đi kèm: {os.path.splitext(exported_file)[0] + '_metadata.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

