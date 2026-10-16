#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Công cụ trực quan hóa đặc trưng ResNet-18 (Feature Activation Map) & Kiểm tra độ nhạy thị giác
Mục đích:
1. Kiểm tra xem ResNet-18 có tập trung phát hiện đúng vật thể (vật gắp, khăn, cốc...) và tay kẹp robot hay không.
2. Xuất ảnh so sánh: RGB gốc | Bản đồ độ sâu Depth | Heatmap đặc trưng Layer 3 | Heatmap đặc trưng Layer 4 đè lên ảnh thật.
3. Hỗ trợ chạy trên cả file .hdf5 (tập dataset) hoặc một file ảnh bất kỳ (.png, .jpg).

Cách sử dụng:
    # 1. Chạy trên 1 episode .hdf5 bất kỳ trong dataset:
    python scripts/visualize_features.py --episode path/to/episode_xxx.hdf5 --step 50

    # 2. Tự động lấy file đầu tiên trong thư mục dataset:
    python scripts/visualize_features.py --dataset_dir data_set/data_set

    # 3. Kèm checkpoint đã train để kiểm tra trọng số thực tế:
    python scripts/visualize_features.py --checkpoint checkpoints/act_openarm/best_checkpoint.pth
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Đảm bảo root repo nằm trong sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from act_pipeline.models.backbone import RGBDConvNeXtBackbone, build_vision_backbone
from act_pipeline.data.preprocess import preprocess_rgbd


def parse_args():
    parser = argparse.ArgumentParser(description="Kiểm tra khả năng phát hiện vật thể của Vision Backbone trong ACT")
    parser.add_argument("--episode", type=str, default=None, help="Đường dẫn file .hdf5 cần kiểm tra")
    parser.add_argument("--dataset_dir", type=str, default="data_set/data_set", help="Thư mục dataset chứa các file .hdf5")
    parser.add_argument("--step", type=int, default=50, help="Vị trí bước timestep trong episode (mặc định bước 50)")
    parser.add_argument("--camera", type=str, default="chest", choices=["chest", "front"], help="Camera cần trực quan hóa: chest hoặc front (mặc định chest)")
    parser.add_argument("--backbone_type", type=str, default="yolov8_nano", choices=["yolov8_nano", "yolov8_small", "resnet18", "resnet34", "resnet50", "convnext_nano"], help="Loại mạng thị giác (yolov8_nano, yolov8_small, resnet18, resnet34, resnet50, convnext_nano)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint .pth đã train (nếu không có sẽ dùng pretrained)")
    parser.add_argument("--output", type=str, default="vision_detection_check.png", help="Đường dẫn file ảnh xuất kết quả")
    parser.add_argument("--img_height", type=int, default=480, help="Chiều cao ảnh chuẩn hóa (mặc định 480)")
    parser.add_argument("--img_width", type=int, default=640, help="Chiều rộng ảnh chuẩn hóa (mặc định 640)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu hoặc cuda")
    return parser.parse_args()


def load_sample_from_hdf5(hdf5_path: str, step: int, camera: str = "chest"):
    """Trích xuất 1 frame RGB và Depth từ file HDF5 episode theo camera chỉ định."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        # Tìm key hình ảnh theo quy ước DATA_FORMAT_SPECIFICATION.md
        cam_key = None
        obs_images = f["/observations/images"]
        candidates = [f"{camera}_rgb", camera]
        for c in candidates:
            if c in obs_images:
                cam_key = f"/observations/images/{c}"
                break
        if cam_key is None:
            # Fallback nếu không có key trùng camera
            for k in ["chest_rgb", "chest", "front_rgb", "front"]:
                if k in obs_images:
                    cam_key = f"/observations/images/{k}"
                    break
        if cam_key is None:
            keys = list(obs_images.keys())
            if not keys:
                raise ValueError(f"Không tìm thấy ảnh camera trong {hdf5_path}")
            cam_key = f"/observations/images/{keys[0]}"

        rgb_dset = f[cam_key]
        num_steps = rgb_dset.shape[0]
        step = max(0, min(step, num_steps - 1))
        rgb = np.array(rgb_dset[step])

        # Tìm Depth tương ứng nếu có
        depth = None
        depth_candidates = [f"{camera}_depth", f"{camera}_d"]
        for dc in depth_candidates:
            if dc in obs_images:
                depth = np.array(obs_images[dc][step])
                break
        if depth is None and "depths" in f["/observations"]:
            depth_key = cam_key.replace("images", "depths")
            if depth_key in f:
                depth = np.array(f[depth_key][step])

        if depth is None:
            # Fallback nếu không tìm thấy depth
            depth = np.full((rgb.shape[0], rgb.shape[1]), 600.0, dtype=np.float32)

        return rgb, depth, step, num_steps


def get_feature_activation_map(feature_tensor: torch.Tensor) -> np.ndarray:
    """
    Tính toán Heatmap kích hoạt từ feature tensor [B, C, H_feat, W_feat].
    Sử dụng trung bình năng lượng kích hoạt qua các channels (Channel Mean Activation).
    """
    # Lấy sample đầu tiên [C, H_feat, W_feat]
    feat = feature_tensor[0]
    # Tính L2-norm hoặc trung bình độ lớn kích hoạt của 512 channels
    activation = torch.norm(feat, dim=0).detach().cpu().numpy()
    
    # Chuẩn hóa về dải [0, 1]
    act_min, act_max = activation.min(), activation.max()
    if act_max - act_min > 1e-6:
        activation = (activation - act_min) / (act_max - act_min)
    else:
        activation = np.zeros_like(activation)
        
    return activation


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Nạp dữ liệu ảnh đầu vào
    if args.image and os.path.exists(args.image):
        img_bgr = cv2.imread(args.image)
        rgb_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        depth_raw = np.full((rgb_raw.shape[0], rgb_raw.shape[1]), 600.0, dtype=np.float32)
        info_str = f"Ảnh tĩnh: {os.path.basename(args.image)}"
    else:
        ep_path = args.episode
        if ep_path is None or not os.path.exists(ep_path):
            files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.hdf5")))
            if not files:
                # Tìm đệ quy trong thư mục
                files = sorted(glob.glob(f"{args.dataset_dir}/**/*.hdf5", recursive=True))
            if not files:
                print(f"[X] Lỗi: Không tìm thấy file .hdf5 nào trong {args.dataset_dir}")
                sys.exit(1)
            ep_path = files[0]

        rgb_raw, depth_raw, step_idx, total_steps = load_sample_from_hdf5(ep_path, args.step, camera=args.camera)
        info_str = f"File: {os.path.basename(ep_path)} | Camera: {args.camera} | Timestep: {step_idx}/{total_steps}"

    print("=" * 80)
    print(f"🔍 KIỂM TRA ĐẶC TRƯNG THỊ GIÁC RESNET-18 (VISUAL DETECTION CHECK)")
    print("=" * 80)
    print(f"[*] Nguồn ảnh  : {info_str}")
    print(f"[*] Kích thước : {rgb_raw.shape[1]}x{rgb_raw.shape[0]} px")
    print(f"[*] Thiết bị   : {device}")

    # 2. Tiền xử lý RGB-D (chuẩn hóa kênh & kích thước 480x640)
    rgbd_tensor = preprocess_rgbd(
        rgb_raw=rgb_raw,
        depth_raw=depth_raw,
        target_size=(args.img_height, args.img_width),
        depth_min_m=0.2,
        depth_max_m=1.2,
    ).unsqueeze(0).to(device)  # [1, 4, H, W]

    # 3. Khởi tạo Backbone
    backbone_type = args.backbone_type
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_cfg = ckpt.get("config", {})
        if "backbone_type" in saved_cfg:
            backbone_type = saved_cfg["backbone_type"]

    backbone = build_vision_backbone(
        backbone_type=backbone_type,
        d_model=512,
        freeze_backbone=False,
        in_channels=4,
    ).to(device)

    # Nạp checkpoint nếu có
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_dict[k.replace("backbone.", "")] = v
            elif k.startswith("module.backbone."):
                backbone_dict[k.replace("module.backbone.", "")] = v
        if backbone_dict:
            backbone.load_state_dict(backbone_dict, strict=False)
            print(f"[✓] Đã nạp trọng số backbone từ checkpoint: {args.checkpoint}")
        else:
            print("[!] Checkpoint không có tiền tố backbone, sử dụng trọng số mặc định.")

    backbone.eval()

    # 4. Trích xuất feature map từng tầng
    with torch.no_grad():
        if backbone_type.startswith("yolov8"):
            stem_out = backbone.stem(rgbd_tensor)
            s1 = backbone.stage1(stem_out)
            s2 = backbone.stage2(s1)
            feat_l3 = backbone.stage3(s2)  # Stage 3: P4 (16x)
            feat_l4 = backbone.sppf(feat_l3)  # SPPF: Đa tỉ lệ
            l3_name = "2. Stage 3 (Biên cạnh & Tọa độ vật thể P4)"
            l4_name = "3. SPPF (Tâm vật thể & Đa tỉ lệ gắp nhặt)"
        elif backbone_type.startswith("resnet") or backbone_type == "resnet":
            stem_out = backbone.conv1(rgbd_tensor)
            x = backbone.relu(backbone.bn1(stem_out))
            x = backbone.maxpool(x)
            s1 = backbone.layer1(x)
            s2 = backbone.layer2(s1)
            feat_l3 = backbone.layer3(s2)      # Layer 3: 16x
            feat_l4 = backbone.layer4(feat_l3)  # Layer 4: 16x hoặc 32x
            l3_name = "2. Layer 3 (Đặc trưng không gian trung gian 16x)"
            l4_name = "3. Layer 4 (Đặc trưng ngữ nghĩa sâu)"
        else:
            stem_out = backbone.backbone.stem(rgbd_tensor)
            s0 = backbone.backbone.stages[0](stem_out)
            s1 = backbone.backbone.stages[1](s0)
            feat_l3 = backbone.backbone.stages[2](s1)      # Stage 2: 16x (viền cạnh, cấu trúc không gian 3D)
            feat_l4 = backbone.backbone.stages[3](feat_l3)  # Stage 3: 32x (ngữ nghĩa vật thể & gripper)
            l3_name = "2. Stage 2 (Viền cạnh & Hình học không gian 16x)"
            l4_name = "3. Stage 3 (Vị trí vật thể & Tay kẹp 32x)"

    # 5. Chuyển đổi thành Heatmap
    heat_l3 = get_feature_activation_map(feat_l3)
    heat_l4 = get_feature_activation_map(feat_l4)

    # Resize Heatmap về kích thước ảnh (mặc định 480x640)
    h, w = args.img_height, args.img_width
    heat_l3_resized = cv2.resize(heat_l3, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_l4_resized = cv2.resize(heat_l4, (w, h), interpolation=cv2.INTER_CUBIC)

    # Chuẩn bị ảnh hiển thị gốc (resize về h x w để đồng bộ)
    rgb_resized = cv2.resize(rgb_raw, (w, h), interpolation=cv2.INTER_AREA)

    # Tạo Overlay Heatmap lên ảnh gốc bằng Colormap JET
    heat_l4_uint8 = np.uint8(255 * heat_l4_resized)
    color_heat_l4 = cv2.applyColorMap(heat_l4_uint8, cv2.COLORMAP_JET)
    color_heat_l4_rgb = cv2.cvtColor(color_heat_l4, cv2.COLOR_BGR2RGB)
    
    # Trộn màu: 60% ảnh gốc + 40% heatmap màu
    overlay_l4 = cv2.addWeighted(rgb_resized, 0.6, color_heat_l4_rgb, 0.4, 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Kiểm tra Đặc trưng Thị giác ({backbone_type.upper()})\n{info_str}", fontsize=13, fontweight="bold")

    # Ô 1: Ảnh gốc RGB
    axes[0, 0].imshow(rgb_resized)
    axes[0, 0].set_title(f"1. Ảnh Camera Gốc ({args.camera} RGB {w}x{h})", fontsize=11)
    axes[0, 0].axis("off")

    # Ô 2: Heatmap Stage 2 (Đặc trưng viền, cấu trúc hình học vật thể)
    im3 = axes[0, 1].imshow(heat_l3_resized, cmap="magma")
    axes[0, 1].set_title(l3_name, fontsize=11)
    axes[0, 1].axis("off")
    fig.colorbar(im3, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Ô 3: Heatmap Stage 3 (Đặc trưng ngữ nghĩa cao cấp - Vị trí vật & Tay kẹp)
    im4 = axes[1, 0].imshow(heat_l4_resized, cmap="jet")
    axes[1, 0].set_title(l4_name, fontsize=11)
    axes[1, 0].axis("off")
    fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Ô 4: Heatmap phủ lên ảnh thật (Overlay)
    axes[1, 1].imshow(overlay_l4)
    axes[1, 1].set_title("4. Overlay (CONVNEXT-NANO: Vùng Đỏ = Nơi tập trung)", fontsize=11)
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print(f"🎉 ĐÃ XUẤT ẢNH TRỰC QUAN HÓA THÀNH CÔNG:")
    print(f"   👉 File: {os.path.abspath(args.output)}")
    print("=" * 80)
    print("💡 HƯỚNG DẪN ĐÁNH GIÁ KẾT QUẢ TRÊN Ô SỐ 4 (OVERLAY):")
    print("   + ✅ TỐT: Vùng sáng ĐỎ / VÀNG phủ trúng vào: Vật thể cần gắp (khăn, khối hộp...) và Tay kẹp.")
    print("   + ⚠️ CẢNH BÁO: Vùng sáng tập trung vào bức tường, bóng đổ sàn nhà hoặc mép khung camera.")
    print("=" * 80)


if __name__ == "__main__":
    main()
