#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 Dataset Inspection & ACT Model Compatibility Validator
Kiểm tra cấu trúc và tính tương thích của file HDF5 với kiến trúc mô hình ACT OpenArm Bimanual (16-DOF).

Cách sử dụng:
    # 1. Bóc thử file HDF5 đầu tiên trong một thư mục:
    python inspect_hdf5.py --folder data_set/pick_and_place

    # 2. Bóc thử một file HDF5 chỉ định:
    python inspect_hdf5.py --file data_set/pick_and_place/episode_0.hdf5

    # 3. Chạy thử nghiệm forward mô hình ACT (kiểm tra cả cờ --use_3d_features):
    python inspect_hdf5.py --folder data_set/pick_and_place --test_forward
"""

import os
import sys
import glob
import argparse
from typing import Dict, Any, List, Optional, Tuple

# Cấu hình UTF-8 chống lỗi encoding trên Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import h5py
import torch

# Đảm bảo import được act_pipeline
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PARENT_ROOT = os.path.dirname(REPO_ROOT)
if PARENT_ROOT not in sys.path:
    sys.path.insert(0, PARENT_ROOT)

try:
    from act_pipeline.config import ModelConfig
    from act_pipeline.models.act_model import ACTPolicy
    from act_pipeline.data.dataset import BimanualEpisodicDataset
    from act_pipeline.data.preprocess import preprocess_rgbd
    from act_pipeline.data.resample import extract_timestamps
    ACT_PIPELINE_AVAILABLE = True
except ImportError as e:
    ACT_PIPELINE_AVAILABLE = False
    IMPORT_ERR = str(e)


# Danh sách 16 khớp theo quy chuẩn ACT OpenArm
JOINT_NAMES = [
    # Tay trái (Left Arm: J1..J7 + Gripper)
    "L_Joint_1_Base", "L_Joint_2_Shoulder", "L_Joint_3_Elbow_Pitch",
    "L_Joint_4_Elbow_Roll", "L_Joint_5_Wrist_Pitch", "L_Joint_6_Wrist_Yaw",
    "L_Joint_7_Wrist_Roll", "L_Gripper",
    # Tay phải (Right Arm: J1..J7 + Gripper)
    "R_Joint_1_Base", "R_Joint_2_Shoulder", "R_Joint_3_Elbow_Pitch",
    "R_Joint_4_Elbow_Roll", "R_Joint_5_Wrist_Pitch", "R_Joint_6_Wrist_Yaw",
    "R_Joint_7_Wrist_Roll", "R_Gripper",
]


class HDF5Inspector:
    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)
        self.results: Dict[str, Any] = {}
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.passes: List[str] = []

    def inspect_file(self) -> Dict[str, Any]:
        """Bóc tách toàn diện cấu trúc file HDF5 và đánh giá từng tiêu chí."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Không tìm thấy file HDF5: {self.file_path}")

        file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)

        with h5py.File(self.file_path, "r") as f:
            # 1. Quét cây cấu trúc HDF5
            tree_info = []
            def _walker(name, obj):
                if isinstance(obj, h5py.Dataset):
                    tree_info.append({
                        "name": "/" + name,
                        "type": "Dataset",
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "chunks": obj.chunks,
                        "compression": obj.compression,
                    })
                elif isinstance(obj, h5py.Group):
                    tree_info.append({
                        "name": "/" + name,
                        "type": "Group",
                        "members": list(obj.keys()),
                    })

            f.visititems(_walker)

            # 2. Kiểm tra Action Dataset (/action)
            action_info = self._check_action(f)

            # 3. Kiểm tra Trạng thái góc khớp (/observations/qpos)
            qpos_info = self._check_qpos(f)

            # 4. Kiểm tra Vận tốc khớp (/observations/qvel)
            qvel_info = self._check_qvel(f)

            # 5. Kiểm tra Camera Feeds (/observations/images)
            cam_info = self._check_cameras(f)

            # 6. Kiểm tra Tần số lấy mẫu và Timestamps
            T = action_info.get("length", 0)
            timing_info = self._check_timing(f, T)

        self.results = {
            "file_path": self.file_path,
            "file_size_mb": file_size_mb,
            "tree": tree_info,
            "action": action_info,
            "qpos": qpos_info,
            "qvel": qvel_info,
            "cameras": cam_info,
            "timing": timing_info,
            "passes": self.passes,
            "warnings": self.warnings,
            "issues": self.issues,
        }
        return self.results

    def _check_action(self, f: h5py.File) -> Dict[str, Any]:
        info: Dict[str, Any] = {"found": False}
        if "action" not in f:
            self.issues.append("THIẾU DATASET: Không tìm thấy '/action' trong file HDF5!")
            return info

        ds = f["action"]
        info["found"] = True
        info["shape"] = ds.shape
        info["dtype"] = str(ds.dtype)
        info["length"] = ds.shape[0]

        if len(ds.shape) != 2:
            self.issues.append(f"SAI SỐ CHIỀU: '/action' phải có 2 chiều [T, action_dim], nhưng nhận được ndim={len(ds.shape)}")
            return info

        T, action_dim = ds.shape
        info["action_dim"] = action_dim

        # Kiểm tra 16 bậc tự do (Chuẩn OpenArm)
        if action_dim == 16:
            self.passes.append(f"'/action' có đúng chuẩn 16 chiều (16-DOF Bimanual): shape = [{T}, 16]")
        elif action_dim == 14:
            self.warnings.append(f"'/action' có 14 chiều (Chuẩn ALOHA cũ). Model hiện tại cấu hình cho 16-DOF (2 tay x 8 motor).")
        else:
            self.issues.append(f"SAI KÍCH THƯỚC: '/action' có {action_dim} chiều (kỳ vọng 16 cho OpenArm 16-DOF).")

        # Thống kê giá trị
        data = ds[:].astype(np.float32)
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        if has_nan or has_inf:
            self.issues.append("DỮ LIỆU BỊ HỎNG: '/action' chứa NaN hoặc Inf!")
        else:
            info["min"] = float(np.min(data))
            info["max"] = float(np.max(data))
            info["mean"] = float(np.mean(data))
            info["std"] = float(np.std(data))
            self.passes.append(f"'/action' không có NaN/Inf, dải giá trị: [{info['min']:.3f}, {info['max']:.3f}]")

        return info

    def _check_qpos(self, f: h5py.File) -> Dict[str, Any]:
        info: Dict[str, Any] = {"found": False}
        qpos_key = None
        for k in ["observations/qpos", "qpos"]:
            if k in f:
                qpos_key = k
                break

        if qpos_key is None:
            self.issues.append("THIẾU DATASET: Không tìm thấy '/observations/qpos' trong file HDF5!")
            return info

        ds = f[qpos_key]
        info["found"] = True
        info["key"] = qpos_key
        info["shape"] = ds.shape
        info["dtype"] = str(ds.dtype)

        if len(ds.shape) != 2:
            self.issues.append(f"SAI SỐ CHIỀU: '{qpos_key}' phải có 2 chiều [T, 16], nhận được ndim={len(ds.shape)}")
            return info

        T, qpos_dim = ds.shape
        info["qpos_dim"] = qpos_dim

        if qpos_dim == 16:
            self.passes.append(f"'{qpos_key}' có đúng chuẩn 16 góc khớp: shape = [{T}, 16]")
        else:
            self.issues.append(f"SAI KÍCH THƯỚC: '{qpos_key}' có {qpos_dim} chiều (kỳ vọng 16 cho OpenArm 16-DOF).")

        data = ds[:].astype(np.float32)
        if np.isnan(data).any() or np.isinf(data).any():
            self.issues.append(f"DỮ LIỆU BỊ HỎNG: '{qpos_key}' chứa NaN hoặc Inf!")
        else:
            info["min"] = float(np.min(data))
            info["max"] = float(np.max(data))
            self.passes.append(f"'{qpos_key}' dải góc khớp thực tế: [{info['min']:.2f}, {info['max']:.2f}] rad")

        return info

    def _check_qvel(self, f: h5py.File) -> Dict[str, Any]:
        info: Dict[str, Any] = {"found": False}
        for k in ["observations/qvel", "qvel"]:
            if k in f:
                ds = f[k]
                info["found"] = True
                info["key"] = k
                info["shape"] = ds.shape
                info["dtype"] = str(ds.dtype)
                self.passes.append(f"Tìm thấy '{k}': shape = {ds.shape}")
                break
        if not info["found"]:
            self.warnings.append("Không tìm thấy '/observations/qvel' (Tùy chọn: ACT không bắt buộc qvel để huấn luyện).")
        return info

    def _check_cameras(self, f: h5py.File) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "has_images_group": False,
            "chest": None,
            "front": None,
            "depth_available": False,
        }

        if "observations/images" not in f and "images" not in f:
            self.issues.append("THIẾU DATASET THỊ GIÁC: Không tìm thấy group '/observations/images'!")
            return info

        obs_img = f["observations/images"] if "observations/images" in f else f["images"]
        info["has_images_group"] = True
        available_keys = list(obs_img.keys())
        info["keys"] = available_keys

        # 1. Kiểm tra Camera Chest (Trước ngực)
        chest_found = False
        chest_shape = None
        has_chest_depth = False

        if "chest_rgb" in obs_img and "chest_depth" in obs_img:
            chest_found = True
            chest_shape = obs_img["chest_rgb"].shape
            has_chest_depth = True
            info["chest"] = {"rgb": "chest_rgb", "depth": "chest_depth", "shape": chest_shape}
            self.passes.append(f"Camera CHEST: Tách biệt RGB {obs_img['chest_rgb'].shape} và Depth {obs_img['chest_depth'].shape}")
        elif "chest" in obs_img:
            chest_found = True
            chest_shape = obs_img["chest"].shape
            if chest_shape[-1] == 4:
                has_chest_depth = True
                info["chest"] = {"type": "RGB-D 4 channels", "shape": chest_shape}
                self.passes.append(f"Camera CHEST: Ghép sẵn 4 kênh RGB-D {chest_shape}")
            else:
                has_chest_depth = "chest_depth" in obs_img
                info["chest"] = {"type": "RGB 3 channels", "shape": chest_shape, "has_depth": has_chest_depth}
                self.passes.append(f"Camera CHEST: Ảnh màu RGB {chest_shape} (Depth đi kèm: {'Có' if has_chest_depth else 'Tự sinh mặc định'})")
        elif len(available_keys) > 0:
            # Fallback lấy key đầu tiên
            first_key = available_keys[0]
            chest_found = True
            chest_shape = obs_img[first_key].shape
            info["chest"] = {"fallback_key": first_key, "shape": chest_shape}
            self.warnings.append(f"Camera CHEST: Sử dụng key '{first_key}' {chest_shape} làm camera chính.")

        if not chest_found:
            self.issues.append("THIẾU CAMERA: Không tìm thấy bất kỳ luồng ảnh camera ngực nào!")

        # 2. Kiểm tra Camera Front (Đối diện nhìn robot)
        front_found = False
        if "front_rgb" in obs_img or "front" in obs_img:
            front_key = "front_rgb" if "front_rgb" in obs_img else "front"
            front_shape = obs_img[front_key].shape
            has_front_depth = ("front_depth" in obs_img) or (front_shape[-1] == 4)
            front_found = True
            info["front"] = {"key": front_key, "shape": front_shape, "has_depth": has_front_depth}
            self.passes.append(f"Camera FRONT: Tìm thấy camera đối diện '{front_key}' {front_shape} (Depth: {'Có' if has_front_depth else 'Không'})")
        else:
            info["front"] = None
            self.warnings.append("Camera FRONT: Không tìm thấy camera đối diện 'front'. Pipeline sẽ tự động nhân bản (duplicate) camera chest làm camera thứ 2.")

        info["depth_available"] = has_chest_depth

        # Kiểm tra độ phân giải chuẩn (480x640)
        if chest_shape is not None:
            H, W = chest_shape[1], chest_shape[2]
            if (H, W) == (480, 640):
                self.passes.append(f"ĐỘ PHÂN GIẢI CHUẨN: Ảnh camera đạt đúng chuẩn thiết kế {H}x{W} (480x640)")
            else:
                self.warnings.append(f"Độ phân giải ảnh là {H}x{W} (khác chuẩn 480x640). Pipeline sẽ tự động Letterbox/Resize về 480x640.")

        return info

    def _check_timing(self, f: h5py.File, T: int) -> Dict[str, Any]:
        t_raw, has_ts, ts_stats = extract_timestamps(f, T, default_hz=50.0)
        dt = np.diff(t_raw) if len(t_raw) > 1 else np.array([0.02])
        mean_dt = float(np.mean(dt))
        effective_hz = 1.0 / mean_dt if mean_dt > 0 else 50.0
        jitter = ts_stats.get("jitter_detected", False)

        info = {
            "has_timestamps": has_ts,
            "effective_hz": effective_hz,
            "mean_dt_ms": mean_dt * 1000.0,
            "min_dt_ms": float(np.min(dt)) * 1000.0 if len(dt) > 0 else 20.0,
            "max_dt_ms": float(np.max(dt)) * 1000.0 if len(dt) > 0 else 20.0,
            "jitter_detected": jitter,
        }

        if has_ts:
            if not jitter and 48.0 <= effective_hz <= 52.0:
                self.passes.append(f"TẦN SỐ ĐỒNG ĐỀU CHUẨN: {effective_hz:.1f} Hz (dt={mean_dt*1000:.1f}ms), không bị rớt khung hình.")
            else:
                self.warnings.append(
                    f"Tần số thực tế: {effective_hz:.1f} Hz (Jitter={jitter}, dải dt=[{info['min_dt_ms']:.1f}ms - {info['max_dt_ms']:.1f}ms]). "
                    f"Tính năng auto_resample của pipeline sẽ tự động nội suy đồng đều về chuẩn 50.0 Hz."
                )
        else:
            self.warnings.append("File không có mảng timestamps riêng. Giả định tần số cố định 50.0 Hz (20ms/bước).")

        return info

    def run_pipeline_test(self, use_3d_features: bool = False) -> bool:
        """Thử nghiệm nạp mẫu thực tế qua BimanualEpisodicDataset và Forward qua ACTPolicy."""
        if not ACT_PIPELINE_AVAILABLE:
            print(f"[!] Không thể chạy test pipeline do lỗi import: {IMPORT_ERR}")
            return False

        print("\n" + "=" * 70)
        print(f"[*] THỰC NGHIỆM TÍCH HỢP DATASET & MÔ HÌNH ACT (3D Features: {'BẬT' if use_3d_features else 'TẮT'}):")
        print("=" * 70)

        # 1. Mock stats nếu chưa có dataset_stats.pkl
        stats = {
            "action_mean": np.zeros(16, dtype=np.float32),
            "action_std": np.ones(16, dtype=np.float32),
            "qpos_mean": np.zeros(16, dtype=np.float32),
            "qpos_std": np.ones(16, dtype=np.float32),
        }

        try:
            dataset = BimanualEpisodicDataset(
                file_paths=[self.file_path],
                stats=stats,
                chunk_size=50,
                step_stride=1,
                target_img_size=(480, 640),
                auto_resample=True,
            )
            print(f"  [✓] Khởi tạo BimanualEpisodicDataset thành công! Tổng số mẫu (chunks): {len(dataset)}")

            sample = dataset[0]
            img = sample["image"].unsqueeze(0)   # [1, 2, 4, 480, 640]
            qpos = sample["qpos"].unsqueeze(0)   # [1, 16]
            actions = sample["actions"].unsqueeze(0) # [1, 50, 16]
            is_pad = sample["is_pad"].unsqueeze(0)   # [1, 50]

            print(f"  [✓] Trích xuất mẫu dataset[0]:")
            print(f"      - Image Tensor  : {list(img.shape)} (2 Cameras x 4 Channels RGB-D)")
            print(f"      - Qpos Tensor   : {list(qpos.shape)} (16 Joint Angles)")
            print(f"      - Action Chunk  : {list(actions.shape)} (Chunk k=50 tương lai)")
            print(f"      - Padding Mask  : {list(is_pad.shape)}")

            # 2. Khởi tạo mô hình ACTPolicy và Forward thử nghiệm
            cfg = ModelConfig(
                in_channels=4,
                num_cameras=2,
                backbone_type="yolov8_nano",
                freeze_backbone=True,
                yolo_stride=32, # Stride 32 để test nhanh
                d_model=512,
                decoder_layers=2,
                chunk_size=50,
                action_dim=16,
                qpos_dim=16,
                use_3d_features=use_3d_features,
            )
            model = ACTPolicy(cfg).eval()

            with torch.no_grad():
                pred_actions, mu, logvar = model(image=img, qpos=qpos, actions=actions, is_pad=is_pad)
                eval_actions = model.predict_action_chunk(image=img, qpos=qpos)

            print(f"  [✓] Forward Train thành công! Pred Actions: {list(pred_actions.shape)}, mu: {list(mu.shape)}")
            print(f"  [✓] Forward Inference thành công! Pred Action Chunk: {list(eval_actions.shape)}")
            print(f"  --> KẾT LUẬN: File HDF5 hoàn toàn tương thích 100% với Mô hình ACT!")
            return True

        except Exception as e:
            print(f"  [X] LỖI khi chạy thử nghiệm pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_report(self):
        """In báo cáo định dạng trực quan, rõ ràng trên Console."""
        res = self.results
        print("\n" + "=" * 80)
        print("   🔍 BÁO CÁO PHÂN TÍCH & KIỂM TRA ĐỊNH DẠNG FILE HDF5 CHO MÔ HÌNH ACT")
        print("=" * 80)
        print(f"[*] Đường dẫn file    : {res['file_path']}")
        print(f"[*] Kích thước file   : {res['file_size_mb']:.2f} MB")
        print("-" * 80)

        print("[*] BẢNG CẤU TRÚC DỮ LIỆU BÊN TRONG HDF5:")
        for item in res["tree"]:
            if item["type"] == "Dataset":
                comp_str = f"({item['compression']})" if item['compression'] else ""
                print(f"    ├─ 📄 {item['name']:<30} Shape: {str(item['shape']):<18} Type: {item['dtype']} {comp_str}")
            elif item["type"] == "Group":
                print(f"    📁 {item['name']}/  (Chứa: {', '.join(item['members'])})")

        print("-" * 80)
        print("[*] ĐÁNH GIÁ CÁC TIÊU CHÍ KỸ THUẬT (CHECKLIST):")

        # In các tiêu chí ĐẠT (PASS)
        for p in self.passes:
            print(f"    [✓ PASS] {p}")

        # In các CẢNH BÁO (WARNING)
        for w in self.warnings:
            print(f"    [⚠️ WARN] {w}")

        # In các LỖI (FAIL)
        for i in self.issues:
            print(f"    [❌ FAIL] {i}")

        print("-" * 80)
        if len(self.issues) == 0:
            print("🎉 KẾT QUẢ: FILE HDF5 HỢP LỆ VÀ SẴN SÀNG HUẤN LUYỆN (100% COMPATIBLE)!")
        else:
            print(f"⛔ KẾT QUẢ: PHÁT HIỆN {len(self.issues)} LỖI CẦN KHẮC PHỤC TRƯỚC KHI TRAIN!")
        print("=" * 80 + "\n")


def find_first_hdf5(folder_path: str) -> Optional[str]:
    """Tìm file HDF5 đầu tiên trong một thư mục."""
    patterns = [
        os.path.join(folder_path, "*.hdf5"),
        os.path.join(folder_path, "**", "*.hdf5"),
        os.path.join(folder_path, "*.h5"),
    ]
    for p in patterns:
        files = glob.glob(p, recursive=True)
        if len(files) > 0:
            return sorted(files)[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Công cụ bóc thử và kiểm tra định dạng HDF5 cho ACT OpenArm")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", type=str, help="Thư mục chứa các file .hdf5 (tự động lấy file đầu tiên)")
    group.add_argument("--file", type=str, help="Đường dẫn trực tiếp tới 1 file .hdf5 cần kiểm tra")
    
    parser.add_argument("--index", type=int, default=0, help="Chỉ số file cần lấy nếu dùng --folder (mặc định 0)")
    parser.add_argument("--test_forward", action="store_true", help="Chạy thử nghiệm forward pass qua Dataset và ACT Policy")
    parser.add_argument("--use_3d_features", action="store_true", help="Kiểm tra với cờ 3D Point Features bật")

    args = parser.parse_args()

    target_file = None
    if args.file:
        target_file = args.file
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"[X] LỖI: Thư mục không tồn tại: {args.folder}")
            sys.exit(1)
        files = sorted(glob.glob(os.path.join(args.folder, "**", "*.hdf5"), recursive=True))
        if len(files) == 0:
            files = sorted(glob.glob(os.path.join(args.folder, "*.hdf5")))
        if len(files) == 0:
            print(f"[X] LỖI: Không tìm thấy file .hdf5 nào trong thư mục: {args.folder}")
            sys.exit(1)
        
        idx = max(0, min(args.index, len(files) - 1))
        target_file = files[idx]
        print(f"[*] Tìm thấy {len(files)} file .hdf5 trong thư mục. Đang kiểm tra file [{idx+1}/{len(files)}]:")
        print(f"    -> {target_file}")

    inspector = HDF5Inspector(target_file)
    inspector.inspect_file()
    inspector.print_report()

    if args.test_forward:
        inspector.run_pipeline_test(use_3d_features=args.use_3d_features)


if __name__ == "__main__":
    main()
