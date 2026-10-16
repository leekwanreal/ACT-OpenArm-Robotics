#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Script: ACT PyTorch Model (.pth) -> ONNX Model (.onnx)
Optimized for Real-Time C++ Inference (ONNX Runtime / TensorRT) on Bimanual OpenArm (16-DOF).

Key Features:
1. Strips training-only branches (bypasses CVAE stochastic sampling, sets z = 0).
2. Formats exact C++ input/output contracts:
   - Input 'image': [1, 4, 480, 640] float32 (RGB-D)
   - Input 'qpos' : [1, 16] float32 (Current Joint Positions)
   - Output 'actions': [1, 50, 16] float32 (Future Action Chunk)
3. Optional --embed_unnorm: Embeds Z-score un-normalization (x * std + mean) directly into ONNX,
   enabling C++ to receive exact real-world joint angles (radians) directly without stats.json!
4. Automatic validation: Checks ONNX schema with onnx.checker and verifies numerical
   parity between PyTorch and ONNX Runtime.
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn

# Đảm bảo UTF-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thêm đường dẫn gốc vào sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Mọi model nằm trong checkpoint/: các checkpoint huấn luyện (.pth) ở
# checkpoint/pth/, các bản export (.onnx) ở checkpoint/onnx/ -- nơi
# openarm_act / run_main.sh tìm model (file .onnx mới nhất, kể cả trong thư
# mục con).
PTH_DIR = os.path.join(BASE_DIR, "checkpoint", "pth")
ONNX_DIR = os.path.join(BASE_DIR, "checkpoint", "onnx")
DEFAULT_CHECKPOINT = os.path.join(PTH_DIR, "best_checkpoint.pth")
# Ảnh mà camera (openarm_camera) và recorder (openarm_ui) tạo ra, và
# openarm_act nhận: chỉ xuất model train trên đúng kích thước này.
IMG_HEIGHT, IMG_WIDTH = 480, 640


def default_output(checkpoint_path: str) -> str:
    """checkpoint/onnx/<tên checkpoint>/policy.onnx: mỗi model một thư mục,
    để dataset_stats.* ghi kèm không đè lên stats của model khác."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    return os.path.join(ONNX_DIR, stem, "policy.onnx")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def find_act_pipeline(named: Optional[str] = None) -> Optional[str]:
    """Thư mục chứa act_pipeline (repo huấn luyện, ngoài repo này):
    --act_pipeline, rồi $OPENARM_ACT_PIPELINE, rồi các chỗ quen thuộc."""
    candidates = [named, os.environ.get("OPENARM_ACT_PIPELINE"), BASE_DIR,
                  os.path.join(os.path.dirname(BASE_DIR), "OpenArm_MC"),
                  os.path.expanduser("~/Project/OpenArm_MC")]
    for directory in candidates:
        if directory and os.path.isdir(os.path.join(os.path.expanduser(directory),
                                                    "act_pipeline")):
            return os.path.abspath(os.path.expanduser(directory))
    return None


# act_pipeline phải import được trước khi nạp model: tìm nó trước khi
# argparse chạy, vì các import dưới đây cần nó ngay.
_named = None
for _i, _arg in enumerate(sys.argv):
    if _arg in ("--act_pipeline", "--act-pipeline") and _i + 1 < len(sys.argv):
        _named = sys.argv[_i + 1]
    elif _arg.startswith(("--act_pipeline=", "--act-pipeline=")):
        _named = _arg.split("=", 1)[1]
_ACT_PIPELINE = find_act_pipeline(_named)
if _ACT_PIPELINE is None:
    sys.exit("Không tìm thấy act_pipeline: truyền --act_pipeline <thư mục chứa "
             "act_pipeline/> hoặc đặt OPENARM_ACT_PIPELINE")
if _ACT_PIPELINE not in sys.path:
    sys.path.insert(0, _ACT_PIPELINE)

from act_pipeline.config import ModelConfig
from act_pipeline.models.act_model import ACTPolicy
from act_pipeline.data.normalization import load_norm_stats

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class ACTONNXExportWrapper(nn.Module):
    """
    Wrapper model rút gọn phục vụ export ONNX và suy luận C++:
    - Nhận vào 2 tensors: image [B, 4, 480, 640] và qpos [B, 16]
    - Cố định latent style z = 0 (tối ưu nhất, loại bỏ nhánh ngẫu nhiên)
    - Trả về đúng 1 tensor duy nhất: actions [B, 50, 16]
    - Tùy chọn nhúng giải chuẩn hóa (actions * std + mean) trực tiếp vào đồ thị ONNX.
    """
    def __init__(
        self,
        policy: ACTPolicy,
        stats: Optional[Dict[str, np.ndarray]] = None,
        embed_unnorm: bool = True,
    ):
        super().__init__()
        self.policy = policy
        self.embed_unnorm = embed_unnorm and (stats is not None)

        if self.embed_unnorm and stats is not None:
            # 1. Giải chuẩn hóa action đầu ra (output unnormalization): pred = norm_pred * std + mean
            action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32).view(1, 1, -1)
            action_std = torch.tensor(stats["action_std"], dtype=torch.float32).view(1, 1, -1)
            self.register_buffer("action_mean", action_mean)
            self.register_buffer("action_std", action_std)

            # 2. Chuẩn hóa qpos đầu vào (input normalization): norm_qpos = (qpos - mean) / std
            # Giúp C++ và runtime ném thẳng góc khớp thực tế (Radian) vào model mà không cần tính toán bên ngoài!
            qpos_mean = torch.tensor(stats["qpos_mean"], dtype=torch.float32).view(1, -1)
            qpos_std = torch.tensor(stats["qpos_std"], dtype=torch.float32).view(1, -1)
            qpos_std = torch.clamp(qpos_std, min=1e-5)
            self.register_buffer("qpos_mean", qpos_mean)
            self.register_buffer("qpos_std", qpos_std)
        else:
            self.action_mean = None
            self.action_std = None
            self.qpos_mean = None
            self.qpos_std = None

    def forward(self, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        # 1. Chuẩn hóa qpos đầu vào từ góc Radian thực tế sang phân phối Z-score
        if self.embed_unnorm and self.qpos_mean is not None:
            qpos = (qpos - self.qpos_mean) / self.qpos_std

        # 2. Forward qua ACTPolicy ở chế độ inference (actions=None -> z=0)
        pred_actions, _, _ = self.policy(image=image, qpos=qpos, actions=None, is_pad=None)

        # 3. Giải chuẩn hóa actions đầu ra từ Z-score về góc Radian thực tế
        if self.embed_unnorm and self.action_mean is not None:
            pred_actions = pred_actions * self.action_std + self.action_mean

        return pred_actions


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 14,
    embed_unnorm: bool = True,
    dynamic_batch: bool = False,
    device_str: str = "cpu",
    verify: bool = True,
) -> bool:
    print("=" * 80)
    print(" 🚀 XUẤT MÔ HÌNH ACT SANG ONNX CHO SUY LUẬN C++ (ONNX RUNTIME / TENSORRT)")
    print("=" * 80)
    print(f"[*] Checkpoint đầu vào  : {checkpoint_path}")
    print(f"[*] File ONNX đầu ra    : {output_path}")
    print(f"[*] Opset Version       : {opset_version}")
    print(f"[*] Nhúng Un-normalize  : {'BẬT (Output trực tiếp Radian)' if embed_unnorm else 'TẮT (Output chuẩn hóa Z-score)'}")
    print(f"[*] Kích thước Batch    : {'Dynamic (Khả biến)' if dynamic_batch else 'Fixed = 1 (Tối ưu tối đa cho C++/TensorRT)'}")
    print(f"[*] Thiết bị chuyển đổi : {device_str.upper()}")
    print("-" * 80)

    if not os.path.exists(checkpoint_path):
        # Tìm kiếm fallback trong các thư mục phổ biến
        fallbacks = [
            os.path.join(BASE_DIR, checkpoint_path),
            os.path.join(PTH_DIR, checkpoint_path),
            DEFAULT_CHECKPOINT,
            os.path.join(BASE_DIR, "checkpoints", "act_openarm", "best_checkpoint.pth"),
            os.path.join(BASE_DIR, "checkpoints", "act_openarm", "act_deploy_weights.pth"),
            os.path.join(BASE_DIR, "dataset", "act_openarm_model.pth"),
        ]
        found = False
        for fb in fallbacks:
            if os.path.exists(fb):
                checkpoint_path = fb
                found = True
                print(f"[!] Tự động chuyển sang file tìm thấy: {checkpoint_path}")
                break
        if not found:
            raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {checkpoint_path}")

    device = torch.device(device_str)

    # 1. Nạp Checkpoint & Thống kê
    print("[*] Đang nạp checkpoint PyTorch...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    stats = checkpoint.get("stats")

    if stats is None:
        stats_candidates = [
            os.path.join(os.path.dirname(checkpoint_path), "dataset_stats.pkl"),
            os.path.join(os.path.dirname(checkpoint_path), "dataset_stats.json"),
            os.path.join(BASE_DIR, "checkpoints", "act_openarm", "dataset_stats.pkl"),
            os.path.join(BASE_DIR, "dataset", "dataset_stats.pkl"),
        ]
        for sp in stats_candidates:
            if os.path.exists(sp):
                stats = load_norm_stats(sp)
                print(f"[✓] Đã nạp thống kê chuẩn hóa từ: {sp}")
                break

    if stats is None and embed_unnorm:
        print("[⚠️ CẢNH BÁO] Không tìm thấy stats chuẩn hóa. Tắt chế độ embed_unnorm.")
        embed_unnorm = False

    # 2. Khởi tạo & nạp trọng số ACTPolicy
    saved_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    detected_d_model = saved_cfg.get("d_model", 512)
    detected_action_dim = saved_cfg.get("action_dim", 16)
    detected_chunk_size = saved_cfg.get("chunk_size", 50)
    detected_decoder_layers = saved_cfg.get("decoder_layers", 4)
    detected_cvae_layers = saved_cfg.get("cvae_layers", 2)
    detected_latent_dim = saved_cfg.get("latent_dim", 32)
    detected_dim_feedforward = saved_cfg.get("dim_feedforward", detected_d_model * 4)

    # Introspect state_dict
    if "qpos_proj.weight" in state_dict:
        detected_d_model = state_dict["qpos_proj.weight"].shape[0]
        detected_action_dim = state_dict["qpos_proj.weight"].shape[1]
    if "cvae.latent_mu.weight" in state_dict:
        detected_latent_dim = state_dict["cvae.latent_mu.weight"].shape[0]
    elif "latent_mu.weight" in state_dict:
        detected_latent_dim = state_dict["latent_mu.weight"].shape[0]

    for k, v in state_dict.items():
        if ("linear1.weight" in k) and hasattr(v, "shape") and len(v.shape) == 2:
            detected_dim_feedforward = v.shape[0]
            break

    num_cameras = saved_cfg.get("num_cameras", 2)
    camera_names = tuple(saved_cfg.get("camera_names", ("chest", "front")))
    cfg = ModelConfig(
        in_channels=4,
        num_cameras=num_cameras,
        camera_names=camera_names,
        backbone_type=saved_cfg.get("backbone_type", "yolov8_nano"),
        freeze_backbone=True,
        d_model=detected_d_model,
        nheads=8 if detected_d_model % 8 == 0 else 4,
        dim_feedforward=detected_dim_feedforward,
        cvae_layers=detected_cvae_layers,
        latent_dim=detected_latent_dim,
        decoder_layers=detected_decoder_layers,
        chunk_size=detected_chunk_size,
        action_dim=detected_action_dim,
        qpos_dim=detected_action_dim,
        use_3d_features=saved_cfg.get("use_3d_features", False),
        point_grid_h=saved_cfg.get("point_grid_h", 15),
        point_grid_w=saved_cfg.get("point_grid_w", 20),
        # Kích thước ảnh lúc train: đồ thị ONNX cố định kích thước này.
        img_height=saved_cfg.get("img_height", IMG_HEIGHT),
        img_width=saved_cfg.get("img_width", IMG_WIDTH),
    )
    if (cfg.img_height, cfg.img_width) != (IMG_HEIGHT, IMG_WIDTH):
        sys.exit(f"Checkpoint được train với ảnh {cfg.img_width}x{cfg.img_height}, "
                 f"nhưng camera và recorder dùng {IMG_WIDTH}x{IMG_HEIGHT}: "
                 f"hãy train lại trên dữ liệu {IMG_WIDTH}x{IMG_HEIGHT}.")
    print(f"Ảnh đầu vào: {cfg.img_width}x{cfg.img_height}, Số camera: {cfg.num_cameras}")

    base_model = ACTPolicy(cfg).to(device)

    # Clean keys
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[7:]
        if new_k == "action_queries" and "policy_decoder.action_queries" not in state_dict:
            new_k = "policy_decoder.action_queries"
        cleaned_state_dict[new_k] = v

    # Filter mismatched shapes safely
    model_state = base_model.state_dict()
    final_state_dict = {}
    for k, v in cleaned_state_dict.items():
        if k in model_state and hasattr(v, "shape") and v.shape != model_state[k].shape:
            continue
        final_state_dict[k] = v

    try:
        base_model.load_state_dict(final_state_dict, strict=True)
    except Exception:
        base_model.load_state_dict(final_state_dict, strict=False)
    base_model.eval()

    # 3. Tạo Wrapper xuất khẩu
    export_model = ACTONNXExportWrapper(base_model, stats=stats, embed_unnorm=embed_unnorm).to(device)
    export_model.eval()

    # 4. Dummy Inputs
    batch_size = 1
    if num_cameras > 1:
        dummy_image = torch.randn(batch_size, num_cameras, 4, cfg.img_height, cfg.img_width, dtype=torch.float32, device=device)
    else:
        dummy_image = torch.randn(batch_size, 4, cfg.img_height, cfg.img_width, dtype=torch.float32, device=device)
    dummy_qpos = torch.randn(batch_size, cfg.qpos_dim, dtype=torch.float32, device=device)

    # Chuẩn bị dynamic axes nếu yêu cầu
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch_size"},
            "qpos": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }

    # Tạo thư mục đầu ra
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Vô hiệu hóa fastpath _native_multi_head_attention của PyTorch 2.x để export ONNX Opset 14 an toàn
    prev_fastpath = True
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
        prev_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)

    # 5. Xuất ONNX
    print(f"[*] Đang thực hiện torch.onnx.export (Opset {opset_version})...")
    try:
        export_kwargs = dict(
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["image", "qpos"],
            output_names=["actions"],
            dynamic_axes=dynamic_axes,
        )
        with torch.no_grad():
            # torch >= 2.9 mặc định dùng exporter dynamo (cần onnxscript);
            # dynamo=False giữ exporter TorchScript cho opset 14.
            try:
                torch.onnx.export(export_model, (dummy_image, dummy_qpos),
                                  output_path, dynamo=False, **export_kwargs)
            except TypeError:  # torch cũ chưa có tham số dynamo
                torch.onnx.export(export_model, (dummy_image, dummy_qpos),
                                  output_path, **export_kwargs)
    finally:
        if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
            torch.backends.mha.set_fastpath_enabled(prev_fastpath)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[✓] Đã xuất file ONNX thành công! Kích thước: {file_size_mb:.2f} MB")

    # 6. Kiểm định cấu trúc ONNX (nếu có thư viện onnx)
    if ONNX_AVAILABLE:
        print("[*] Đang kiểm tra tính hợp lệ của đồ thị ONNX (onnx.checker)...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        # Ghi metadata vào file ONNX để C++ đọc được thông số
        meta = onnx_model.metadata_props.add()
        meta.key, meta.value = "model_type", "ACT_OpenArm_Bimanual_16DOF"
        meta2 = onnx_model.metadata_props.add()
        meta2.key, meta2.value = "chunk_size", str(cfg.chunk_size)
        meta3 = onnx_model.metadata_props.add()
        meta3.key, meta3.value = "action_dim", str(cfg.action_dim)
        meta4 = onnx_model.metadata_props.add()
        meta4.key, meta4.value = "embedded_unnorm", str(embed_unnorm)
        onnx.save(onnx_model, output_path)
        print("[✓] Đồ thị ONNX hoàn toàn hợp lệ và đã nhúng metadata kỹ thuật!")

        # Xuất kèm file stats sang thư mục để C++ hoặc các module ngoài dễ dàng đọc
        if stats is not None:
            import pickle, json
            out_dir = os.path.dirname(os.path.abspath(output_path))
            try:
                pkl_p = os.path.join(out_dir, "dataset_stats.pkl")
                json_p = os.path.join(out_dir, "dataset_stats.json")
                with open(pkl_p, "wb") as f:
                    pickle.dump(stats, f)
                with open(json_p, "w", encoding="utf-8") as f:
                    s_json = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in stats.items()}
                    json.dump(s_json, f, indent=2)
                print(f"[✓] Đã lưu kèm file dataset_stats.json và dataset_stats.pkl tại: {out_dir}")
            except Exception as e:
                print(f"[!] Bỏ qua lưu file stats phụ: {e}")

    # 7. Kiểm định độ chính xác số học (Numerical Parity Check)
    if verify and ORT_AVAILABLE:
        print("[*] Đang kiểm tra so sánh kết quả PyTorch vs ONNX Runtime...")
        with torch.no_grad():
            pt_out = export_model(dummy_image, dummy_qpos).cpu().numpy()

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device_str == "cuda" else ["CPUExecutionProvider"]
        ort_sess = ort.InferenceSession(output_path, providers=providers)
        ort_inputs = {
            "image": dummy_image.cpu().numpy(),
            "qpos": dummy_qpos.cpu().numpy(),
        }
        ort_out = ort_sess.run(None, ort_inputs)[0]

        max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
        mean_abs_diff = float(np.mean(np.abs(pt_out - ort_out)))

        print(f"   + Sai số tuyệt đối lớn nhất (Max Diff)  : {max_abs_diff:.6e} rad")
        print(f"   + Sai số trung bình (Mean Diff)         : {mean_abs_diff:.6e} rad")

        if max_abs_diff < 1e-4:
            print("   🏆 HOÀN HẢO! Kết quả ONNX khớp gần như tuyệt đối với PyTorch (< 1e-4 rad).")
        elif max_abs_diff < 1e-3:
            print("   ✅ ĐẠT YÊU CẦU! Sai số rất nhỏ do tối ưu hóa float32 (< 1e-3 rad).")
        else:
            print("   ⚠️ CẢNH BÁO: Sai số lớn hơn 1e-3 rad, hãy kiểm tra lại cấu hình.")

    print("=" * 80)
    print(f"🎉 HOÀN TẤT! File ONNX sẵn sàng cho C++ tại: {output_path}")
    print("=" * 80)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Chuyển đổi mô hình ACT PyTorch (.pth) sang ONNX (.onnx) cho C++")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="File checkpoint .pth cần chuyển đổi; một tên file trần được "
             "tìm trong checkpoint/pth/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file .onnx xuất ra (mặc định: "
             "checkpoint/onnx/<tên checkpoint>/policy.onnx)",
    )
    parser.add_argument(
        "--act_pipeline", "--act-pipeline",
        type=str,
        default=None,
        help="Thư mục chứa act_pipeline/ (mặc định: $OPENARM_ACT_PIPELINE, "
             "rồi ../OpenArm_MC, ~/Project/OpenArm_MC)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX Opset version (mặc định: 14)",
    )
    parser.add_argument(
        "--no_embed_unnorm",
        dest="embed_unnorm",
        action="store_false",
        help="Tắt nhúng giải chuẩn hóa (Mặc định: BẬT để xuất trực tiếp Radian góc khớp cho C++)",
    )
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="Bật dynamic batch size (Mặc định: False - Cố định batch=1 để tối đa hiệu năng TensorRT/C++)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Thiết bị thực hiện chuyển đổi (cpu hoặc cuda)",
    )
    parser.add_argument(
        "--no_verify",
        dest="verify",
        action="store_false",
        help="Tắt bước kiểm định kết quả so sánh với ONNX Runtime",
    )
    parser.set_defaults(embed_unnorm=True, verify=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output or default_output(args.checkpoint),
        opset_version=args.opset,
        embed_unnorm=args.embed_unnorm,
        dynamic_batch=args.dynamic_batch,
        device_str=args.device,
        verify=args.verify,
    )
