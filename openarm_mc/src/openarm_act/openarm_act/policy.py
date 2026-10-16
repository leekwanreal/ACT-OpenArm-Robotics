# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What the node calls to turn one observation into a chunk of actions.

The node hands a policy the observation exactly as openarm_ui's recorder
writes it into a dataset:

    rgb    uint8   [H, W, 3]   RGB
    depth  uint16  [H, W]      millimetres, 0 = no reading
    qpos   float32 [16]        left_j1..j7, left_gripper, right_j1..j7,
                               right_gripper; rad, grippers as stroke in m

and expects back [chunk_size, 16] absolute joint targets in the same order
and units, actions[0] being for the step the observation was taken at.

The ACT policies below reproduce act_pipeline (the training code) exactly:

    image  [1, 4, H, W]   RGB / 255 normalised with ImageNet mean and std,
                          then depth in metres clipped to [0.2, 1.2] and
                          scaled to [0, 1]; resized after normalising, as
                          preprocess_rgbd does
    qpos   [1, 16]        (qpos - qpos_mean) / qpos_std
    ->     [1, 50, 16]    actions * action_std + action_mean

OnnxPolicy runs an export made with utils/export_onnx.py; ActPipelinePolicy loads
the training checkpoint (.pth) with act_pipeline itself. The node picks one by
its import path (`policy` parameter, "module:Class").
"""

import importlib
import json
import pickle
from pathlib import Path

import numpy as np

NUM_JOINTS = 16
GRIPPERS = (7, 15)
# How openarm_can relates finger stroke to the gripper motor's angle.
GRIPPER_MAX_STROKE_M = 0.043
GRIPPER_OPEN_RAD = -1.20

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

# Keys utils/export_onnx.py writes into the model's metadata.
META_STATS = "openarm_act.stats"
META_CONFIG = "openarm_act.config"
# "True" when the graph normalises qpos and un-normalises the actions itself.
META_EMBEDDED = "embedded_unnorm"
# What the export writes next to the .onnx.
SIDE_STATS = ("dataset_stats.pkl", "dataset_stats.json")


def identity_stats():
    """Stats that leave qpos and actions as they are."""
    zero = np.zeros(NUM_JOINTS, dtype=np.float32)
    one = np.ones(NUM_JOINTS, dtype=np.float32)
    return {"qpos_mean": zero, "qpos_std": one,
            "action_mean": zero, "action_std": one}


def load_policy(spec, options):
    """Build the policy named "module:Class" with the node's options."""
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"policy must be 'module:Class', not {spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**options)


class Policy:
    """The interface the node drives. Subclasses implement `predict`.

    `cameras` names the cameras the policy reads (openarm_camera's roles),
    the first being the one predict()'s rgb and depth come from. A policy
    that reads more than one also gets the others:

        predict(rgb, depth, qpos, cameras={"front": (rgb, depth)})

    and one that reads a single camera is called as it always was.
    """

    chunk_size = 50
    cameras = ("chest", "front")

    def predict(self, rgb, depth, qpos, cameras=None):
        raise NotImplementedError

    def reset(self):
        """Called when the node (re)starts a run: drop any internal state."""

    def describe(self):
        """One line for the log: what was loaded."""
        return type(self).__name__


class HoldPolicy(Policy):
    """Predicts staying where the arm is.

    No model and no torch: for trying the camera, state and chunking plumbing
    against the simulator before a checkpoint is involved.
    """

    def __init__(self, chunk_size=50, **_ignored):
        self.chunk_size = int(chunk_size)

    def predict(self, rgb, depth, qpos):
        return np.repeat(np.asarray(qpos, dtype=np.float32)[None],
                         self.chunk_size, axis=0)


# ------------------------------------------------------------------- stats

def normalize_stats(raw, source="stats"):
    """mean/std for qpos and action as float32 [16], from whatever held them."""
    stats = {}
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        if key not in raw:
            raise KeyError(f"{source} has no {key!r}")
        value = raw[key]
        if hasattr(value, "detach"):            # a torch tensor
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape != (NUM_JOINTS,):
            raise ValueError(f"{key} in {source} has shape {value.shape}, "
                             f"expected ({NUM_JOINTS},)")
        stats[key] = value
    return stats


def load_stats(path):
    """act_pipeline's stats: dataset_stats.pkl, a .json, or an .npz."""
    path = Path(path).expanduser()
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
    elif path.suffix == ".npz":
        raw = dict(np.load(path))
    elif path.suffix in (".pt", ".pth"):
        raise ValueError(f"{path}: stats in a torch file need ActPipelinePolicy, "
                         f"or save them as .json")
    else:
        with open(path, "rb") as handle:
            raw = pickle.load(handle)
    return normalize_stats(raw, str(path))


def resolve_gripper_unit(unit, stats):
    """"stroke_m" or "rad": what the model's gripper values are in.

    The recorder writes strokes (0..0.043 m); a dataset made elsewhere may hold
    the motor angle (0..-1.20 rad) instead. "auto" tells them apart by the
    dataset's own numbers, which cannot be off by a factor of thirty.
    """
    if unit in ("stroke_m", "rad"):
        return unit
    if unit != "auto":
        raise ValueError(f"gripper_unit must be auto, stroke_m or rad, not {unit!r}")
    largest = max(abs(float(stats[key][index]))
                  for key in ("qpos_mean", "action_mean") for index in GRIPPERS)
    return "stroke_m" if largest <= 0.06 else "rad"


def stroke_to_rad(stroke):
    ratio = np.clip(np.asarray(stroke) / GRIPPER_MAX_STROKE_M, 0.0, 1.0)
    return ratio * GRIPPER_OPEN_RAD


def rad_to_stroke(angle):
    ratio = np.clip(np.asarray(angle) / GRIPPER_OPEN_RAD, 0.0, 1.0)
    return ratio * GRIPPER_MAX_STROKE_M


def close_grippers(actions, below, to=0.0):
    """A chunk whose gripper strokes under `below` (m) close to `to` instead.

    A model trained on the measured joints (action = qpos, as the recorder
    writes it) learns the stroke the fingers stopped at on the object --
    its width -- not the one teleop commanded, which was shut. Sent as is,
    that stroke reaches the object and squeezes nothing; closed further,
    as in the demonstrations, the gripper's force limit does the holding.
    `below` <= 0 leaves the chunk alone.
    """
    actions = np.array(actions, dtype=np.float64)
    if below <= 0:
        return actions
    for index in GRIPPERS:
        column = actions[:, index]
        actions[:, index] = np.where(column < below, to, column)
    return actions


# -------------------------------------------------------------- preprocess

def preprocess_rgbd(rgb, depth, size=(480, 640), depth_min_m=0.2,
                    depth_max_m=1.2):
    """act_pipeline.data.preprocess.preprocess_rgbd without torch.

    rgb uint8 [H, W, 3], depth [H, W] in mm (uint16) or m (float) ->
    float32 [4, size[0], size[1]]. Normalised first and resized second, in
    that order, because that is what the model was trained on.
    """
    import cv2

    rgb = (np.asarray(rgb, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    depth = np.asarray(depth)
    if depth.dtype == np.uint16 or np.nanmax(depth) > 20.0:
        depth_m = depth.astype(np.float32) / 1000.0
    else:
        depth_m = depth.astype(np.float32)
    depth_m = np.nan_to_num(depth_m, nan=0.0)
    depth_norm = (np.clip(depth_m, depth_min_m, depth_max_m) - depth_min_m) \
        / (depth_max_m - depth_min_m)

    rgbd = np.concatenate([rgb, depth_norm[..., None]], axis=2)    # [H, W, 4]
    height, width = size
    if rgbd.shape[:2] != (height, width):
        # cv2's bilinear uses half-pixel centres, as F.interpolate does with
        # align_corners=False.
        rgbd = cv2.resize(rgbd, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgbd.transpose(2, 0, 1), dtype=np.float32)


class ActPolicy(Policy):
    """Normalisation and gripper units, shared by every way of running ACT."""

    def __init__(self, stats, chunk_size=50, image_size=(480, 640),
                 depth_range_m=(0.2, 1.2), gripper_unit="auto", in_channels=4):
        if in_channels not in (3, 4):
            raise ValueError(f"the model takes {in_channels} image channels; "
                             f"only RGB (3) and RGB-D (4) are known")
        self.in_channels = int(in_channels)
        self.stats = stats
        self.chunk_size = int(chunk_size)
        self.image_size = tuple(int(v) for v in image_size)
        self.depth_range_m = tuple(float(v) for v in depth_range_m)
        self.gripper_unit = resolve_gripper_unit(gripper_unit, stats)

    def model_qpos(self, qpos):
        """Topic-unit qpos -> the normalised [1, 16] the network takes."""
        qpos = np.array(qpos, dtype=np.float32)
        if self.gripper_unit == "rad":
            qpos[list(GRIPPERS)] = stroke_to_rad(qpos[list(GRIPPERS)])
        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos[None].astype(np.float32)

    def model_image(self, rgb, depth):
        image = preprocess_rgbd(rgb, depth, self.image_size, *self.depth_range_m)
        # An RGB-only model takes the first three channels, as its conv1 does.
        return image[None, :self.in_channels]

    def model_images(self, rgb, depth, cameras=None):
        chest_img = preprocess_rgbd(rgb, depth, self.image_size, *self.depth_range_m)[:self.in_channels]
        if cameras and "front" in cameras and cameras["front"] is not None:
            f_rgb, f_depth = cameras["front"]
            front_img = preprocess_rgbd(f_rgb, f_depth, self.image_size, *self.depth_range_m)[:self.in_channels]
            return np.stack([chest_img, front_img], axis=0)[None, ...]
        return chest_img[None, None, ...]

    def actions_from(self, out):
        """The network's [1, chunk, 16] -> joint targets in topic units."""
        actions = np.asarray(out, dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != NUM_JOINTS:
            raise ValueError(f"model returned {np.shape(out)}, "
                             f"expected [1, chunk, {NUM_JOINTS}]")
        actions = actions * self.stats["action_std"] + self.stats["action_mean"]
        if self.gripper_unit == "rad":
            actions[:, list(GRIPPERS)] = rad_to_stroke(actions[:, list(GRIPPERS)])
        return actions[:self.chunk_size]

    def describe(self):
        height, width = self.image_size
        kind = "RGB-D" if self.in_channels == 4 else "RGB"
        return (f"{type(self).__name__}: chunk {self.chunk_size}, {kind} "
                f"{width}x{height}, depth {self.depth_range_m[0]}-"
                f"{self.depth_range_m[1]} m, gripper in {self.gripper_unit}")


# -------------------------------------------------------------------- ONNX

_ONNX_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
}


class OnnxPolicy(ActPolicy):
    """An ACT model exported to ONNX, run with onnxruntime.

    Takes `image` [1, 4, H, W] and `qpos` [1, 16]; an export with other names
    is matched by rank, or by `input_names` ("image=x,qpos=y"). Stats come
    from the model's metadata when export_onnx.py wrote them there, otherwise
    from `stats_file`, which then must be given.
    """

    def __init__(self, checkpoint, stats_file="", device="cuda",
                 input_names="", chunk_size=50, gripper_unit="auto",
                 cpu_threads=4, **_ignored):
        import onnxruntime as ort       # only this policy needs onnxruntime

        if not checkpoint:
            raise ValueError("OnnxPolicy needs a checkpoint (.onnx)")
        path = str(Path(checkpoint).expanduser())
        options = ort.SessionOptions()
        if cpu_threads > 0:
            # Left alone, onnxruntime takes every core and keeps them busy,
            # which starves the 400 Hz controller on the same machine.
            options.intra_op_num_threads = int(cpu_threads)
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            path, sess_options=options, providers=self._providers(ort, device))
        self.device = ("cuda" if "CUDAExecutionProvider" in
                       self.session.get_providers() else "cpu")

        meta = self.session.get_modelmeta().custom_metadata_map
        config = json.loads(meta.get(META_CONFIG, "{}"))
        self.embedded = meta.get(META_EMBEDDED, "").lower() == "true"
        if stats_file:
            stats = load_stats(stats_file)
        elif META_STATS in meta:
            stats = normalize_stats(json.loads(meta[META_STATS]), path)
        else:
            beside = [Path(path).with_name(name) for name in SIDE_STATS]
            found = next((p for p in beside if p.exists()), None)
            if found is not None:
                stats = load_stats(found)
            elif self.embedded:
                stats = None
            else:
                raise ValueError(f"{path} carries no normalisation stats; pass "
                                 f"stats_file (act_pipeline's dataset_stats.pkl), "
                                 f"or export with utils/export_onnx.py")
        if self.embedded:
            # Normalising again here would do it twice. The dataset's stats,
            # when there are any, still tell the gripper unit.
            if stats is not None:
                gripper_unit = resolve_gripper_unit(gripper_unit, stats)
            elif gripper_unit == "auto":
                gripper_unit = "stroke_m"   # what the recorder writes
            stats = identity_stats()

        self.inputs = self._match_inputs(self.session.get_inputs(), input_names)
        self.output = self.session.get_outputs()[0].name

        # The exported graph fixes the image size; trust it over the config.
        image_shape = self.inputs["image"].shape
        height = config.get("img_height", 480)
        width = config.get("img_width", 640)
        channels = config.get("in_channels", 4)
        if len(image_shape) == 4 and all(isinstance(v, int) for v in image_shape[2:]):
            height, width = image_shape[2], image_shape[3]
        if len(image_shape) == 4 and isinstance(image_shape[1], int):
            channels = image_shape[1]
        super().__init__(
            stats,
            chunk_size=config.get("chunk_size", chunk_size),
            image_size=(height, width),
            depth_range_m=(config.get("depth_min_m", 0.2),
                           config.get("depth_max_m", 1.2)),
            gripper_unit=gripper_unit,
            in_channels=channels)

    @staticmethod
    def _providers(ort, device):
        available = ort.get_available_providers()
        if device == "auto":
            device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        if device.startswith("cuda"):
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    f"device {device!r} asked for, but this onnxruntime has "
                    f"only {available}. Install onnxruntime-gpu, or pass "
                    f"device:=cpu")
            index = int(device.partition(":")[2] or 0)
            return [("CUDAExecutionProvider", {"device_id": index}),
                    "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _match_inputs(declared, input_names=""):
        """Map image and qpos to the model's input descriptions."""
        by_name = {meta.name: meta for meta in declared}
        if len(declared) != 2:
            raise ValueError(f"expected two model inputs (image, qpos), got "
                             f"{sorted(by_name)}")
        if input_names:
            matched = {}
            for pair in input_names.split(","):
                role, _, name = (part.strip() for part in pair.partition("="))
                if role not in ("image", "qpos"):
                    raise ValueError(f"input_names: unknown role {role!r}; "
                                     f"expected image or qpos")
                if name not in by_name:
                    raise ValueError(f"input_names: the model has no input "
                                     f"{name!r}; it has {sorted(by_name)}")
                matched[role] = by_name[name]
            if set(matched) != {"image", "qpos"}:
                raise ValueError("input_names must name both image and qpos")
            return matched
        if "image" in by_name and "qpos" in by_name:
            return {"image": by_name["image"], "qpos": by_name["qpos"]}
        # Otherwise by rank: the image is the 4-D or 5-D one.
        ranked = sorted(declared, key=lambda meta: len(meta.shape))
        if len(ranked[1].shape) not in (4, 5):
            raise ValueError(f"cannot tell which of {sorted(by_name)} is the "
                             f"image; set input_names")
        return {"qpos": ranked[0], "image": ranked[1]}

    def describe(self):
        return super().describe() + (", normalised in the graph"
                                      if self.embedded else "") + \
            f", on {self.device}"

    def predict(self, rgb, depth, qpos, cameras=None):
        input_rank = len(self.inputs["image"].shape)
        if input_rank == 5:
            image_arr = self.model_images(rgb, depth, cameras=cameras)
        else:
            image_arr = self.model_image(rgb, depth)
        arrays = {"image": image_arr,
                  "qpos": self.model_qpos(qpos)}
        feed = {meta.name: arrays[role].astype(
                    _ONNX_DTYPES.get(meta.type, np.float32), copy=False)
                for role, meta in self.inputs.items()}
        return self.actions_from(self.session.run([self.output], feed)[0])


# ------------------------------------------------------- act_pipeline (.pth)

class ActPipelinePolicy(ActPolicy):
    """The training checkpoint, run by act_pipeline itself.

    For checking an ONNX export against the original, or for running without
    one. Needs torch and torchvision, and act_pipeline importable: pass
    `act_pipeline_path`, the directory holding the act_pipeline package.
    """

    def __init__(self, checkpoint, stats_file="", device="cuda",
                 act_pipeline_path="", chunk_size=50, gripper_unit="auto",
                 cpu_threads=4, **_ignored):
        from .act_checkpoint import load_act_model
        import torch

        if cpu_threads > 0:
            torch.set_num_threads(int(cpu_threads))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, stats, cfg = load_act_model(
            checkpoint, act_pipeline_path, device)
        if stats_file:
            stats = load_stats(stats_file)
        from act_pipeline.data.preprocess import preprocess_rgbd as reference
        self.torch = torch
        self.device = torch.device(device)
        self.reference_preprocess = reference
        super().__init__(stats, chunk_size=cfg.chunk_size,
                         image_size=(cfg.img_height, cfg.img_width),
                         depth_range_m=(cfg.depth_min_m, cfg.depth_max_m),
                         gripper_unit=gripper_unit,
                         in_channels=cfg.in_channels)

    def model_image(self, rgb, depth):
        image = self.reference_preprocess(
            rgb, depth, depth_min_m=self.depth_range_m[0],
            depth_max_m=self.depth_range_m[1], target_size=self.image_size)
        return image[:self.in_channels].unsqueeze(0).numpy()

    def model_images(self, rgb, depth, cameras=None):
        chest_img = self.reference_preprocess(
            rgb, depth, depth_min_m=self.depth_range_m[0],
            depth_max_m=self.depth_range_m[1], target_size=self.image_size)[:self.in_channels]
        if cameras and "front" in cameras and cameras["front"] is not None:
            f_rgb, f_depth = cameras["front"]
            front_img = self.reference_preprocess(
                f_rgb, f_depth, depth_min_m=self.depth_range_m[0],
                depth_max_m=self.depth_range_m[1], target_size=self.image_size)[:self.in_channels]
            return self.torch.stack([chest_img, front_img], dim=0).unsqueeze(0).numpy()
        return chest_img.unsqueeze(0).unsqueeze(0).numpy()

    def predict(self, rgb, depth, qpos, cameras=None):
        torch = self.torch
        num_cams = getattr(self.model, "num_cameras", 2)
        if num_cams > 1:
            img_np = self.model_images(rgb, depth, cameras=cameras)
        else:
            img_np = self.model_image(rgb, depth)
        image = torch.from_numpy(img_np).to(self.device)
        qpos = torch.from_numpy(self.model_qpos(qpos)).to(self.device)
        with torch.inference_mode():
            out, _, _ = self.model(image=image, qpos=qpos, actions=None)
        return self.actions_from(out.float().cpu().numpy())
