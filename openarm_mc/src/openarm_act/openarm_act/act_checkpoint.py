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

"""Load an act_pipeline training checkpoint the way its own tools do.

Mirrors OpenArm_MC's tools/infer_robot.py: the model's shape is read back
from the state dict (d_model, decoder and CVAE depth, latent size) rather
than trusted from the saved config, and the stats come from the checkpoint,
or from dataset_stats.pkl beside it. Needs torch, torchvision and the
act_pipeline package.
"""

import os
import sys
from pathlib import Path


def import_act_pipeline(path=""):
    """Make `import act_pipeline` work, from `path` if it is given."""
    if path:
        root = str(Path(path).expanduser().resolve())
        if (Path(root) / "act_pipeline").is_dir() and root not in sys.path:
            sys.path.insert(0, root)
        elif Path(root).name == "act_pipeline" and str(Path(root).parent) not in sys.path:
            sys.path.insert(0, str(Path(root).parent))
    try:
        import act_pipeline  # noqa: F401
    except ImportError as error:
        raise ImportError(
            f"cannot import act_pipeline ({error}); pass act_pipeline_path, the "
            f"directory that holds it (e.g. ~/Project/OpenArm_MC)") from error


def _layers(state_dict, *needles):
    indices = [int(key.split(".layers.")[1].split(".")[0])
               for key in state_dict
               if ".layers." in key and all(n in key for n in needles)]
    return max(indices) + 1 if indices else None


def load_act_model(checkpoint, act_pipeline_path="", device="cpu"):
    """-> (ACTPolicy in eval mode, stats dict, ModelConfig)."""
    import_act_pipeline(act_pipeline_path)
    import torch
    from act_pipeline.config import ModelConfig
    from act_pipeline.data.normalization import load_norm_stats
    from act_pipeline.models.act_model import ACTPolicy

    from .policy import normalize_stats

    path = os.path.expanduser(checkpoint)
    data = torch.load(path, map_location=device, weights_only=False)
    state_dict = data.get("model_state_dict", data) if isinstance(data, dict) else data

    stats = data.get("stats") if isinstance(data, dict) else None
    if stats is None:
        beside = os.path.join(os.path.dirname(path), "dataset_stats.pkl")
        if not os.path.exists(beside):
            raise FileNotFoundError(
                f"{path} has no stats and there is no dataset_stats.pkl beside it")
        stats = load_norm_stats(beside)
    stats = normalize_stats(stats, path)

    # act_pipeline's train.py saves its command line (vars(args)), whose
    # names match ModelConfig's; the weights themselves are the better
    # witness where they can say, so they win over it.
    saved = data.get("config", {}) if isinstance(data, dict) else {}
    saved = saved if isinstance(saved, dict) else vars(saved)
    shapes = {key.removeprefix("module."): tuple(value.shape)
              for key, value in state_dict.items() if hasattr(value, "shape")}

    d_model = saved.get("d_model", 512)
    qpos_dim = saved.get("qpos_dim", 16)
    action_dim = saved.get("action_dim", 16)
    chunk_size = saved.get("chunk_size", 50)
    latent_dim = saved.get("latent_dim", 32)
    in_channels = saved.get("in_channels", 4)
    if "qpos_proj.weight" in shapes:
        d_model, qpos_dim = shapes["qpos_proj.weight"]
    if "policy_decoder.action_head.weight" in shapes:
        action_dim = shapes["policy_decoder.action_head.weight"][0]
    for key in ("policy_decoder.action_queries", "action_queries"):
        if key in shapes and len(shapes[key]) == 3:
            chunk_size = shapes[key][1]
            break
    for key in ("cvae.latent_mu.weight", "latent_mu.weight"):
        if key in shapes:
            latent_dim = shapes[key][0]
            break
    if "backbone.conv1_adapter.weight" in shapes:
        in_channels = shapes["backbone.conv1_adapter.weight"][1]
    decoder_layers = (_layers(state_dict, "policy_decoder")
                      or _layers(state_dict, "decoder")
                      or saved.get("decoder_layers", 4))
    cvae_layers = _layers(state_dict, "cvae") or saved.get("cvae_layers", 2)

    cfg = ModelConfig(
        in_channels=in_channels,
        backbone_type=saved.get("backbone_type", "resnet18"),
        freeze_backbone=True,
        img_height=saved.get("img_height", 480),
        img_width=saved.get("img_width", 640),
        d_model=d_model,
        nheads=saved.get("nheads", 8 if d_model % 8 == 0 else 4),
        dim_feedforward=saved.get("dim_feedforward", d_model * 4),
        cvae_layers=cvae_layers,
        latent_dim=latent_dim,
        decoder_layers=decoder_layers,
        chunk_size=chunk_size,
        action_dim=action_dim,
        qpos_dim=qpos_dim,
        depth_min_m=saved.get("depth_min_m", 0.2),
        depth_max_m=saved.get("depth_max_m", 1.2),
    )
    model = ACTPolicy(cfg)

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key == "action_queries" and "policy_decoder.action_queries" not in state_dict:
            key = "policy_decoder.action_queries"
        cleaned[key] = value

    # Strict, and loudly so. Skipping the tensors that do not fit would load
    # -- and leave those layers at their random initial weights, driving the
    # arm with a network nobody trained.
    expected = model.state_dict()
    mismatched = [f"{key}: checkpoint {tuple(value.shape)}, model "
                  f"{tuple(expected[key].shape)}"
                  for key, value in cleaned.items()
                  if key in expected and hasattr(value, "shape")
                  and tuple(value.shape) != tuple(expected[key].shape)]
    missing = sorted(set(expected) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(expected))
    if mismatched or missing or unexpected:
        problems = mismatched[:5] + [f"missing {k}" for k in missing[:5]] \
            + [f"unexpected {k}" for k in unexpected[:5]]
        raise ValueError(
            f"{path} does not fit the network its shapes describe "
            f"({len(mismatched)} mismatched, {len(missing)} missing, "
            f"{len(unexpected)} unexpected): " + "; ".join(problems))
    model.load_state_dict(cleaned, strict=True)
    model.to(device).eval()
    return model, stats, cfg
