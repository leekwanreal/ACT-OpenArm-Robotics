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

"""Which action each step gets, and what the model is fed.

The preprocessing and ensembling must be act_pipeline's, number for number;
the torch comparison itself lives outside the package (it needs torch), so
these pin the formulas down without it.
"""

import math
import pickle

import numpy as np
import pytest

from openarm_act.chunking import ActionChunker
from openarm_act.policy import (
    GRIPPERS, HoldPolicy, IMAGENET_MEAN, IMAGENET_STD, OnnxPolicy, load_stats,
    preprocess_rgbd, rad_to_stroke, resolve_gripper_unit, stroke_to_rad,
)


def chunk(value, steps=5, dof=2):
    return np.full((steps, dof), float(value))


# ----------------------------------------------------------------- chunking

def test_actions_line_up_with_the_observation_step():
    chunker = ActionChunker(5)
    chunker.add(10, np.arange(5)[:, None] * np.ones((1, 2)))
    assert chunker.action(9) is None
    assert chunker.action(10)[0] == 0
    assert chunker.action(14)[0] == 4
    assert chunker.action(15) is None


def test_without_ensembling_the_newest_chunk_wins():
    chunker = ActionChunker(5, temporal_ensemble=False)
    chunker.add(0, chunk(1.0))
    chunker.add(2, chunk(3.0))
    assert chunker.action(3)[0] == 3.0


def test_ensembling_matches_act_pipeline_temporal_ensemble():
    """TemporalEnsemblePolicy weighs a chunk exp(-m * steps since it was made)."""
    m = 0.5
    chunker = ActionChunker(5, ensemble_k=m)            # newest by default
    chunker.add(0, chunk(0.0))
    chunker.add(2, chunk(1.0))
    w_old, w_new = math.exp(-m * 3), math.exp(-m * 1)   # at step 3
    assert chunker.action(3)[0] == pytest.approx(w_new / (w_old + w_new))
    assert chunker.action(3)[0] > 0.5


def test_paper_weighting_favours_the_oldest():
    chunker = ActionChunker(5, ensemble_k=0.5, weighting="oldest")
    chunker.add(2, chunk(1.0))
    chunker.add(0, chunk(0.0))
    assert chunker.action(3)[0] < 0.5


def test_prune_and_remaining():
    chunker = ActionChunker(5)
    chunker.add(0, chunk(0.0))
    chunker.add(3, chunk(1.0))
    assert chunker.remaining(4) == 4
    chunker.prune(6)
    assert len(chunker.chunks) == 1
    chunker.prune(8)
    assert chunker.remaining(8) == 0


def test_hold_policy_repeats_the_state():
    qpos = np.arange(16, dtype=np.float32)
    actions = HoldPolicy(chunk_size=50).predict(None, None, qpos)
    assert actions.shape == (50, 16)
    assert np.all(actions == qpos)


# --------------------------------------------------------------- preprocess

def test_preprocess_is_act_pipelines_formula():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[..., 0] = 255
    depth = np.array([[0, 100, 200, 700, 1200, 5000]] * 4, np.uint16)
    image = preprocess_rgbd(rgb, depth, size=(4, 6))
    assert image.shape == (4, 4, 6) and image.dtype == np.float32
    expected_rgb = (np.array([1.0, 0.0, 0.0]) - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(image[:3, 0, 0], expected_rgb)
    # mm -> m, clip to [0.2, 1.2], scale to [0, 1]; no reading reads as near.
    assert np.allclose(image[3, 0], [0.0, 0.0, 0.0, 0.5, 1.0, 1.0])


def test_preprocess_resizes_after_normalising():
    rgb = np.full((720, 1280, 3), 128, np.uint8)
    depth = np.full((720, 1280), 700, np.uint16)
    image = preprocess_rgbd(rgb, depth, size=(480, 640))
    assert image.shape == (4, 480, 640)
    assert np.allclose(image[3], 0.5)


# ------------------------------------------------------------ gripper, stats

def test_gripper_unit_is_read_off_the_stats():
    stats = {key: np.zeros(16, np.float32) for key in
             ("qpos_mean", "qpos_std", "action_mean", "action_std")}
    stats["qpos_mean"][list(GRIPPERS)] = 0.02            # metres of stroke
    assert resolve_gripper_unit("auto", stats) == "stroke_m"
    stats["qpos_mean"][list(GRIPPERS)] = -0.6            # motor radians
    assert resolve_gripper_unit("auto", stats) == "rad"
    assert resolve_gripper_unit("stroke_m", stats) == "stroke_m"


def test_stroke_and_angle_convert_both_ways():
    stroke = np.array([0.0, 0.0215, 0.043])
    assert np.allclose(stroke_to_rad(stroke), [0.0, -0.6, -1.2])
    assert np.allclose(rad_to_stroke(stroke_to_rad(stroke)), stroke)


def test_stats_load_from_act_pipelines_pickle(tmp_path):
    raw = {key: np.arange(16, dtype=np.float64) for key in
           ("qpos_mean", "qpos_std", "action_mean", "action_std")}
    raw["num_episodes"] = 3
    path = tmp_path / "dataset_stats.pkl"
    path.write_bytes(pickle.dumps(raw))
    stats = load_stats(path)
    assert stats["qpos_std"].dtype == np.float32 and stats["qpos_std"].shape == (16,)


# --------------------------------------------------------------- onnx inputs

class _Input:
    """Stands in for onnxruntime's NodeArg: a name, a shape and a type."""

    def __init__(self, name, shape, type="tensor(float)"):
        self.name, self.shape, self.type = name, list(shape), type


def roles(inputs, input_names=""):
    matched = OnnxPolicy._match_inputs(inputs, input_names)
    return {role: meta.name for role, meta in matched.items()}


def test_onnx_inputs_by_name():
    inputs = [_Input("qpos", (1, 16)), _Input("image", (1, 4, 480, 640))]
    assert roles(inputs) == {"image": "image", "qpos": "qpos"}


def test_onnx_inputs_by_rank_when_names_say_nothing():
    inputs = [_Input("input_0", (1, 4, 480, 640)), _Input("input_1", (1, 16))]
    assert roles(inputs) == {"image": "input_0", "qpos": "input_1"}


def test_onnx_inputs_named_explicitly():
    inputs = [_Input("a", (1, 16)), _Input("b", (1, 16))]
    assert roles(inputs, "image=b, qpos=a") == {"image": "b", "qpos": "a"}
    with pytest.raises(ValueError):
        roles(inputs)                               # neither is 4-D
    with pytest.raises(ValueError):
        roles(inputs, "image=b")                    # qpos left out


# ---------------------------------------------------- normalisation in graph

class _Session:
    """Stands in for an InferenceSession whose model repeats qpos twice."""

    def __init__(self, metadata):
        self.metadata = metadata

    def get_modelmeta(self):
        return type("Meta", (), {"custom_metadata_map": self.metadata})()

    def get_inputs(self):
        return [_Input("image", (1, 4, 8, 8)), _Input("qpos", (1, 16))]

    def get_outputs(self):
        return [_Input("actions", (1, 2, 16))]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, outputs, feed):
        return [np.repeat(feed["qpos"][:, None], 2, axis=1)]


def onnx_policy(monkeypatch, tmp_path, metadata):
    """An OnnxPolicy on _Session, dataset_stats.pkl beside it as the export
    writes it: stats that visibly change the numbers, means small enough
    to read as gripper strokes."""
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "InferenceSession",
                        lambda *args, **kwargs: _Session(metadata))
    stats = {"qpos_mean": np.full(16, 0.05), "qpos_std": np.full(16, 2.0),
             "action_mean": np.full(16, 0.01), "action_std": np.full(16, 3.0)}
    (tmp_path / "dataset_stats.pkl").write_bytes(pickle.dumps(stats))
    model = tmp_path / "policy.onnx"
    model.write_bytes(b"")
    return OnnxPolicy(checkpoint=str(model), device="cpu")


def observe(qpos):
    rgb = np.zeros((8, 8, 3), np.uint8)
    depth = np.full((8, 8), 600, np.uint16)
    return rgb, depth, np.asarray(qpos, np.float32)


def test_graph_that_normalises_itself_is_not_normalised_again(monkeypatch, tmp_path):
    policy = onnx_policy(monkeypatch, tmp_path, {"embedded_unnorm": "True"})
    qpos = np.linspace(-1.0, 1.0, 16)
    qpos[list(GRIPPERS)] = 0.02                     # strokes, in range
    actions = policy.predict(*observe(qpos))
    assert actions.shape == (2, 16)
    np.testing.assert_allclose(actions, np.stack([qpos, qpos]), atol=1e-6)


def test_bare_graph_takes_the_stats_beside_it(monkeypatch, tmp_path):
    policy = onnx_policy(monkeypatch, tmp_path, {})
    qpos = np.linspace(-1.0, 1.0, 16)
    expected = (qpos - 0.05) / 2.0 * 3.0 + 0.01     # normalised once, back once
    np.testing.assert_allclose(policy.predict(*observe(qpos))[0], expected,
                               atol=1e-5)


# ------------------------------------------------------------ which model

def test_device_auto_takes_cuda_only_when_the_runtime_has_it():
    class Runtime:
        def __init__(self, providers):
            self.providers = providers

        def get_available_providers(self):
            return self.providers

    with_cuda = Runtime(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert OnnxPolicy._providers(with_cuda, "auto")[0][0] == "CUDAExecutionProvider"
    assert OnnxPolicy._providers(Runtime(["CPUExecutionProvider"]), "auto") == \
        ["CPUExecutionProvider"]


def test_newest_model_in_checkpoint_unless_one_is_named(monkeypatch, tmp_path):
    import os
    from openarm_act import models

    monkeypatch.setattr(models, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.delenv(models.MODEL_ENV, raising=False)
    assert models.find_model() == ""
    old, new = tmp_path / "old.onnx", tmp_path / "new.onnx"
    for age, path in ((100, old), (0, new)):
        path.write_bytes(b"")
        os.utime(path, (path.stat().st_mtime - age,) * 2)
    (tmp_path / "policy.onnx.bak").write_bytes(b"")     # not a model
    assert models.find_model() == str(new)
    # An export in a folder of its own, as export_onnx.py writes it.
    (tmp_path / "run2").mkdir()
    newest = tmp_path / "run2" / "policy.onnx"
    newest.write_bytes(b"")
    os.utime(newest, (new.stat().st_mtime + 10,) * 2)
    assert models.find_model() == str(newest)
    monkeypatch.setenv(models.MODEL_ENV, str(old))
    assert models.find_model() == str(old)


# ----------------------------------------------------------------- grasping

def test_a_grasp_predicted_as_the_objects_width_closes():
    """Trained on qpos, a model predicts the stroke the fingers stopped at."""
    from openarm_act.policy import close_grippers
    actions = np.zeros((3, 16))
    actions[:, 7] = [0.043, 0.030, 0.024]       # left: open, then on the box
    actions[:, 15] = [0.043, 0.040, 0.036]      # right: still open enough
    actions[:, 3] = 0.024                       # an arm joint is not a gripper
    closed = close_grippers(actions, below=0.035)
    np.testing.assert_allclose(closed[:, 7], [0.043, 0.0, 0.0])
    np.testing.assert_allclose(closed[:, 15], actions[:, 15])
    np.testing.assert_allclose(closed[:, 3], 0.024)
    np.testing.assert_allclose(close_grippers(actions, 0.035, to=0.005)[1:, 7], 0.005)


def test_closing_off_leaves_the_chunk_alone():
    from openarm_act.policy import close_grippers
    actions = np.full((2, 16), 0.02)
    np.testing.assert_array_equal(close_grippers(actions, below=0.0), actions)
