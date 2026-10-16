# Copyright 2026 Enactic, Inc.
# Licensed under the Apache License, Version 2.0.

import os

from openarm_ui import mujoco_visuals
from openarm_ui.model_import import file_digest, safe_stem, same_file, unique_dir


def test_a_file_name_becomes_a_safe_folder_name():
    assert safe_stem("best_checkpoint (1).pth") == "best_checkpoint_1"
    assert safe_stem("../../etc/passwd.onnx") == "passwd"
    assert safe_stem("###.onnx") == "model"


def test_a_taken_name_gets_a_number(tmp_path):
    assert unique_dir(str(tmp_path), "run") == str(tmp_path / "run")
    (tmp_path / "run").mkdir()
    (tmp_path / "run_2.pth").write_bytes(b"")
    assert unique_dir(str(tmp_path), "run") == str(tmp_path / "run_2")
    assert unique_dir(str(tmp_path), "run", ".pth") == str(tmp_path / "run.pth")
    (tmp_path / "run.pth").write_bytes(b"")
    assert unique_dir(str(tmp_path), "run", ".pth") == str(tmp_path / "run_3.pth")


def test_the_same_model_is_found_not_kept_twice(tmp_path):
    (tmp_path / "a").mkdir()
    kept = tmp_path / "a" / "policy.onnx"
    kept.write_bytes(b"weights" * 100)
    other = tmp_path / "b.onnx"
    other.write_bytes(b"WEIGHTS" * 100)          # same size, other bytes
    (tmp_path / ".incoming").mkdir()
    (tmp_path / ".incoming" / "x.onnx").write_bytes(b"weights" * 100)
    digest = file_digest(str(kept))
    assert same_file(str(tmp_path), ".onnx", 700, digest) == str(kept)
    assert same_file(str(tmp_path), ".onnx", 701, digest) is None


def test_the_simulators_parts_replace_each_collada_mesh():
    path = mujoco_visuals.find_mjcf()
    if path is None:
        return                      # openarm_mujoco is not beside this checkout
    manifest, root = mujoco_visuals.parse(path)
    parts = manifest["meshes"]["link4.dae"]
    assert [p["material"] for p in parts] == ["matte_black", "metal_silver",
                                              "matte_black"]
    assert all(os.path.isfile(os.path.join(root, p["file"])) for p in parts)
    assert manifest["materials"]["matte_black"]["rgba"][0] < 0.3
