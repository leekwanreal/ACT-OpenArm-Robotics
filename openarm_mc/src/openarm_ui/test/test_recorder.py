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

import json

import h5py
import numpy as np
import pytest

from openarm_ui.data_recorder import BackgroundEpisodeWriter, EpisodeWriter
from openarm_ui.episode_compress import compress

H, W = 6, 8


def frames(i, offset=0):
    return (np.full((H, W, 3), (i + offset) % 256, np.uint8),
            np.full((H, W), 300 + i + offset, np.uint16))


def sample(i, cameras=("chest",)):
    """(images, qpos, qvel, effort, action, timestamp_ns), as append takes."""
    images = {camera: frames(i, 100 * k) for k, camera in enumerate(cameras)}
    joints = np.full(16, i / 10.0, np.float32)
    return images, joints, joints * 2, joints * 3, joints * 4, 1_000_000_000 + i * 20_000_000


def written(writer, count):
    for i in range(count):
        writer.append(*sample(i))
    return writer.finalize()


def check_final(path, count):
    with h5py.File(path, "r") as f:
        assert f.attrs["complete"]
        assert f.attrs["samples"] == count
        assert f.attrs["duration_seconds"] == pytest.approx((count - 1) * 0.02)
        rgb = f["observations/images/chest_rgb"]
        assert rgb.shape == (count, H, W, 3) and rgb.compression == "lzf"
        assert f["observations/images/chest_depth"].compression == "lzf"
        for i in (0, count // 2, count - 1):
            expected = sample(i)
            assert np.array_equal(rgb[i], expected[0]["chest"][0])
            assert np.array_equal(f["observations/images/chest_depth"][i], expected[0]["chest"][1])
            assert np.allclose(f["action"][i], expected[4])
            assert f["timestamp_ns"][i] == expected[5]


def test_writer_keeps_every_sample_past_its_growth_step(tmp_path):
    count = EpisodeWriter.GROW + 7
    writer = EpisodeWriter(tmp_path, "ep", img_height=H, img_width=W)
    partial = written(writer, count)
    with h5py.File(partial, "r") as f:
        assert not f.attrs["complete"]
        assert f["timestamp_ns"].shape == (count,)
        assert f["observations/images/chest_rgb"].compression is None
    final = compress(partial)
    assert final == tmp_path / "ep.hdf5" and not partial.exists()
    check_final(final, count)


def test_background_writer_then_compress(tmp_path):
    writer = BackgroundEpisodeWriter(output_dir=tmp_path, episode_name="bg",
                                     img_height=H, img_width=W)
    partial = written(writer, 60)
    assert writer.sample_count == 60 and partial == writer.partial_path
    writer.compress_in_background().wait(timeout=60)
    check_final(writer.final_path, 60)


def test_background_writer_refuses_an_existing_episode(tmp_path):
    (tmp_path / "taken.hdf5").write_bytes(b"")
    with pytest.raises(FileExistsError):
        BackgroundEpisodeWriter(output_dir=tmp_path, episode_name="taken",
                                img_height=H, img_width=W)


def test_a_killed_recording_is_recovered_up_to_its_last_sample(tmp_path):
    writer = EpisodeWriter(tmp_path, "killed", img_height=H, img_width=W)
    for i in range(30):
        writer.append(*sample(i))
    writer.file.flush()
    writer.file.close()             # no finalize: datasets still GROW long
    final = compress(writer.partial_path)
    check_final(final, 30)


def test_an_armed_writer_opens_its_episode_only_when_told(tmp_path):
    # Forked in standby, before the episode has a name; the file appears
    # only when recording starts.
    writer = BackgroundEpisodeWriter(deferred=True, img_height=H, img_width=W)
    assert not any(tmp_path.iterdir())
    writer.open(tmp_path, "armed")
    assert (tmp_path / "armed.partial.hdf5").exists()
    partial = written(writer, 12)
    with h5py.File(partial, "r") as f:
        assert f["timestamp_ns"].shape == (12,)


def test_an_armed_writer_never_started_leaves_nothing(tmp_path):
    writer = BackgroundEpisodeWriter(deferred=True, img_height=H, img_width=W)
    assert writer.finalize() is None
    assert not writer.process.is_alive()
    assert not any(tmp_path.iterdir())


def test_an_armed_writer_refuses_an_episode_that_exists(tmp_path):
    (tmp_path / "taken.hdf5").write_bytes(b"")
    writer = BackgroundEpisodeWriter(deferred=True, img_height=H, img_width=W)
    with pytest.raises(FileExistsError):
        writer.open(tmp_path, "taken")


def test_commands_are_kept_beside_the_action_and_nan_where_missing(tmp_path):
    writer = EpisodeWriter(tmp_path, "cmd", img_height=H, img_width=W,
                           command_topic="/openarm/joint_commands")
    command = np.arange(16, dtype=np.float32)
    command[7] = np.nan                       # a source without a gripper
    first = sample(0)
    writer.append(*first, command=command)
    writer.append(*sample(1))                 # no command at all
    partial = writer.finalize()
    with h5py.File(partial, "r") as f:
        stored = f["joint_commands"][:]
        assert f.attrs["joint_commands_topic"] == "/openarm/joint_commands"
        assert np.array_equal(stored[0][:7], command[:7]) and np.isnan(stored[0][7])
        assert np.isnan(stored[1]).all()
        assert np.allclose(f["action"][0], first[4])     # action unchanged


# ------------------------------------------------------------ two cameras

def test_two_cameras_each_get_their_datasets(tmp_path):
    cameras = ("chest", "front")
    writer = EpisodeWriter(tmp_path, "two", img_height=H, img_width=W, cameras=cameras)
    for i in range(5):
        writer.append(*sample(i, cameras))
    final = compress(writer.finalize())
    with h5py.File(final, "r") as f:
        images = f["observations/images"]
        assert sorted(images) == ["chest_depth", "chest_rgb", "front_depth", "front_rgb"]
        assert all(images[name].compression == "lzf" for name in images)
        assert np.array_equal(images["front_rgb"][3], frames(3, 100)[0])
        assert json.loads(f.attrs["cameras_json"]) == ["chest", "front"]
        assert json.loads(f.attrs["virtual_cameras_json"]) == []


def test_one_camera_is_the_file_from_before(tmp_path):
    """chest alone: chest_rgb and chest_depth, nothing else."""
    writer = EpisodeWriter(tmp_path, "one", img_height=H, img_width=W)
    writer.append(*sample(0))
    with h5py.File(writer.finalize(), "r") as f:
        assert sorted(f["observations/images"]) == ["chest_depth", "chest_rgb"]


def test_a_frame_repeated_is_sent_once_and_written_every_time(tmp_path):
    """The camera at 25 Hz, the samples at 50: each frame comes twice."""
    cameras = ("chest", "front")
    writer = BackgroundEpisodeWriter(output_dir=tmp_path, episode_name="zoh",
                                     img_height=H, img_width=W, cameras=cameras)
    expected = []
    for i in range(20):
        images, *rest = sample(i // 2, cameras)        # a new frame every other sample
        if i % 2:
            images = held                               # the very same arrays
        held = images
        writer.append(images, *rest)
        expected.append(images)
    partial = writer.finalize()
    with h5py.File(partial, "r") as f:
        for i, images in enumerate(expected):
            for camera in cameras:
                assert np.array_equal(f[f"observations/images/{camera}_rgb"][i], images[camera][0])
                assert np.array_equal(f[f"observations/images/{camera}_depth"][i], images[camera][1])


def test_a_camera_virtual_in_any_sample_is_noted(tmp_path):
    """Known per sample, not at the start: an episode may begin before the
    first frame of a camera has arrived."""
    cameras = ("chest", "front")
    writer = BackgroundEpisodeWriter(deferred=True, img_height=H, img_width=W,
                                     cameras=cameras)
    writer.open(tmp_path, "virt")
    writer.append(*sample(0, cameras))
    writer.append(*sample(1, cameras), virtual=("front",))
    with h5py.File(writer.finalize(), "r") as f:
        assert json.loads(f.attrs["virtual_cameras_json"]) == ["front"]


def test_late_samples_and_gaps_are_counted(tmp_path):
    """A clean 50 Hz episode says 0 and 0; one that filled slots late, or
    jumped one, says how many."""
    writer = EpisodeWriter(tmp_path, "grid", img_height=H, img_width=W)
    images, *rest = sample(0)
    step = 20_000_000
    for k, late in ((0, False), (1, False), (2, True), (3, False), (6, False)):
        joints = rest[:4]
        writer.append(images, *joints, 1_000_000_000 + k * step, late=late)
    with h5py.File(writer.finalize(), "r") as f:
        assert f.attrs["late_samples"] == 1
        assert f.attrs["gaps"] == 1                  # 3 -> 6 skipped two slots
