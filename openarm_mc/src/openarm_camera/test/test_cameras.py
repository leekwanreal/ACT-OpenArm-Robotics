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

"""cameras.yaml, as setup_camera.sh writes it, and the cameras it starts."""

import pytest

from openarm_camera import cameras


def write(tmp_path, text):
    path = tmp_path / "cameras.yaml"
    path.write_text(text)
    return path


TWO = '''# written by setup_camera.sh
cameras:
  chest:
    serial: "123456789012"
  front:
    serial: "034567890123"
'''


def test_topics():
    assert cameras.rgb_topic("front") == "/openarm/camera/front/rgb"
    assert cameras.depth_topic("chest", "/x/") == "/x/chest/depth"
    assert cameras.raw_topics("front") == (
        "/camera/front/color/image_raw",
        "/camera/front/aligned_depth_to_color/image_raw")


def test_names_come_back_in_role_order():
    assert cameras.parse_names("front,chest") == ("chest", "front")
    assert cameras.parse_names(" front ") == ("front",)
    assert cameras.parse_names("") == ()
    with pytest.raises(ValueError, match="unknown camera 'side'"):
        cameras.parse_names("chest,side")
    with pytest.raises(ValueError, match="twice"):
        cameras.parse_names("chest,chest")


CHEST, FRONT = "123456789012", "034567890123"


def test_two_cameras_by_serial(tmp_path):
    path = write(tmp_path, TWO)
    assert cameras.load(path) == {"chest": {"serial": CHEST},
                                  "front": {"serial": FRONT}}
    both = [CHEST, FRONT]
    assert cameras.plan("auto", path=path, plugged=both) == [
        ("chest", CHEST, "realsense"), ("front", FRONT, "realsense")]
    assert cameras.plan("front", path=path, plugged=both) == [
        ("front", FRONT, "realsense")]


def test_a_camera_not_plugged_in_is_virtual(tmp_path):
    path = write(tmp_path, TWO)
    assert cameras.plan("auto", path=path, plugged=[CHEST]) == [
        ("chest", CHEST, "realsense"), ("front", FRONT, "virtual")]
    # none: only the real ones; all: none of them real.
    assert cameras.plan("auto", "none", path, plugged=[CHEST]) == [
        ("chest", CHEST, "realsense")]
    assert [s for _, _, s in cameras.plan("auto", "all", path, plugged=[CHEST, FRONT])] \
        == ["virtual", "virtual"]
    assert cameras.plan("auto", "front", path, plugged=[CHEST, FRONT])[1][2] == "virtual"
    # Nothing could be listed (no rs-enumerate-devices): every camera virtual.
    assert [s for _, _, s in cameras.plan("auto", path=path, plugged=None)] \
        == ["virtual", "virtual"]


def test_an_unquoted_serial_stays_text(tmp_path):
    path = write(tmp_path, "cameras:\n  chest:\n    serial: 123456789012\n")
    assert cameras.load(path)["chest"]["serial"] == "123456789012"


def test_no_file_is_the_one_camera_setup(tmp_path):
    """One RealSense and no cameras.yaml: it is chest, by its serial."""
    missing = tmp_path / "none.yaml"
    assert cameras.load(missing) == {}
    assert cameras.plan("auto", path=missing, plugged=[CHEST]) == [
        ("chest", CHEST, "realsense"), ("front", "", "virtual")]
    assert cameras.plan("auto", path=missing, plugged=[]) == [
        ("chest", "", "virtual"), ("front", "", "virtual")]


def test_two_cameras_without_serials_are_not_guessed(tmp_path):
    """Which is chest? Neither is opened: run setup_camera.sh."""
    plan = cameras.plan("auto", path=tmp_path / "none.yaml", plugged=[CHEST, FRONT])
    assert [source for _, _, source in plan] == ["virtual", "virtual"]


def test_a_lone_role_takes_the_lone_new_camera(tmp_path):
    path = write(tmp_path, 'cameras:\n  chest:\n    serial: "1"\n')
    assert cameras.plan("front", path=path, plugged=["1", FRONT]) == [
        ("front", FRONT, "realsense")]


# rs-enumerate-devices -S, as v2.58.4 prints it.
LISTING = """Device Name                   Serial Number       Firmware Version
Intel RealSense D435          123456789012        5.16.0.1
Intel RealSense D435          034567890123        5.16.0.1
Device info: 
    Name                          : \tIntel RealSense D435
    Serial Number                 : \t123456789012
    Physical Port                 : \t/sys/devices/pci0000:00/0000:00:14.0/usb2/2-1/2-1:1.0/video4linux/video0
    Usb Type Descriptor           : \t3.2

Device info: 
    Name                          : \tIntel RealSense D435
    Serial Number                 : \t034567890123
    Usb Type Descriptor           : \t3.2

"""


def test_the_device_listing_is_read():
    assert cameras.parse_devices(LISTING) == [CHEST, FRONT]
    assert cameras.parse_devices("No device detected. Is it plugged in?\n") == []


def test_no_listing_tool_is_none(monkeypatch):
    monkeypatch.setenv("OPENARM_RS_ENUMERATE", "/nonexistent/rs-enumerate-devices")
    assert cameras.plugged_serials() is None


def test_a_virtual_frame_is_known_by_its_frame_id():
    assert cameras.is_virtual("front_virtual")
    assert not cameras.is_virtual("camera_color_optical_frame")


def test_a_bad_file_says_what_is_wrong(tmp_path):
    with pytest.raises(ValueError, match="unknown camera 'side'"):
        cameras.load(write(tmp_path, 'cameras:\n  side:\n    serial: "1"\n'))
    with pytest.raises(ValueError, match="needs a serial"):
        cameras.load(write(tmp_path, "cameras:\n  chest: {}\n"))
    with pytest.raises(ValueError, match="same serial"):
        cameras.load(write(tmp_path, 'cameras:\n  chest:\n    serial: "1"\n'
                                     '  front:\n    serial: "1"\n'))
    with pytest.raises(ValueError, match="cameras:"):
        cameras.load(write(tmp_path, "- 1\n"))


def test_the_environment_names_another_file(tmp_path, monkeypatch):
    path = write(tmp_path, TWO)
    monkeypatch.setenv("OPENARM_CAMERAS_FILE", str(path))
    assert cameras.config_path() == path
    assert set(cameras.load()) == {"chest", "front"}


def test_save_writes_what_load_reads(tmp_path):
    path = tmp_path / "cameras.yaml"
    cameras.save({"front": {"serial": FRONT}, "chest": {"serial": CHEST}}, path)
    text = path.read_text()
    assert text.index("chest:") < text.index("front:")      # role order
    assert f'serial: "{FRONT}"' in text                      # leading zero kept
    assert cameras.load(path) == {"chest": {"serial": CHEST}, "front": {"serial": FRONT}}
    cameras.save({}, path)
    assert cameras.load(path) == {}
    with pytest.raises(ValueError, match="same serial"):
        cameras.save({"chest": {"serial": "1"}, "front": {"serial": "1"}}, path)


def test_swap_exchanges_the_roles(tmp_path):
    path = write(tmp_path, TWO)
    assert cameras.swap(path) == {"chest": {"serial": FRONT}, "front": {"serial": CHEST}}
    assert cameras.load(path) == {"chest": {"serial": FRONT}, "front": {"serial": CHEST}}
    cameras.swap(path)
    assert cameras.load(path)["chest"]["serial"] == CHEST     # twice is back
    lone = write(tmp_path, 'cameras:\n  chest:\n    serial: "1"\n')
    assert cameras.swap(lone) == {"front": {"serial": "1"}}
