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

"""The robot as the simulator draws it, for the dashboard's 3D view.

The page's URDF points at COLLADA meshes whose materials are plain black
and white. openarm_mujoco's MJCF splits each of those meshes into OBJ parts,
each with its own material (matte_black, metal_silver), in the same frame
as the URDF visual (a geom's pos is the visual's origin, the scale the
same). So every .dae can be swapped for its parts and the arm looks the way
it does in MuJoCo, and stays that way when the MJCF changes.

    manifest()  {"materials": {name: {rgba, shininess, reflectance}},
                 "meshes": {"link4.dae": [{"file": "arm/link4_0.obj",
                                           "material": "matte_black"}, ...]}}

file is relative to the MJCF's meshes/visual/, which web_server.py serves
under /mujoco/. Kept free of ROS so it can be tested on its own.
"""

import os
import re
import xml.etree.ElementTree as ET

MJCF_NAME = os.path.join("v1", "openarm_bimanual.xml")
# A part's file -> the .dae it is a piece of: link4_0.obj -> link4.dae.
PART = re.compile(r"^(?P<stem>.+)_\d+\.obj$")


def find_mjcf():
    """openarm_mujoco's bimanual MJCF, installed or beside this package."""
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory("openarm_mujoco"), MJCF_NAME))
    except Exception:                   # noqa: BLE001 -- not installed
        pass
    here = os.path.dirname(os.path.realpath(__file__))
    candidates.append(os.path.join(here, "..", "..", "openarm_mujoco", MJCF_NAME))
    for path in candidates:
        if os.path.isfile(path):
            return os.path.realpath(path)
    return None


def parse(mjcf_path):
    """-> (manifest, mesh_root), or raises ValueError."""
    try:
        root = ET.parse(mjcf_path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"cannot read {mjcf_path}: {error}") from error
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_root = os.path.join(os.path.dirname(mjcf_path), meshdir, "visual")

    materials = {}
    for material in root.iter("material"):
        rgba = [float(v) for v in material.get("rgba", "1 1 1 1").split()]
        materials[material.get("name")] = {
            "rgba": rgba,
            "shininess": float(material.get("shininess", 0.5)),
            "reflectance": float(material.get("reflectance", 0.0)),
        }
    files = {}                          # mesh name -> file under visual/
    for mesh in root.iter("mesh"):
        name, file = mesh.get("name"), mesh.get("file", "")
        if file.startswith("visual/"):
            files[name] = file[len("visual/"):]

    meshes = {}
    for geom in root.iter("geom"):
        if geom.get("class") != "visual" or geom.get("type") != "mesh":
            continue
        file = files.get(geom.get("mesh"))
        match = PART.match(os.path.basename(file or ""))
        if not match:
            continue
        dae = match.group("stem") + ".dae"
        parts = meshes.setdefault(dae, [])
        part = {"file": file, "material": geom.get("material", "default_material")}
        # Both arms list the same parts (mirrored by the URDF's scale on the
        # page, by a _left mesh here): keep each once.
        if part["file"] not in (p["file"] for p in parts):
            parts.append(part)
    for parts in meshes.values():
        parts.sort(key=lambda part: part["file"])
    if not meshes:
        raise ValueError(f"{mjcf_path} has no visual mesh geoms")
    used = {part["material"] for parts in meshes.values() for part in parts}
    return ({"materials": {name: materials[name] for name in used
                           if name in materials},
             "meshes": meshes}, os.path.realpath(mesh_root))


def manifest():
    """-> (manifest, mesh_root) from the installed MJCF, or (None, None)."""
    path = find_mjcf()
    if path is None:
        return None, None
    try:
        return parse(path)
    except ValueError:
        return None, None
