import os
from glob import glob

from setuptools import setup

package_name = "openarm_ui"


def web_data_files():
    """Install web/ keeping its directory layout, meshes and all."""
    entries = []
    for root, _dirs, files in os.walk("web"):
        if not files:
            continue
        target = os.path.join("share", package_name, root)
        entries.append((target, [os.path.join(root, name) for name in files]))
    return entries


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ] + web_data_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nguyenton",
    maintainer_email="nguyentoan10601@gmail.com",
    description="Web dashboard for OpenArm.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ui_server = openarm_ui.ui_server:main",
        ],
    },
)
