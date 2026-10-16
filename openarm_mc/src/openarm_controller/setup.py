from glob import glob

from setuptools import setup

package_name = "openarm_controller"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nguyenton",
    maintainer_email="nguyentoan10601@gmail.com",
    description="CAN controller for OpenArm.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "controller = openarm_controller.controller_node:main",
        ],
    },
)
