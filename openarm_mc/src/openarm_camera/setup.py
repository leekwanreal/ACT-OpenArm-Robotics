from glob import glob

from setuptools import setup

package_name = "openarm_camera"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nguyenton",
    maintainer_email="nguyentoan10601@gmail.com",
    description="OpenArm's RealSense cameras (chest, front): drivers and RGB-D preprocessors.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rgbd_preprocessor = openarm_camera.rgbd_preprocessor:main",
            "virtual_camera = openarm_camera.virtual_camera:main",
        ],
    },
)
