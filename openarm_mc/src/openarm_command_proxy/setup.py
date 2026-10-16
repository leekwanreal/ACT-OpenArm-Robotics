from glob import glob

from setuptools import setup

package_name = "openarm_command_proxy"

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
    description="Command source selection and merging for OpenArm.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "command_proxy = openarm_command_proxy.command_proxy:main",
        ],
    },
)
