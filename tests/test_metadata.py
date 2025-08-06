import tomllib
import sys
import os
from importlib import metadata

# Add src directory to path to import open_data_pvnet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import open_data_pvnet


def test_version_consistency():
    """
    Verify that the dynamic version works correctly and matches the version in __init__.py
    """
    # Read the dynamic version configuration from pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)
    
    # Verify that dynamic versioning is configured
    assert "version" in pyproject_data["project"]["dynamic"], (
        "Dynamic versioning should be configured in pyproject.toml"
    )
    
    # Get the version from the installed package metadata
    try:
        package_version = metadata.version("open-data-pvnet")
    except metadata.PackageNotFoundError:
        # If package is not installed, skip this check
        package_version = None

    # Read version from the __init__.py file
    init_version = open_data_pvnet.__version__

    # If package is installed, verify versions match
    if package_version:
        assert package_version == init_version, (
            f"Version mismatch: installed package has {package_version}, "
            f"but __init__.py has {init_version}"
        )
    
    # Verify that setuptools dynamic configuration is correct
    dynamic_config = pyproject_data["tool"]["setuptools"]["dynamic"]
    assert "version" in dynamic_config, (
        "Version should be configured in [tool.setuptools.dynamic]"
    )
    assert dynamic_config["version"]["attr"] == "open_data_pvnet.__version__", (
        "Dynamic version should point to open_data_pvnet.__version__"
    )
