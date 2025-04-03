import tomllib
import open_data_pvnet


def test_version_consistency():
    """
    Verify that the version in pyproject.toml matches the version in __init__.py
    """
    # Read version from pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)
    pyproject_version = pyproject_data["project"]["version"]

    # Read version from the __init__.py file
    init_version = open_data_pvnet.__version__

    # Assert both versions are the same
    assert pyproject_version == init_version, (
        f"Version mismatch: pyproject.toml has {pyproject_version}, "
        f"but __init__.py has {init_version}"
    )
