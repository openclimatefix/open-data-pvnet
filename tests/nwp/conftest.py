"""Conftest for NWP tests.

Pre-mocks ocf_data_sampler submodules that may differ across versions,
ensuring gfs_dataset.py can be imported regardless of the installed
ocf_data_sampler version.
"""

import sys
from unittest.mock import MagicMock

# -- Pre-mock ocf_data_sampler submodules that gfs_dataset.py imports --------
# This must happen BEFORE any test module imports gfs_dataset.

_MODULES_TO_MOCK = [
    "ocf_data_sampler.constants",
    "ocf_data_sampler.torch_datasets",
    "ocf_data_sampler.torch_datasets.utils",
    "ocf_data_sampler.torch_datasets.utils.valid_time_periods",
]

for mod_name in _MODULES_TO_MOCK:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Ensure NWP_MEANS / NWP_STDS are dicts so patch.dict works in tests
sys.modules["ocf_data_sampler.constants"].NWP_MEANS = {}
sys.modules["ocf_data_sampler.constants"].NWP_STDS = {}
