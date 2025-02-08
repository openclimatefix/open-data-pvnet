import re
import logging
import xarray as xr
from typing import Dict, List, Optional
from ocf_data_sampler.numpy_sample.nwp import NWPSampleKey
from zarr.storage import ZipStore

logger = logging.getLogger(__name__)

# Variable mapping for Met Office data
METOFFICE_VARIABLE_MAP = {
    "cloud_amount_of_high_cloud": "high_type_cloud_area_fraction",
    "cloud_amount_of_low_cloud": "low_type_cloud_area_fraction",
    "cloud_amount_of_medium_cloud": "medium_type_cloud_area_fraction",
    "cloud_amount_of_total_cloud": "cloud_area_fraction",
    "radiation_flux_in_longwave_downward_at_surface": "surface_downwelling_longwave_flux_in_air",
    "radiation_flux_in_shortwave_total_downward_at_surface": "surface_downwelling_shortwave_flux_in_air",
    "radiation_flux_in_uv_downward_at_surface": "surface_downwelling_ultraviolet_flux_in_air",
    "snow_depth_water_equivalent": "lwe_thickness_of_surface_snow_amount",
    "temperature_at_screen_level": "air_temperature",
    "wind_direction_at_10m": "wind_from_direction",
    "wind_speed_at_10m": "wind_speed",
}


def create_dynamic_variable_mapping(
    zarr_groups: List[str], store: ZipStore, chunks: Optional[Dict], consolidated: bool
) -> Dict[str, str]:
    """
    Dynamically maps variable names from Zarr groups to NWPSampleKey formatted names.

    Args:
        zarr_groups: List of Zarr group names extracted from the dataset.
        store: The Zarr store containing the dataset.
        chunks: Chunking configuration for opening Zarr files.
        consolidated: Whether the Zarr dataset is consolidated.

    Returns:
        Dictionary mapping internal dataset variables to NWPSampleKey formatted names.
    """
    from open_data_pvnet.utils.data_downloader import open_zarr_group

    variable_mapping = {}

    for group in zarr_groups:
        match = re.search(r"PT\d+H\d+M-(.*).zarr", group)
        if match:
            file_var_name = match.group(1)
            target_var_name = METOFFICE_VARIABLE_MAP.get(file_var_name, file_var_name)

            try:
                group_ds = open_zarr_group(store, group, chunks, consolidated)
                for var in group_ds.variables:
                    if target_var_name in var:
                        variable_mapping[var] = f"{NWPSampleKey.nwp}.{target_var_name}"
                        break
                else:
                    logger.warning(f"No match found for '{file_var_name}' in {group}")

            except Exception as e:
                logger.error(f"Could not open group {group}: {e}")

    return variable_mapping


def prepare_nwp_dataset_for_ocf(
    ds: xr.Dataset,
    zarr_groups: List[str],
    store: ZipStore,
    chunks: Optional[Dict] = None,
    consolidated: bool = True,
) -> xr.Dataset:
    """
    Prepares the merged NWP dataset for use with ocf-data-sampler.

    Args:
        ds: The merged xarray dataset containing NWP data.
        zarr_groups: List of Zarr group names extracted from the dataset.
        store: The Zarr store containing the dataset.
        chunks: Chunking configuration for opening Zarr files.
        consolidated: Whether the Zarr dataset is consolidated.

    Returns:
        The transformed dataset compatible with ocf-data-sampler.

    Raises:
        ValueError: If required coordinates are missing.
    """
    variable_mapping = create_dynamic_variable_mapping(zarr_groups, store, chunks, consolidated)
    ds = ds.rename(variable_mapping)

    required_coords = ["projection_x_coordinate", "projection_y_coordinate", "time"]
    missing_coords = [coord for coord in required_coords if coord not in ds.coords]

    if missing_coords:
        raise ValueError(f"Missing required coordinates: {', '.join(missing_coords)}")

    return ds
