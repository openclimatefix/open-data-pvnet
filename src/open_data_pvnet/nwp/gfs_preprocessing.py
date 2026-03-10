import xarray as xr
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def preprocess_gfs_data(input_path: str, output_path: str, year: int) -> None:
    """
    Preprocess GFS data to ensure correct longitude range and dimension structure.
    
    Args:
        input_path: Path to input GFS data
        output_path: Path to save processed data
        year: Year of data to process
    """
    logger.info(f"Starting GFS preprocessing for year {year}")
    
    # Load GFS dataset
    gfs = xr.open_mfdataset(f"{input_path}/{year}*.zarr.zip", engine="zarr")
    logger.info(f"Loaded GFS data with shape: {gfs.dims}")
    
    # Step 1: Fix longitude range to [0, 360)
    gfs['longitude'] = ((gfs['longitude'] + 360) % 360)
    gfs = gfs.sortby('longitude')  # Ensure longitudes are sorted
    
    # Step 2: Select UK region (with buffer)
    gfs = gfs.sel(
        latitude=slice(65, 45),  # Include buffer around UK
        longitude=slice(350, 362)  # Wrap around 360° (350° to 2°)
    )
    
    # Step 3: Ensure data validity
    if len(gfs.latitude) == 0 or len(gfs.longitude) == 0:
        raise ValueError("No data found after selecting UK region")
    
    # Step 4: Stack variables into channel dimension
    gfs = gfs.to_array(dim="channel")
    
    # Step 5: Optimize chunking for performance
    chunk_sizes = {
        'init_time_utc': 1,
        'step': 4,
        'channel': -1,  # Keep all channels together
        'latitude': 1,
        'longitude': 1
    }
    
    # Remove existing chunk encoding
    for var in gfs.variables:
        if 'chunks' in gfs[var].encoding:
            del gfs[var].encoding['chunks']
    
    gfs = gfs.chunk(chunk_sizes)
    
    # Step 6: Save processed data
    output_file = Path(output_path) / f"gfs_uk_{year}.zarr"
    logger.info(f"Saving processed data to {output_file}")
    gfs.to_zarr(output_file)
    
    logger.info(f"Completed processing for {year}")
    logger.info(f"Final dimensions: {gfs.dims}")

def main():
    input_base = "/mnt/storage_b/nwp/gfs/global"
    output_base = "uk"
    
    for year in [2023, 2024]:
        try:
            preprocess_gfs_data(input_base, output_base, year)
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
