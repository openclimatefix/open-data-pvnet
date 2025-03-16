"""Test script for the DWD data source implementation."""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys

# Add the project root to the path so we can import the module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.open_data_pvnet.nwp.dwd import get_data_source

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dwd_data_loading():
    """Test loading DWD data."""
    
    # Define test parameters
    base_dir = os.path.join(os.path.dirname(__file__), "../data")
    config_path = os.path.join(os.path.dirname(__file__), "../configs/dwd_data_config.yaml")
    
    # Create sample target coordinates (Berlin, Munich, Hamburg)
    target_coords = pd.DataFrame({
        "site_id": [1, 2, 3],
        "latitude": [52.520008, 48.137154, 53.551086],
        "longitude": [13.404954, 11.576124, 9.993682],
        "name": ["Berlin", "Munich", "Hamburg"]
    })
    
    # Define time parameters
    start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(hours=24)
    
    # Create data source
    dwd_source = get_data_source(base_dir, config_path)
    
    # Load data
    logger.info(f"Loading DWD data for {start_dt} to {end_dt}")
    dataset = dwd_source.load_nwp_data(
        start_dt=start_dt,
        end_dt=end_dt,
        target_coords=target_coords
    )
    
    # Print dataset info
    logger.info("Dataset loaded successfully")
    print(dataset)
    
    # Plot some results if direct and diffuse radiation are available
    if "direct_radiation" in dataset and "diffuse_radiation" in dataset:
        plt.figure(figsize=(12, 6))
        
        for i, location in enumerate(target_coords["name"]):
            plt.subplot(1, 3, i+1)
            
            # Extract data for this location
            direct = dataset["direct_radiation"].isel(location=i)
            diffuse = dataset["diffuse_radiation"].isel(location=i)
            total = direct + diffuse
            
            # Plot
            plt.plot(dataset.time, direct, label="Direct")
            plt.plot(dataset.time, diffuse, label="Diffuse")
            plt.plot(dataset.time, total, label="Total", linestyle="--")
            
            plt.title(location)
            plt.xlabel("Time")
            plt.ylabel("Radiation (W/m²)")
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "dwd_radiation_test.png"))
        plt.show()
    
    # Check if dataset has expected dimensions and variables
    assert "time" in dataset.dims, "Dataset should have a time dimension"
    assert "location" in dataset.dims, "Dataset should have a location dimension"
    
    # Check if dataset has expected coordinates
    assert dataset.dims["location"] == len(target_coords), "Number of locations should match"
    
    logger.info("DWD data test completed successfully")
    return dataset

if __name__ == "__main__":
    # Run the test
    try:
        result = test_dwd_data_loading()
        print("Test completed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise