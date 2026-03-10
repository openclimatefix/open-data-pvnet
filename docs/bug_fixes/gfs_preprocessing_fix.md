# GFS Preprocessing Data Format Fix

## Issues
1. Dimension naming conflict between 'variable' and 'channel'
2. Longitude range mismatch (-180 to 180 vs 0 to 360)
3. Data structure incompatibility with model expectations

## Solution

### 1. Correct Preprocessing Script
```python
import xarray as xr

def preprocess_gfs(year: int):
    gfs = xr.open_mfdataset(f"/mnt/storage_b/nwp/gfs/global/{year}*.zarr.zip", engine="zarr")
    
    # Fix longitude range
    gfs['longitude'] = ((gfs['longitude'] + 360) % 360)
    
    # Select UK region (in 0-360 range)
    gfs = gfs.sel(
        latitude=slice(65, 45),
        longitude=slice(0, 360)
    )
    
    # Stack variables into channel dimension
    gfs = gfs.to_array(dim="channel")  # Use channel instead of variable
    
    # Optimize chunking
    gfs = gfs.chunk({
        'init_time_utc': len(gfs.init_time_utc),
        'step': 10,
        'latitude': 1,
        'longitude': 1
    })
    
    return gfs
```

### 2. Expected Data Structure
- Dimensions: (init_time_utc, step, channel, latitude, longitude)
- Longitude range: [0, 360)
- Single stacked DataArray with channel dimension

### 3. Verification
```python
ds = xr.open_zarr("path/to/gfs.zarr")
assert "channel" in ds.dims
assert 0 <= ds.longitude.min() < ds.longitude.max() <= 360
```