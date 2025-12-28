# Pull Request

## Description

Extends PVNet to support the United States by adding data ingestion for U.S. solar generation (EIA API) and GFS weather data processing. Enables training/validation for U.S. regions using the same CLI as UK.

**Key Changes:**
- **EIA Data Ingestion**: `fetch_eia_data.py` and `collect_eia_data.py` to fetch hourly solar generation by Balancing Authority (7 major ISOs: CAISO, ERCOT, PJM, MISO, NYISO, ISO-NE, SPP)
- **GFS Processing**: Complete pipeline to download GFS GRIB2 from NOAA S3, convert to Zarr with channel filtering, supports `--region us` and `--region global`
- **US Config**: Added `gfs_us_data_config.yaml` for US-specific GFS settings
- **CLI Integration**: Extended GFS provider with `--region` flag (defaults to "global" for backward compatibility)

**Fixes:**
- Fixed incomplete global region handling (removed `pass`, unified processing)
- Implemented channel filtering from config
- Improved error handling and config validation
- Code cleanup and file management improvements

## Fixes #

Fixes #103

## How Has This Been Tested?

- **Unit tests**: Added `test_eia_fetcher.py` and `test_collect_eia.py` covering API client, data collection, pagination, and error handling
- **Integration**: Verified GFS download from NOAA S3, GRIB→Zarr conversion, CLI `--region us` flag, and backward compatibility
- **Code quality**: Formatted with `black`, linted with `ruff`, Google-style docstrings

- [x] Yes, I have tested this code
- [x] Yes, I have tested plotting changes (if data processing is affected)

## Checklist

- [x] My code follows OCF's coding style guidelines ([coding_style.md](https://github.com/openclimatefix/.github/blob/main/coding_style.md))
- [x] I have performed a self-review of my own code
- [x] I have made corresponding changes to the documentation
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] I have checked my code and corrected any misspellings

