# Branch Comparison Analysis: main vs usa

## Executive Summary

This document compares the `main` and `usa` branches to assess whether the USA branch achieves the stated purpose of extending PVNet to the United States, and evaluates if all changes are necessary.

## Purpose Statement (from requirements)

**Goal**: Add U.S. geography support to PVNet for training/evaluation and inference for U.S. regions.

**Scope includes**:
1. ✅ Ingesting historical U.S. solar generation time series for training/validation
2. ✅ Aligning those series with corresponding GFS features
3. ⚠️ Defining geographic units for inference (nationwide, BA, ISO/RTO, or state level)
4. ⚠️ Training and validating PVNet on U.S. data; reporting performance by region and season
5. ✅ Packaging configuration so U.S. runs can be triggered via the same CLI/infra as the UK

## Files Changed

### New Files Added (3)
1. `src/open_data_pvnet/configs/gfs_us_data_config.yaml` - US-specific GFS configuration
2. `src/open_data_pvnet/scripts/fetch_eia_data.py` - EIA API client for fetching solar generation data
3. `src/open_data_pvnet/scripts/collect_eia_data.py` - Script to collect and store EIA data in zarr/netcdf format

### Modified Files (3)
1. `src/open_data_pvnet/main.py` - Added `--region` argument support for GFS provider
2. `src/open_data_pvnet/nwp/gfs.py` - Implemented GFS data fetching and processing (was previously NotImplementedError)
3. `src/open_data_pvnet/scripts/archive.py` - Updated to pass region parameter to process_gfs_data

## Detailed Analysis

### ✅ Achievements

#### 1. EIA Data Ingestion (REQUIREMENT MET)
- **`fetch_eia_data.py`**: Complete implementation of EIA API client
  - Fetches hourly solar generation data by Balancing Authority (BA)
  - Supports pagination for large datasets
  - Handles multiple BA codes (CISO, ERCO, PJM, MISO, NYIS, ISNE, SWPP)
  - Returns structured pandas DataFrame with timestamps, BA codes, and generation values

- **`collect_eia_data.py`**: Data collection and storage script
  - Fetches EIA data for specified date ranges and BA codes
  - Adds approximate latitude/longitude centroids for each BA
  - Converts to xarray Dataset and saves as NetCDF or Zarr
  - Output path: `src/open_data_pvnet/data/target_eia_data.zarr` (matches PVNet config)

**Assessment**: ✅ **NECESSARY** - Core requirement for US solar generation data ingestion.

#### 2. GFS Weather Data Processing (REQUIREMENT MET)
- **`gfs_us_data_config.yaml`**: US-specific configuration
  - Defines S3 bucket for NOAA GFS data (`noaa-gfs-bdp-pds`)
  - Specifies local output directory (`tmp/gfs/us`)
  - Same channel list as global config (14 channels: dlwrf, dswrf, hcc, lcc, mcc, prate, r, t, tcc, u10, u100, v10, v100, vis)
  - 3-hour resolution, 6 forecast steps (18 hours total)

- **`gfs.py` implementation**:
  - `fetch_gfs_data()`: Downloads GFS GRIB2 files from NOAA S3 bucket
  - `convert_grib_to_zarr()`: Converts GRIB files to Zarr format
  - `process_gfs_data()`: Main processing function with region support

**Assessment**: ✅ **NECESSARY** - Required for aligning EIA targets with GFS weather features.

#### 3. CLI Integration (REQUIREMENT MET)
- **`main.py`**: Added `--region` argument for GFS provider
  - Choices: `["global", "us"]`
  - Default: `"global"` (maintains backward compatibility)
  - Integrated into archive operation

- **`archive.py`**: Updated to pass region to `process_gfs_data()`

**Assessment**: ✅ **NECESSARY** - Enables US runs via same CLI as UK (core requirement).

### ⚠️ Issues and Concerns

#### 1. Incomplete GFS Processing Implementation

**Location**: `src/open_data_pvnet/nwp/gfs.py`

**Issues**:
- Line 120-122: Global region logic is incomplete (just `pass`)
  ```python
  else:
      # Existing global logic?
      pass
  ```
  - This means `--region global` will not work properly
  - **Impact**: Breaks existing functionality for global GFS processing

**Assessment**: ⚠️ **INCOMPLETE** - Needs to be fixed or the global path should be handled differently.

#### 2. GRIB to Zarr Conversion Issues

**Location**: `src/open_data_pvnet/nwp/gfs.py`, `convert_grib_to_zarr()` function

**Issues**:
- Lines 46-74: Extensive commented-out code explaining GRIB file structure
- Line 46: `needed_channels` is defined but never used for filtering
- Line 86: Concatenation uses `dim="step"` but comment suggests uncertainty about `valid_time`
- No channel filtering/mapping implemented (channels list in config is ignored)
- No error handling for missing channels

**Assessment**: ⚠️ **INCOMPLETE** - Function works but doesn't fully utilize configuration. May need refinement.

#### 3. Geographic Units Definition

**Current State**: 
- EIA data is collected at **Balancing Authority (BA)** level
- 7 major ISOs/RTOs are supported: CISO, ERCO, PJM, MISO, NYIS, ISNE, SWPP
- Approximate centroids are hardcoded in `collect_eia_data.py`

**Requirement**: "Defining geographic units for inference (nationwide, BA, ISO/RTO, or state level, whichever is best supported by data)"

**Assessment**: ⚠️ **PARTIALLY MET** - BA level is implemented, but:
- No nationwide aggregation
- No state-level support
- No explicit ISO/RTO grouping (though BAs map to ISOs)
- Hardcoded centroids may not be accurate

**Recommendation**: Document that BA level is chosen as it's the most granular and internally consistent from EIA API.

#### 4. Training/Validation Integration

**Current State**:
- EIA data collection script exists
- GFS data processing exists
- PVNet configuration exists (`us_configuration.yaml`)

**Missing**:
- No explicit training scripts or validation code in this branch
- No performance reporting by region/season
- No alignment verification between EIA timestamps and GFS timestamps

**Assessment**: ⚠️ **NOT FULLY ADDRESSED** - Data ingestion is ready, but training/validation integration is not in this branch. This may be intentional if it's handled in PVNet core codebase.

#### 5. Unused Code/Comments

**Location**: `src/open_data_pvnet/nwp/gfs.py`

- Lines 49-58: Extensive commented-out reasoning about GRIB file structure
- Line 117: Commented cleanup code `# shutil.rmtree(files[0].parent)`

**Assessment**: ⚠️ **MINOR** - Should be cleaned up or converted to proper documentation.

#### 6. Missing Error Handling

**Location**: `src/open_data_pvnet/nwp/gfs.py`, `convert_grib_to_zarr()`

- No validation that required channels exist in GRIB files
- No handling for empty datasets after filtering
- No validation of output Zarr structure

**Assessment**: ⚠️ **SHOULD BE IMPROVED** - Error handling would make debugging easier.

### ✅ Necessary Changes Summary

All changes appear necessary for the stated purpose:

1. **EIA data scripts** - ✅ Required for US solar generation data
2. **US GFS config** - ✅ Required for US-specific GFS processing
3. **GFS processing implementation** - ✅ Required (was NotImplementedError)
4. **CLI region support** - ✅ Required for triggering US runs
5. **Archive script update** - ✅ Required to pass region parameter

## Recommendations

### Critical Fixes Needed

1. **Fix global region handling** in `process_gfs_data()`:
   - Either implement global logic or raise NotImplementedError with clear message
   - Or route global to existing implementation if it exists elsewhere

2. **Complete GRIB conversion**:
   - Implement channel filtering based on config
   - Add proper dimension handling (step vs valid_time)
   - Add error handling and validation

### Nice-to-Have Improvements

1. **Documentation**:
   - Add docstrings explaining BA-level choice
   - Document EIA API usage and rate limits
   - Document GFS S3 bucket structure

2. **Code cleanup**:
   - Remove commented-out code in `convert_grib_to_zarr()`
   - Add proper error messages
   - Add logging for missing channels

3. **Testing**:
   - Add tests for EIA data fetching (tests exist: `test_eia_fetcher.py`, `test_collect_eia.py`)
   - Add tests for GFS processing
   - Add integration tests

4. **Geographic units**:
   - Consider making BA centroids configurable
   - Document why BA level was chosen
   - Consider adding aggregation functions for nationwide/state level if needed later

## Conclusion

### Overall Assessment: ✅ **MOSTLY ACHIEVES PURPOSE**

**Strengths**:
- Core data ingestion (EIA + GFS) is implemented
- CLI integration enables US runs via same infrastructure
- Configuration is properly structured
- Code follows existing patterns

**Gaps**:
- Incomplete global region handling (may break existing functionality)
- GRIB conversion needs refinement
- Training/validation integration not in this branch (may be intentional)
- Geographic units documentation needed

**Verdict**: The changes are **absolutely necessary** for the stated purpose, but some **incomplete implementations** need to be fixed before merging. The branch successfully lays the foundation for US support, but requires completion of the GFS processing logic and proper handling of the global region case.

