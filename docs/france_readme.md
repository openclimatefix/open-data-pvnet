## France Solar Data Pipeline for PVNet
This edit/ contribution adds support for France RTE solar generation data to the project.

## Changes
- Added France data processing script
- Created admin region metadata CSV
- Updated data pipeline to use integer location_ids
- Added inspection script for validation

## Data API
The Definitive datasets follow the format:
https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_{Region}_Annuel-Definitif_{Year}.zip

The consolidate datasets follow the format:
https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_{Region}_En-cours-Consolide.zip

Note that TCH (le Taux de CHarge), which refers to the actual production compared to installed solar capacity is only available from 2020. Hence, initially we use 2020 to 2024 (5 years) of data. 

## Summer Time Behavior
When transitioning to summer time (e.g. 26 Mar 2023 2:00 to 03:00), entries between 2:00 and 3:00 are duplicated.
When transitioning back to winter time (e.g. 29 Oct 2023 3:00 to 2:00), data entries are ambiguous and 2 timesteps will be missing.

### ZARR File 
The converted zarr file is available on huggingface, link:
https://huggingface.co/datasets/hhhn2/France_PV_data

### Scripts
| File | Description |
|------|-------------|
| `scripts/download_mendeley_india.py` | Dataset download instructions |
| `scripts/process_india_data.py` | Excel → Zarr conversion |
| `scripts/test_india_pipeline.py` | Pipeline validation tests |
| `scripts/train_india_baseline.py` | Solar-only baseline model |

## Testing
- Ran process_france_data.py successfully
- Validated output with inspect_france_training_pipeline.py

## Data Processing Results
Data Quality

Generation (MW):
  Shape: (12, 87696)
  Range: [0.00, 4002.00] MW
  Mean: 174.98 MW
  NaN count: 120 (0.01%)

Capacity (MWp):
  Shape: (12, 87696)
  Range: [122.70, 6000.00] MWp
  Mean: 1170.15 MWp
  NaN count: 0 (0.00%)

Per-Region Statistics

0:
Generation: [0.0, 2194.0] MW, Mean: 238.2 MW, NaN: 0.0%
Capacity: 1655.3 MWp, NaN: 0.0%

1:
  Generation: [0.0, 883.0] MW, Mean: 78.7 MW, NaN: 0.0%
  Capacity: 537.8 MWp, NaN: 0.0%

2:
  Generation: [0.0, 568.0] MW, Mean: 49.3 MW, NaN: 0.0%
  Capacity: 364.3 MWp, NaN: 0.0%

3:
  Generation: [0.0, 975.0] MW, Mean: 97.5 MW, NaN: 0.0%
  Capacity: 665.7 MWp, NaN: 0.0%

4:
  Generation: [0.0, 1337.0] MW, Mean: 134.2 MW, NaN: 0.0%
  Capacity: 998.7 MWp, NaN: 0.0%

5:
  Generation: [0.0, 629.0] MW, Mean: 49.0 MW, NaN: 0.0%
  Capacity: 361.4 MWp, NaN: 0.0%

6:
  Generation: [0.0, 306.0] MW, Mean: 26.8 MW, NaN: 0.0%
  Capacity: 218.7 MWp, NaN: 0.0%

7:
  Generation: [0.0, 464.0] MW, Mean: 32.1 MW, NaN: 0.0%
  Capacity: 247.4 MWp, NaN: 0.0%

8:
  Generation: [0.0, 4002.0] MW, Mean: 534.9 MW, NaN: 0.0%
  Capacity: 3524.9 MWp, NaN: 0.0%

9:
  Generation: [0.0, 3287.0] MW, Mean: 438.6 MW, NaN: 0.0%
  Capacity: 2799.3 MWp, NaN: 0.0%

10:
  Generation: [0.0, 1213.0] MW, Mean: 118.8 MW, NaN: 0.0%
  Capacity: 860.4 MWp, NaN: 0.0%

11:
  Generation: [0.0, 1942.0] MW, Mean: 301.8 MW, NaN: 0.0%
  Capacity: 1807.9 MWp, NaN: 0.0%

## Next Steps
