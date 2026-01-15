# Training a PVNet model for a new country

This document describes the end-to-end steps to add a **new country** to the pipeline: acquire PV generation data, acquire NWP (start with GFS), build configs, train, evaluate, and (optionally) publish weights.

## What you need before you start

- A target country (preferably with ISO code and a clear geographic boundary/bounding box).
- A PV generation dataset (the more history the better; ~5 years is ideal).
- A plan for NWP data (start with GFS; other sources can be added later).
- A place to store/share data (e.g., S3) so others can reproduce and train.

## High-level steps

1. Choose a country (from the project’s supported/desired country list). **TODO:** link to the list.
2. Search online to confirm there is an accessible PV generation data source (and its license/terms).
3. Download PV generation data (more is better; ~5 years ideal).
4. Upload PV data to shared storage (e.g., S3) so others can reuse it; update docs with the data location.
5. Gather weather (NWP) data:
   - Prefer reading from shared storage if it already exists, or
   - Download yourself (start with GFS via `nwp-consumer`; easiest but not necessarily best).
6. Trim NWP data down to the country region (plus a buffer).
7. Create `ocf-data-sampler` and PVNet configs like [here](https://github.com/openclimatefix/open-data-pvnet/tree/main/src/open_data_pvnet/configs/PVNet_configs) for the new country.
8. Train the model.
9. Evaluate the model.
10. (Optional) Store model weights publicly and document how to use them.

## Step 1–4: PV generation data

### Minimum requirements

- Data must be convertible into the **generation Zarr schema** described in the appendix.
- Timestamps must be in **UTC**.
- Units:
  - `generation_mw` in **MW**
  - `capacity_mwp` in **MW peak**
- Each site/region must have stable coordinates (`latitude`, `longitude`) and a stable `location_id`.

### Recommendations (strongly suggested)

- Prefer higher temporal resolution (e.g., 15-min / 30-min) if available.
- Include a consistent site list over time where possible.
- Keep raw source files somewhere (even if you transform into Zarr), for audit/reproducibility.
- Record data license/attribution and any restrictions.

### Uploading / sharing the dataset

Upload the prepared Zarr (and any metadata) to shared storage (e.g., S3). Then update this repo docs to include:

- Country
- Data source + license/terms
- Time coverage
- Temporal resolution
- Storage location (e.g., S3 path)

Once the model has been locally trained and tested with the dataset, you can contact [Peter](https://github.com/peterdudfield).
**Note**: Please only reach out to upload dataset once the model training pipelines run with the dataset without any errors.

## Step 5–6: NWP data (start with GFS)

### Options

- **Preferred (if available):** use existing NWP data already uploaded/shared for this country.
- **Otherwise:** download using `nwp-consumer` (GFS is easiest to start with).

### Trimming

After acquiring NWP, trim to:

- Country bounding box + buffer (to capture weather systems just outside borders).
- The same time range (or at least overlapping) as the PV generation data.

**TODO:** document the exact trimming script/command used in this repo. [this needs to be opened as an issue to add a common cli script]

## Step 7: Configs

You will need [PVNet training config](<(https://github.com/openclimatefix/open-data-pvnet/tree/main/src/open_data_pvnet/configs/PVNet_configs)>) (model + dataset +training hyperparams) for the new country.
**Note** : The above is old configs, new configs to be updated soon.

Checklist:

- [ ] Country code/name matches naming convention used in the repo
- [ ] Generation data paths correct
- [ ] NWP paths correct
- [ ] Time ranges overlap
- [ ] Locations count and coordinates look sane
- [ ] Splits are deterministic and documented

## Step 8: Train

Run training using the new country configs.

You can use [`run.py`](https://github.com/openclimatefix/open-data-pvnet/blob/main/run.py) to run the model by updating the configs.

## Step 9: Evaluate

With the default configs, validation plots would be automatically reported to WANDB, please share your results in github.

## Step 10: Publish weights (optional)

If you publish weights:

- Provide a stable download location
- Record the exact training dataset versions (generation + NWP)
- Record the commit hash of code used
- Add a short “how to run inference” pointer (or link)

---

# Appendix A: Generation data format (required)

Generation data schema: a **Zarr file** with the following data variables and dimensions/coordinates.

Dimensions:

- `(time_utc, location_id)`

Data variables:

- `generation_mw (time_utc, location_id)`: `float32` representing the generation in MW
- `capacity_mwp (time_utc, location_id)`: `float32` representing the capacity in MW peak

Coordinates:

- `time_utc (time_utc)`: `datetime64[ns]` representing the time in UTC
- `location_id (location_id)`: `int` representing the location IDs
- `longitude (location_id)`: `float` representing the longitudes of the locations
- `latitude (location_id)`: `float` representing the latitudes of the locations

Notes:

- Missing generation values should be encoded as `NaN`.
- If capacity is constant per location, it can be repeated along `time_utc` (still shaped `(time_utc, location_id)`).
- Ensure `time_utc` is monotonic increasing and has no duplicates.

---

# Appendix B: Questions to answer for each new country (fill these in)

Country:

- Name:
- ISO code:
- Bounding box (lat/lon min/max):
- Target sites/regions count:

PV generation data:

- Data source + link:
- Time coverage:
- Temporal resolution:
- What is a “location” (plant? substation? region aggregate?):
- Capacity definition (nameplate? AC/DC? time-varying?):
- Known issues (missing periods, curtailment, outages, daylight-saving artifacts, etc.):

NWP:

- Source (GFS/other):
- Variables/levels used:
- Time coverage:
- Spatial resolution:
- Storage location (if shared):

**NOTE**: Please do not try to attempt in one PR, we recommend to open PR at each step and contact us (@siddharth7113 or @peterdudfield) for help.
