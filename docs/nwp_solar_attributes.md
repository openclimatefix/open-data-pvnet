| Attribute | CF Standard Name | Description | Rationale |
|-----------|------------------|-------------|-----------|
| 1. | `surface_downwelling_shortwave_flux_in_air` | Downward shortwave radiation flux (W/m²) | Core driver of PV output; universally included. |
| 2. | `surface_diffuse_downwelling_shortwave_flux_in_air` | Diffuse shortwave radiation flux (W/m²) | Refines radiation input for cloudy conditions; derivable if not directly available. |
| 3. | `surface_downwelling_longwave_flux_in_air` | Downward longwave radiation flux (W/m²) | Affects panel temperature. |
| 4. | `air_temperature` | 2-meter temperature (K) | Impacts panel efficiency; universally included. |
| 5. | `cloud_area_fraction` | Total cloud cover (%) | Affects radiation; universally included. |
| 6. | `high_type_cloud_area_fraction` | High cloud cover (%) | Scatters radiation; universally included. |
| 7. | `medium_type_cloud_area_fraction` | Medium cloud cover (%) | Reduces radiation; universally included. |
| 8. | `low_type_cloud_area_fraction` | Low cloud cover (%) | Blocks radiation; universally included. |
| 9. | `lwe_thickness_of_snowfall_amount` | Snow depth (m) | Reduces output if panels are covered; universally included. |
| 10. | `precipitation_flux` | Total precipitation rate (kg/m²/s) | Indicates rain/snow and cloudiness. |
| 11. | `eastward_wind` | 10-meter zonal wind (m/s) | Aids cooling and dust removal; universally included. |
| 12. | `northward_wind` | 10-meter meridional wind (m/s) | Complements zonal wind; universally included. |
| 13. | `relative_humidity` | Relative humidity (%) | Affects atmospheric clarity. |
| 14. | `visibility_in_air` | Visibility (m) | Indicates fog/haze. |
| 15. | `atmosphere_optical_thickness_due_to_aerosol` | Aerosol optical depth (dimensionless) | Reduces radiation in dusty/polluted regions; enhances clear-sky forecasts. |
| 16. | `surface_albedo` | Surface albedo (dimensionless) | Affects reflected radiation for bifacial panels; relevant in snowy areas. |