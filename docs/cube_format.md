# Cube format

Both applications share one file layout: a compressed NetCDF4 (HDF5) file, one file
per map, named `{stem}_{timestamp}.nc`. Section 2.4 of the article describes it in
more detail.

## Contents

* `mapping(bands, y, x)` — `uint16` counts, dark-corrected during acquisition, one
  full spectrum per pixel. Darks are subtracted while scanning and are not stored;
* `bands(bands)` — the wavelength axis in nm, with the six spectrometers concatenated
  in ascending order and without gaps;
* global attributes — the acquisition parameters: `step_size_mm`, `steps_per_mm`,
  `step_trigger_steps`, `laser_frequency_hz`, `x_feed_mm_min`, `x_travel_mm`,
  `integration_time_ms`, `integration_delay_ns`, `averages`, `mapping_X_points`,
  `mapping_Y_points`, `mapping_X_size_mm`, `mapping_Y_size_mm`, `bands_total`,
  `devices_total` and `created_utc`;
* one subgroup per spectrometer under `devices` (`device_000` … `device_005`), each
  with its serial number, firmware and DLL versions, first and last wavelength and
  pixel count.

Two attributes show how the shot grid was set. `step_trigger_steps` is the division
factor of the pulse controller, so the shot pitch is
`step_trigger_steps / steps_per_mm` millimetres. And `x_travel_mm` is larger than the
mapped width, 62 mm for a 60 mm map, the extra 2 mm being the over-travel where the
stage accelerates and decelerates outside the sample.

## Reading a cube

```python
import xarray as xr

cube = xr.open_dataset("map_20260217_1032.nc")
print(cube.attrs["step_size_mm"], cube.sizes)

fe = cube["mapping"].sel(bands=404.581, method="nearest")   # Fe I 404.581 nm
fe.plot(robust=True)
```

Cubes are chunked along the spectral axis (256 bands × 1 row × 64 columns), which
makes a single wavelength plane cheap to read and the full cube expensive: a
megapixel map is tens of gigabytes uncompressed. Slice, then compute.

For anything beyond a quick look, use the
[LIBS Hypercube Explorer](https://github.com/Niphargusproject/LIBS_hypercube_explorer):
it does peak isolation
with baseline removal, masking, normalization, element ratios, RGB composites,
k-means clustering and co-registration with a photograph of the sample.

## Alongside the cube

* progress PNGs, one per finished line, showing a chosen band while the map grows;
* a pulse-count log, the number of triggers the controller reported against the
  number of shots expected per line. A mismatch is the signature of the
  synchronization problem shown in Fig. 11 of the article.
