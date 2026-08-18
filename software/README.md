# Software

`acquisition_app/` is the Fused LIBS acquisition application: it talks to the laser,
the motion controller, the pulse controller, the six spectrometers and the camera,
drives the scan, and writes the hyperspectral cube. Licensed GPL-3.0-only, except the
Avantes wrapper and the logos described in `../LICENSING.md`.

The post-processing application is **not** here. It has its own repository,
[LIBS_hypercube_explorer](https://github.com/Niphargusproject/LIBS_hypercube_explorer):
masking, normalization, elemental maps, ratios, RGB composites, k-means clustering and
co-registration with a photograph of the sample.

## Files

| File | Role |
| --- | --- |
| `app_init.py` | start-up: paths, file checks, logging. This is the entry point |
| `ui_main_window.py`, `mapping_gui.py` | main window and mapping panels (PyQt5) |
| `config.py`, `config.json` | typed configuration and the saved settings of the instrument |
| `device_handlers.py` | discovery and bookkeeping of spectrometers, serial devices and camera |
| `hardware_control.py` | laser (Telnet), motion controller and pulse controller (serial) commands |
| `mapping_engine.py` | the scan loop: arming, line moves, callback collection, NetCDF4 cube writer, progress previews, fault recovery |
| `data_processing.py` | dark correction, wavelength concatenation, band sums |
| `avaspec.py`, `avaspec_fix_winfunctype.py` | ctypes wrapper for the Avantes AvaSpec DLL |
| `test_full_hardware.py` | end-to-end check of laser, motion, pulse controller and spectrometers |
| `test_spectrometer_stress.py` | spectrometer-only soak test, no laser and no motion |
| `help.html` | in-application manual: parameters, file layout, fault recovery |
| `Bruniquel_LIBS_improved.py` | earlier single-file version, kept for reference |
| `launch_LIBS.bat` | Windows launcher: activates the conda base environment and starts the single-file version above |
| `mapping_flowchart.md`, `LIBS_mapping_paper.md` | development notes on the scan loop and the mapping logic |

## Requirements

Windows, because the Avantes and Basler drivers are Windows-only. Python 3.12 and the
packages in `acquisition_app/requirements.txt`. The proprietary `avaspecx64.dll` is
**not** included: copy it from your Avantes installation next to `avaspec.py`.
`pypylon` is only needed for the camera view. Long maps need memory for the growing
cube and its previews, so 16 GB of RAM is comfortable.

## Running

Connect the devices, check the serial ports and the laser IP address in `config.json`,
then:

```
cd acquisition_app
python app_init.py
```

Section 6 of the article gives the operating procedure.

Two habits are worth keeping. Run `test_full_hardware.py` after any rewiring, before
putting a sample in. And watch the pulse count that the application compares with the
expected number of shots at the end of every line: that is what tells you no trigger
was missed.

## Cube layout

Output is one compressed NetCDF4 (HDF5) file per map, `{stem}_{timestamp}.nc`, plus
progress PNGs, one per finished line, and a pulse-count log. The file holds:

* `mapping(bands, y, x)` — `uint16` counts, dark-corrected during acquisition, one
  full spectrum per pixel. Darks are subtracted while scanning and are not stored;
* `bands(bands)` — the wavelength axis in nm, the six spectrometers concatenated in
  ascending order and without gaps;
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
stage accelerates and decelerates outside the sample. Section 2.4 of the article
describes the format in more detail.

Reading a plane with xarray:

```python
import xarray as xr

cube = xr.open_dataset("map_20260217_1032.nc")
fe = cube["mapping"].sel(bands=404.581, method="nearest")   # Fe I 404.581 nm
fe.plot(robust=True)
```

Cubes are chunked along the spectral axis (256 bands × 1 row × 64 columns), which
makes a single wavelength plane cheap to read and the whole cube expensive: a
megapixel map is tens of gigabytes uncompressed. Slice, then compute.
