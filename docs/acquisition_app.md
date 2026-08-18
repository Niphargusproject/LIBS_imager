# The acquisition application

The Fused LIBS acquisition application runs the instrument: it talks to the laser,
the motion controller, the pulse controller, the six spectrometers and the camera,
drives the scan, and writes the hyperspectral cube. Its files sit at the top level of
this repository.

| File | Role |
| --- | --- |
| `app_init.py` | start-up: paths, file checks, logging. This is the entry point |
| `ui_main_window.py`, `mapping_gui.py` | main window and mapping panels (PyQt5) |
| `config.py`, `config.json` | typed configuration and the saved settings of the instrument |
| `device_handlers.py` | discovery and bookkeeping of spectrometers, serial devices and camera |
| `hardware_control.py` | laser (Telnet), motion controller and pulse controller (serial) commands |
| `mapping_engine.py` | the scan loop: arming, line moves, callback collection, NetCDF4 cube writer, progress previews, fault recovery |
| `data_processing.py` | dark correction, wavelength concatenation, band sums |
| `avaspec.py`, `avaspec_fix_winfunctype.py` | ctypes wrapper for the Avantes AvaSpec DLL, see `LICENSING.md` |
| `test_full_hardware.py` | end-to-end check of laser, motion, pulse controller and spectrometers |
| `test_spectrometer_stress.py` | spectrometer-only soak test, no laser and no motion |
| `help.html` | in-application manual: parameters, file layout, fault recovery |
| `launch_LIBS.bat` | Windows launcher |
| `Bruniquel_LIBS_improved.py` | earlier single-file version of the application, kept for reference |
| `mapping_flowchart.md`, `LIBS_mapping_paper.md` | development notes on the scan loop and the mapping logic |

## Requirements

Windows, because the Avantes and Basler drivers are Windows-only. Python 3.12 and
the packages in `requirements.txt`. The proprietary `avaspecx64.dll` is **not**
included: copy it from your Avantes installation next to `avaspec.py`. `pypylon` is
only needed for the camera view. Long maps need memory for the growing cube and its
previews, so 16 GB of RAM is comfortable.

## Running

Connect the devices, check the serial ports and the laser IP address in
`config.json`, then:

```
python app_init.py
```

or double-click `launch_LIBS.bat`. Section 6 of the article gives the operating
procedure. Output is one NetCDF4 file per map, `{stem}_{timestamp}.nc`, plus progress
PNGs and a pulse-count log, described in `docs/cube_format.md`.

Two habits are worth keeping. Run `test_full_hardware.py` after any rewiring, before
putting a sample in. And read the pulse count that the application compares with the
expected number of shots at the end of every line: it is what tells you that no
trigger was missed.
