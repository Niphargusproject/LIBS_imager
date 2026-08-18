# LIBS elemental imager

Design files, firmware and software of a laboratory laser-induced breakdown
spectroscopy (LIBS) elemental imager built from low-cost CNC-derived components,
with a fully hardware-triggered acquisition chain.

The instrument moves the sample under a fixed laser and fixed collection optics on
three ball-screw linear stages driven by the open-source Grbl_ESP32 firmware. The
stepper step pulses of the scan axis are counted and divided by an Arduino-based
pulse controller, which triggers the laser, and the laser in turn triggers the
spectrometers. Shot spacing is therefore locked to stage position instead of to
software timing. A Python/PyQt5 application streams dark-corrected,
wavelength-concatenated spectra into a compressed NetCDF4 hyperspectral cube; a
second application reads those cubes and produces elemental maps, ratios and RGB
composites.

Although it is demonstrated here for LIBS, the motion, triggering and data-cube
chain is technique-agnostic: it can drive other point-by-point mapping methods such
as UV fluorescence, FTIR or Raman imaging.

This repository is the reference implementation for the article:

> C. Burlet, S. Bengattat, X. Devleeschouwer, H. Goethals, T. Goovaerts, M. Rossi,
> G. Stasi, A. Tovar, Y. Vanbrabant, S. Verheyden, R. Barros, S. Nachtergaele,
> E. Pirard. *A Laser-Induced Breakdown Spectroscopy (LIBS) elemental imager built
> from CNC-derived components with a fully hardware-triggered acquisition chain.*
> HardwareX (submitted).

This repository is the living version of the instrument and is where development
continues. The frozen state that produced the results in the article, together with one
measured example data cube, is archived separately on Zenodo,
[10.5281/zenodo.21986499](https://doi.org/10.5281/zenodo.21986499).

## Repository layout

| Path | Content |
| --- | --- |
| `cad/` | STEP models of the laser assembly, the mounting and sample-stage adapter plates, the electronics enclosure and the end-stop board |
| `schematics/` | wiring of the trigger chain and of the internal controller |
| `firmware/` | Arduino sketches of the pulse controller and of the internal controller, and the Grbl_ESP32 settings of the motion controller |
| `software/acquisition_app/` | the acquisition application: device control, mapping engine, NetCDF4 cube writer, GUI |
| `example_data/` | how to get the example cube; the cube itself is too large for GitHub and lives on Zenodo |
| `bill_of_materials.csv` | bill of materials of the article, semicolon-delimited UTF-8 |
| `LICENSES/` | full text of every license used here; see [`LICENSING.md`](LICENSING.md) |

## Quick start

**Motion and triggering.** Flash `firmware/pulse_counter_INT0/` to an Arduino Uno
and `firmware/Libs_leds_distance_sensor_controller/` to an Arduino Nano with the
Arduino IDE, then restore `firmware/Grbl_ESP32_configuration.txt` on the MKS DLC32
board and check it with `$$`. Keep the trigger cable disconnected from the laser
while testing. `firmware/README.md` documents the pin assignment and the serial
command sets.

**Acquisition.** Windows, Python 3.12. The proprietary Avantes `avaspecx64.dll` is not
redistributed here: copy it from your Avantes installation next to `avaspec.py`. Set
the serial ports and the laser IP address in `config.json`, then:

```
cd software/acquisition_app
pip install -r requirements.txt
python app_init.py
```

**Post-processing.** The cubes are read by the LIBS Hypercube Explorer, which has its
own repository. It needs no instrument and no proprietary library, so it also runs on
Linux and macOS:

```
git clone https://github.com/Niphargusproject/LIBS_hypercube_explorer.git
cd LIBS_hypercube_explorer
pip install -r requirements.txt
python Hypercube_explorer.py
```

Output is one NetCDF4 file per map, `{stem}_{timestamp}.nc`, plus progress PNGs and a
pulse-count log. [`software/README.md`](software/README.md) describes the file layout,
and [`example_data/README.md`](example_data/README.md) explains how to fetch a real
cube to try the chain on.

## Safety

The instrument uses a Class 4 pulsed Nd:YAG laser. Build and operate it only with
appropriate engineering controls, an interlocked enclosure, eye protection matched
to 1064 nm and a trained laser operator, following IEC 60825-1 or ANSI Z136.1 and
your institution's laser safety rules. Section 8 of the article describes the
practice used here; it is an illustration, not a substitute for your own risk
assessment.

## Licensing

| Part | License |
| --- | --- |
| Hardware, i.e. everything in `cad/` | CERN-OHL-W-2.0 |
| Firmware and software | GPL-3.0-only |
| Documentation, tables and data | CC-BY-4.0 |
| Third-party icons in `software/acquisition_app/images/` | MIT (Feather, Cole Bemis) |
| Institutional logos and splash screens | not covered, see [`LICENSING.md`](LICENSING.md) |
| `avaspec.py`, `avaspec_fix_winfunctype.py` | Avantes SDK terms, see [`LICENSING.md`](LICENSING.md) |

Per-file licensing is machine-readable in `REUSE.toml`, following the
[REUSE](https://reuse.software) specification. The root `LICENSE` file is the GPL,
which covers the bulk of the repository; `LICENSING.md` explains the exceptions.

## Citing

Cite the article for the design and the Zenodo record for the files. GitHub's
"Cite this repository" button reads `CITATION.cff`.

## Related

* Post-processing application: <https://github.com/Niphargusproject/LIBS_hypercube_explorer>
* Motion firmware: Grbl_ESP32 / FluidNC, <https://github.com/bdring/FluidNC>

## Funding and contact

Developed at the Geological Survey of Belgium (Royal Belgian Institute of Natural
Sciences) within the BELSPO BRAIN-be projects LIBSCREEN and LEAP.

Christian Burlet — cburlet@naturalsciences.be
