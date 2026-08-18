# Licensing

This repository mixes hardware, code, documentation and a few third-party files, so
one license would not fit. Each file has exactly one license, declared in
machine-readable form in `REUSE.toml` following the
[REUSE 3.3 specification](https://reuse.software/spec/). The full text of every
license is in `LICENSES/`, named after its SPDX identifier.

The root `LICENSE` file holds the GPL, because most files in the repository are
code and because GitHub shows only one license per repository. It does not override
the per-file declarations below.

## What is under which license

| Files | SPDX identifier | Why |
| --- | --- | --- |
| `cad/**` | `CERN-OHL-W-2.0` | CERN Open Hardware License, weakly reciprocal: the usual choice for open hardware, and the one recommended for HardwareX design files |
| `*.py`, `firmware/**/*.ino`, `hypercube_explorer/*.py`, `launch_LIBS.bat`, `config.json`, `requirements.txt` | `GPL-3.0-only` | the license the applications were released under |
| `*.md`, `docs/**`, `*.html`, `bill_of_materials.csv`, `firmware/Grbl_ESP32_configuration.txt`, help figures | `CC-BY-4.0` | documentation and tabular data, reusable with attribution |
| `images/*.svg` | `MIT` | Feather icon set, © 2013–2017 Cole Bemis |
| `avaspec.py`, `avaspec_fix_winfunctype.py` | `LicenseRef-Avantes-SDK` | wrapper for a proprietary library, see below |
| logos, splash screens, `images/icon_aconvert.ico` | `LicenseRef-Logos-Trademarks` | institutional logos are trademarks, see below |

Check a single file with the REUSE tool:

```
pipx run reuse lint
pipx run reuse spdx > sbom.spdx
```

## Two cases that need your attention before you reuse this code

**`avaspec.py` and `avaspec_fix_winfunctype.py`.** These are a ctypes wrapper
around `avaspecx64.dll`, the proprietary Avantes AvaSpec library. The wrapper
follows the structure and the constant names of the Python example distributed with
the Avantes AvaSpec-DLL SDK, so it is best treated as derived from it rather than as
original GPL code. Its use is governed by the terms you accepted with the Avantes
SDK. Neither the DLL nor the Avantes manual is redistributed in this repository:
install the SDK from Avantes and copy `avaspecx64.dll` next to `avaspec.py`. If you
want to reuse the wrapper in another project, clarify the terms with Avantes first.

**Logos and splash screens.** `LOGO-LEAP-RGB-160x90.png`, `splash.png`,
`images/logo.png`, `images/splash.png`, `images/icon_aconvert.ico`,
`hypercube_explorer/assets/newlogo_gsb.png` and
`hypercube_explorer/assets/N-be-logo_complet_png_transparent.png` carry the marks of
the Royal Belgian Institute of Natural Sciences, the Geological Survey of Belgium
and the BELSPO LEAP project. Trademarks are not licensed by CC-BY or the GPL. You
may keep them when you run the applications as they are, but replace them with your
own if you distribute a modified version, so that your build is not presented as
ours.

## Other things not included here

* **Laser control documentation.** The Lumibird/Quantel Viron command set is covered
  by the manufacturer's manual. The Telnet commands actually used are visible in
  `hardware_control.py`.
* **Basler camera runtime.** The optional micro-view camera needs `pypylon` and the
  Basler pylon runtime, installed separately.
* **Hyperspectral cubes.** Full cubes are tens of gigabytes each. One example cube
  is archived on Zenodo under CC-BY-4.0, see `example_data/README.md`; the others
  are available from the corresponding author on request.

## Copyright

Unless a file says otherwise, the copyright is:

```
SPDX-FileCopyrightText: 2021-2026 Royal Belgian Institute of Natural Sciences
                        (Geological Survey of Belgium)
```
