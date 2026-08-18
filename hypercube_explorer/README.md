# LIBS Hypercube Explorer

A desktop application for the interactive visualization, processing, and export of **LIBS (Laser-Induced Breakdown Spectroscopy) hypercubes** stored as NetCDF (`.nc`) files.

Originally developed for **speleothem and geoscience mapping workflows**, it provides band imaging, spectral data extraction, photo co-registration, multi-element composite generation, normalization, interactive masking, baseline inspection, ROI management, and project (experiment) snapshotting — all in a step-by-step tabbed UI.

---

## Table of contents

- [Features](#features)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Input data format](#input-data-format)
- [Project files & experiments](#project-files--experiments)
- [Element line database](#element-line-database)
- [Documentation](#documentation)
---

## Features

- **Step-by-step tabbed workflow** (Info → Photo → Mask → Normalize → Map Explorer → Data Extraction → Composite → Cube Subset → Cube Utils).
- **Hypercube band imaging** with interactive band selection, integration windows, colormaps, and low-signal warnings.
- **Photo co-registration** of a reference photograph onto LIBS coordinates via 4-point homography.
- **Interactive masking** (threshold, paint tools, save/load `.npy` masks) with cube-wide application.
- **Baseline correction** with several local and cube-wide methods (rubber-band, polynomial, asymmetric least squares, …) and a dedicated **Baseline Inspector**.
- **Normalization** with multiple spectral-based methods (total area, internal standard, SNV, …) following published best practices [1, 2, 4].
- **Spectrum / line / peak extraction** along arbitrary lines or pixels drawn on the photo or LIBS map, with peak detection and **peak height vs. peak area** integration.
- **Multi-element composite overlays** (RGB-style blending, per-channel gain/colormap).
- **ROI management** for repeatable spectral integrations.
- **Built-in periodic table widget** with a curated database of common LIBS lines, including critical raw materials (CRM) and rare-earth elements.
- **Project / experiment system** (`.hcxproj`) — snapshot the entire UI state (selections, masks, ROIs, normalization, view, baseline) and switch between experiments.
- **Unified export dialog** for figures, raw 1:1 images, plot data (CSV / XLSX), and cube subsets (NetCDF).
- **NetCDF cube subsetting** to create lighter `.nc` files with only the bands and spatial regions you need.

---

## Workflow

The application is organized as a linear tab sequence that mirrors a typical LIBS analysis pipeline:

1. **Info** — load a NetCDF cube and inspect its metadata.
2. **Photo** — load and co-register a reference photo (4-point homography).
3. **Mask** — define a binary spatial mask (threshold + paint), and apply it to the cube.
4. **Normalize** — choose and apply a spectral normalization method.
5. **Map Explorer** — browse element/band maps interactively.
6. **Data Extraction** — extract spectra / line profiles / peak intensities along drawn shapes.
7. **Composite** — build multi-element RGB-style overlays.
8. **Cube Subset** — export a reduced NetCDF cube.
9. **Cube Utils** — additional cube-wide utilities (e.g. compression).

---

## Installation

### Requirements

| Dependency | Purpose |
|---|---|
| Python 3.8+ | Runtime |
| PyQt5 | GUI framework |
| xarray | NetCDF cube loading and manipulation |
| numpy | Array computation |
| matplotlib | Interactive canvases and plots |
| pandas | CSV/XLSX export of line/spectrum data |
| scipy | Peak detection and spatial median filter |
| Pillow | Raw 1:1 image export |
| netCDF4 *(optional)* | Reading spectrometer device metadata |

### Quick install (pip)

```bash
git clone https://github.com/<your-user>/LIBS_hypercube_explorer.git
cd LIBS_hypercube_explorer
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install pyqt5 xarray numpy matplotlib pandas scipy pillow netCDF4
```

A `requirements.txt` can be generated from your environment with:

```bash
pip freeze > requirements.txt
```

---

## Usage

Launch the application from the project root:

```bash
python Hypercube_explorer.py
```

Then:

1. **File → Open LIBS cube…** and select a NetCDF (`.nc`) file.
2. Step through the tabs in order (1 → 9).
3. Use **Project → Add Experiment (snapshot)…** to save the current UI state, and **Load Experiment…** to restore it later.
4. Use **File → Export…** for figures, plot data, or cube subsets.

A complete user guide is available in **Help → User Guide…** or by opening [`help.html`](help.html) directly in a browser.

---

## Input data format

The application expects **NetCDF (`.nc`)** hypercubes with:

- a `bands` coordinate containing wavelength values **in nanometres**;
- one or more 2-D or 3-D data variables representing spectral intensity maps (i.e. a `(y, x, bands)` cube or per-band 2-D slices).

NetCDF global attributes (e.g. creation date, integration time, laser frequency, step size, mapping dimensions) are surfaced automatically in the **Info** tab.

---

## Project files & experiments

Projects are saved as `.hcxproj` files. A project can contain multiple **experiments**, each one a complete snapshot of the UI state, including:

- selected element line / band / integration window
- normalization method and parameters
- baseline method and parameters
- mask
- ROIs
- composite layer configuration
- view positions (zoom/pan)

Use the **Project** menu to create, open, save, snapshot, rename, and delete experiments.

---

## Element line database

The app ships with a curated database of LIBS emission lines (atomic and ionic) for:

- Common major elements (H, C, N, O, Na, Mg, Al, Si, P, S, Cl, K, Ca, Ti, Mn, Fe, Cu, Zn, Sr, …)
- **Critical Raw Materials (CRM)** (Be, Co, Cr, Ga, Ge, Hf, In, Mo, Nb, Ni, Pb, Pt, Sb, Sc, Sn, Ta, V, W, Y, Zr, …)
- **Rare-earth elements** (Ce, Dy, Er, Eu, Gd, Ho, La, Lu, Nd, Pr, Sm, Tb, Tm, Yb)
- Specialized extended Ca / Mg lines for carbonates / speleothem work

Lines can be selected from a **compact interactive periodic-table widget** in the Map Explorer.

---

## Documentation

A standalone HTML user guide is bundled with the app: [`help.html`](help.html).
It covers each tab in detail, baseline and normalization methods, ROI workflows, projects/experiments, and export options.

---


---

## Acknowledgements

Developed at the **Royal Belgian Institute of Natural Sciences** with the help of Claude Code.
