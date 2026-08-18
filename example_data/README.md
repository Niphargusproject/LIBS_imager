# Example data

The example cube is **not in this repository**: it is 1.35 GB, well over what GitHub
accepts. It is archived on Zenodo with the rest of the design package, under
CC-BY-4.0.

> `Vedrin_01_0.2mm.nc` — Zenodo DOI to be added on publication

## What it is

An iron ore sample from the Vedrin mine (Belgium), mapped over its full 60 × 40 mm
area at 0.2 mm pitch. It is a complete, unedited output of the acquisition software,
so it doubles as a reference for the file format.

| Property | Value |
| --- | --- |
| Pixels (x × y) | 300 × 200 = 60 000 spectra |
| Mapped area | 60 × 40 mm, shot pitch 0.2 mm |
| Spectral bands | 20 472, from 195.5 to 1031.1 nm, six spectrometers concatenated without gaps |
| Laser | 10 Hz, stage feed 120 mm/min, one shot every 160 step pulses (0.2 mm × 800 steps/mm) |
| Detection | integration time 0.01 ms, delay after trigger 3 µs, 1 accumulation per pixel |
| File size | 1.35 GB compressed, 2.46 GB uncompressed |

`docs/cube_format.md` describes the layout of the file.

## Trying it out

Download the cube from the Zenodo record into this folder, then open it with the
[LIBS Hypercube Explorer](https://github.com/Niphargusproject/LIBS_hypercube_explorer):

```
git clone https://github.com/Niphargusproject/LIBS_hypercube_explorer.git
cd LIBS_hypercube_explorer
python Hypercube_explorer.py
```

Follow the tabs from left to right: load, optionally co-register a photograph, mask
the pixels that are off the sample, normalize, and extract maps. The `help.html` of
that repository explains the options.

In Python, without the application:

```python
import xarray as xr

cube = xr.open_dataset("Vedrin_01_0.2mm.nc")
fe = cube["mapping"].sel(bands=404.581, method="nearest")   # Fe I 404.581 nm
fe.plot(robust=True)
```

The cube is chunked along the spectral axis (256 bands × 1 row × 64 columns), so
reading one wavelength plane takes about 0.2 s. Slice before you compute.

The other maps shown in the article are tens of gigabytes each and are available
from the corresponding author on request.
