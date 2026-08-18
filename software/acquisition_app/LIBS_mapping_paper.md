# A Synchronized Hardware-Triggered LIBS Mapping System: Implementation and Data Pipeline

**Technical note / Methods paper**

---

## Abstract

We describe the design and implementation of a laser-induced breakdown spectroscopy (LIBS) mapping system in which shot timing is fully determined by a hardware trigger chain: a CNC stage provides step pulses to a pulse controller, which issues trigger signals to both the laser and multiple Avantes spectrometers. The software arms devices, starts motion, and collects spectra via a callback-based acquisition path, writing a continuous hyperspectral cube to NetCDF4/HDF5 with embedded metadata and optional progress previews [6–8]. The architecture avoids software timing jitter, supports multi-device wavelength concatenation and dark correction, and includes automatic laser fault detection and recovery. The platform is designed as **open-source, open-hardware**: motion control is based on GRBL-compatible firmware on a GRBL32-class controller [14–16], the pulse controller is based on an Arduino Uno [13], and mechanical parts are built from low-cost, scalable gantry components with shared 3D models (e.g. via GitHub) consistent with open science hardware practices [17–21,29]. We document the trigger chain, data flow, file format, and key design choices (callback vs. polling, memory-efficient progress imaging, and robust cleanup) so that the approach can be reproduced, scaled, and adapted to other raster-scan measurement techniques.

**Keywords:** LIBS; mapping; hyperspectral; hardware trigger; NetCDF; Avantes; CNC; synchronization.

---

## 1. Introduction

Laser-induced breakdown spectroscopy (LIBS) is widely used for elemental mapping of solid samples [1,2], including imaging and mapping workflows that generate large hyperspectral datasets [3–5]. In typical implementations, the sample is moved under a fixed laser beam (or the beam is scanned), and at each position one or more spectra are acquired. For high spatial resolution over large areas, the number of shots and the volume of spectral data become large; synchronization between laser firing, stage position, and spectrometer integration must be reliable over long runs.

Many systems use software-triggered acquisition: the computer sends a fire command, waits, then reads the spectrometer(s). This introduces latency and jitter and can become a bottleneck at high repetition rates. An alternative is to drive the entire sequence from the motion system: the CNC (or equivalent) produces step pulses that are divided to generate a trigger at every *N* steps, and that trigger is fanned out to the laser and to the spectrometers. The software then only arms the hardware, starts the motion, and collects results as they become available. Shot timing is determined entirely by the stage speed and the step-to-trigger ratio, so it is deterministic and decoupled from software scheduling.

We implemented such a system using a GRBL-controlled CNC stage, a custom pulse controller, a Viron laser, and multiple Avantes spectrometers. The control and acquisition software is written in Python [11] and integrates with a PyQt5-based laboratory application [12]. Data are written in near real time to a NetCDF4 file, forming a hyperspectral cube (bands × *y* × *x*) with metadata and device information [6–8]. In addition to describing the mapping process, we highlight the system’s open-hardware philosophy—shared designs, commodity components, and modifiable controller firmware—aligned with the open science hardware community [17–20]. This document describes the trigger chain, callback-based acquisition, file format, and fault-recovery strategy, with the aim of providing a reference for reproducible, scalable LIBS mapping and for adaptation to other raster-scan measurement modalities.

---

## 2. Materials and Methods

### 2.1 Hardware chain and trigger synchronization

The mapping system comprises four hardware subsystems used in a single coordinated workflow:

1. **CNC stage and motion controller (GRBL32-class, open source / open hardware)**  
   A three-axis gantry stage driven by stepper motors. The X-axis is used for the main scan direction along each line; the Y-axis advances between lines. Motion is commanded over a serial link using GRBL-compatible jog commands. In our implementation the controller is a **GRBL32-class board** (32-bit GRBL ecosystem) running open-source firmware (e.g. GRBL [14] or 32-bit derivatives such as grblHAL or FluidNC) [15,16]. The controller emits step pulses (e.g. 800 steps/mm) that are available as a digital output. The gantry is built from commodity, easily sourced components (e.g. extrusion/rails/motors) and is readily scalable by changing rail lengths and workspace dimensions, consistent with open labware and open hardware practices [19–22].

2. **Pulse controller (Arduino Uno)**  
   A custom unit based on an Arduino Uno microcontroller [13] that receives step pulses from the CNC and, after a programmable number of steps (*step_trigger*), outputs a single trigger pulse. That pulse is fanned out to the laser (external trigger input) and to the spectrometer(s) (external trigger input for integration start). Thus, one trigger is issued every *step_trigger* motor steps, i.e. every spatial step of size *step_size* = *step_trigger* / *steps_per_mm* (mm). Using a widely available open microcontroller platform simplifies replication and modification of the trigger logic [13,19].

3. **Laser (Viron)**  
   The laser is controlled via a TCP (telnet-like) interface. For mapping it is placed in external-trigger mode: it fires on each trigger pulse and remains in a “fire” state for the whole mapping run. No software command is sent between scan lines; only the first line involves a warm-up sequence (standby → ready → fire). This minimizes command traffic and avoids CAN-bus overrun issues observed when sending frequent commands.

4. **Spectrometers (Avantes)**  
   One or more Avantes spectrometers are used, each with its own wavelength range. They are configured for external trigger and a fixed number of scans per line (*x_points*). The SDK provides a callback API: when a scan completes, a DLL callback runs and the application can move the result into a thread-safe queue. The mapping thread blocks on the queue instead of polling, which reduces CPU load and avoids accumulated polling overhead over long runs.

The trigger chain is therefore:

```
CNC step pulses  →  Pulse controller (divide by step_trigger)  →  Trigger out
                                                                       ├→ Laser (fire)
                                                                       └→ Spectrometers (start integration)
```

The software does not control shot timing; it only sets *step_trigger*, starts the CNC motion, and collects spectra as they are delivered by the callback mechanism.

### 2.1.1 Open-source/open-hardware implementation and scalability

The mapping platform is distributed as **open source** (software) and **open hardware** (mechanical and electronic designs) in the sense of the OSHWA definition: design files are made publicly available so others can study, modify, make, and redistribute them [17]. Community frameworks such as GOSH emphasize accessibility, local manufacturability, repairability, and reproducibility as core goals for scientific instrumentation [18].

In practice, the system’s scalability and modifiability come from three design choices:

- **Motion control in the GRBL ecosystem:** GRBL provides a well-documented, widely adopted motion control stack; 32-bit GRBL-compatible controllers (GRBL32-class) extend capabilities while preserving the G-code/jog model [14–16].
- **Commodity, modular gantry mechanics:** The stage is built from low-cost, easily replaceable components; scaling the mapping area is primarily a mechanical redesign (rail/extrusion length and cable management), not a software rewrite. This approach mirrors established open hardware strategies in scientific tooling [19–22].
- **Programmable trigger logic:** Implementing the pulse controller on Arduino Uno reduces barriers to modifying trigger division, gating, or adding auxiliary outputs (e.g. camera strobe, shutter, or safety interlocks) [13,19].

Design files (including 3D models) can be hosted on collaborative platforms such as GitHub for versioned distribution and citation support (e.g. `CITATION.cff`) [28,29]. Hardware designs can be licensed using established open hardware licenses (e.g. CERN OHL v2 or TAPR OHL), enabling reuse and reciprocity while clarifying redistribution terms [23,24].

### 2.2 Mapping parameters and derived quantities

User-defined parameters include:

- **Spatial:** *step_size* (mm), *mapping_X_size*, *mapping_Y_size* (mm).
- **Laser:** *laser_frequency* (Hz), *line_start_wait_s* (s) — wait after first fire before starting the first line.
- **Spectrometer:** *integration_time_ms*, *integration_delay_ns*, *averages* (typically 1 for mapping).
- **CNC:** *steps_per_mm*, *return_feed_mm_min* (mm/min for the return jog).

Derived quantities used in the implementation are:

- *mapping_X_points* = *mapping_X_size* / *step_size*  
  (number of shots per line)
- *mapping_Y_points* = *mapping_Y_size* / *step_size*  
  (number of lines)
- *x_speed_mm_s* = *step_size* × *laser_frequency*  
  (scan speed in mm/s so that one shot occurs per step)
- *x_feed_mm_min* = *x_speed_mm_s* × 60  
  (GRBL feed in mm/min)
- *step_trigger* = round(*steps_per_mm* × *step_size*)  
  (steps per trigger pulse)
- *x_travel_mm* = *mapping_X_size* + 10 × *step_size*  
  (forward travel including a margin so that extra triggers are available after the last pixel; these “margin” shots compensate for occasional laser misfires)

The CNC jog distance is *x_travel_mm* + 1 mm (additional safety margin). After the line, the stage returns to the start at *return_feed_mm_min*.

### 2.3 Software workflow

The mapping run is executed in a dedicated background thread. Pre-established connections (serial for CNC and pulse controller, TCP for laser, USB for spectrometers) are passed in; the mapping code does not open or close them.

**One-time setup:**

1. Laser: login, configure for mapping (external trigger, burst off, appropriate quiescent/prefire settings), then for the first line only: standby → wait ready → fire → wait *line_start_wait_s*.
2. Create the NetCDF file (see 2.5).
3. Capture a dark spectrum per device (software trigger, one scan per spectrometer).
4. Build pixel ranges and wavelength concatenation for all devices; initialise the callback-based collector.

**Per-line loop (for each *y* = 0 … *mapping_Y_points* − 1):**

1. If *y* = 0 and first attempt: run full laser standby-and-fire sequence; otherwise no laser commands (laser stays in fire).
2. Arm spectrometers: `AVS_PrepareMeasure` (external trigger, *x_points* scans) and `AVS_MeasureCallback` (callback + queue per handle). Drain any leftover items from the queues from the previous line.
3. Arm pulse controller: send step divisor (*step_trigger*) and start.
4. Start CNC: jog X by the computed distance at *x_feed_mm_min*.
5. For each *x* = 0 … *mapping_X_points* − 1: block on the collector’s queues until one scan is available from every device; dark-correct and merge into a single band vector; write that vector to the NetCDF at (*x*, *y*); release the per-*x* buffer.
6. After the last *x*: call `AVS_StopMeasure` on all devices (critical for resetting DLL/USB state and avoiding desynchronisation over many lines).
7. Wait for CNC idle (forward jog, including margin, complete). Then stop the pulse controller, jog X back to line start at return feed, wait for idle, flush NetCDF.
8. If at any *x* no data arrive within *shot_timeout_s* (e.g. 30 s), treat as laser fault: stop triggers, stop measure, wait CNC, return stage, raise timeout. The main loop then runs a recovery sequence (see 2.4) and retries the same line up to *max_line_retries* times.
9. After a successful line: update progress; move Y by *step_size*; every *progress_every_lines* lines, generate a progress PNG (see 2.6) and optionally upload via FTP.

**Cleanup (always, in a `finally` block):**  
Laser: send stop. Close the NetCDF file. The GUI then disconnects the laser and resumes camera/illumination if they were paused for mapping.

### 2.4 Laser fault detection and recovery

If the laser stops firing (e.g. fault or interlock), the spectrometers stop receiving triggers and the callback-based collector does not receive new data. After *shot_timeout_s* seconds at one *x*, a `ShotTimeoutError` is raised. The mapping thread:

1. Stops the pulse controller and calls `AVS_StopMeasure` on all spectrometers.
2. Waits for the CNC to finish the current jog and returns the stage to the line start.
3. Runs a recovery sequence: login → `$STOP` → wait 30 s → reconfigure for mapping → `$STANDBY` → wait 5 s → poll status until “ready” (up to 120 s) → `$FIRE` → wait *line_start_wait_s*.
4. Retries the same line. If timeouts persist after *max_line_retries* attempts, the mapping aborts with an error.

All laser commands are separated by a minimum gap (e.g. 1 s) to avoid CAN-bus overrun. A 0.5 s delay is applied after login.

### 2.5 Data format (NetCDF4)

The output file is NetCDF4 with:

- **Dimensions:** `bands`, `y`, `x`.  
  `bands` is the total number of spectral channels (sum of pixel counts over all devices after concatenation). `y` and `x` are *mapping_Y_points* and *mapping_X_points*.

- **Variables:**  
  - `mapping` (`uint16`, dimensions `bands` × `y` × `x`): hyperspectral cube. Stored with zlib compression (level 4), chunked (e.g. 256 bands × 1 × 64) and shuffle for efficient partial I/O.  
  - `bands` (`float64`, dimension `bands`): wavelength (nm) for each channel in concatenated order.

- **Global attributes:** Creation time (UTC), integration time, delay, averages, laser frequency, step size, steps per mm, step trigger, travel and feed settings, grid dimensions, band count, device count, progress selection and tolerance.

- **Group `devices`:** One subgroup per device (`device_000`, …) with attributes: handle, serial, user_friendly_name, FPGA/firmware/DLL versions, wavelength range, pixel count.

Each pixel (*x*, *y*) is written as a single vector: dark-corrected intensities from all devices, merged in device order, clipped to [0, 65535] and stored as `u2`. Writes are incremental; `sync()` is called after each line. NetCDF4 builds on HDF5 to provide a self-describing container with compression and chunked I/O [6–8].

### 2.6 Progress preview and memory use

Every *progress_every_lines* lines, a preview image is generated from the current NetCDF file: a user-selected element (e.g. Sr) defines wavelength bands (with a tolerance in nm); the corresponding bands are summed over the cube. To avoid loading the full cube into memory, bands are summed one at a time (read one band slab, add to an accumulator, discard slab). The result is saved as a PNG with Matplotlib [9]; the figure is explicitly closed in a `finally` block to prevent figure leaks. The same PNG can be uploaded to an FTP server for remote monitoring.

### 2.7 Callback-based acquisition

The Avantes SDK allows either polling (`AVS_PollScan`) or callback (`AVS_MeasureCallback`). We use the callback: for each device, a small callback stores a result token in a per-handle queue; the mapping thread calls `queue.get(timeout=shot_timeout_s)` for each device in turn. This avoids busy-waiting and reduces CPU use and sensitivity to thread scheduling over long runs. The callback function reference is kept on the collector object to prevent garbage collection of the ctypes trampoline. After each line, `AVS_StopMeasure` is called on every device so that internal buffers are cleared and devices stay synchronized over hundreds of lines.

### 2.8 Extending the platform beyond LIBS

Because the system is a general raster-scan positioning platform with a deterministic hardware trigger chain, it can be adapted to other spatially resolved measurements by replacing the probe head and reusing the motion + synchronization infrastructure. Examples include:

- **Raman mapping** (substituting a Raman probe and spectrometer; scanning strategies and throughput constraints are well described in Raman imaging reviews) [25].
- **X-ray fluorescence (XRF) mapping** (mounting an XRF source/detector head on the gantry or using a fixed head with stage motion) [26].
- **Laser ablation ICP-MS (LA-ICP-MS) imaging** (using a raster ablation source with stage motion and synchronizing acquisition to line scans) [27].

More broadly, the same architecture can support reflectance/hyperspectral imaging, laser-induced fluorescence, photoluminescence mapping, or automated microscopy, provided the measurement subsystem offers either hardware trigger inputs or robust timestamping; open hardware microscopy projects provide precedents for scalable, locally manufacturable mechatronics and control software [22].

---

## 3. Results and design validation

### 3.1 Timing and throughput

For a nominal configuration (e.g. *step_size* = 0.1 mm, *laser_frequency* = 20 Hz, *mapping_X_size* = 20 mm, *mapping_Y_size* = 3 mm, *steps_per_mm* = 800, *return_feed_mm_min* = 100):

- *mapping_X_points* = 200, *mapping_Y_points* = 30.
- Scan speed = 0.1 × 20 = 2 mm/s; *x_feed_mm_min* = 120 mm/min.
- *step_trigger* = 80 steps; one shot every 50 ms.
- Forward travel ≈ 22 mm → forward time ≈ 11 s; return ≈ 13 s; total per line ≈ 24 s (plus arm and flush). Scan efficiency (forward scan time vs. total line time) is on the order of 40%; the rest is return and overhead.

The software provides duration and file-size estimates before the run using the same formulas as in Section 2.2 and a simple compression model (e.g. 0.6× raw size for the main array).

### 3.2 Stability over long runs

Without calling `AVS_StopMeasure` after each line, the spectrometers’ internal state can diverge after on the order of 200 lines (observed as timeouts or missing data). With `AVS_StopMeasure` in a `finally` block after every line (success or failure), runs of hundreds of lines complete without desynchronisation. The callback-based path and the disciplined laser command pacing (no commands between lines, 1 s gap between commands, 30 s cool-down after stop before standby) have been used to run full multi-hour maps without CAN-bus or communication errors.

### 3.3 Resource and integration notes

- **Memory:** Per-line buffers are released after each pixel; progress PNG generation uses incremental band summation and explicit figure close; no unbounded in-memory growth is expected over long runs.
- **Camera:** The GUI pauses the laboratory camera during mapping and resumes it in the finalization step (on both normal completion and error), so the camera lifecycle is consistent.
- **Identities:** When a subset of spectrometers is selected, device metadata (identities) is filtered to match the selected handles so that NetCDF device groups correspond to the devices actually used.

---

## 4. Discussion

The main design choices were:

1. **Hardware-driven timing** so that shot rate and spatial step are set by the stage and pulse controller, not by software loops.
2. **Callback-based spectrometer readout** to avoid polling overhead and improve reliability over long acquisitions.
3. **Single fire state for the laser** for the whole map, with a full recovery sequence only on timeout, to minimise command traffic and CAN-bus load.
4. **Systematic `AVS_StopMeasure`** after every line to keep spectrometer state and buffers consistent across many lines.
5. **NetCDF4** for a self-describing, chunked, compressed cube with metadata and device info, suitable for downstream analysis in Python, MATLAB, or other tools.
6. **Progress preview** with bounded memory (incremental band sum) and explicit cleanup of Matplotlib figures.

In addition, the system is intentionally aligned with open hardware/open science hardware principles—commodity components, shareable CAD, and modifiable firmware—so the mapping area can be scaled and the platform can be repurposed to other raster-scan modalities without redesigning the full stack [17–22,25–27].

Limitations include: the return phase is slower than the forward scan, so scan efficiency is below 50%; the current implementation does not support on-the-fly change of step size or laser frequency mid-run; and app exit during mapping may block until the user stops the run or the thread finishes, unless the application explicitly sets a stop event and waits for the thread with a timeout on close.

---

## 5. Conclusion

We have described a synchronized, hardware-triggered LIBS mapping implementation in which the CNC stage drives the trigger chain and the software arms devices, starts motion, and collects spectra via callbacks into a streaming NetCDF4 cube. The design avoids software timing jitter, supports multi-device concatenation and dark correction, and includes fault detection and recovery. The document provides enough detail on parameters, trigger chain, data format, and software workflow to reproduce or adapt the approach for other LIBS mapping systems.

---

## References

1. Cremers, D. A., & Radziemski, L. J. (2006). *Handbook of Laser-Induced Breakdown Spectroscopy*. Wiley.
2. Noll, R. (2012). *Laser-Induced Breakdown Spectroscopy*. Springer.
3. Limbeck, A., Brunnbauer, L., Lohninger, H., Pořízka, P., Modlitbová, P., Kaiser, J., Janovszky, P., Kéri, A., & Galbács, G. (2021). Methodology and applications of elemental mapping by laser induced breakdown spectroscopy. *Analytica Chimica Acta*, 1147, 72–98. https://doi.org/10.1016/j.aca.2020.12.054
4. Senesi, G. S. (2014). Laser-Induced Breakdown Spectroscopy (LIBS) applied to terrestrial and extraterrestrial analogue geomaterials with emphasis to minerals and rocks. *Earth-Science Reviews*. https://doi.org/10.1016/j.earscirev.2014.09.008
5. Galbács, G. (2015). A critical review of recent progress in analytical laser-induced breakdown spectroscopy. *Analytical and Bioanalytical Chemistry*, 407, 7537–7562. https://doi.org/10.1007/s00216-015-8855-3
6. Rew, R. K., & Davis, G. P. (1990). NetCDF: An interface for scientific data access. *IEEE Computer Graphics and Applications*, 10(4), 76–82. https://doi.org/10.1109/38.56302
7. Eaton, B., Gregory, J., Drach, B., Taylor, K., Hankin, S., et al. (2025). *NetCDF Climate and Forecast (CF) Metadata Conventions* (Version 1.13). CF Community. https://doi.org/10.5281/zenodo.17801666
8. The HDF Group. (2024). *HDF5 File Format Specification* (v3.0). https://support.hdfgroup.org/HDF5/doc/H5.format.html
9. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
10. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2
11. Van Rossum, G., & Drake, F. L. (2009). *Python 3 Reference Manual*. CreateSpace.
12. Riverbank Computing. (2024). *PyQt5 Documentation*. https://www.riverbankcomputing.com/static/Docs/PyQt5/
13. Arduino. (n.d.). *Arduino Uno Rev3*. https://docs.arduino.cc/hardware/uno-rev3
14. Jeon, S. (Sonny) and contributors. (2019). *GRBL (gnea/grbl)* (v1.1h). GitHub repository. https://github.com/gnea/grbl
15. grblHAL/core contributors. (n.d.). *grblHAL core*. GitHub repository. https://github.com/grblHAL/core
16. Dring, B. and contributors. (n.d.). *FluidNC*. GitHub repository and documentation. https://github.com/bdring/FluidNC ; http://wiki.fluidnc.com/home
17. Open Source Hardware Association (OSHWA). (n.d.). *Open Source Hardware Definition 1.0*. https://oshwa.org/resources/open-source-hardware-definition/
18. Dosemagen, S., Liboiron, M., & Molloy, J. (2017). Gathering for Open Science Hardware 2016. *Journal of Open Hardware*, 1(1), 5. https://doi.org/10.5334/joh.5
19. Pearce, J. M. (2012). Building research equipment with free, open-source hardware. *Science*, 337(6100), 1303–1304. https://doi.org/10.1126/science.1228183
20. Baden, T., Chagas, A. M., Gage, G., Marzullo, T., Prieto-Godino, L. L., & Euler, T. (2015). Open Labware: 3-D printing your own lab equipment. *PLOS Biology*, 13(3), e1002086. https://doi.org/10.1371/journal.pbio.1002086
21. Jones, R., Haufe, P., Sells, E., Iravani, P., Olliver, V., Palmer, C., & Bowyer, A. (2011). RepRap – the replicating rapid prototyper. *Robotica*, 29(1), 177–191. https://doi.org/10.1017/S026357471000069X
22. Collins, J. T., Knapper, J., Stirling, J., et al. (2020). Robotic microscopy for everyone: the OpenFlexure microscope. *Biomedical Optics Express*, 11(5), 2447–2460. https://doi.org/10.1364/BOE.385729
23. CERN. (2020). *CERN Open Hardware Licence v2* (OHL-P/W/S). License texts: https://ohwr.org/cern_ohl_p_v2.txt ; https://ohwr.org/cern_ohl_w_v2.txt ; https://ohwr.org/cern_ohl_s_v2.txt
24. TAPR. (2007). *TAPR Open Hardware License v1.0*. https://files.tapr.org/OHL/TAPR_Open_Hardware_License_v1.0.txt
25. Opilik, L., Schmid, T., & Zenobi, R. (2013). Modern Raman imaging: Vibrational spectroscopy on the micrometer and nanometer scales. *Annual Review of Analytical Chemistry*, 6, 379–398. https://doi.org/10.1146/annurev-anchem-062012-092646
26. Vanhoof, C., Bacon, J. R., Fittschen, U. E. A., & Vincze, L. (2022). Atomic spectrometry update: review of advances in X-ray fluorescence spectrometry and its special applications. *Journal of Analytical Atomic Spectrometry*, 37, 1761–1775. https://doi.org/10.1039/D2JA90035A
27. Becker, J. S., Matusch, A., & Wu, B. (2014). Bioimaging mass spectrometry of trace elements – recent advance and applications of LA-ICP-MS: A review. *Analytica Chimica Acta*, 835, 1–18. https://doi.org/10.1016/j.aca.2014.04.048
28. Chacon, S., & Straub, B. (2014). *Pro Git* (2nd ed.). Apress.
29. GitHub Docs. (n.d.). About citation files (`CITATION.cff`). https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files

---

## Data and code availability

The mapping engine, GUI components, and flowchart documentation are part of the Fused LIBS application. NetCDF4 files produced by the system contain full acquisition parameters and device metadata for reproducibility.
