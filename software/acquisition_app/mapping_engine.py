# -*- coding: utf-8 -*-
"""
LIBS Mapping Engine
-------------------
Class-based mapping engine for streamed NetCDF acquisition with GRBL CNC,
pulse controller, Viron laser, and Avantes spectrometers.

Used by ``mapping_gui.MappingThread`` inside the main application.
"""
import os
import time
import queue as _queue
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from netCDF4 import Dataset
import ftplib

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import avaspec_fix_winfunctype as avs
from hardware_control import (
    wait_cnc_idle, grbl_stream_jog_x, grbl_stream_jog_y,
    movement_trigger_start, movement_trigger_stop,
    movement_trigger_reset_count, movement_trigger_get_count,
)


class ShotTimeoutError(Exception):
    """Raised when waiting for spectrometer data times out (likely laser fault)."""
    pass


# ============================
# Element band dictionary
# ============================
BAND_DICT = {
    "Si": [251.61, 288.16],
    "Al": [308.22, 394.40, 396.15],
    "Fe": [248.33, 371.99, 374.55, 382.04, 438.35, 516.89],
    "Mg": [279.55, 285.21, 517.27],
    "Ca": [393.37, 396.85, 422.67],
    "Na": [588.995, 589.592],
    "K":  [766.49, 769.90],
    "Ti": [334.94, 337.28],
    "Mn": [403.08, 293.31],
    "O":  [777.19, 844.62],
    "Ba": [455.40, 493.41, 553.55, 614.17],
    "Sr": [407.77, 421.55, 460.73],
    "Cu": [324.75, 327.40],
    "Zn": [213.86, 468.01, 472.22],
    "P":  [213.62, 214.91],
    "S":  [906.89, 910.48, 912.74, 937.42],   # S II lines above 900 nm
    "Li": [610.36, 670.78],
    "Pb": [280.20, 283.31, 363.96, 368.35, 405.78],
}


# ============================
# MappingParams dataclass
# ============================
@dataclass
class MappingParams:
    """All parameters needed for a mapping run, replacing 30+ module-level variables."""
    # Mapping dimensions
    step_size: float = 0.10
    mapping_X_size: float = 20.0
    mapping_Y_size: float = 3.0

    # Laser & timing
    laser_frequency: float = 20.0
    integration_time_ms: float = 0.1
    integration_delay_ns: int = 1000
    averages: int = 1

    # CNC / pulse
    steps_per_mm: int = 800
    return_feed_mm_min: float = 100.0

    # Laser
    laser_single_shot_mode: bool = False
    line_start_wait_s: float = 5.0  # seconds to wait at start of each line after fire

    # Output
    save_dir: str = "output_job"
    save_stem: str = "mapping_run"
    progress_every_lines: int = 5
    progress_selection: str = "Sr"
    progress_tol_nm: float = 0.3

    # FTP (optional)
    ftp_host: Optional[str] = None
    ftp_user: Optional[str] = None
    ftp_pass: Optional[str] = None
    ftp_path: str = "."

    # Fault recovery
    max_line_retries: int = 3
    shot_timeout_s: float = 30.0
    resync_every_n_lines: int = 50

    # Verbosity
    verbose: bool = True

    # --- Derived (computed after construction) ---
    @property
    def mapping_X_points(self) -> int:
        return int(self.mapping_X_size / self.step_size)

    @property
    def mapping_Y_points(self) -> int:
        return int(self.mapping_Y_size / self.step_size)

    @property
    def x_travel_mm(self) -> float:
        return self.mapping_X_size + (self.step_size * 10.0)

    @property
    def x_speed_mm_s(self) -> float:
        return self.step_size * self.laser_frequency

    @property
    def x_feed_mm_min(self) -> float:
        return self.x_speed_mm_s * 60.0

    @property
    def step_trigger(self) -> int:
        return int(round(self.steps_per_mm * self.step_size))

    @property
    def mapping_filename_nc(self) -> str:
        # Cache so the timestamp is consistent across multiple accesses
        if not hasattr(self, '_cached_filename_nc'):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            object.__setattr__(self, '_cached_filename_nc', f"{self.save_stem}_{stamp}.nc")
        return self._cached_filename_nc


# ============================
# Utility functions
# ============================
def ns_to_fpga_ticks(ns):
    """Convert nanoseconds to 48 MHz FPGA tick units (~20.83 ns/tick)."""
    return int(round(ns / 20.83)) if ns else 0


# ============================
# Estimation functions (standalone, no hardware needed)
# ============================
def estimate_file_size(bands, mapping_X_points, mapping_Y_points, num_devices=1):
    """Estimate compressed NetCDF file size in bytes/MB/GB."""
    mapping_data_raw = bands * mapping_X_points * mapping_Y_points * 2
    bands_data = bands * 8
    metadata_overhead = 10 * 1024
    device_overhead = num_devices * 2 * 1024
    uncompressed_size = mapping_data_raw + bands_data + metadata_overhead + device_overhead
    compression_ratio = 0.6
    estimated_size = uncompressed_size * compression_ratio
    return {
        'uncompressed_bytes': uncompressed_size,
        'estimated_bytes': estimated_size,
        'uncompressed_mb': uncompressed_size / (1024 * 1024),
        'estimated_mb': estimated_size / (1024 * 1024),
        'uncompressed_gb': uncompressed_size / (1024 * 1024 * 1024),
        'estimated_gb': estimated_size / (1024 * 1024 * 1024),
        'mapping_data_raw': mapping_data_raw,
        'compression_ratio': compression_ratio,
    }


def estimate_mapping_duration(mapping_X_points, mapping_Y_points, x_travel_mm, step_size,
                              x_feed_mm_min, return_feed_mm_min=60.0, y_feed_mm_min=60.0):
    """Estimate total mapping duration."""
    x_feed_mm_s = x_feed_mm_min / 60.0
    return_feed_mm_s = return_feed_mm_min / 60.0
    y_feed_mm_s = y_feed_mm_min / 60.0

    actual_x_travel = x_travel_mm + 5.0
    forward_scan_time = actual_x_travel / x_feed_mm_s
    return_time = actual_x_travel / return_feed_mm_s
    y_movement_time = step_size / y_feed_mm_s if y_feed_mm_s > 0 else 0
    overhead_per_line = 2.0

    time_per_line = forward_scan_time + return_time + overhead_per_line
    y_movements = max(0, mapping_Y_points - 1)
    total_scan_time = time_per_line * mapping_Y_points
    total_y_movement_time = y_movement_time * y_movements
    total_duration_sec = total_scan_time + total_y_movement_time

    return {
        'total_seconds': total_duration_sec,
        'hours': int(total_duration_sec // 3600),
        'minutes': int((total_duration_sec % 3600) // 60),
        'seconds': int(total_duration_sec % 60),
        'time_per_line': time_per_line,
        'forward_scan_time': forward_scan_time,
        'return_time': return_time,
        'y_movement_time': y_movement_time,
        'overhead_per_line': overhead_per_line,
    }


# ============================
# AvaSpec helpers
# ============================
def _avs_safe_init(log_fn=None):
    log = log_fn or print
    try:
        if hasattr(avs, "AVS_Init"):
            log("[AVS] AVS_Init(0)")
            avs.AVS_Init(0)
    except Exception as e:
        log(f"[AVS] Init note: {e}")
    try:
        if hasattr(avs, "AVS_UpdateUSBDevices"):
            log("[AVS] AVS_UpdateUSBDevices()")
            avs.AVS_UpdateUSBDevices()
    except Exception as e:
        log(f"[AVS] USB refresh note: {e}")


def _bytes_to_str(b):
    return bytes(b).split(b"\x00", 1)[0].decode("utf-8", "ignore")


def _get_num_pixels(handle):
    npx = avs.AVS_GetNumPixels(handle)
    try:
        return int(npx)
    except Exception:
        if isinstance(npx, (tuple, list)) and len(npx) > 0:
            return int(npx[0])
        return int(npx)


def discover_and_sort_by_range(log_fn=None):
    """Discover AvaSpec spectrometers and return (handles_sorted, wavelengths, identities)."""
    log = log_fn or print
    log("[AVS] Discovering devices ...")
    _avs_safe_init(log_fn=log)
    id_list = avs.AVS_GetList()
    log(f"[AVS] AVS_GetList -> {len(id_list)} device(s).")
    if not id_list:
        time.sleep(0.3)
        _avs_safe_init(log_fn=log)
        id_list = avs.AVS_GetList()
        log(f"[AVS] Retry AVS_GetList -> {len(id_list)} device(s).")
    if not id_list:
        raise RuntimeError("No AvaSpec spectrometers found (AVS_GetList empty).")

    handles, wavelengths = [], {}
    for i, dev in enumerate(id_list):
        sn = _bytes_to_str(dev.SerialNumber) if hasattr(dev, "SerialNumber") else ""
        nm = _bytes_to_str(dev.UserFriendlyName) if hasattr(dev, "UserFriendlyName") else ""
        log(f"[AVS] Activating {i}: serial={sn} name={nm}")
        h = avs.AVS_Activate(dev)
        handles.append(h)
        try:
            if hasattr(avs, "AVS_UseHighResAdc"):
                avs.AVS_UseHighResAdc(h, True)
                log(f"[AVS]   High-res ADC enabled for handle={h}")
        except Exception as e:
            log(f"[AVS]   High-res ADC enable failed for handle={h}: {e}")
        px = _get_num_pixels(h)
        wl = np.asarray(avs.AVS_GetLambda(h), dtype=np.float64)[:px]
        wavelengths[h] = wl
        log(f"[AVS]   handle={h} pixels={px} lambda=[{wl[0]:.2f} .. {wl[-1]:.2f}] nm")

    handles_sorted = sorted(handles, key=lambda hh: float(wavelengths[hh][0]))
    log(f"[AVS] Sorted {len(handles_sorted)} device(s) by start lambda.")
    identities = []
    for dev in id_list:
        try:
            identities.append({
                "serial": _bytes_to_str(dev.SerialNumber),
                "name": _bytes_to_str(dev.UserFriendlyName),
            })
        except Exception:
            identities.append({"serial": "", "name": ""})
    return handles_sorted, wavelengths, identities


# ============================
# Measurement config helpers
# ============================
def build_meas_config(start_px, stop_px, int_ms, int_delay_fpga, navg,
                      trig_mode, trig_source, edge=True, laser_delay_fpga=0):
    cfg = avs.MeasConfigType()
    cfg.m_StartPixel = int(start_px)
    cfg.m_StopPixel = int(stop_px)
    cfg.m_IntegrationTime = float(int_ms)
    cfg.m_IntegrationDelay = int(int_delay_fpga)
    cfg.m_NrAverages = int(max(1, navg))
    cfg.m_CorDynDark_m_Enable = 0
    cfg.m_CorDynDark_m_ForgetPercentage = 0
    cfg.m_Smoothing_m_SmoothPix = 0
    cfg.m_Smoothing_m_SmoothModel = 0
    cfg.m_SaturationDetection = 1
    cfg.m_Trigger_m_Mode = int(trig_mode)
    cfg.m_Trigger_m_Source = int(trig_source)
    cfg.m_Trigger_m_SourceType = 0 if edge else 1
    cfg.m_Control_m_StrobeControl = 0
    cfg.m_Control_m_LaserDelay = int(laser_delay_fpga)
    cfg.m_Control_m_LaserWidth = 0
    cfg.m_Control_m_LaserWaveLength = 0.0
    cfg.m_Control_m_StoreToRam = 0
    return cfg


def make_cfg_single_scan_parallel(start_px, stop_px, int_ms, int_delay_fpga, navg,
                                  laser_delay_fpga=0):
    return build_meas_config(start_px, stop_px, int_ms, int_delay_fpga, navg,
                             trig_mode=2, trig_source=0, edge=True,
                             laser_delay_fpga=laser_delay_fpga)


def _get_scope(handle):
    ret = avs.AVS_GetScopeData(handle)
    if isinstance(ret, tuple) and len(ret) == 2:
        ts, spec = ret
    else:
        ts, spec = 0, ret
    return int(ts), np.asarray(spec, dtype=np.float32)


def capture_dark_once(handles, device_pixel_ranges, int_ms, int_delay_fpga, navg, log_fn=None):
    log = log_fn or print
    log("[AVS] Capturing dark (single scan per device, using full pixel range) ...")
    for h in handles:
        start_px, stop_px = device_pixel_ranges[h]
        cfg = build_meas_config(start_px, stop_px, int_ms, int_delay_fpga, navg,
                                trig_mode=0, trig_source=0, edge=True, laser_delay_fpga=0)
        err = avs.AVS_PrepareMeasure(h, cfg)
        log(f"[AVS] PrepareMeasure(dark) handle={h} px={start_px}-{stop_px} -> {err}")
        if err != 0:
            raise RuntimeError(f"AVS_PrepareMeasure(dark) failed {h} err={err}")
    for h in handles:
        err = avs.AVS_Measure(h, 0, 1)
        log(f"[AVS] Measure(dark) handle={h} -> {err}")
        if err != 0:
            raise RuntimeError(f"AVS_Measure(dark) failed {h} err={err}")
    darks, pending = {}, set(handles)
    while pending:
        for h in list(pending):
            if avs.AVS_PollScan(h):
                _, spec = _get_scope(h)
                darks[h] = spec
                pending.discard(h)
                log(f"[AVS] Dark received handle={h} (remaining {len(pending)})")
        time.sleep(0.001)
    log("[AVS] Dark capture complete.")
    return darks


def arm_line_single_scan_parallel(handles, x_points, device_pixel_ranges, int_ms,
                                  int_delay_fpga, navg, laser_delay_fpga, log_fn=None):
    log = log_fn or print
    log(f"[AVS] Arming line: Single Scan, external trigger, x_points={x_points}")
    for h in handles:
        start_px, stop_px = device_pixel_ranges[h]
        cfg = make_cfg_single_scan_parallel(start_px, stop_px, int_ms, int_delay_fpga, navg,
                                            laser_delay_fpga=laser_delay_fpga)
        err = avs.AVS_PrepareMeasure(h, cfg)
        log(f"[AVS] PrepareMeasure(handle={h}) px={start_px}-{stop_px} -> {err}")
        if err != 0:
            raise RuntimeError(f"AVS_PrepareMeasure failed {h} err={err}")
    for h in handles:
        err = avs.AVS_Measure(h, 0, int(x_points))
        log(f"[AVS] Measure(handle={h}, scans={x_points}) -> {err}")
        if err != 0:
            raise RuntimeError(f"AVS_Measure failed {h} err={err}")


def collect_next_shot(handles_sorted, on_spectrum, shot_index, timeout=30.0):
    """Collect one spectrum from each spectrometer for the given shot index.

    Raises ``ShotTimeoutError`` if no data arrives within *timeout* seconds,
    which typically indicates a laser fault (trigger chain interrupted).
    """
    pending = set(handles_sorted)
    deadline = time.time() + timeout
    while pending:
        for h in list(pending):
            if avs.AVS_PollScan(h):
                ts, spec = _get_scope(h)
                on_spectrum(h, shot_index, ts, spec)
                pending.discard(h)
        if time.time() > deadline:
            raise ShotTimeoutError(
                f"Timeout ({timeout:.0f}s) at shot x={shot_index}, "
                f"still waiting on {len(pending)} spectrometer(s)")
        time.sleep(0.001)


# ============================
# Callback-based collection
# ============================
class CallbackCollector:
    """Callback-based spectrometer data collection.

    Uses ``AVS_MeasureCallback`` instead of ``AVS_Measure`` + ``AVS_PollScan``
    polling.  The DLL invokes a callback from its own thread whenever a scan
    completes; the callback pushes a notification into a per-handle
    ``queue.Queue``.  The mapping thread blocks on ``Queue.get(timeout=…)``
    instead of busy-polling, which is both faster and more reliable for long
    runs (no accumulated DLL polling overhead).

    Create **once** per mapping run and reuse across all Y lines.
    """

    def __init__(self, handles, log_fn=None):
        self._log = log_fn or (lambda msg: None)
        self._ready = {h: _queue.Queue() for h in handles}

        def _on_ready(handle_ptr, result_ptr):
            h = handle_ptr[0]
            q = self._ready.get(h)
            if q is not None:
                q.put(result_ptr[0])

        # prevent garbage-collection of the ctypes trampoline
        self._cb = avs.AVS_MeasureCallbackFunc(_on_ready)

    def drain_queues(self, handles, max_per_handle=5000):
        """Discard any remaining tokens in the callback queues (e.g. after StopMeasure).
        Bounded to avoid spinning forever if the DLL keeps posting."""
        for h in handles:
            q = self._ready.get(h)
            if q is None:
                continue
            n = 0
            while n < max_per_handle:
                try:
                    q.get_nowait()
                    n += 1
                except _queue.Empty:
                    break
            if n > 0:
                self._log(f"[AVS] Drained {n} stale callback(s) for handle {h}")

    def arm(self, handles, x_points, device_pixel_ranges, int_ms,
            int_delay_fpga, navg, laser_delay_fpga=0):
        """Prepare + start callback-based measurement for one scan line."""
        self.drain_queues(handles)

        self._log(f"[AVS] Arming line (callback): x_points={x_points}")
        for h in handles:
            start_px, stop_px = device_pixel_ranges[h]
            cfg = make_cfg_single_scan_parallel(
                start_px, stop_px, int_ms, int_delay_fpga, navg,
                laser_delay_fpga=laser_delay_fpga)
            err = avs.AVS_PrepareMeasure(h, cfg)
            self._log(f"[AVS] PrepareMeasure(handle={h}) "
                      f"px={start_px}-{stop_px} -> {err}")
            if err != 0:
                raise RuntimeError(
                    f"AVS_PrepareMeasure failed {h} err={err}")

        for h in handles:
            err = avs.AVS_MeasureCallback(h, self._cb, int(x_points))
            self._log(f"[AVS] MeasureCallback(handle={h}, "
                      f"scans={x_points}) -> {err}")
            if err != 0:
                raise RuntimeError(
                    f"AVS_MeasureCallback failed {h} err={err}")

    def collect_next_shot(self, handles_sorted, on_spectrum, shot_index,
                          timeout=30.0):
        """Block until every handle delivers one scan, then read the data."""
        deadline = time.time() + timeout
        for h in handles_sorted:
            remaining = max(0.001, deadline - time.time())
            try:
                _status = self._ready[h].get(timeout=remaining)
            except _queue.Empty:
                raise ShotTimeoutError(
                    f"Timeout ({timeout:.0f}s) at shot x={shot_index}, "
                    f"still waiting on handle {h}")
            ts, spec = _get_scope(h)
            on_spectrum(h, shot_index, ts, spec)


# ============================
# NetCDF writer
# ============================
class NetCDFCubeWriter:
    def __init__(self, nc_path, bands, nx, ny, wavelengths_concat, metadata: dict,
                 device_info_list, log_fn=None):
        log = log_fn or print
        log(f"[NC] Creating NetCDF: {nc_path}")
        self.nc = Dataset(nc_path, "w", format="NETCDF4")
        self.nc.createDimension("bands", bands)
        self.nc.createDimension("y", ny)
        self.nc.createDimension("x", nx)
        self.var = self.nc.createVariable(
            "mapping", "u2", ("bands", "y", "x"),
            zlib=True, complevel=4,
            chunksizes=(min(bands, 256), 1, min(64, nx)),
            shuffle=True,
        )
        self.bands_var = self.nc.createVariable("bands", "f8", ("bands",))
        self.bands_var[:] = np.asarray(wavelengths_concat, dtype=np.float64)
        for k, v in metadata.items():
            try:
                self.nc.setncattr(k, v)
            except Exception:
                pass
        self.devg = self.nc.createGroup("devices")
        for i, dev in enumerate(device_info_list):
            g = self.devg.createGroup(f"device_{i:03d}")
            for k, v in dev.items():
                try:
                    g.setncattr(k, v)
                except Exception:
                    pass
        log("[NC] File initialized.")

    def write_xy(self, x_idx, y_idx, vector_uint16):
        self.var[:, y_idx, x_idx] = vector_uint16

    def flush(self):
        self.nc.sync()

    def close(self):
        self.nc.close()


# ============================
# Progress PNG + optional FTP
# ============================
def save_progress_png(nc_path, out_png_path, selection="Sr", tol_nm=0.3, log_fn=None):
    log = log_fn or print
    log(f"[PNG] Building progress preview ({selection}, +/-{tol_nm} nm) ...")
    with Dataset(nc_path, "r") as nc:
        data = nc.variables["mapping"]
        wl = nc.variables["bands"][:]
        keys = [selection] if isinstance(selection, str) else list(selection)
        chosen = []
        for key in keys:
            chosen.extend(BAND_DICT.get(key, []))
        band_mask = np.zeros_like(wl, dtype=bool)
        for line_nm in chosen:
            band_mask |= (np.abs(wl - line_nm) <= float(tol_nm))
        title = f"Preview: {', '.join(keys)} (+/-{tol_nm} nm)"
        if not band_mask.any():
            expand = float(tol_nm)
            for _ in range(5):
                expand *= 2.0
                band_mask = np.zeros_like(wl, dtype=bool)
                for line_nm in chosen:
                    band_mask |= (np.abs(wl - line_nm) <= expand)
                if band_mask.any():
                    title += f" [expanded {expand:.2f} nm]"
                    break
            if not band_mask.any():
                band_mask = np.ones_like(wl, dtype=bool)
                title += " [fallback all]"

        # Sum selected bands one-at-a-time so memory stays flat regardless of
        # how many bands match (avoids loading the whole cube into RAM).
        band_indices = np.where(band_mask)[0]
        ny, nx = data.shape[1], data.shape[2]
        img = np.zeros((ny, nx), dtype=np.float64)
        for bi in band_indices:
            slab = np.ma.filled(data[int(bi), :, :], 0)
            img += slab.astype(np.float64)
        img = img.astype(np.float32)

        if not np.isfinite(img).any():
            img = np.zeros((ny, nx), dtype=np.float32)

    aspect_ratio = max(ny, 1) / max(nx, 1)
    fig_w = 8
    fig_h = max(2, min(12, fig_w * aspect_ratio + 1))
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    try:
        plt.imshow(img, origin="upper", aspect="equal", vmin=0, vmax=100000)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(out_png_path)
    finally:
        plt.close(fig)
    log(f"[PNG] Saved: {out_png_path}")


def ftp_upload_png(local_png, host, user, pw, remote_path=".", log_fn=None):
    log = log_fn or print
    log(f"[FTP] Uploading {local_png} to {host}:{remote_path}")
    with ftplib.FTP(host) as ftp:
        ftp.login(user=user, passwd=pw)
        ftp.cwd(remote_path)
        with open(local_png, "rb") as f:
            ftp.storbinary(f"STOR {os.path.basename(local_png)}", f)
    log("[FTP] Upload OK.")


# ============================
# MappingEngine class
# ============================
class MappingEngine:
    """Runs a complete LIBS mapping acquisition.

    Usage::

        params = MappingParams(step_size=0.1, mapping_X_size=20, ...)
        engine = MappingEngine(params)
        engine.run(cnc, pulse, laser, handles, wavelengths, identities)
    """

    def __init__(self, params: MappingParams, log_fn: Optional[Callable] = None):
        self.p = params
        self._log_fn = log_fn

    def log(self, msg):
        if self.p.verbose and self._log_fn:
            self._log_fn(msg)
        elif self.p.verbose:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}", flush=True)

    # ------------------------------------------------------------------
    def run(self, cnc, pulse, laser, handles_sorted, wavelengths, identities,
            stop_event=None, progress_callback=None, selected_handles=None):
        """Execute the full mapping loop.

        Parameters
        ----------
        cnc : serial.Serial  -- pre-opened CNC serial connection
        pulse : serial.Serial -- pre-opened pulse controller serial
        laser : hardware_control.VironLaser (or compatible) -- pre-connected laser
        handles_sorted : list -- spectrometer handles sorted by wavelength
        wavelengths : dict -- {handle: np.ndarray of wavelengths}
        identities : list[dict] -- spectrometer identity dicts
        stop_event : threading.Event, optional -- set to request graceful stop
        progress_callback : callable(int), optional -- called with progress 0-100
        selected_handles : list, optional -- subset of handles to use
        """
        p = self.p

        # Filter to selected handles if provided (keep identity per handle in order)
        if selected_handles:
            identity_by_handle = dict(zip(handles_sorted, identities)) if identities else {}
            handles_sorted = [h for h in handles_sorted if h in selected_handles]
            wavelengths = {h: wl for h, wl in wavelengths.items() if h in handles_sorted}
            identities = [identity_by_handle[h] for h in handles_sorted if h in identity_by_handle]
            # Pad if identities were missing for some handles
            while len(identities) < len(handles_sorted):
                identities.append({})

        if not handles_sorted:
            raise RuntimeError("[AVS] No spectrometers available for mapping.")

        os.makedirs(p.save_dir, exist_ok=True)
        nc_fullpath = os.path.join(p.save_dir, p.mapping_filename_nc)
        progress_png_path = os.path.join(p.save_dir, "progress.png")
        pulse_log_path = os.path.join(p.save_dir, "pulse_count_log.csv")
        with open(pulse_log_path, "w", encoding="utf-8") as f:
            f.write("line_y,expected_shots,pulse_count,status,note\n")
        self.log(f"[PULSE] Per-line pulse log: {pulse_log_path}")

        mapping_X_points = p.mapping_X_points
        mapping_Y_points = p.mapping_Y_points

        self.log(f"=== START ===  X={mapping_X_points}, Y={mapping_Y_points}, "
                 f"step={p.step_size} mm, feed={p.x_feed_mm_min:.2f} mm/min")
        self.log(f"[ENV] save_dir={p.save_dir}  file={p.mapping_filename_nc}")

        # Duration estimate
        est = estimate_mapping_duration(
            mapping_X_points, mapping_Y_points, p.x_travel_mm, p.step_size,
            p.x_feed_mm_min, return_feed_mm_min=p.return_feed_mm_min,
            y_feed_mm_min=p.return_feed_mm_min)
        self.log("=" * 60)
        self.log("[ESTIMATION] Mapping Duration Estimation:")
        self.log(f"  Mapping size: {mapping_X_points} x {mapping_Y_points} points "
                 f"({p.mapping_X_size} x {p.mapping_Y_size} mm)")
        self.log(f"  Forward scan speed: {p.x_feed_mm_min:.2f} mm/min")
        self.log(f"  Return speed: {p.return_feed_mm_min:.2f} mm/min")
        self.log(f"  Time per line: {est['time_per_line']:.2f} s")
        self.log(f"  Total estimated duration: {est['hours']:d}h {est['minutes']:d}m "
                 f"{est['seconds']:d}s ({est['total_seconds']:.1f} s)")
        self.log("=" * 60)

        # Configure laser (fire will happen at the start of the first line)
        self.log("[LASER] Using pre-established connection")
        laser.ensure_session()
        laser.configure_for_mapping(single_shot=p.laser_single_shot_mode)
        tup, decoded, raw, texts = laser.status_readable()
        if tup:
            self.log(f"[LASER] Configured. STATUS: {raw}")
            if texts and "OK" not in texts:
                self.log(f"[LASER] Active messages: {texts}")

        # Build device info
        devinfo_list = []
        for idx, h in enumerate(handles_sorted):
            try:
                FPGAv, FWv, DLLv = avs.AVS_GetVersionInfo(h)
            except Exception:
                FPGAv = FWv = DLLv = ""
            wl = wavelengths[h]
            info = {
                "handle": int(h),
                "serial": identities[idx].get("serial", "") if idx < len(identities) else "",
                "user_friendly_name": identities[idx].get("name", "") if idx < len(identities) else "",
                "fpga_version": FPGAv.decode("utf-8", "ignore") if hasattr(FPGAv, "decode") else str(FPGAv),
                "fw_version": FWv.decode("utf-8", "ignore") if hasattr(FWv, "decode") else str(FWv),
                "dll_version": DLLv.decode("utf-8", "ignore") if hasattr(DLLv, "decode") else str(DLLv),
                "wavelength_start_nm": float(wl[0]),
                "wavelength_stop_nm": float(wl[-1]),
                "pixels": int(len(wl)),
            }
            devinfo_list.append(info)
            self.log(f"[AVS] Dev{idx}: handle={info['handle']} serial={info['serial']} "
                     f"name={info['user_friendly_name']}")

        # Pixel ranges and wavelength concatenation
        device_pixel_ranges = {}
        for h in handles_sorted:
            pixel_count = len(wavelengths[h])
            device_pixel_ranges[h] = (0, pixel_count - 1)
            self.log(f"[ROI] Device {h}: full pixel range 0 to {pixel_count - 1}")

        wavelengths_concat = []
        for h in handles_sorted:
            wavelengths_concat.extend(wavelengths[h].tolist())
            self.log(f"[BANDS] Device {h}: added {len(wavelengths[h])} wavelengths")
        channel_number = len(wavelengths_concat)
        self.log(f"[BANDS] total channels={channel_number}")

        # File size estimation
        file_est = estimate_file_size(channel_number, mapping_X_points, mapping_Y_points,
                                      num_devices=len(handles_sorted))
        self._log_file_estimate(file_est)

        # Timing ticks
        integration_delay_ticks = ns_to_fpga_ticks(p.integration_delay_ns)

        # Metadata
        metadata = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "integration_time_ms": float(p.integration_time_ms),
            "integration_delay_ns": int(p.integration_delay_ns),
            "integration_delay_fpga_ticks": int(integration_delay_ticks),
            "averages": int(p.averages),
            "laser_frequency_hz": float(p.laser_frequency),
            "step_size_mm": float(p.step_size),
            "steps_per_mm": int(p.steps_per_mm),
            "step_trigger_steps": int(p.step_trigger),
            "x_travel_mm": float(p.x_travel_mm),
            "x_feed_mm_min": float(p.x_feed_mm_min),
            "mapping_X_points": int(mapping_X_points),
            "mapping_Y_points": int(mapping_Y_points),
            "mapping_X_size_mm": float(p.mapping_X_size),
            "mapping_Y_size_mm": float(p.mapping_Y_size),
            "bands_total": int(channel_number),
            "devices_total": int(len(handles_sorted)),
            "progress_selection": str(p.progress_selection),
            "progress_tol_nm": float(p.progress_tol_nm),
        }

        # Create NetCDF writer
        ncw = NetCDFCubeWriter(nc_fullpath, channel_number, mapping_X_points, mapping_Y_points,
                               wavelengths_concat, metadata, devinfo_list, log_fn=self.log)
        self.log(f"[NC] Wavelengths: {len(wavelengths_concat)} values, "
                 f"first={wavelengths_concat[0]:.2f} nm, last={wavelengths_concat[-1]:.2f} nm")

        # Capture dark
        dark_dict = capture_dark_once(handles_sorted, device_pixel_ranges,
                                      p.integration_time_ms, integration_delay_ticks,
                                      p.averages, log_fn=self.log)

        # Compute offsets
        device_offsets, offset = [], 0
        for h in handles_sorted:
            start_px, stop_px = device_pixel_ranges[h]
            span_device = stop_px - start_px + 1
            device_offsets.append((h, offset, offset + span_device))
            self.log(f"[OFFSET] Device {h}: offset {offset} to {offset + span_device - 1}")
            offset += span_device

        # Callback collector (reused across all lines)
        collector = CallbackCollector(handles_sorted, log_fn=self.log)
        self.log("[AVS] Callback collector initialised.")

        # Main scan loop (with automatic retry on laser fault / timeout)
        y_coords = np.arange(mapping_Y_points, dtype=int)
        try:
            for y in y_coords:
                if stop_event and stop_event.is_set():
                    self.log("[STOP] Stop requested, aborting mapping.")
                    break

                # Periodic full spectrometer DLL reset to prevent drift
                if (y > 0 and p.resync_every_n_lines > 0
                        and y % p.resync_every_n_lines == 0):
                    self.log(f"[RESYNC] Periodic re-init at y={y} "
                             f"(every {p.resync_every_n_lines} lines)")
                    old_handles = list(handles_sorted)
                    handles_sorted, wavelengths = \
                        self._reinit_spectrometers(
                            handles_sorted, wavelengths)
                    device_pixel_ranges = {
                        h: (0, len(wavelengths[h]) - 1)
                        for h in handles_sorted}
                    device_offsets, dark_dict, collector, new_ch = \
                        self._rebuild_scan_state(
                            handles_sorted, wavelengths,
                            device_pixel_ranges, dark_dict,
                            old_handles)
                    if new_ch != channel_number:
                        raise RuntimeError(
                            f"Channel count changed after resync: "
                            f"{channel_number} -> {new_ch}")

                for attempt in range(1, p.max_line_retries + 1):
                    try:
                        self._scan_single_line(
                            y, handles_sorted, device_offsets, dark_dict,
                            device_pixel_ranges, mapping_X_points,
                            mapping_Y_points, channel_number, ncw, cnc, pulse,
                            laser, p.x_travel_mm, p.x_feed_mm_min,
                            p.return_feed_mm_min, p.integration_time_ms,
                            integration_delay_ticks,
                            p.averages, p.step_trigger, collector, pulse_log_path,
                            is_first_line=(y == 0 and attempt == 1))
                        break  # line completed successfully
                    except ShotTimeoutError:
                        self.log(
                            f"[LINE] y={y} attempt {attempt}/{p.max_line_retries}"
                            f" FAILED (timeout)")
                        if attempt >= p.max_line_retries:
                            raise RuntimeError(
                                f"Line y={y} failed after {p.max_line_retries} "
                                f"retries — possible persistent laser fault.")
                        if stop_event and stop_event.is_set():
                            self.log("[STOP] Stop requested during recovery.")
                            break
                        self.log("[LINE] Attempting full system recovery "
                                 f"before retry {attempt + 1}/"
                                 f"{p.max_line_retries} ...")
                        self._recover_laser(laser)
                        old_handles = list(handles_sorted)
                        handles_sorted, wavelengths = \
                            self._reinit_spectrometers(
                                handles_sorted, wavelengths)
                        device_pixel_ranges = {
                            h: (0, len(wavelengths[h]) - 1)
                            for h in handles_sorted}
                        device_offsets, dark_dict, collector, new_ch = \
                            self._rebuild_scan_state(
                                handles_sorted, wavelengths,
                                device_pixel_ranges,
                                dark_dict, old_handles)
                        if new_ch != channel_number:
                            raise RuntimeError(
                                f"Channel count changed after recovery: "
                                f"{channel_number} -> {new_ch}")

                if stop_event and stop_event.is_set():
                    self.log("[STOP] Stop requested, aborting mapping.")
                    break

                # Progress callback
                if progress_callback and mapping_Y_points > 0:
                    progress_callback(int(((y + 1) / mapping_Y_points) * 100))

                # Move Y
                if y < y_coords[-1]:
                    self.log(f"[LINE] Moving Y by +{p.step_size:.3f} mm to next line...")
                    grbl_stream_jog_y(cnc, p.step_size, p.return_feed_mm_min, log_fn=self.log)
                    wait_cnc_idle(cnc, log_fn=self.log)

                # Progress PNG
                if p.progress_every_lines and ((y + 1) % p.progress_every_lines == 0):
                    save_progress_png(nc_fullpath, progress_png_path,
                                      selection=p.progress_selection, tol_nm=p.progress_tol_nm,
                                      log_fn=self.log)
                    if p.ftp_host and p.ftp_user and p.ftp_pass:
                        try:
                            ftp_upload_png(progress_png_path, p.ftp_host, p.ftp_user,
                                          p.ftp_pass, p.ftp_path, log_fn=self.log)
                        except Exception as e:
                            self.log(f"[FTP] Upload failed: {e}")
        finally:
            # Stop laser (don't close -- GUI manages the connection)
            try:
                laser.ensure_session()
                laser._send("$STOP")
                self.log("[LASER] Laser stopped.")
            except Exception as e:
                self.log(f"[LASER] Error stopping laser: {e}")
            ncw.close()
            self.log("=== STOP ===")

    # ------------------------------------------------------------------
    def _scan_single_line(self, y, handles_sorted, device_offsets, dark_dict,
                          device_pixel_ranges, mapping_X_points, mapping_Y_points,
                          channel_number, ncw, cnc, pulse, laser, x_travel_mm,
                          x_feed_mm_min, return_feed_mm_min, integration_time_ms,
                          integration_delay_ticks, averages,
                          step_trigger, collector, pulse_log_path,
                          is_first_line=False):
        """Scan a single line (y-coordinate).

        Uses callback-based collection (``AVS_MeasureCallback``) via the
        supplied *collector*.  The laser stays in ``$FIRE`` for the entire
        mapping — no laser commands are sent between lines.  Only the
        first line performs the full ``standby_and_fire`` warm-up.  If a
        timeout occurs the caller invokes ``_recover_laser`` which does a
        full ``$STOP`` → ``$STANDBY`` → ``wait_ready`` → ``$FIRE`` cycle.
        """
        p = self.p

        if is_first_line:
            self.log(f"[LINE] y={y}  first line — full standby+fire ...")
            laser.standby_and_fire(strict_warnings=False,
                                   wait_after=float(p.line_start_wait_s))

        self.log(f"[LINE] y={y}  arming (callback) ...")
        collector.arm(
            handles_sorted, mapping_X_points, device_pixel_ranges,
            integration_time_ms, integration_delay_ticks, averages,
            laser_delay_fpga=0)

        if not movement_trigger_reset_count(pulse, log_fn=self.log):
            self._append_pulse_log(
                pulse_log_path, y, mapping_X_points, "",
                "reset_failed", "pulse counter RESETCOUNT command failed")
            raise ShotTimeoutError(
                f"Line y={y} failed: pulse counter RESETCOUNT command failed")
        movement_trigger_start(pulse, step_trigger, log_fn=self.log)
        grbl_stream_jog_x(cnc, x_travel_mm + 1, x_feed_mm_min, log_fn=self.log)

        merged_vecs = {}

        def on_spectrum_save(handle, x_idx, tstamp, spec):
            if x_idx not in merged_vecs:
                merged_vecs[x_idx] = np.zeros(channel_number, dtype=np.float32)
            start_px, stop_px = device_pixel_ranges[handle]
            roi = np.asarray(spec[start_px:stop_px + 1], dtype=np.float32)
            dark = dark_dict.get(handle)
            if dark is not None:
                roi = roi - dark[start_px:stop_px + 1].astype(np.float32)
            for hh, lo, hi in device_offsets:
                if hh == handle:
                    merged_vecs[x_idx][lo:hi] = roi
                    break

        line_failed = False
        line_failure_reason = ""
        trigger_stopped = False
        x_coords = np.arange(mapping_X_points, dtype=int)
        try:
            for x in x_coords:
                collector.collect_next_shot(handles_sorted, on_spectrum_save, x,
                                           timeout=p.shot_timeout_s)
                out = np.clip(merged_vecs[x], 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
                ncw.write_xy(x, y, out)
                del merged_vecs[x]
            # Stop triggers immediately after the last acquired shot.
            # The X jog includes extra travel margin; stopping here prevents
            # additional off-map laser shots while motion decelerates/finishes.
            movement_trigger_stop(pulse, log_fn=self.log)
            trigger_stopped = True
        except ShotTimeoutError as exc:
            line_failed = True
            line_failure_reason = str(exc)
            self.log(f"[LINE] y={y} TIMEOUT: {exc}")
            try:
                movement_trigger_stop(pulse, log_fn=self.log)
                trigger_stopped = True
            except Exception as stop_exc:
                self.log(f"[LINE] Trigger stop error: {stop_exc}")
        finally:
            # Always stop spectrometer measurements to reset DLL/USB state.
            # Without this, internal buffers accumulate across lines and the
            # spectrometers desynchronize after many consecutive cycles.
            for h in handles_sorted:
                try:
                    avs.AVS_StopMeasure(h)
                except Exception as stop_exc:
                    self.log(f"[LINE] AVS_StopMeasure({h}) error: {stop_exc}")
            # Settle so DLL/USB can flush; then drain any late callbacks
            # so they don't leak into the next line.
            time.sleep(0.50)
            collector.drain_queues(handles_sorted)
            try:
                wait_cnc_idle(cnc, log_fn=self.log)
            except Exception as exc:
                self.log(f"[LINE] Wait idle error: {exc}")
            if not trigger_stopped:
                try:
                    movement_trigger_stop(pulse, log_fn=self.log)
                except Exception as exc:
                    self.log(f"[LINE] Trigger stop error: {exc}")
            pulse_count = movement_trigger_get_count(pulse, log_fn=self.log)
            if pulse_count is None:
                line_failed = True
                if not line_failure_reason:
                    line_failure_reason = (
                        "pulse counter COUNT command failed")
                self.log(f"[LINE] y={y} pulse counter read failed")
            elif pulse_count < int(mapping_X_points):
                line_failed = True
                line_failure_reason = (
                    f"pulse count below expected (expected at least "
                    f"{mapping_X_points}, got {pulse_count})")
                self.log(f"[LINE] y={y} {line_failure_reason}")
            self._append_pulse_log(
                pulse_log_path, y, mapping_X_points,
                "" if pulse_count is None else pulse_count,
                "failed" if line_failed else "ok",
                line_failure_reason if line_failed else "")
            try:
                grbl_stream_jog_x(cnc, -(x_travel_mm + 1), return_feed_mm_min,
                                  log_fn=self.log)
                wait_cnc_idle(cnc, quiet=True, log_fn=self.log)
            except Exception as exc:
                self.log(f"[LINE] Return jog error: {exc}")

        if line_failed:
            if not line_failure_reason:
                line_failure_reason = "timed out"
            raise ShotTimeoutError(f"Line y={y} failed: {line_failure_reason}")

        ncw.flush()
        self.log(f"[LINE] y={y} complete.")

    # ------------------------------------------------------------------
    def _recover_laser(self, laser):
        """Full laser recovery after a timeout: $STOP → $STANDBY → wait_ready → $FIRE.

        Uses generous delays between every command (enforced by
        ``VironLaser.CMD_GAP_S``) and allows up to 2 minutes for the
        laser to report ready after a fault.
        """
        p = self.p
        self.log("=" * 60)
        self.log("[RECOVERY] Laser timeout — starting full recovery ...")

        self.log("[RECOVERY] Sending $STOP ...")
        laser.ensure_session()
        try:
            laser._send("$STOP", expect_ack=False)
        except Exception as exc:
            self.log(f"[RECOVERY] $STOP error: {exc}")

        self.log("[RECOVERY] Waiting 30 s for laser to settle ...")
        time.sleep(30.0)

        self.log("[RECOVERY] Re-configuring laser for mapping ...")
        laser.configure_for_mapping(single_shot=p.laser_single_shot_mode)

        self.log("[RECOVERY] Sending $STANDBY ...")
        laser._send("$STANDBY", expect_ack=False)
        time.sleep(120.0)
        # let's not risk a problem here, just wait for 120 seconds
        # self.log("[RECOVERY] Waiting for laser ready (up to 120 s) ...")
        # laser.wait_ready(max_wait=120.0, poll=2.0, strict_warnings=False)

        self.log("[RECOVERY] Sending $FIRE ...")
        laser._send("$FIRE", expect_ack=False)
        time.sleep(float(p.line_start_wait_s))

        self.log("[RECOVERY] Laser recovery complete.")
        self.log("=" * 60)

    # ------------------------------------------------------------------
    def _reinit_spectrometers(self, old_handles, old_wavelengths):
        """Full AVS library teardown + re-init to flush DLL/USB state.

        After many hundreds of arm/stop measurement cycles the Avantes DLL
        can accumulate internal buffer state that causes spectrometer
        triggers to desynchronize from the laser.  This performs a complete
        StopMeasure → Deactivate → Done → Init → Activate cycle, matching
        the reactivated devices back to the original selection by wavelength
        range so that handle values changing is transparent to the caller.

        Returns ``(new_handles_sorted, new_wavelengths)``.
        """
        self.log("=" * 60)
        self.log("[RESYNC] Full spectrometer DLL re-initialization ...")

        old_ranges = [
            (float(old_wavelengths[h][0]), float(old_wavelengths[h][-1]))
            for h in old_handles
        ]

        for h in old_handles:
            try:
                avs.AVS_StopMeasure(h)
            except Exception as exc:
                self.log(f"[RESYNC] AVS_StopMeasure({h}): {exc}")
        time.sleep(0.5)

        for h in old_handles:
            try:
                avs.AVS_Deactivate(h)
            except Exception as exc:
                self.log(f"[RESYNC] AVS_Deactivate({h}): {exc}")
        time.sleep(0.5)

        try:
            avs.AVS_Done()
        except Exception as exc:
            self.log(f"[RESYNC] AVS_Done: {exc}")
        time.sleep(1.0)

        new_handles_all, new_wl_all, _ = discover_and_sort_by_range(
            log_fn=self.log)

        matched = []
        used = set()
        for old_start, old_stop in old_ranges:
            best_h, best_diff = None, float('inf')
            for h in new_handles_all:
                if h in used:
                    continue
                diff = (abs(float(new_wl_all[h][0]) - old_start)
                        + abs(float(new_wl_all[h][-1]) - old_stop))
                if diff < best_diff:
                    best_diff = diff
                    best_h = h
            if best_h is not None:
                matched.append(best_h)
                used.add(best_h)

        for h in new_handles_all:
            if h not in used:
                try:
                    avs.AVS_Deactivate(h)
                except Exception:
                    pass

        new_wavelengths = {h: new_wl_all[h] for h in matched}
        self.log(f"[RESYNC] Matched {len(matched)} device(s) to original "
                 f"selection")
        self.log("=" * 60)
        return matched, new_wavelengths

    # ------------------------------------------------------------------
    def _rebuild_scan_state(self, handles_sorted, wavelengths,
                            device_pixel_ranges, old_dark_dict,
                            old_handles):
        """Rebuild device offsets, remap dark spectra, and create a fresh
        CallbackCollector after a spectrometer re-initialization.

        The dark is captured only once per mapping, so instead of
        recapturing we remap the existing *old_dark_dict* (keyed by old
        handles) onto the new handles by matching wavelength order.
        """
        device_offsets = []
        offset = 0
        for h in handles_sorted:
            start_px, stop_px = device_pixel_ranges[h]
            span = stop_px - start_px + 1
            device_offsets.append((h, offset, offset + span))
            self.log(f"[RESYNC] Device {h}: offset {offset} to "
                     f"{offset + span - 1}")
            offset += span
        channel_number = offset

        new_dark_dict = {}
        for new_h, old_h in zip(handles_sorted, old_handles):
            if old_h in old_dark_dict:
                new_dark_dict[new_h] = old_dark_dict[old_h]
        self.log(f"[RESYNC] Dark spectra remapped from {len(old_dark_dict)} "
                 f"old handle(s) to {len(new_dark_dict)} new handle(s)")

        collector = CallbackCollector(handles_sorted, log_fn=self.log)
        self.log("[RESYNC] Scan state rebuilt (offsets + collector)")
        return device_offsets, new_dark_dict, collector, channel_number

    # ------------------------------------------------------------------
    def _append_pulse_log(self, pulse_log_path, y, expected, pulse_count,
                          status, note=""):
        """Append one line-level pulse parity record to CSV."""
        safe_note = str(note).replace(",", ";")
        with open(pulse_log_path, "a", encoding="utf-8") as f:
            f.write(f"{int(y)},{int(expected)},{pulse_count},{status},{safe_note}\n")

    # ------------------------------------------------------------------
    def _log_file_estimate(self, file_est):
        self.log("[ESTIMATION] File Size Estimation:")
        compression_reduction = (1.0 - file_est['compression_ratio']) * 100
        if file_est['estimated_mb'] < 1:
            self.log(f"  Estimated: {file_est['estimated_bytes'] / 1024:.2f} KB "
                     f"(compressed, ~{compression_reduction:.0f}% reduction)")
        elif file_est['estimated_mb'] < 1024:
            self.log(f"  Estimated: {file_est['estimated_mb']:.2f} MB "
                     f"(compressed, ~{compression_reduction:.0f}% reduction)")
        else:
            self.log(f"  Estimated: {file_est['estimated_gb']:.2f} GB "
                     f"(compressed, ~{compression_reduction:.0f}% reduction)")
        self.log(f"  Main data array: {file_est['mapping_data_raw'] / (1024 * 1024):.2f} MB (uint16)")
        self.log("=" * 60)


