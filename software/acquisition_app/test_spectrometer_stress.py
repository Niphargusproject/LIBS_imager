# -*- coding: utf-8 -*-
"""
Spectrometer stress-test  (no laser, no CNC, software trigger)
--------------------------------------------------------------
Simulates a full mapping grid using only the spectrometers to isolate
whether the data-transfer pipeline can sustain the required throughput.

Usage:
    python test_spectrometer_stress.py                       # defaults: 500 x 350
    python test_spectrometer_stress.py --nx 500 --ny 350     # explicit
    python test_spectrometer_stress.py --nx 100 --ny 50 --mode poll   # polling fallback

The script discovers all connected AvaSpec spectrometers, arms them for
*software-triggered* scans (trig_mode=0, no external trigger needed),
collects data shot-by-shot exactly like the mapping engine, and reports
timing and any failures.  Data is discarded (not written to disk) so
that disk I/O cannot be a confounding factor.
"""
import os
import sys
import time
import argparse
import queue as _queue
from pathlib import Path

# Ensure local modules are importable
_here = Path(__file__).parent.absolute()
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
try:
    os.add_dll_directory(str(_here))
except AttributeError:
    pass

import threading
import numpy as np
import avaspec_fix_winfunctype as avs

try:
    from pypylon import pylon
except Exception:
    pylon = None


# ── Basler camera USB load ──────────────────────────────────────────────

class CameraLoad:
    """Open a Basler camera and continuously grab frames in the background.

    This saturates the USB bus the same way the real app does during mapping,
    so USB bandwidth contention with the spectrometers is replicated.
    """

    def __init__(self, log):
        self._log = log
        self._running = False
        self._thread = None
        self._cam = None
        self._frame_count = 0
        self._error_count = 0

    def start(self):
        if pylon is None:
            self._log("[CAM] pypylon not available — skipping camera load")
            return False
        try:
            tl = pylon.TlFactory.GetInstance()
            devices = tl.EnumerateDevices()
            if not devices:
                self._log("[CAM] No Basler camera detected — skipping camera load")
                return False
            self._cam = pylon.InstantCamera(tl.CreateDevice(devices[0]))
            self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="CamStress",
                                           daemon=True)
            self._thread.start()
            self._log("[CAM] Basler camera opened — continuous grab running")
            return True
        except Exception as exc:
            self._log(f"[CAM] Failed to start camera: {exc}")
            return False

    def _loop(self):
        while self._running:
            try:
                if not (self._cam and self._cam.IsGrabbing()):
                    time.sleep(0.1)
                    continue
                grab = self._cam.RetrieveResult(2000, pylon.TimeoutHandling_Return)
                if grab is None:
                    time.sleep(0.005)
                    continue
                try:
                    if grab.GrabSucceeded():
                        _img = self._converter.Convert(grab).GetArray()
                        self._frame_count += 1
                    else:
                        self._error_count += 1
                finally:
                    grab.Release()
            except Exception:
                self._error_count += 1
                time.sleep(0.01)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cam:
            try:
                self._cam.StopGrabbing()
                self._cam.Close()
            except Exception:
                pass
        self._log(f"[CAM] Stopped — {self._frame_count} frames grabbed, "
                  f"{self._error_count} errors")

    @property
    def frames(self):
        return self._frame_count


# ── helpers (copied from mapping_engine to keep this self-contained) ─────

class ShotTimeoutError(Exception):
    pass


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


def _get_scope(handle):
    ret = avs.AVS_GetScopeData(handle)
    if isinstance(ret, tuple) and len(ret) == 2:
        ts, spec = ret
    else:
        ts, spec = 0, ret
    return int(ts), np.asarray(spec, dtype=np.float32)


def build_meas_config_sw(start_px, stop_px, int_ms, navg):
    """Software-triggered config (trig_mode=0): scan starts immediately."""
    cfg = avs.MeasConfigType()
    cfg.m_StartPixel = int(start_px)
    cfg.m_StopPixel = int(stop_px)
    cfg.m_IntegrationTime = float(int_ms)
    cfg.m_IntegrationDelay = 0
    cfg.m_NrAverages = int(max(1, navg))
    cfg.m_CorDynDark_m_Enable = 0
    cfg.m_CorDynDark_m_ForgetPercentage = 0
    cfg.m_Smoothing_m_SmoothPix = 0
    cfg.m_Smoothing_m_SmoothModel = 0
    cfg.m_SaturationDetection = 0
    cfg.m_Trigger_m_Mode = 0        # software trigger
    cfg.m_Trigger_m_Source = 0
    cfg.m_Trigger_m_SourceType = 0
    cfg.m_Control_m_StrobeControl = 0
    cfg.m_Control_m_LaserDelay = 0
    cfg.m_Control_m_LaserWidth = 0
    cfg.m_Control_m_LaserWaveLength = 0.0
    cfg.m_Control_m_StoreToRam = 0
    return cfg


# ── discover spectrometers ───────────────────────────────────────────────

def discover(log):
    log("[AVS] AVS_Init(0)")
    avs.AVS_Init(0)
    log("[AVS] AVS_UpdateUSBDevices()")
    avs.AVS_UpdateUSBDevices()
    id_list = avs.AVS_GetList()
    log(f"[AVS] Found {len(id_list)} device(s)")
    if not id_list:
        time.sleep(0.5)
        avs.AVS_UpdateUSBDevices()
        id_list = avs.AVS_GetList()
        log(f"[AVS] Retry: {len(id_list)} device(s)")
    if not id_list:
        raise RuntimeError("No AvaSpec spectrometers found.")

    handles, wavelengths, pixel_ranges = [], {}, {}
    for i, dev in enumerate(id_list):
        sn = _bytes_to_str(dev.SerialNumber)
        nm = _bytes_to_str(dev.UserFriendlyName)
        h = avs.AVS_Activate(dev)
        handles.append(h)
        try:
            avs.AVS_UseHighResAdc(h, True)
        except Exception:
            pass
        px = _get_num_pixels(h)
        wl = np.asarray(avs.AVS_GetLambda(h), dtype=np.float64)[:px]
        wavelengths[h] = wl
        pixel_ranges[h] = (0, px - 1)
        log(f"  [{i}] handle={h}  serial={sn}  name={nm}  "
            f"pixels={px}  range={wl[0]:.1f}-{wl[-1]:.1f} nm")

    handles_sorted = sorted(handles, key=lambda hh: float(wavelengths[hh][0]))
    return handles_sorted, wavelengths, pixel_ranges


# ── callback collector (same as mapping_engine) ─────────────────────────

class CallbackCollector:
    def __init__(self, handles):
        self._ready = {h: _queue.Queue() for h in handles}

        def _on_ready(handle_ptr, result_ptr):
            h = handle_ptr[0]
            q = self._ready.get(h)
            if q is not None:
                q.put(result_ptr[0])

        self._cb = avs.AVS_MeasureCallbackFunc(_on_ready)

    def arm(self, handles, x_points, pixel_ranges, int_ms):
        for h in handles:
            q = self._ready[h]
            while not q.empty():
                try:
                    q.get_nowait()
                except _queue.Empty:
                    break
        for h in handles:
            start_px, stop_px = pixel_ranges[h]
            cfg = build_meas_config_sw(start_px, stop_px, int_ms, navg=1)
            err = avs.AVS_PrepareMeasure(h, cfg)
            if err != 0:
                raise RuntimeError(f"PrepareMeasure failed h={h} err={err}")
        for h in handles:
            err = avs.AVS_MeasureCallback(h, self._cb, int(x_points))
            if err != 0:
                raise RuntimeError(f"MeasureCallback failed h={h} err={err}")

    def collect(self, handles, shot_index, timeout=30.0):
        deadline = time.time() + timeout
        for h in handles:
            remaining = max(0.001, deadline - time.time())
            try:
                _status = self._ready[h].get(timeout=remaining)
            except _queue.Empty:
                raise ShotTimeoutError(
                    f"TIMEOUT at shot x={shot_index}, handle {h}")
            _ts, _spec = _get_scope(h)


# ── polling collector (fallback, for comparison) ────────────────────────

class PollCollector:
    def arm(self, handles, x_points, pixel_ranges, int_ms):
        for h in handles:
            start_px, stop_px = pixel_ranges[h]
            cfg = build_meas_config_sw(start_px, stop_px, int_ms, navg=1)
            err = avs.AVS_PrepareMeasure(h, cfg)
            if err != 0:
                raise RuntimeError(f"PrepareMeasure failed h={h} err={err}")
        for h in handles:
            err = avs.AVS_Measure(h, 0, int(x_points))
            if err != 0:
                raise RuntimeError(f"Measure failed h={h} err={err}")

    def collect(self, handles, shot_index, timeout=30.0):
        pending = set(handles)
        deadline = time.time() + timeout
        while pending:
            for h in list(pending):
                if avs.AVS_PollScan(h):
                    _ts, _spec = _get_scope(h)
                    pending.discard(h)
            if time.time() > deadline:
                raise ShotTimeoutError(
                    f"TIMEOUT at shot x={shot_index}, "
                    f"still waiting on {len(pending)} handle(s)")
            time.sleep(0.0005)


# ── main stress test ────────────────────────────────────────────────────

def run_stress(nx, ny, int_ms, freq_hz, mode, timeout_s, use_camera, log):
    handles, wavelengths, pixel_ranges = discover(log)
    n_devices = len(handles)
    total_channels = sum(len(wavelengths[h]) for h in handles)
    shot_interval = 1.0 / freq_hz if freq_hz > 0 else 0.0

    cam = CameraLoad(log)
    cam_active = False
    if use_camera:
        cam_active = cam.start()

    log(f"\n{'='*60}")
    log(f"STRESS TEST: {nx} x {ny} = {nx*ny:,} shots/device")
    log(f"  Devices:      {n_devices}")
    log(f"  Channels:     {total_channels}")
    log(f"  Integration:  {int_ms} ms")
    log(f"  Target freq:  {freq_hz} Hz  (interval {shot_interval*1000:.1f} ms)")
    log(f"  Mode:         {mode}")
    log(f"  Camera USB:   {'ACTIVE' if cam_active else 'OFF'}")
    log(f"  Shot timeout: {timeout_s} s")
    log(f"  Total scans:  {nx*ny*n_devices:,}")
    log(f"{'='*60}\n")

    if mode == "callback":
        collector = CallbackCollector(handles)
    else:
        collector = PollCollector()

    total_shots = 0
    failed_lines = []
    t_run_start = time.time()

    for y in range(ny):
        t_line_start = time.time()
        try:
            collector.arm(handles, nx, pixel_ranges, int_ms)

            for x in range(nx):
                t_shot_due = t_line_start + x * shot_interval
                collector.collect(handles, x, timeout=timeout_s)
                total_shots += 1
                # Pace to target frequency so timing matches real mapping
                wait = t_shot_due + shot_interval - time.time()
                if wait > 0:
                    time.sleep(wait)

        except ShotTimeoutError as exc:
            t_fail = time.time() - t_line_start
            log(f"  *** FAIL  y={y:4d}  after {t_fail:.1f}s  |  {exc}")
            failed_lines.append(y)
        except Exception as exc:
            t_fail = time.time() - t_line_start
            log(f"  *** ERROR y={y:4d}  after {t_fail:.1f}s  |  {exc}")
            failed_lines.append(y)
        finally:
            for h in handles:
                try:
                    avs.AVS_StopMeasure(h)
                except Exception:
                    pass

        t_line = time.time() - t_line_start
        rate = nx / t_line if t_line > 0 else 0

        if (y + 1) % 10 == 0 or y == ny - 1 or y in failed_lines:
            elapsed = time.time() - t_run_start
            pct = (y + 1) / ny * 100
            cam_info = f"  cam_frames={cam.frames}" if cam_active else ""
            log(f"  y={y:4d}/{ny}  ({pct:5.1f}%)  "
                f"line={t_line:6.2f}s  rate={rate:6.1f} shots/s  "
                f"elapsed={elapsed:7.1f}s  fails={len(failed_lines)}"
                f"{cam_info}")

        if len(failed_lines) >= 10:
            log("\n  Too many failures — aborting early.")
            break

    cam.stop()

    t_total = time.time() - t_run_start
    log(f"\n{'='*60}")
    log(f"RESULTS")
    log(f"  Lines completed: {ny - len(failed_lines)}/{ny}")
    log(f"  Total shots OK:  {total_shots:,}")
    log(f"  Failed lines:    {failed_lines if failed_lines else 'NONE'}")
    log(f"  Total time:      {t_total:.1f}s")
    if total_shots > 0:
        log(f"  Avg rate:        {total_shots / t_total:.1f} shots/s")
    if cam_active:
        log(f"  Camera frames:   {cam.frames}")
    log(f"{'='*60}")

    return len(failed_lines) == 0


# ── entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spectrometer stress-test (software trigger, no laser/CNC)")
    parser.add_argument("--nx", type=int, default=500,
                        help="Shots per line (X points, default 500)")
    parser.add_argument("--ny", type=int, default=350,
                        help="Number of lines (Y points, default 350)")
    parser.add_argument("--int-ms", type=float, default=0.010,
                        help="Integration time in ms (default 0.010)")
    parser.add_argument("--freq-hz", type=float, default=200.0,
                        help="Target acquisition frequency in Hz (default 200)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-shot timeout in seconds (default 30)")
    parser.add_argument("--mode", choices=["callback", "poll"], default="callback",
                        help="Collection mode (default: callback)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Skip Basler camera (no USB contention test)")
    args = parser.parse_args()

    log_path = os.path.join(str(_here), "stress_test_log.txt")
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Log file: {log_path}")
    try:
        ok = run_stress(args.nx, args.ny, args.int_ms, args.freq_hz,
                        args.mode, args.timeout, not args.no_camera, log)
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        log("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        log(f"\nFATAL: {exc}")
        import traceback
        log(traceback.format_exc())
        sys.exit(2)
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
