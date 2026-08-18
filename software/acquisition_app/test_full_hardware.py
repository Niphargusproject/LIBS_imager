# -*- coding: utf-8 -*-
"""
Full hardware integration test  (laser + CNC + pulse + spectrometers)
---------------------------------------------------------------------
Runs a real mapping-like loop with all hardware active but writes no
NetCDF — data is checked in-memory and discarded.  This isolates
timing / synchronisation issues between the trigger chain and the
spectrometer data pipeline.

The CNC jogs forward, the pulse controller fires step-triggers, the
laser fires, and the spectrometers acquire on the external trigger
exactly as in a real mapping run.

Usage (reads defaults from config.json):
    python test_full_hardware.py
    python test_full_hardware.py --nx 500 --ny 10         # short run
    python test_full_hardware.py --ny 350 --mode poll      # polling fallback
    python test_full_hardware.py --no-camera               # skip Basler
"""
import os
import sys
import json
import time
import argparse
import threading
import queue as _queue
from pathlib import Path

_here = Path(__file__).parent.absolute()
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
try:
    os.add_dll_directory(str(_here))
except AttributeError:
    pass

import numpy as np
import serial
import avaspec_fix_winfunctype as avs

from hardware_control import (
    VironLaser,
    grbl_cmd, get_grbl_status, wait_cnc_idle,
    grbl_stream_jog_x, grbl_stream_jog_y,
    pulse_open, pulse_handshake,
    movement_trigger_start, movement_trigger_stop,
)

try:
    from pypylon import pylon
except Exception:
    pylon = None


# ── logging ─────────────────────────────────────────────────────────────

_log_file = None

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")


# ── load config.json defaults ───────────────────────────────────────────

def load_config():
    cfg_path = _here / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}


# ── camera background load (same as stress test) ───────────────────────

class CameraLoad:
    def __init__(self):
        self._running = False
        self._thread = None
        self._cam = None
        self.frames = 0
        self.errors = 0

    def start(self):
        if pylon is None:
            log("[CAM] pypylon not available — skipping")
            return False
        try:
            tl = pylon.TlFactory.GetInstance()
            devices = tl.EnumerateDevices()
            if not devices:
                log("[CAM] No Basler camera found — skipping")
                return False
            self._cam = pylon.InstantCamera(tl.CreateDevice(devices[0]))
            self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._conv = pylon.ImageFormatConverter()
            self._conv.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            log("[CAM] Basler camera grab running")
            return True
        except Exception as exc:
            log(f"[CAM] Failed: {exc}")
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
                        _ = self._conv.Convert(grab).GetArray()
                        self.frames += 1
                    else:
                        self.errors += 1
                finally:
                    grab.Release()
            except Exception:
                self.errors += 1
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
        log(f"[CAM] Stopped — {self.frames} frames, {self.errors} errors")


# ── spectrometer helpers ────────────────────────────────────────────────

class ShotTimeoutError(Exception):
    pass

def _bytes_to_str(b):
    return bytes(b).split(b"\x00", 1)[0].decode("utf-8", "ignore")

def _get_num_pixels(h):
    npx = avs.AVS_GetNumPixels(h)
    try:
        return int(npx)
    except Exception:
        return int(npx[0]) if isinstance(npx, (tuple, list)) else int(npx)

def _get_scope(handle):
    ret = avs.AVS_GetScopeData(handle)
    if isinstance(ret, tuple) and len(ret) == 2:
        ts, spec = ret
    else:
        ts, spec = 0, ret
    return int(ts), np.asarray(spec, dtype=np.float32)

def ns_to_fpga_ticks(ns):
    return int(round(ns / 20.83)) if ns else 0


def discover_spectrometers():
    log("[AVS] AVS_Init(0)")
    avs.AVS_Init(0)
    avs.AVS_UpdateUSBDevices()
    id_list = avs.AVS_GetList()
    log(f"[AVS] Found {len(id_list)} device(s)")
    if not id_list:
        time.sleep(0.5)
        avs.AVS_UpdateUSBDevices()
        id_list = avs.AVS_GetList()
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


# ── measurement config (external trigger, same as mapping_engine) ───────

def build_meas_config_ext(start_px, stop_px, int_ms, int_delay_fpga, navg):
    """External-trigger config (trig_mode=2, source=0, edge)."""
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
    cfg.m_Trigger_m_Mode = 2          # external trigger
    cfg.m_Trigger_m_Source = 0
    cfg.m_Trigger_m_SourceType = 0    # edge
    cfg.m_Control_m_StrobeControl = 0
    cfg.m_Control_m_LaserDelay = 0
    cfg.m_Control_m_LaserWidth = 0
    cfg.m_Control_m_LaserWaveLength = 0.0
    cfg.m_Control_m_StoreToRam = 0
    return cfg


# ── callback collector ──────────────────────────────────────────────────

class CallbackCollector:
    def __init__(self, handles):
        self._ready = {h: _queue.Queue() for h in handles}
        def _on_ready(handle_ptr, result_ptr):
            h = handle_ptr[0]
            q = self._ready.get(h)
            if q is not None:
                q.put(result_ptr[0])
        self._cb = avs.AVS_MeasureCallbackFunc(_on_ready)

    def arm(self, handles, x_points, pixel_ranges, int_ms, int_delay_fpga):
        for h in handles:
            q = self._ready[h]
            while not q.empty():
                try:
                    q.get_nowait()
                except _queue.Empty:
                    break
        log(f"[AVS] Arming (callback, ext trigger): x_points={x_points}")
        for h in handles:
            start_px, stop_px = pixel_ranges[h]
            cfg = build_meas_config_ext(start_px, stop_px, int_ms,
                                        int_delay_fpga, navg=1)
            err = avs.AVS_PrepareMeasure(h, cfg)
            log(f"[AVS] PrepareMeasure(handle={h}) "
                f"px={start_px}-{stop_px} -> {err}")
            if err != 0:
                raise RuntimeError(f"PrepareMeasure failed h={h} err={err}")
        for h in handles:
            err = avs.AVS_MeasureCallback(h, self._cb, int(x_points))
            log(f"[AVS] MeasureCallback(handle={h}, scans={x_points}) -> {err}")
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
                    f"TIMEOUT shot x={shot_index}, handle {h}")
            _ts, _spec = _get_scope(h)


# ── poll collector (fallback) ───────────────────────────────────────────

class PollCollector:
    def arm(self, handles, x_points, pixel_ranges, int_ms, int_delay_fpga):
        log(f"[AVS] Arming (poll, ext trigger): x_points={x_points}")
        for h in handles:
            start_px, stop_px = pixel_ranges[h]
            cfg = build_meas_config_ext(start_px, stop_px, int_ms,
                                        int_delay_fpga, navg=1)
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
                    f"TIMEOUT shot x={shot_index}, "
                    f"{len(pending)} handle(s) pending")
            time.sleep(0.0005)


# ── main integration test ───────────────────────────────────────────────

def run_test(args, cfg):
    # ── Mapping parameters ──
    step_size    = args.step_size
    x_size       = step_size * args.nx
    laser_freq   = args.laser_freq
    steps_per_mm = args.steps_per_mm
    int_ms       = args.int_ms
    int_delay_ns = args.int_delay_ns
    line_wait_s  = args.line_wait
    return_feed  = args.return_feed

    nx = args.nx
    ny = args.ny
    x_travel     = x_size + step_size * 10
    x_feed       = step_size * laser_freq * 60.0    # mm/min
    step_trigger = int(round(steps_per_mm * step_size))
    int_delay_fpga = ns_to_fpga_ticks(int_delay_ns)
    shot_timeout = args.timeout

    # ── Read connection settings from config.json (or CLI overrides) ──
    serial_cfg = cfg.get("serial", {})
    laser_cfg  = cfg.get("laser", {})

    cnc_port   = args.cnc_port   or serial_cfg.get("cnc_port", "COM8")
    cnc_baud   = args.cnc_baud   or serial_cfg.get("cnc_baudrate", 115200)
    pulse_port = args.pulse_port or serial_cfg.get("pulse_port", "COM3")
    pulse_baud = args.pulse_baud or serial_cfg.get("pulse_baudrate", 115200)
    laser_host = args.laser_host or laser_cfg.get("host", "192.168.103.103")
    laser_port = args.laser_port or laser_cfg.get("port", 23)
    laser_login = args.laser_login or laser_cfg.get("login_code", "")

    log(f"\n{'='*60}")
    log(f"FULL HARDWARE TEST")
    log(f"  Grid:          {nx} x {ny}  (step={step_size} mm)")
    log(f"  X travel:      {x_travel:.1f} mm  @ {x_feed:.1f} mm/min")
    log(f"  Laser freq:    {laser_freq} Hz")
    log(f"  Step trigger:  {step_trigger} steps")
    log(f"  Integration:   {int_ms} ms  delay={int_delay_ns} ns")
    log(f"  Line wait:     {line_wait_s} s")
    log(f"  Return feed:   {return_feed} mm/min")
    log(f"  Shot timeout:  {shot_timeout} s")
    log(f"  CNC:           {cnc_port} @ {cnc_baud}")
    log(f"  Pulse:         {pulse_port} @ {pulse_baud}")
    log(f"  Laser:         {laser_host}:{laser_port}")
    log(f"{'='*60}\n")

    # ── Camera ──
    cam = CameraLoad()
    cam_active = cam.start() if not args.no_camera else False

    # ── Spectrometers ──
    handles, wavelengths, pixel_ranges = discover_spectrometers()
    total_ch = sum(len(wavelengths[h]) for h in handles)
    log(f"[AVS] {len(handles)} device(s), {total_ch} total channels")

    if args.mode == "callback":
        collector = CallbackCollector(handles)
    else:
        collector = PollCollector()

    # ── CNC ──
    log(f"[CNC] Opening {cnc_port} ...")
    cnc = serial.Serial(cnc_port, int(cnc_baud), timeout=1)
    time.sleep(2)
    cnc.reset_input_buffer()
    cnc.write(b"\r\n\r\n")
    time.sleep(1)
    cnc.reset_input_buffer()
    wait_cnc_idle(cnc, log_fn=log)
    log("[CNC] Ready")

    # ── Pulse controller ──
    pulse = pulse_open(pulse_port, int(pulse_baud), log_fn=log)
    pulse_handshake(pulse, step_trigger=step_trigger, log_fn=log)
    log("[PULSE] Ready")

    # ── Laser ──
    laser = VironLaser(laser_host, int(laser_port), login=laser_login)
    laser.connect()
    laser.configure_for_mapping(single_shot=False)
    tup, decoded, raw, texts = laser.status_readable()
    log(f"[LASER] Connected — STATUS: {raw}")
    if texts:
        log(f"[LASER] TEXTS: {texts}")

    # ── Run ──
    total_shots = 0
    failed_lines = []
    t_run_start = time.time()

    try:
        for y in range(ny):
            t_line = time.time()
            line_ok = False
            try:
                # Fire laser
                log(f"[LINE] y={y}  STANDBY + FIRE ...")
                laser.standby_and_fire(strict_warnings=False,
                                       wait_after=float(line_wait_s))

                # Arm spectrometers (external trigger)
                collector.arm(handles, nx, pixel_ranges, int_ms,
                              int_delay_fpga)

                # Start trigger chain + CNC jog
                movement_trigger_start(pulse, step_trigger, log_fn=log)
                grbl_stream_jog_x(cnc, x_travel + 1, x_feed, log_fn=log)

                # Collect shots
                for x in range(nx):
                    collector.collect(handles, x, timeout=shot_timeout)
                    total_shots += 1

                line_ok = True

            except ShotTimeoutError as exc:
                log(f"  *** FAIL  y={y}  |  {exc}")
                failed_lines.append(y)
                try:
                    movement_trigger_stop(pulse, log_fn=log)
                except Exception:
                    pass

            except Exception as exc:
                log(f"  *** ERROR y={y}  |  {exc}")
                failed_lines.append(y)

            finally:
                # Always: stop spectrometers, wait CNC, stop triggers, return
                for h in handles:
                    try:
                        avs.AVS_StopMeasure(h)
                    except Exception:
                        pass
                try:
                    wait_cnc_idle(cnc, quiet=True, log_fn=log)
                except Exception:
                    pass
                if line_ok:
                    try:
                        movement_trigger_stop(pulse, log_fn=log)
                    except Exception:
                        pass
                try:
                    grbl_stream_jog_x(cnc, -(x_travel + 1), return_feed,
                                      log_fn=log)
                    wait_cnc_idle(cnc, quiet=True, log_fn=log)
                except Exception as exc:
                    log(f"[LINE] Return error: {exc}")

                # Laser back to standby
                try:
                    laser._send("$STANDBY", expect_ack=False)
                except Exception:
                    pass

            dt = time.time() - t_line
            rate = nx / dt if dt > 0 else 0

            if (y + 1) % 5 == 0 or y == ny - 1 or y in failed_lines:
                elapsed = time.time() - t_run_start
                pct = (y + 1) / ny * 100
                cam_info = f"  cam={cam.frames}" if cam_active else ""
                log(f"  y={y:4d}/{ny}  ({pct:5.1f}%)  "
                    f"line={dt:6.1f}s  rate={rate:5.1f} shots/s  "
                    f"elapsed={elapsed:7.1f}s  fails={len(failed_lines)}"
                    f"{cam_info}")

            # Move Y (unless last line)
            if y < ny - 1 and y not in failed_lines:
                grbl_stream_jog_y(cnc, step_size, return_feed, log_fn=log)
                wait_cnc_idle(cnc, quiet=True, log_fn=log)

            if len(failed_lines) >= 10:
                log("\n  Too many failures — aborting.")
                break

    finally:
        # Cleanup
        try:
            laser._send("$STOP")
            log("[LASER] Stopped")
        except Exception:
            pass
        try:
            laser.close()
        except Exception:
            pass
        try:
            cnc.close()
        except Exception:
            pass
        try:
            pulse.close()
        except Exception:
            pass
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
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Full hardware integration test "
                    "(laser + CNC + pulse + spectrometers)")

    # Grid
    parser.add_argument("--nx", type=int, default=500)
    parser.add_argument("--ny", type=int, default=10,
                        help="Number of Y lines (default 10 for safety)")
    parser.add_argument("--step-size", type=float, default=0.10)
    parser.add_argument("--steps-per-mm", type=int, default=800)

    # Laser & timing
    parser.add_argument("--laser-freq", type=float, default=15.0,
                        help="Laser frequency Hz (default 15)")
    parser.add_argument("--int-ms", type=float, default=0.010,
                        help="Integration time ms (default 0.010)")
    parser.add_argument("--int-delay-ns", type=int, default=1000,
                        help="Integration delay ns (default 1000)")
    parser.add_argument("--line-wait", type=float, default=5.0,
                        help="Wait after FIRE before scanning (default 5s)")
    parser.add_argument("--return-feed", type=float, default=100.0,
                        help="Return/Y feed mm/min (default 100)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-shot timeout seconds (default 30)")

    # Connection overrides (default from config.json)
    parser.add_argument("--cnc-port", type=str, default="")
    parser.add_argument("--cnc-baud", type=int, default=0)
    parser.add_argument("--pulse-port", type=str, default="")
    parser.add_argument("--pulse-baud", type=int, default=0)
    parser.add_argument("--laser-host", type=str, default="")
    parser.add_argument("--laser-port", type=int, default=0)
    parser.add_argument("--laser-login", type=str, default="")

    # Options
    parser.add_argument("--mode", choices=["callback", "poll"],
                        default="callback")
    parser.add_argument("--no-camera", action="store_true")

    args = parser.parse_args()

    global _log_file
    log_path = os.path.join(str(_here), "full_hardware_test_log.txt")
    _log_file = open(log_path, "w", buffering=1)

    log(f"Log file: {log_path}")
    try:
        ok = run_test(args, cfg)
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
        _log_file.close()


if __name__ == "__main__":
    main()
