# -*- coding: utf-8 -*-
"""
Hardware control helpers for the LIBS application.

Contains:
* Viron laser TCP control (VironLaser class + status decode tables)
* GRBL CNC serial helpers (open, handshake, jog, wait idle)
* Pulse controller serial helpers (open, handshake, trigger start/stop)
"""
import socket
import time
import serial


# ===================================================================
# Logging helper (used as default when callers don't supply log_fn)
# ===================================================================
_VERBOSE = True


def _log(msg):
    if _VERBOSE:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


# ===================================================================
# Viron laser status decode tables
# ===================================================================
STATE1_BITS = {
    7: ("Fire Mode", ("Disabled", "Fire")),
    6: ("Standby Mode", ("Stop", "Standby")),
    5: ("Diode Trigger", ("Internal", "External")),
    4: ("Q-Switch Source", ("Internal", "External")),
    3: ("Divide-By", ("Normal", "Divide-By")),
    2: ("Burst Mode", ("Continuous", "Burst")),
    1: ("Q-Switch", ("Disabled", "Enabled")),
    0: ("Not Ready", ("Ready", "Not Ready")),  # read-only
}

STATE2_BITS = {
    7: ("UV Illumination", ("Disabled", "Enabled")),
    6: ("Remote Q-Switch (RO)", ("Normal Q-switch", "Q-switch off")),
    4: ("BLE Session", ("No Session", "Temp")),
    3: ("Diode TEC", ("Off", "Run")),
    2: ("LAN Session", ("No Session", "Temp")),
    1: ("NLO Oven 2", ("Off", "Run")),
    0: ("NLO Oven 1", ("Off", "Run")),
}

FAULT_A = {
    7: "Remote Interlock",
    6: "Laser Temperature Range Fault",
    5: "Charge Fault",
    4: "Diode Current Fault",
    3: "Diode Temperature High/Low",
    2: "Diode Temperature Control Fault",
    1: "System Interlock: System/TEC",
    0: "System Interlock: Laser",
}

FAULT_B = {b: f"Fault2 bit {b}" for b in range(8)}

WARN_C = {
    7: "External Lamp PRF High",
    6: "Laser Temperature Warning",
    5: "Pre-Lase/Q-Switch inhibited",
    4: "CAN Bus Illegal ID or data",
    3: "CAN Bus Overrun",
    2: "Diode Current Limit Warning",
    1: "Reserved (log only)",
    0: "Diode/TEC Temp Warning",
}

WARN_D = {
    7: "NLO Oven 2 out of tolerance",
    6: "NLO Oven 2 timeout (oven off)",
    5: "NLO Oven 2 over temp (oven off)",
    4: "NLO Oven 2 open sensor (oven off)",
    3: "NLO Oven 1 out of tolerance",
    2: "NLO Oven 1 timeout (oven off)",
    1: "NLO Oven 1 over temp (oven off)",
    0: "NLO Oven 1 open sensor (oven off)",
}


def _bits_set(byte_val):
    """Return list of bit positions that are set in *byte_val*."""
    return [b for b in range(8) if (byte_val >> b) & 1]


# ===================================================================
# VironLaser class
# ===================================================================
class VironLaser:
    """Viron laser control over raw TCP socket (replaces telnetlib)."""

    # Minimum gap between consecutive commands to avoid CAN bus overrun
    CMD_GAP_S = 1.0

    def __init__(self, host, port=23, login="", timeout=5.0):
        self.host = host
        self.port = port
        self.login = login
        self.timeout = float(timeout)
        self.sock = None
        self._last_send_t = 0.0

    # ---------------- I/O core ----------------
    def _send(self, cmd, expect_ack=True, quiet=False, timeout=1.0):
        """Send a command and optionally read a reply. Never blocks forever."""
        if not self.sock:
            raise RuntimeError("Laser socket not connected.")
        # Pace commands to prevent CAN bus overrun on the laser controller
        elapsed = time.time() - self._last_send_t
        if elapsed < self.CMD_GAP_S:
            time.sleep(self.CMD_GAP_S - elapsed)
        data = (cmd + "\r").encode("ascii")
        self.sock.sendall(data)
        self._last_send_t = time.time()
        if not expect_ack:
            if not quiet:
                print(f"[LASER] {cmd} -> (no-ack)", flush=True)
            return ""
        resp = b""
        deadline = time.time() + float(timeout)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self.sock.settimeout(remaining)
            try:
                chunk = self.sock.recv(1)
                if not chunk:
                    break
                resp += chunk
                if resp.endswith((b"\r", b"\n")):
                    break
            except socket.timeout:
                break
        text = resp.decode("ascii", errors="ignore").strip()
        if not quiet:
            print(f"[LASER] {cmd} -> {text}", flush=True)
        return text

    def _readline_with_timeout(self, timeout=0.5, inter_char_timeout=0.2, max_bytes=256):
        """Read until CR/LF or timeout. Never blocks forever."""
        resp = b""
        end = time.time() + float(timeout)
        while len(resp) < max_bytes and time.time() < end:
            self.sock.settimeout(inter_char_timeout)
            try:
                chunk = self.sock.recv(1)
                if not chunk:
                    break
                resp += chunk
                if resp.endswith((b"\r", b"\n")):
                    break
            except socket.timeout:
                break
        return resp.decode("ascii", errors="ignore").strip()

    # ---------------- Connection ----------------
    def connect(self):
        print(f"[LASER] Connecting to {self.host}:{self.port} (timeout={self.timeout}s) ...")
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.login:
            self._send(f"$LOGIN {self.login}")
        return True

    def ensure_session(self):
        """Re-send the LOGIN command to keep the session alive.

        The Viron laser drops the authenticated session after a period of
        inactivity.  Calling this before critical commands (e.g. $FIRE)
        guarantees the session is active.  It is harmless if already logged in.
        """
        if not self.sock:
            raise RuntimeError("Laser socket not connected.")
        if self.login:
            self._send(f"$LOGIN {self.login}", quiet=True)
            time.sleep(0.5)

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None
            print("[LASER] Socket closed.")

    # ---------------- Status helpers ----------------
    def wait_ready(self, max_wait=600.0, poll=0.5, strict_warnings=True):
        print("[LASER] wait_ready(): monitoring STATUS until ready...", flush=True)
        t0 = time.time()
        last_log = 0.0
        last_raw = "<no status>"
        while True:
            tup, decoded, raw, texts = self.status_readable()
            last_raw = raw or last_raw
            if tup:
                SS, XX, AA, BB, CC, DD = tup
                not_ready = (SS & 0x01) == 0x01
                any_fault = (AA != 0) or (BB != 0)
                any_warn = (CC != 0) or (DD != 0)
                ready_now = (not any_fault) and (not not_ready) and (
                    (not any_warn) if strict_warnings else True
                )
                if ready_now:
                    print(f"\n[LASER] Ready. STATUS: {raw}", flush=True)
                    s1 = ", ".join(decoded.get("State1", [])[:4])
                    s2 = ", ".join(decoded.get("State2", [])[:4])
                    print(f"[LASER]   {s1}\n[LASER]   {s2}", flush=True)
                    return tup
                now = time.time()
                if now - last_log >= 1.0:
                    last_log = now
                    print(f"\n[LASER] Waiting... STATUS: {raw}", flush=True)
                    faults = "; ".join(decoded.get("Faults", []) or [])
                    warns = "; ".join(decoded.get("Warnings", []) or [])
                    if faults:
                        print("[LASER]   Faults: " + faults, flush=True)
                    if warns:
                        print("[LASER]   Warnings: " + warns, flush=True)
                    if texts and texts.strip().upper() != "OK":
                        print(f"[LASER]   TEXTS: {texts}", flush=True)
            else:
                print(".", end="", flush=True)
            if (time.time() - t0) > max_wait:
                raise RuntimeError(
                    f"\nLaser not ready after {max_wait:.0f}s; last STATUS='{last_raw}'"
                )
            time.sleep(poll)

    # ---------------- Config & control ----------------
    def configure_for_mapping(self, single_shot=False):
        self.ensure_session()
        self._send("$TRIG EI")
        self._send("$BURST 0")
        self._send("$QSPRE 1")
        self._send("$QSON 2" if single_shot else "$QSON 1")

    def standby_and_fire(self, strict_warnings=True, wait_after=5.0,
                         max_wait=90.0):
        self.ensure_session()
        print("[LASER] Sending STANDBY...", flush=True)
        self._send("$STANDBY", expect_ack=False)
        time.sleep(2.0)
        print("[LASER] Entering wait_ready()...", flush=True)
        self.wait_ready(max_wait=max_wait, poll=0.5, strict_warnings=strict_warnings)
        print("[LASER] Ready; sending FIRE...", flush=True)
        self._send("$FIRE", expect_ack=False)
        if wait_after and wait_after > 0:
            print(f"[LASER] Waiting {wait_after:.1f} seconds before starting first line...", flush=True)
            time.sleep(wait_after)

    def stop_and_close(self):
        self.ensure_session()
        try:
            self._send("$STOP")
            self._send("$LOGOUT", expect_ack=False)
        finally:
            self.close()

    def stop_only(self):
        self.ensure_session()
        """Stop laser without closing the socket."""
        self._send("$STOP", expect_ack=False)

    # ---------------- Status parsing ----------------
    def parse_status_line(self, s):
        """Parse a $STATUS reply. Returns (SS,XX,AA,BB,CC,DD) and a decoded dict."""
        cleaned = s.replace("$STATUS", "").replace("=", " ").replace(",", " ")
        tokens = [
            t for t in cleaned.split()
            if all(c in "0123456789ABCDEFabcdef" for c in t)
        ]
        if len(tokens) < 3:
            return None, {}
        try:
            w1, w2, w3 = (int(t, 16) for t in tokens[:3])
        except ValueError:
            return None, {}
        SS, XX = (w1 >> 8) & 0xFF, w1 & 0xFF
        AA, BB = (w2 >> 8) & 0xFF, w2 & 0xFF
        CC, DD = (w3 >> 8) & 0xFF, w3 & 0xFF
        decoded = {"State1": [], "State2": [], "Faults": [], "Warnings": []}
        for b in range(8):
            if b in STATE1_BITS:
                name, (zero_txt, one_txt) = STATE1_BITS[b]
                val = (SS >> b) & 1
                decoded["State1"].append(f"{name}: {one_txt if val else zero_txt}")
        for b in range(8):
            if b in STATE2_BITS:
                name, (zero_txt, one_txt) = STATE2_BITS[b]
                val = (XX >> b) & 1
                decoded["State2"].append(f"{name}: {one_txt if val else zero_txt}")
        for b in _bits_set(AA):
            decoded["Faults"].append(f"AA:{b} {FAULT_A.get(b, f'Unknown Fault bit {b}')}")
        for b in _bits_set(BB):
            decoded["Faults"].append(f"BB:{b} {FAULT_B.get(b, f'Unknown Fault2 bit {b}')}")
        for b in _bits_set(CC):
            decoded["Warnings"].append(f"CC:{b} {WARN_C.get(b, f'Unknown Warning bit {b}')}")
        for b in _bits_set(DD):
            decoded["Warnings"].append(f"DD:{b} {WARN_D.get(b, f'Unknown Warning bit {b}')}")
        return (SS, XX, AA, BB, CC, DD), decoded

    def status_readable(self):
        """Query STATUS twice (manual requirement). Returns (tuple, decoded, raw, texts)."""
        self._send("$STATUS ?", expect_ack=False, quiet=True)
        _ = self._readline_with_timeout(timeout=0.7)
        time.sleep(0.05)
        self._send("$STATUS ?", expect_ack=False, quiet=True)
        s2 = self._readline_with_timeout(timeout=0.7)
        tup, decoded = self.parse_status_line(s2)
        texts = ""
        if tup:
            _, _, AA, BB, CC, DD = tup
            if AA or BB or CC or DD:
                self._send("$TEXTS ?", expect_ack=False, quiet=True)
                texts = self._readline_with_timeout(timeout=0.7)
        return tup, decoded, s2, texts

    def _status_twice(self):
        _ = self._send("$STATUS ?", expect_ack=False, quiet=True)
        time.sleep(0.05)
        s2 = self._send("$STATUS ?", expect_ack=False, quiet=True)
        return s2


# ===================================================================
# GRBL CNC helpers
# ===================================================================
def grbl_cmd(ser, cmd, log_fn=None):
    """Send a single GRBL command."""
    log = log_fn or _log
    if not cmd.endswith("\n"):
        cmd += "\n"
    log(f"[GRBL] >> {cmd.strip()}")
    ser.write(cmd.encode("utf-8"))
    ser.flush()


def get_grbl_status(ser, quiet=False, log_fn=None):
    """Query GRBL status and return parsed parts list."""
    log = log_fn or _log
    ser.reset_input_buffer()
    ser.write(b"?\n")
    resp = ser.readline().decode("utf-8", errors="ignore").strip()
    s = resp.replace("<", "").replace(">", "").replace("|", ",").replace(":", ",")
    parts = s.split(",") if s else []
    if not quiet:
        log(f"[GRBL] Status raw: {resp}")
    return parts


def wait_cnc_idle(ser, quiet=False, log_fn=None):
    """Block until GRBL reports Idle state."""
    log = log_fn or _log
    if not quiet:
        log("[GRBL] Waiting for Idle ...")
    time.sleep(0.2)
    status = get_grbl_status(ser, quiet=quiet, log_fn=log)
    guard = 0
    while not status or status[0] not in ("Idle", "Check"):
        time.sleep(0.2)
        status = get_grbl_status(ser, quiet=quiet, log_fn=log)
        guard += 1
        if guard > 600:
            raise RuntimeError("GRBL did not reach Idle state in time.")
    if not quiet:
        log("[GRBL] Idle.")
    return status


def grbl_stream_jog_x(ser, dist_mm, feed_mm_min, log_fn=None):
    """Send a G91 jog command in X."""
    grbl_cmd(ser, f"$J=G91 X{dist_mm:.3f} F{feed_mm_min:.2f}", log_fn=log_fn)


def grbl_stream_jog_y(ser, dist_mm, feed_mm_min, log_fn=None):
    """Send a G91 jog command in Y."""
    grbl_cmd(ser, f"$J=G91 Y{dist_mm:.3f} F{feed_mm_min:.2f}", log_fn=log_fn)


# ===================================================================
# Pulse controller helpers
# ===================================================================
def pulse_open(port, baud, log_fn=None):
    """Open a serial connection to the pulse controller."""
    log = log_fn or _log
    log(f"[PULSE] Opening {port} @ {baud}...")
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    log("[PULSE] Port opened.")
    return ser


def pulse_handshake(ser, step_trigger=None, timeout=2.0, log_fn=None):
    """Verify communication with the pulse controller."""
    log = log_fn or _log
    log("[PULSE] Handshake PING ...")
    t0 = time.time()
    ser.write(b"?\n")
    ser.flush()
    while time.time() - t0 < timeout:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            log(f"[PULSE] << {line}")
            return True
    log("[PULSE] No PING response, sending STEP/STOP ...")
    ser.reset_input_buffer()
    trigger = step_trigger if step_trigger is not None else 80
    ser.write(f"STEP {trigger}\n".encode("utf-8"))
    ser.flush()
    ser.write(b"STOP\n")
    ser.flush()
    log("[PULSE] Fallback handshake sent.")
    return True


def movement_trigger_start(pulse_ser, step_trigger, log_fn=None):
    """Arm the pulse controller and start triggering."""
    log = log_fn or _log
    log(f"[PULSE] START with STEP={step_trigger}")
    time.sleep(0.05)
    pulse_ser.write(f"STEP {step_trigger}\n".encode("utf-8"))
    line = pulse_ser.readline().decode("utf-8", errors="ignore").strip()
    print(line)
    pulse_ser.flush()
    time.sleep(0.02)
    pulse_ser.write(b"START\n")
    pulse_ser.flush()


def movement_trigger_stop(pulse_ser, log_fn=None):
    """Stop pulse controller triggering."""
    log = log_fn or _log
    log("[PULSE] STOP")
    time.sleep(0.02)
    pulse_ser.write(b"STOP\n")
    pulse_ser.flush()


def movement_trigger_reset_count(pulse_ser, log_fn=None):
    """Reset emitted-trigger counter on pulse controller."""
    log = log_fn or _log
    log("[PULSE] RESETCOUNT")
    try:
        pulse_ser.reset_input_buffer()
        pulse_ser.write(b"RESETCOUNT\n")
        pulse_ser.flush()
        line = pulse_ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            log(f"[PULSE] << {line}")
        return "RESETCOUNT" in line.upper()
    except Exception as exc:
        log(f"[PULSE] RESETCOUNT failed: {exc}")
        return False


def movement_trigger_get_count(pulse_ser, log_fn=None):
    """Read emitted-trigger counter from pulse controller."""
    log = log_fn or _log
    log("[PULSE] COUNT")
    try:
        pulse_ser.reset_input_buffer()
        pulse_ser.write(b"COUNT\n")
        pulse_ser.flush()
        line = pulse_ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            log(f"[PULSE] << {line}")
        cleaned = line.replace(":", " ")
        for tok in reversed(cleaned.split()):
            if tok.isdigit():
                return int(tok)
        return None
    except Exception as exc:
        log(f"[PULSE] COUNT failed: {exc}")
        return None
