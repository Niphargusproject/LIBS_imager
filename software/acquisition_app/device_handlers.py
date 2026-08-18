# -*- coding: utf-8 -*-
"""
Device handlers for LIBS application
Handles camera, spectrometer, laser, and serial communication
"""
import logging
import time
import threading
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import serial

# Local modules
from config import AppConfig
import avaspec_fix_winfunctype as avs
from hardware_control import (
    VironLaser, STATE1_BITS, STATE2_BITS, FAULT_A, FAULT_B, WARN_C, WARN_D,
    pulse_open, pulse_handshake,
)


class SpectrometerHandler:
    """Handles spectrometer operations"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger('LIBS_App.Spectrometer')
        self.handles = []
        self.num_pixels_per_device = []
        self.wavelengths_per_device = []
        self.device_ranges = []  # list of (min_lambda, max_lambda)
        self.device_order = []   # indices sorted by range
        self.device_order_all = []
        self.device_info: Dict[int, Dict[str, Any]] = {}
        self._library_initialized = False  # Track if AVS_Init has been called
        # Initialize spectrometer at startup
        self.initialize()
        
    def initialize(self) -> bool:
        """Initialize spectrometer devices"""
        try:
            # Deactivate existing handles before reinitializing
            for handle in self.handles:
                try:
                    avs.AVS_Deactivate(handle)
                except Exception as e:
                    self.logger.warning(f'Failed to deactivate existing spectrometer handle: {e}')
            
            # Clear existing handles
            self.handles = []
            self.num_pixels_per_device = []
            self.wavelengths_per_device = []
            self.device_ranges = []
            self.device_order = []
            self.device_order_all = []
            self.device_info = {}
            
            # Close library if already initialized
            if self._library_initialized:
                try:
                    avs.AVS_Done()
                    self._library_initialized = False
                except Exception as e:
                    self.logger.warning(f'Failed to close library before reinitializing: {e}')
            
            # Initialize library and activate all devices found
            init_result = avs.AVS_Init(0)
            
            # Check if initialization was successful (positive number = device count, negative = error)
            if init_result < 0:
                self.logger.error(f'AVS_Init failed with error code: {init_result}')
                return False
            
            self._library_initialized = True
            count = avs.AVS_UpdateUSBDevices()
            
            if count <= 0:
                self.logger.error('No Avantes spectrometer devices found')
                return False
            
            devices = avs.AVS_GetList(max(1, count))
            if not devices:
                self.logger.error('No Avantes spectrometer detected')
                return False
            
            # AVS_GetList returns a ctypes array structure that supports indexing
            # Iterate over devices by index
            for i in range(count):
                try:
                    # Access device by index from the array structure
                    dev = devices[i]
                    
                    # Verify this is a valid AvsIdentityType object
                    if not hasattr(dev, 'SerialNumber'):
                        self.logger.warning(f'Device at index {i} is not a valid AvsIdentityType')
                        continue
                    
                    handle = avs.AVS_Activate(dev)
                    
                    # Check if handle is valid (INVALID_AVS_HANDLE_VALUE = 1000)
                    if handle == avs.INVALID_AVS_HANDLE_VALUE:
                        self.logger.error(f'Failed to activate spectrometer {i}: invalid handle returned')
                        continue

                    # Enable high resolution ADC (16-bit) if supported
                    try:
                        if hasattr(avs, "AVS_UseHighResAdc"):
                            avs.AVS_UseHighResAdc(handle, True)
                            self.logger.info(f"High-res ADC enabled for spectrometer {i}")
                    except Exception as adc_error:
                        self.logger.warning(f"Failed to enable high-res ADC for spectrometer {i}: {adc_error}")
                    
                    npx = avs.AVS_GetNumPixels(handle)
                    lambdas = list(avs.AVS_GetLambda(handle))[: npx]
                    self.handles.append(handle)
                    self.num_pixels_per_device.append(npx)
                    self.wavelengths_per_device.append(lambdas)
                    rng = (min(lambdas or [0]), max(lambdas or [0]))
                    self.device_ranges.append(rng)
                    serial = self._decode_identity_field(getattr(dev, 'SerialNumber', None))
                    friendly = self._decode_identity_field(getattr(dev, 'UserFriendlyName', None))
                    label = friendly or serial or f"Spectrometer {i+1}"
                    range_text = self._format_range_text(rng)
                    description = f"{label} ({range_text})" if range_text else label
                    fpga_version = firmware_version = dll_version = ""
                    try:
                        ver_info = avs.AVS_GetVersionInfo(handle)
                        if isinstance(ver_info, (list, tuple)) and len(ver_info) >= 3:
                            fpga_version = self._decode_identity_field(ver_info[0])
                            firmware_version = self._decode_identity_field(ver_info[1])
                            dll_version = self._decode_identity_field(ver_info[2])
                    except Exception as ver_error:
                        self.logger.debug(f"Unable to read version info for spectrometer {i}: {ver_error}")
                    self.device_info[i] = {
                        'index': i,
                        'label': label,
                        'serial': serial,
                        'friendly_name': friendly,
                        'pixels': npx,
                        'handle': handle,
                        'range': rng,
                        'range_text': range_text,
                        'description': description,
                        'fpga_version': fpga_version,
                        'firmware_version': firmware_version,
                        'dll_version': dll_version,
                        'wavelength_start_nm': rng[0],
                        'wavelength_stop_nm': rng[1],
                    }
                except (IndexError, AttributeError, Exception) as e:
                    self.logger.error(f'Failed to activate spectrometer at index {i}: {e}')
            
            # Check if any devices were successfully activated
            if not self.handles:
                self.logger.error('No spectrometers were successfully activated')
                return False
            
            # Order devices by spectral range (min then max)
            self.device_order_all = sorted(range(len(self.handles)), key=lambda i: (self.device_ranges[i][0], self.device_ranges[i][1]))
            self.device_order = list(self.device_order_all)
            ordered_info = [self.device_ranges[i] for i in self.device_order]
            self.logger.info(f'Activated {len(self.handles)} Avantes spectrometer(s) with ranges: {ordered_info}')
            return True
        except Exception as e:
            self.logger.error(f'Error initializing Avantes spectrometers: {e}')
            return False

    @staticmethod
    def _decode_identity_field(value) -> str:
        """Decode Avantes identity field to printable text."""
        if value is None:
            return ""
        try:
            if isinstance(value, str):
                return value.strip('\x00')
            if isinstance(value, (bytes, bytearray)):
                data = bytes(value)
            else:
                data = bytes(value)
            return data.decode('ascii', errors='ignore').strip('\x00')
        except Exception:
            try:
                return str(value).strip('\x00')
            except Exception:
                return ""

    @staticmethod
    def _format_range_text(rng: Tuple[float, float]) -> str:
        """Format wavelength range tuple to text."""
        try:
            start, stop = rng
            return f"{start:.0f}-{stop:.0f} nm"
        except Exception:
            return ""

    def get_device_info(self) -> List[Dict[str, Any]]:
        """Return ordered metadata for detected spectrometers."""
        ordered_indices = self.device_order_all if self.device_order_all else list(range(len(self.handles)))
        info: List[Dict[str, Any]] = []
        for idx in ordered_indices:
            data = dict(self.device_info.get(idx, {}))
            data.setdefault('index', idx)
            data.setdefault('label', f"Spectrometer {idx+1}")
            if 'pixels' not in data and idx < len(self.num_pixels_per_device):
                data['pixels'] = self.num_pixels_per_device[idx]
            if 'handle' not in data and idx < len(self.handles):
                data['handle'] = self.handles[idx]
            rng = data.get('range')
            if not rng and idx < len(self.device_ranges):
                rng = self.device_ranges[idx]
                data['range'] = rng
            range_text = data.get('range_text') or self._format_range_text(rng) if rng else ""
            if range_text:
                data['range_text'] = range_text
            data.setdefault('description', f"{data['label']} ({range_text})" if range_text else data['label'])
            info.append(data)
        return info

    def set_active_devices(self, indices: List[int]) -> bool:
        """Limit active spectrometers to provided handle indices."""
        if not self.handles:
            self.logger.error('Spectrometer not initialized')
            return False
        if not indices:
            self.logger.warning('No spectrometer indices provided; selection unchanged')
            return False
        order_source = self.device_order_all if self.device_order_all else list(range(len(self.handles)))
        unique_ordered = []
        seen = set()
        for idx in order_source:
            if idx in indices and idx not in seen:
                unique_ordered.append(idx)
                seen.add(idx)
        if not unique_ordered:
            self.logger.warning('Spectrometer selection does not match detected devices')
            return False
        self.device_order = unique_ordered
        labels = [self.device_info.get(i, {}).get('label', f"Spectrometer {i+1}") for i in unique_ordered]
        self.logger.info(f"Active spectrometers set to: {labels}")
        return True

    def get_active_device_indices(self) -> List[int]:
        """Return indices of currently active spectrometers."""
        if self.device_order:
            return list(self.device_order)
        return list(range(len(self.handles)))
        
    def prepare_measurement(self, integration_time: int, trigger_delay: int,
                            triggered: bool = True, nr_averages: int = 1) -> bool:
        """Prepare spectrometer for measurement using avaspec.dll"""
        if not self.handles:
            self.logger.error('Spectrometer not initialized')
            return False
        active_indices = self.get_active_device_indices()
        if not active_indices:
            self.logger.error('No active spectrometers selected')
            return False
        try:
            for idx in active_indices:
                handle = self.handles[idx]
                meas = avs.MeasConfigType()
                meas.m_StartPixel = 0
                stop_pix = (self.num_pixels_per_device[idx] or 2048) - 1
                meas.m_StopPixel = stop_pix
                meas.m_IntegrationTime = float(integration_time) / 1000.0  # μs -> ms
                # UI provides trigger_delay in ns. Avantes expects clock ticks (~20.83 ns per tick, legacy behavior).
                if triggered:
                    ticks = int(trigger_delay/20.83)
                else:
                    ticks = 0
                meas.m_IntegrationDelay = ticks
                meas.m_NrAverages = int(max(1, nr_averages))
                meas.m_CorDynDark_m_Enable = 0
                meas.m_CorDynDark_m_ForgetPercentage = 0
                meas.m_Smoothing_m_SmoothPix = 0
                meas.m_Smoothing_m_SmoothModel = 0
                meas.m_SaturationDetection = 0
                # Trigger settings: 
                # Mode 0 = Software trigger (immediate start, no external trigger) - for dark measurements
                # Mode 1 = Hardware trigger (wait for external trigger) - for real measurements with laser
                meas.m_Trigger_m_Mode = int(triggered)  # 0 for dark (no external trigger), 1 for external trigger
                meas.m_Trigger_m_Source = 0
                meas.m_Trigger_m_SourceType = 0
                meas.m_Control_m_StrobeControl = 0
                meas.m_Control_m_LaserDelay = 0
                meas.m_Control_m_LaserWidth = 0
                meas.m_Control_m_LaserWaveLength = 0.0
                meas.m_Control_m_StoreToRam = 0

                avs.AVS_PrepareMeasure(handle, meas)
            self.logger.info(
                f"Spectrometer prepared: int_time={integration_time}μs, trigger_delay={trigger_delay}ns (~{ticks} ticks), triggered={triggered}"
            )
            return True
        except Exception as e:
            self.logger.error(f'Failed to prepare spectrometer: {e}')
            return False
    
    def start_measurement(self) -> bool:
        """Start a single measurement on all devices (arm for external trigger)."""
        if not self.handles:
            self.logger.error('Spectrometer not initialized')
            return False
        active_indices = self.get_active_device_indices()
        if not active_indices:
            self.logger.error('No active spectrometers selected')
            return False
        try:
            for idx in active_indices:
                avs.AVS_Measure(self.handles[idx], 0, 1)
            return True
        except Exception as e:
            self.logger.error(f'Failed to start measurement: {e}')
            return False

    def read_measurement(self, timeout_s: float = 3.0) -> Optional[Dict[str, Any]]:
        """Read one measurement from all devices after they were armed."""
        if not self.handles:
            self.logger.error('Spectrometer not initialized')
            return None
        active_indices = self.get_active_device_indices()
        if not active_indices:
            self.logger.error('No active spectrometers selected')
            return None
        try:
            # Wait for all devices to have data available
            ready = {idx: False for idx in active_indices}
            t0 = time.time()
            while not all(ready.values()):
                for idx in active_indices:
                    if not ready[idx]:
                        ready[idx] = bool(avs.AVS_PollScan(self.handles[idx]))
                if all(ready.values()):
                    break
                if time.time() - t0 > timeout_s:
                    raise TimeoutError('Timeout waiting for spectra')
                time.sleep(0.005)

            # Read in range-ordered device order
            scopes = {}
            ordered_selection = [idx for idx in (self.device_order if self.device_order else active_indices) if idx in ready]
            for ch_idx, dev_idx in enumerate(ordered_selection, start=1):
                handle = self.handles[dev_idx]
                timestamp, spectrum = avs.AVS_GetScopeData(handle)
                values = list(spectrum)[: self.num_pixels_per_device[dev_idx]]
                lambdas = self.wavelengths_per_device[dev_idx]
                scopes[f'ch{ch_idx}'] = type('Scope', (), {'values': values, 'lambdas': lambdas})()

            self.logger.info(f'Spectra acquired successfully from {len(ordered_selection)} active device(s)')
            return scopes
        except Exception as e:
            self.logger.error(f'Failed to read spectra: {e}')
            return None
    
    def process_spectrum_data(self, scopes: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process raw spectrum data into DataFrames"""
        try:
            corrected_spectra_merged = []
            lambdas_merged = []
            
            # Keep device order as provided (already range-ordered)
            for channel_name, spectrum in scopes.items():
                corrected_spectra_merged.extend(spectrum.values)
                lambdas_merged.extend(spectrum.lambdas)
            
            # Create DataFrames
            spectro_data = pd.DataFrame({
                'wavelength_(nm)': lambdas_merged,
                'intensity_dark_corrected': corrected_spectra_merged
            })
            
            # Do not sort; preserve device order to respect range ordering
            
            return spectro_data, pd.DataFrame()  # Return empty dark data for now
            
        except Exception as e:
            self.logger.error(f"Failed to process spectrum data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def close(self):
        """Close spectrometer connections and cleanup"""
        try:
            # Deactivate all handles
            for handle in self.handles:
                try:
                    avs.AVS_Deactivate(handle)
                except Exception as e:
                    self.logger.warning(f'Failed to deactivate spectrometer handle: {e}')
            
            # Clear handles
            self.handles = []
            self.num_pixels_per_device = []
            self.wavelengths_per_device = []
            self.device_ranges = []
            self.device_order = []
            
            # Close library if initialized
            if self._library_initialized:
                try:
                    avs.AVS_Done()
                    self._library_initialized = False
                    self.logger.info("Spectrometer library closed")
                except Exception as e:
                    self.logger.error(f'Failed to close spectrometer library: {e}')
                    
        except Exception as e:
            self.logger.error(f"Error closing spectrometer connections: {e}")


class LaserHandler:
    """Handles laser operations using VironLaser from hardware_control.

    All communication goes through VironLaser (raw TCP socket), eliminating
    the deprecated telnetlib dependency.  Decode tables are imported from
    hardware_control.py -- the single source of truth.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger('LIBS_App.Laser')
        self._laser: Optional[VironLaser] = None
        self.connected = False
        self._mapping_controller = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self, progress_callback=None) -> bool:
        """Connect to laser, wait for ready, standby, configure."""
        try:
            laser = VironLaser(
                self.config.laser.host,
                self.config.laser.port,
                self.config.laser.login_code,
                timeout=self.config.laser.timeout,
            )
            laser.connect()
            self.logger.info("Laser login successful")
            if progress_callback:
                progress_callback(25)

            # Wait until ready (no faults/warnings)
            laser.wait_ready(max_wait=120.0, poll=0.5, strict_warnings=True)
            self.logger.info("Laser ready for standby")
            if progress_callback:
                progress_callback(50)

            # Standby + wait ready again
            laser._send("$STANDBY", expect_ack=False)
            time.sleep(2.0)
            laser.wait_ready(max_wait=30.0, poll=0.5, strict_warnings=True)
            self.logger.info("Laser ready to fire")

            # Default one-shot/multi-shot settings
            laser._send("$QSPRE 1")
            laser._send("$TRIG II")
            laser._send("$QSON 2")

            self._laser = laser
            self.connected = True
            self.logger.info("Laser connection complete and ready to fire")
            if progress_callback:
                progress_callback(100)
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to laser: {e}")
            try:
                laser.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            return False

    def disconnect(self):
        """Disconnect from laser."""
        self.disconnect_mapping_controller()
        if self._laser:
            try:
                self._laser.stop_and_close()
                self.logger.info("Laser disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting laser: {e}")
            finally:
                self._laser = None
                self.connected = False

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def status_readable(self):
        """Return (tuple, decoded, raw, texts) -- delegates to VironLaser."""
        if self._laser:
            return self._laser.status_readable()
        return None, {}, "", "OK"

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def fire(self) -> bool:
        """Fire the laser.

        Re-sends the LOGIN command first because the Viron laser drops
        the authenticated session after a period of inactivity.
        """
        if not self.connected or not self._laser:
            self.logger.error("Laser not connected")
            return False
        try:
            self._laser.ensure_session()
            self._laser._send("$FIRE", expect_ack=False)
            self.logger.info("Laser fired")
            return True
        except Exception as e:
            self.logger.error(f"Failed to fire laser: {e}")
            return False

    def stop_only(self):
        """Send STOP without closing the session."""
        # If a mapping controller is active, stop through it
        if self._mapping_controller is not None and self._mapping_controller is not self:
            try:
                self._mapping_controller.stop_only()
                return
            except Exception as exc:
                self.logger.warning(f"Failed to stop mapping laser controller: {exc}")

        if not self._laser:
            self.logger.info("Laser stop skipped: no active session")
            return
        try:
            self._laser.stop_only()
            self.logger.info("Laser stopped")
        except Exception as exc:
            self.logger.warning(f"Failed to stop laser: {exc}")

    def apply_acquisition_defaults(self):
        """Reapply default one-shot/multi-shot settings ($TRIG II / $QSON 2)."""
        if not self._laser:
            return
        try:
            self._laser._send("$TRIG II")
            self._laser._send("$QSON 2")
            self.logger.debug("Laser acquisition defaults reapplied ($TRIG II, $QSON 2).")
        except Exception as exc:
            self.logger.warning(f"Failed to reapply acquisition defaults: {exc}")

    def configure_for_mapping(self, single_shot: bool = False):
        """Apply mapping-specific laser settings."""
        if not self._laser:
            return
        try:
            self._laser.configure_for_mapping(single_shot=single_shot)
        except Exception as exc:
            self.logger.warning(f"Failed to configure laser for mapping: {exc}")

    def standby_and_fire(self, strict_warnings: bool = True, wait_after: float = 5.0):
        """Send standby, wait ready, then fire."""
        if not self._laser:
            raise RuntimeError("Laser session not available")
        self._laser.standby_and_fire(strict_warnings=strict_warnings, wait_after=wait_after)

    def wait_ready(self, max_wait: float = 120.0, poll: float = 0.5):
        """Wait until the laser reports ready status."""
        if not self._laser:
            return False
        try:
            self._laser.wait_ready(max_wait=max_wait, poll=poll, strict_warnings=True)
            return True
        except RuntimeError:
            return False

    def stop_and_close(self):
        """Compatibility with VironLaser stop/close API."""
        self.disconnect()

    # ------------------------------------------------------------------
    # Mapping controller management
    # ------------------------------------------------------------------
    def connect_mapping_controller(self, single_shot: bool = False) -> bool:
        """Connect mapping-specific laser controller (VironLaser)."""
        if self._mapping_controller is not None:
            return True
        # Re-use the existing session if already connected
        if self.connected and self._laser:
            self._mapping_controller = self._laser
            return True
        try:
            controller = VironLaser(
                self.config.laser.host,
                self.config.laser.port,
                self.config.laser.login_code,
                timeout=self.config.laser.timeout,
            )
            controller.connect()
            controller.configure_for_mapping(single_shot=single_shot)
            controller.wait_ready(max_wait=120.0, poll=0.5, strict_warnings=False)
            self._mapping_controller = controller
            self.logger.info("Mapping laser controller connected")
            return True
        except Exception as exc:
            self.logger.error(f"Failed to connect mapping laser controller: {exc}")
            try:
                controller.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            self._mapping_controller = None
            return False

    def disconnect_mapping_controller(self):
        """Disconnect mapping-specific laser controller."""
        if self._mapping_controller is None:
            return
        try:
            # If the mapping controller is the main laser, just clear the ref
            if self._mapping_controller is self._laser:
                self._mapping_controller = None
                return
            self._mapping_controller.stop_and_close()
        except Exception as exc:
            self.logger.warning(f"Error closing mapping laser controller: {exc}")
        finally:
            self._mapping_controller = None
            self.logger.info("Mapping laser controller disconnected")

    def get_mapping_controller(self):
        """Return the mapping laser controller instance."""
        return self._mapping_controller


class SerialHandler:
    """Handles serial communication with Arduino and CNC"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger('LIBS_App.Serial')
        self.arduino_serial = None
        self.cnc_serial = None
        self.pulse_serial = None
        
    def connect_arduino(self) -> bool:
        """Connect to Arduino"""
        try:
            self.arduino_serial = serial.Serial(
                self.config.serial.arduino_port,
                self.config.serial.arduino_baudrate
            )
            self.logger.info("Arduino connected")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def connect_cnc(self) -> bool:
        """Connect to CNC controller"""
        try:
            self.cnc_serial = serial.Serial(
                self.config.serial.cnc_port,
                self.config.serial.cnc_baudrate
            )
            self.cnc_serial.flushInput()
            self.logger.info("CNC controller connected")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to CNC: {e}")
            return False

    def disconnect_cnc(self):
        """Disconnect CNC serial connection."""
        if not self.cnc_serial:
            return
        try:
            self.cnc_serial.close()
            self.logger.info("CNC controller disconnected")
        except Exception as exc:
            self.logger.error(f"Failed to disconnect CNC: {exc}")
        finally:
            self.cnc_serial = None

    def connect_pulse_controller(self) -> bool:
        """Connect to the external pulse controller used for mapping."""
        if self.pulse_serial:
            return True
        try:
            self.pulse_serial = pulse_open(
                self.config.serial.pulse_port,
                self.config.serial.pulse_baudrate
            )
            pulse_handshake(self.pulse_serial)
            self.logger.info("Pulse controller connected")
            return True
        except Exception as exc:
            self.logger.error(f"Failed to connect to pulse controller: {exc}")
            if self.pulse_serial:
                try:
                    self.pulse_serial.close()
                except Exception:
                    pass
                self.pulse_serial = None
            return False

    def disconnect_pulse_controller(self):
        """Disconnect from the pulse controller."""
        if not self.pulse_serial:
            return
        try:
            self.pulse_serial.close()
            self.logger.info("Pulse controller disconnected")
        except Exception as exc:
            self.logger.error(f"Error disconnecting pulse controller: {exc}")
        finally:
            self.pulse_serial = None
    
    def send_arduino_command(self, command: str) -> bool:
        """Send command to Arduino"""
        if not self.arduino_serial:
            return False
        
        try:
            self.arduino_serial.write(command.encode('utf-8'))
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Arduino command: {e}")
            return False
    
    def send_cnc_command(self, command) -> bool:
        """Send command to CNC controller"""
        if not self.cnc_serial:
            return False
        
        try:
            if isinstance(command, bytes):
                payload = command
            else:
                payload = str(command).encode('utf-8')
            self.cnc_serial.write(payload)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send CNC command: {e}")
            return False
    
    def get_grbl_status(self) -> list:
        """Get GRBL status"""
        if not self.cnc_serial:
            return []
        
        try:
            self.cnc_serial.write(b'? \n')
            response = self.cnc_serial.readline().decode('utf-8')
            response = response.replace(":", ",").replace(">", "").replace("<", "").replace("|", ",")
            status_list = response.split(",")
            self.cnc_serial.flushInput()
            return status_list
        except Exception as e:
            self.logger.error(f"Failed to get GRBL status: {e}")
            return []
    
    def close_all(self):
        """Close all serial connections"""
        if self.arduino_serial:
            try:
                self.arduino_serial.close()
                self.logger.info("Arduino connection closed")
            except Exception as e:
                self.logger.error(f"Error closing Arduino: {e}")
        
        if self.cnc_serial:
            try:
                self.cnc_serial.close()
                self.logger.info("CNC connection closed")
            except Exception as e:
                self.logger.error(f"Error closing CNC: {e}")
        
        if self.pulse_serial:
            try:
                self.pulse_serial.close()
                self.logger.info("Pulse controller connection closed")
            except Exception as e:
                self.logger.error(f"Error closing pulse controller: {e}")
            finally:
                self.pulse_serial = None
