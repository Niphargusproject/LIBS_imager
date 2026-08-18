# -*- coding: utf-8 -*-
"""
Configuration management for LIBS application
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class CameraConfig:
    """Camera configuration settings"""
    micro_img_size: tuple
    disp_scale: int
    disp_msec: int
    crop_region: tuple  # y1, y2, x1, x2
    crosshair_lines: bool


@dataclass
class SpectrometerConfig:
    """Spectrometer configuration settings"""
    channel_number: int
    integration_time_list: List[str]
    trigger_delay_list: List[str]
    profile_steps_list: List[str]
    shots_delay: List[str]
    selected_integration_time: str
    selected_trigger_delay: str
    selected_profile_steps: str
    selected_shots_delay: str


@dataclass
class MappingConfig:
    """Mapping configuration settings"""
    x_size: float
    y_size: float
    step_size: float
    table_steps: List[str]
    mapping_steps: List[str]
    selected_xy_step: str
    selected_z_step: str
    laser_frequency: float
    integration_time_ms: float
    integration_delay_ns: int
    save_dir: str
    save_stem: str
    progress_every_lines: int
    progress_selection: str
    progress_tol_nm: float
    averages: int
    steps_per_mm: int
    return_feed_mm_min: float
    scan_line_error_margin: float
    line_start_wait_s: float
    verbose: bool
    log_every_x: int
    ftp_host: str
    ftp_user: str
    ftp_pass: str
    ftp_path: str
    laser_single_shot_mode: bool


@dataclass
class LaserConfig:
    """Laser configuration settings"""
    host: str
    port: int
    timeout: int
    login_code: str
    # Allow using an external laser trigger without LAN connection
    use_external: bool
    # How long to wait for an external trigger before timing out (seconds)
    external_trigger_timeout_s: float


@dataclass
class SerialConfig:
    """Serial communication configuration"""
    arduino_port: str
    arduino_baudrate: int
    cnc_port: str
    cnc_baudrate: int
    pulse_port: str
    pulse_baudrate: int


@dataclass
class AppConfig:
    """Main application configuration"""
    app_name: str
    dark_mode: bool
    save_path: str
    folder_path: str

    # Sub-configurations
    camera: CameraConfig
    spectrometer: SpectrometerConfig
    mapping: MappingConfig
    laser: LaserConfig
    serial: SerialConfig
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AppConfig':
        """Load configuration from JSON file; all fields required."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed to read config file: {config_path}") from exc

        def _require(container: Dict[str, Any], key: str, section: str):
            if key not in container:
                raise KeyError(f"Missing '{section}.{key}' in config.json")
            return container[key]

        def _require_section(container: Dict[str, Any], key: str) -> Dict[str, Any]:
            value = container.get(key)
            if not isinstance(value, dict):
                raise KeyError(f"Missing '{key}' section in config.json")
            return value

        camera_data = _require_section(data, 'camera')
        spectrometer_data = _require_section(data, 'spectrometer')
        mapping_data = _require_section(data, 'mapping')
        laser_data = _require_section(data, 'laser')
        serial_data = _require_section(data, 'serial')

        camera = CameraConfig(
            micro_img_size=tuple(_require(camera_data, 'micro_img_size', 'camera')),
            disp_scale=_require(camera_data, 'disp_scale', 'camera'),
            disp_msec=_require(camera_data, 'disp_msec', 'camera'),
            crop_region=tuple(_require(camera_data, 'crop_region', 'camera')),
            crosshair_lines=_require(camera_data, 'crosshair_lines', 'camera'),
        )

        spectrometer = SpectrometerConfig(
            channel_number=_require(spectrometer_data, 'channel_number', 'spectrometer'),
            integration_time_list=_require(spectrometer_data, 'integration_time_list', 'spectrometer'),
            trigger_delay_list=_require(spectrometer_data, 'trigger_delay_list', 'spectrometer'),
            profile_steps_list=_require(spectrometer_data, 'profile_steps_list', 'spectrometer'),
            shots_delay=_require(spectrometer_data, 'shots_delay', 'spectrometer'),
            selected_integration_time=_require(spectrometer_data, 'selected_integration_time', 'spectrometer'),
            selected_trigger_delay=_require(spectrometer_data, 'selected_trigger_delay', 'spectrometer'),
            selected_profile_steps=_require(spectrometer_data, 'selected_profile_steps', 'spectrometer'),
            selected_shots_delay=_require(spectrometer_data, 'selected_shots_delay', 'spectrometer'),
        )

        mapping = MappingConfig(
            x_size=_require(mapping_data, 'x_size', 'mapping'),
            y_size=_require(mapping_data, 'y_size', 'mapping'),
            step_size=_require(mapping_data, 'step_size', 'mapping'),
            table_steps=_require(mapping_data, 'table_steps', 'mapping'),
            mapping_steps=_require(mapping_data, 'mapping_steps', 'mapping'),
            selected_xy_step=_require(mapping_data, 'selected_xy_step', 'mapping'),
            selected_z_step=_require(mapping_data, 'selected_z_step', 'mapping'),
            laser_frequency=_require(mapping_data, 'laser_frequency', 'mapping'),
            integration_time_ms=_require(mapping_data, 'integration_time_ms', 'mapping'),
            integration_delay_ns=_require(mapping_data, 'integration_delay_ns', 'mapping'),
            save_dir=_require(mapping_data, 'save_dir', 'mapping'),
            save_stem=_require(mapping_data, 'save_stem', 'mapping'),
            progress_every_lines=_require(mapping_data, 'progress_every_lines', 'mapping'),
            progress_selection=_require(mapping_data, 'progress_selection', 'mapping'),
            progress_tol_nm=_require(mapping_data, 'progress_tol_nm', 'mapping'),
            averages=_require(mapping_data, 'averages', 'mapping'),
            steps_per_mm=_require(mapping_data, 'steps_per_mm', 'mapping'),
            return_feed_mm_min=_require(mapping_data, 'return_feed_mm_min', 'mapping'),
            scan_line_error_margin=_require(mapping_data, 'scan_line_error_margin', 'mapping'),
            line_start_wait_s=float(mapping_data.get('line_start_wait_s', 10)),
            verbose=_require(mapping_data, 'verbose', 'mapping'),
            log_every_x=_require(mapping_data, 'log_every_x', 'mapping'),
            ftp_host=_require(mapping_data, 'ftp_host', 'mapping'),
            ftp_user=_require(mapping_data, 'ftp_user', 'mapping'),
            ftp_pass=_require(mapping_data, 'ftp_pass', 'mapping'),
            ftp_path=_require(mapping_data, 'ftp_path', 'mapping'),
            laser_single_shot_mode=_require(mapping_data, 'laser_single_shot_mode', 'mapping'),
        )

        laser = LaserConfig(
            host=_require(laser_data, 'host', 'laser'),
            port=_require(laser_data, 'port', 'laser'),
            timeout=_require(laser_data, 'timeout', 'laser'),
            login_code=_require(laser_data, 'login_code', 'laser'),
            use_external=_require(laser_data, 'use_external', 'laser'),
            external_trigger_timeout_s=_require(laser_data, 'external_trigger_timeout_s', 'laser'),
        )

        serial = SerialConfig(
            arduino_port=_require(serial_data, 'arduino_port', 'serial'),
            arduino_baudrate=_require(serial_data, 'arduino_baudrate', 'serial'),
            cnc_port=_require(serial_data, 'cnc_port', 'serial'),
            cnc_baudrate=_require(serial_data, 'cnc_baudrate', 'serial'),
            pulse_port=_require(serial_data, 'pulse_port', 'serial'),
            pulse_baudrate=_require(serial_data, 'pulse_baudrate', 'serial'),
        )

        return cls(
            app_name=_require(data, 'app_name', 'root'),
            dark_mode=_require(data, 'dark_mode', 'root'),
            save_path=_require(data, 'save_path', 'root'),
            folder_path=_require(data, 'folder_path', 'root'),
            camera=camera,
            spectrometer=spectrometer,
            mapping=mapping,
            laser=laser,
            serial=serial,
        )
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'app_name': self.app_name,
            'dark_mode': self.dark_mode,
            'save_path': self.save_path,
            'folder_path': self.folder_path,
            'camera': {
                'micro_img_size': self.camera.micro_img_size,
                'disp_scale': self.camera.disp_scale,
                'disp_msec': self.camera.disp_msec,
                'crop_region': self.camera.crop_region,
                'crosshair_lines': self.camera.crosshair_lines
            },
            'spectrometer': {
                'channel_number': self.spectrometer.channel_number,
                'integration_time_list': self.spectrometer.integration_time_list,
                'trigger_delay_list': self.spectrometer.trigger_delay_list,
                'profile_steps_list': self.spectrometer.profile_steps_list,
                'shots_delay': self.spectrometer.shots_delay,
                'selected_integration_time': self.spectrometer.selected_integration_time,
                'selected_trigger_delay': self.spectrometer.selected_trigger_delay,
                'selected_profile_steps': self.spectrometer.selected_profile_steps,
                'selected_shots_delay': self.spectrometer.selected_shots_delay
            },
            'mapping': {
                'x_size': self.mapping.x_size,
                'y_size': self.mapping.y_size,
                'step_size': self.mapping.step_size,
                'table_steps': self.mapping.table_steps,
                'mapping_steps': self.mapping.mapping_steps,
                'selected_xy_step': self.mapping.selected_xy_step,
                'selected_z_step': self.mapping.selected_z_step,
                'laser_frequency': self.mapping.laser_frequency,
                'integration_time_ms': self.mapping.integration_time_ms,
                'integration_delay_ns': self.mapping.integration_delay_ns,
                'save_dir': self.mapping.save_dir,
                'save_stem': self.mapping.save_stem,
                'progress_every_lines': self.mapping.progress_every_lines,
                'progress_selection': self.mapping.progress_selection,
                'progress_tol_nm': self.mapping.progress_tol_nm,
                'averages': self.mapping.averages,
                'steps_per_mm': self.mapping.steps_per_mm,
                'return_feed_mm_min': self.mapping.return_feed_mm_min,
                'scan_line_error_margin': self.mapping.scan_line_error_margin,
                'line_start_wait_s': self.mapping.line_start_wait_s,
                'verbose': self.mapping.verbose,
                'log_every_x': self.mapping.log_every_x,
                'ftp_host': self.mapping.ftp_host,
                'ftp_user': self.mapping.ftp_user,
                'ftp_pass': self.mapping.ftp_pass,
                'ftp_path': self.mapping.ftp_path,
                'laser_single_shot_mode': self.mapping.laser_single_shot_mode
            },
            'laser': {
                'host': self.laser.host,
                'port': self.laser.port,
                'timeout': self.laser.timeout,
                'login_code': self.laser.login_code,
                'use_external': self.laser.use_external,
                'external_trigger_timeout_s': self.laser.external_trigger_timeout_s
            },
            'serial': {
                'arduino_port': self.serial.arduino_port,
                'arduino_baudrate': self.serial.arduino_baudrate,
                'cnc_port': self.serial.cnc_port,
                'cnc_baudrate': self.serial.cnc_baudrate,
                'pulse_port': self.serial.pulse_port,
                'pulse_baudrate': self.serial.pulse_baudrate
            }
        }
        
        config_path = Path(config_path)
        if config_path.parent and not config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open('w') as f:
            json.dump(config_dict, f, indent=4)


# ---------------------------------------------------------------------------
# Lazy config singleton -- only loaded when first accessed, not on import
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = Path(__file__).parent.absolute() / 'config.json'
_config: AppConfig = None  # type: ignore[assignment]


def get_config() -> AppConfig:
    """Return the application config, loading from disk on first call."""
    global _config
    if _config is None:
        _config = AppConfig.from_file(str(_DEFAULT_CONFIG_PATH))
    return _config


class _LazyConfig:
    """Transparent proxy so ``from config import config`` still works."""
    def __getattr__(self, name):
        return getattr(get_config(), name)

    def __setattr__(self, name, value):
        setattr(get_config(), name, value)


config = _LazyConfig()  # type: ignore[assignment]
