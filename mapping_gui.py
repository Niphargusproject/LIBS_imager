# -*- coding: utf-8 -*-
"""
LIBS Mapping GUI components.

Provides reusable widgets and the background mapping thread for the main
application (Bruniquel_LIBS_improved.py):

* ``BasicParamsTab``  -- mapping dimensions, laser/timing, output settings
* ``AdvancedParamsTab`` -- CNC, pulse, laser network, FTP, verbosity
* ``ProgressPlotWidget`` -- matplotlib widget for progress preview images
* ``MappingThread`` -- background thread that drives ``MappingEngine``
"""
import os
import traceback
import threading

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox,
    QGridLayout, QComboBox, QCheckBox, QScrollArea,
    QSizePolicy,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from mapping_engine import (
    MappingParams, MappingEngine,
    discover_and_sort_by_range,
)


# ============================
# ProgressPlotWidget
# ============================
class ProgressPlotWidget(QWidget):
    """Widget to display a progress preview image, using all available space."""

    def __init__(self):
        super().__init__()
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.ax = self.fig.add_subplot(111)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax.text(0.5, 0.5, 'Waiting for progress plot...',
                     ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
        self.canvas.draw()

    def clear_plot(self):
        """Reset the plot to a blank black image."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        self.ax.set_title('Mapping started - waiting for first data...')
        self.ax.axis('off')
        self.canvas.draw()

    def resizeEvent(self, event):
        """Re-draw canvas on widget resize to fill available space."""
        super().resizeEvent(event)
        self.canvas.draw_idle()

    def update_plot(self, png_path):
        """Update plot from PNG file."""
        if os.path.exists(png_path):
            try:
                self.fig.clear()
                self.ax = self.fig.add_subplot(111)
                img = plt.imread(png_path)
                self.ax.imshow(img, aspect='equal')
                self.ax.axis('off')
                self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                self.canvas.draw()
            except Exception as e:
                self.fig.clear()
                self.ax = self.fig.add_subplot(111)
                self.ax.text(0.5, 0.5, f'Error loading plot:\n{str(e)}',
                             ha='center', va='center', fontsize=12,
                             transform=self.ax.transAxes)
                self.canvas.draw()


# ============================
# BasicParamsTab
# ============================
class BasicParamsTab(QWidget):
    """Basic parameters tab (mapping dimensions, laser/timing, output)."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # -- Mapping dimensions --
        mapping_group = QGroupBox("Mapping dimensions")
        mapping_layout = QGridLayout()
        self.step_size = QLineEdit("0.10")
        self.mapping_X_size = QLineEdit("20")
        self.mapping_Y_size = QLineEdit("3")
        mapping_layout.addWidget(QLabel("Step size (mm):"), 0, 0)
        mapping_layout.addWidget(self.step_size, 0, 1)
        mapping_layout.addWidget(QLabel("X size (mm):"), 1, 0)
        mapping_layout.addWidget(self.mapping_X_size, 1, 1)
        mapping_layout.addWidget(QLabel("Y size (mm):"), 2, 0)
        mapping_layout.addWidget(self.mapping_Y_size, 2, 1)
        mapping_group.setLayout(mapping_layout)
        scroll_layout.addWidget(mapping_group)

        # -- Laser and timing --
        laser_group = QGroupBox("Laser and timing")
        laser_layout = QGridLayout()
        self.laser_frequency = QLineEdit("20")
        self.integration_time_ms = QLineEdit("0.1")
        self.integration_delay_ns = QLineEdit("1000")
        self.line_start_wait_s = QLineEdit("5")
        self.line_start_wait_s.setToolTip("Seconds to wait at the start of each line after firing the laser (configurable in config.json).")
        laser_layout.addWidget(QLabel("Laser frequency (Hz):"), 0, 0)
        laser_layout.addWidget(self.laser_frequency, 0, 1)
        laser_layout.addWidget(QLabel("Integration time (ms):"), 1, 0)
        laser_layout.addWidget(self.integration_time_ms, 1, 1)
        laser_layout.addWidget(QLabel("Integration delay (ns):"), 2, 0)
        laser_layout.addWidget(self.integration_delay_ns, 2, 1)
        laser_layout.addWidget(QLabel("Wait before line (s):"), 3, 0)
        laser_layout.addWidget(self.line_start_wait_s, 3, 1)
        laser_group.setLayout(laser_layout)
        scroll_layout.addWidget(laser_group)

        # -- Output settings --
        output_group = QGroupBox("Output settings")
        output_layout = QGridLayout()
        self.save_dir = QLineEdit("output_job")
        self.save_stem = QLineEdit("mapping_run")
        self.progress_every_lines = QLineEdit("5")
        output_layout.addWidget(QLabel("Save directory:"), 0, 0)
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir)
        self.browse_btn = QPushButton("Browse...")
        save_dir_layout.addWidget(self.browse_btn)
        output_layout.addLayout(save_dir_layout, 0, 1)
        output_layout.addWidget(QLabel("File stem:"), 1, 0)
        output_layout.addWidget(self.save_stem, 1, 1)
        output_layout.addWidget(QLabel("Progress update (every N lines):"), 2, 0)
        output_layout.addWidget(self.progress_every_lines, 2, 1)
        output_group.setLayout(output_layout)
        scroll_layout.addWidget(output_group)

        # -- Progress plot settings --
        progress_group = QGroupBox("Progress plot")
        progress_layout = QGridLayout()
        self.progress_selection = QComboBox()
        self.progress_selection.addItems([
            "Si", "Al", "Fe", "Mg", "Ca", "Na", "K", "Ti", "Mn",
            "O", "Ba", "Sr", "Cu", "Zn", "P", "S", "Li", "Pb",
        ])
        self.progress_selection.setCurrentText("Sr")
        self.progress_tol_nm = QLineEdit("0.3")
        progress_layout.addWidget(QLabel("Band preview:"), 0, 0)
        progress_layout.addWidget(self.progress_selection, 0, 1)
        progress_layout.addWidget(QLabel("Tolerance (nm):"), 1, 0)
        progress_layout.addWidget(self.progress_tol_nm, 1, 1)
        progress_group.setLayout(progress_layout)
        scroll_layout.addWidget(progress_group)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    # ---- Getters / setters ----

    def get_params(self):
        """Return all basic parameters as a dict."""
        return {
            'step_size': self._parse_float(self.step_size.text(), 0.1),
            'mapping_X_size': self._parse_float(self.mapping_X_size.text(), 20.0),
            'mapping_Y_size': self._parse_float(self.mapping_Y_size.text(), 3.0),
            'laser_frequency': self._parse_int(self.laser_frequency.text(), 20),
            'integration_time_ms': self._parse_float(self.integration_time_ms.text(), 0.1),
            'integration_delay_ns': self._parse_int(self.integration_delay_ns.text(), 1000),
            'line_start_wait_s': self._parse_float(self.line_start_wait_s.text(), 5.0),
            'averages': 1,
            'save_dir': self.save_dir.text(),
            'save_stem': self.save_stem.text(),
            'progress_every_lines': self._parse_int(self.progress_every_lines.text(), 5),
            'progress_selection': self.progress_selection.currentText(),
            'progress_tol_nm': self._parse_float(self.progress_tol_nm.text(), 0.3),
        }

    @staticmethod
    def _parse_float(value, default):
        try:
            return float(str(value).strip().replace(",", "."))
        except (TypeError, ValueError):
            return default

    @classmethod
    def _parse_int(cls, value, default):
        try:
            return int(cls._parse_float(value, default))
        except (TypeError, ValueError):
            return default

    def set_params(self, params):
        """Populate all fields from a dict."""
        self.step_size.setText(str(params.get('step_size', 0.10)))
        self.mapping_X_size.setText(str(params.get('mapping_X_size', 20)))
        self.mapping_Y_size.setText(str(params.get('mapping_Y_size', 3)))
        self.laser_frequency.setText(str(params.get('laser_frequency', 20)))
        self.integration_time_ms.setText(str(params.get('integration_time_ms', 0.1)))
        self.integration_delay_ns.setText(str(params.get('integration_delay_ns', 1000)))
        self.line_start_wait_s.setText(str(params.get('line_start_wait_s', 5)))
        self.save_dir.setText(params.get('save_dir', 'output_job'))
        self.save_stem.setText(params.get('save_stem', 'mapping_run'))
        self.progress_every_lines.setText(str(params.get('progress_every_lines', 5)))
        idx = self.progress_selection.findText(params.get('progress_selection', 'Sr'))
        if idx >= 0:
            self.progress_selection.setCurrentIndex(idx)
        self.progress_tol_nm.setText(str(params.get('progress_tol_nm', 0.3)))


# ============================
# AdvancedParamsTab
# ============================
class AdvancedParamsTab(QWidget):
    """Advanced parameters tab (CNC, pulse, laser network, FTP, verbosity)."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # -- CNC --
        cnc_group = QGroupBox("CNC Settings")
        cnc_layout = QGridLayout()
        self.steps_per_mm = QLineEdit("800")
        self.return_feed_mm_min = QLineEdit("100")
        cnc_layout.addWidget(QLabel("Steps per mm:"), 0, 0)
        cnc_layout.addWidget(self.steps_per_mm, 0, 1)
        cnc_layout.addWidget(QLabel("Return Feed (mm/min):"), 1, 0)
        cnc_layout.addWidget(self.return_feed_mm_min, 1, 1)
        cnc_group.setLayout(cnc_layout)
        scroll_layout.addWidget(cnc_group)

        # -- Laser --
        laser_group = QGroupBox("Laser Settings")
        laser_layout = QGridLayout()
        self.LASER_QSON_SINGLE_SHOT = QCheckBox("Single Shot Mode")
        laser_layout.addWidget(self.LASER_QSON_SINGLE_SHOT, 0, 0, 1, 2)
        laser_group.setLayout(laser_layout)
        scroll_layout.addWidget(laser_group)

        # -- FTP --
        ftp_group = QGroupBox("FTP Upload (Optional)")
        ftp_layout = QGridLayout()
        self.ftp_host = QLineEdit()
        self.ftp_user = QLineEdit()
        self.ftp_pass = QLineEdit()
        self.ftp_pass.setEchoMode(QLineEdit.Password)
        self.ftp_path = QLineEdit(".")
        self.enable_ftp = QCheckBox("Enable FTP Upload")
        ftp_layout.addWidget(self.enable_ftp, 0, 0, 1, 2)
        ftp_layout.addWidget(QLabel("FTP Host:"), 1, 0)
        ftp_layout.addWidget(self.ftp_host, 1, 1)
        ftp_layout.addWidget(QLabel("FTP User:"), 2, 0)
        ftp_layout.addWidget(self.ftp_user, 2, 1)
        ftp_layout.addWidget(QLabel("FTP Password:"), 3, 0)
        ftp_layout.addWidget(self.ftp_pass, 3, 1)
        ftp_layout.addWidget(QLabel("FTP Path:"), 4, 0)
        ftp_layout.addWidget(self.ftp_path, 4, 1)
        ftp_group.setLayout(ftp_layout)
        scroll_layout.addWidget(ftp_group)

        # -- Verbosity --
        verbosity_group = QGroupBox("Verbosity")
        verbosity_layout = QGridLayout()
        self.VERBOSE = QCheckBox("Verbose Logging")
        self.VERBOSE.setChecked(True)
        verbosity_layout.addWidget(self.VERBOSE, 0, 0, 1, 2)
        verbosity_group.setLayout(verbosity_layout)
        scroll_layout.addWidget(verbosity_group)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    # ---- Getters / setters ----

    def get_params(self):
        params = {
            'steps_per_mm': int(self.steps_per_mm.text()),
            'return_feed_mm_min': float(self.return_feed_mm_min.text()),
            'LASER_QSON_SINGLE_SHOT': self.LASER_QSON_SINGLE_SHOT.isChecked(),
            'VERBOSE': self.VERBOSE.isChecked(),
        }
        if self.enable_ftp.isChecked():
            params['ftp_host'] = self.ftp_host.text()
            params['ftp_user'] = self.ftp_user.text()
            params['ftp_pass'] = self.ftp_pass.text()
            params['ftp_path'] = self.ftp_path.text()
        else:
            params['ftp_host'] = None
            params['ftp_user'] = None
            params['ftp_pass'] = None
            params['ftp_path'] = "."
        return params

    def set_params(self, params):
        self.steps_per_mm.setText(str(params.get('steps_per_mm', 800)))
        self.return_feed_mm_min.setText(str(params.get('return_feed_mm_min', 100)))
        self.LASER_QSON_SINGLE_SHOT.setChecked(params.get('LASER_QSON_SINGLE_SHOT', False))
        self.VERBOSE.setChecked(params.get('VERBOSE', True))
        ftp_host = params.get('ftp_host')
        if ftp_host:
            self.enable_ftp.setChecked(True)
            self.ftp_host.setText(ftp_host)
            self.ftp_user.setText(params.get('ftp_user', ''))
            self.ftp_pass.setText(params.get('ftp_pass', ''))
            self.ftp_path.setText(params.get('ftp_path', '.'))


# ============================
# MappingThread
# ============================
class MappingThread(threading.Thread):
    """Background thread that drives a MappingEngine run.

    Accepts pre-established device connections and spectrometer data from the
    main application. Builds a ``MappingParams`` from the GUI dictionaries
    (no parameter-injection into module globals).
    """

    def __init__(self, params, log_queue, progress_callback=None,
                 cnc_ser=None, pulse_ser=None, laser_obj=None,
                 spectrometer_handles=None, spectrometer_wavelengths=None,
                 spectrometer_identities=None):
        super().__init__()
        self.params = params
        self.log_queue = log_queue
        self.progress_callback = progress_callback
        self.stop_event = threading.Event()
        self.daemon = True

        # Pre-established connections
        self.cnc_ser = cnc_ser
        self.pulse_ser = pulse_ser
        self.laser_obj = laser_obj

        # Pre-discovered spectrometers
        self.spectrometer_handles = spectrometer_handles
        self.spectrometer_wavelengths = spectrometer_wavelengths
        self.spectrometer_identities = spectrometer_identities

    def run(self):
        try:
            # Mapping script plots via Agg (no interactive window)
            import matplotlib
            matplotlib.use("Agg")

            basic = self.params['basic']
            advanced = self.params['advanced']
            selected_handles = self.params.get('spectrometers_selected', [])

            # Build MappingParams from GUI dictionaries
            mp = MappingParams(
                step_size=basic['step_size'],
                mapping_X_size=basic['mapping_X_size'],
                mapping_Y_size=basic['mapping_Y_size'],
                laser_frequency=basic['laser_frequency'],
                integration_time_ms=basic['integration_time_ms'],
                integration_delay_ns=basic['integration_delay_ns'],
                line_start_wait_s=float(basic.get('line_start_wait_s', 5.0)),
                averages=basic['averages'],
                steps_per_mm=advanced['steps_per_mm'],
                return_feed_mm_min=advanced['return_feed_mm_min'],
                laser_single_shot_mode=advanced['LASER_QSON_SINGLE_SHOT'],
                save_dir=basic['save_dir'],
                save_stem=basic['save_stem'],
                progress_every_lines=basic['progress_every_lines'],
                progress_selection=basic['progress_selection'],
                progress_tol_nm=basic['progress_tol_nm'],
                ftp_host=advanced.get('ftp_host'),
                ftp_user=advanced.get('ftp_user'),
                ftp_pass=advanced.get('ftp_pass'),
                ftp_path=advanced.get('ftp_path', '.'),
                verbose=advanced.get('VERBOSE', True),
            )

            # Push messages to queue; main app logs and displays them
            def queued_log(msg):
                self.log_queue.put(msg)

            engine = MappingEngine(mp, log_fn=queued_log)

            # Validate connections
            if self.cnc_ser is None:
                raise RuntimeError("[CNC] No pre-established connection.")
            if self.pulse_ser is None:
                raise RuntimeError("[PULSE] No pre-established connection.")
            if self.laser_obj is None:
                raise RuntimeError("[LASER] No pre-established connection.")

            # Spectrometers
            if self.spectrometer_handles and self.spectrometer_wavelengths:
                handles = self.spectrometer_handles
                wavelengths = self.spectrometer_wavelengths
                identities = self.spectrometer_identities or []
            else:
                handles, wavelengths, identities = discover_and_sort_by_range(log_fn=queued_log)

            # Run mapping
            engine.run(
                cnc=self.cnc_ser,
                pulse=self.pulse_ser,
                laser=self.laser_obj,
                handles_sorted=handles,
                wavelengths=wavelengths,
                identities=identities,
                stop_event=self.stop_event,
                progress_callback=self.progress_callback,
                selected_handles=selected_handles or None,
            )

            if self.progress_callback:
                self.progress_callback(100)
            self.log_queue.put("=== MAPPING COMPLETED SUCCESSFULLY ===")

        except Exception as e:
            error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
            self.log_queue.put(error_msg)
