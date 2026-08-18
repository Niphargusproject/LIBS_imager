# -*- coding: utf-8 -*-
"""
Bruniquel LIBS Control Application (Improved Version)
-----------------------------------------------------
Copyright (c) 2026 Royal Belgian Institute of Natural Sciences

This application provides a graphical user interface (GUI) for controlling and managing
the LIBS (Laser Induced Breakdown Spectroscopy) hardware setup. The software supports 
camera live view and image refresh (with credits to iosoft.blog for inspiration), direct
manual and automated control of hardware, data acquisition, mapping (to be done), and result visualization.

The GUI is built using PyQt5 with programmatic UI definitions.
It is extendable, modular, and includes robust logging and hardware abstraction layers
for reproducible LIBS laboratory operation.

Developed and maintained by the Royal Belgian Institute of Natural Sciences.
"""
import sys
import os
import time
import threading
import queue
import traceback
from pathlib import Path
from typing import List

# Setup paths for library loading
try:
    from app_init import setup_environment
    setup_environment()
except ImportError:
    # Fallback path setup if app_init.py is not available
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    os.add_dll_directory(str(current_dir))

# PyQt5 imports
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMessageBox, QFileDialog, QSplashScreen
)
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtTest import QTest
from PyQt5 import QtSvg  # Ensure SVG plugin is loaded

# Serial communication
import serial

# Basler camera (pypylon)
try:
    from pypylon import pylon
except Exception:
    pylon = None  # Allow app to run without camera installed

# Scientific imports
import numpy as np
import pandas as pd
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)

# Application modules
from config import config
from app_init import setup_logging
from device_handlers import SpectrometerHandler, LaserHandler, SerialHandler
from data_processing import SpectrumProcessor, PlotGenerator, PeakIdentifier
from ui_main_window import Ui_MainWindow
from mapping_gui import BasicParamsTab, AdvancedParamsTab, ProgressPlotWidget, MappingThread

class ImageWidget(QtWidgets.QWidget):
    """Custom widget for displaying camera images with crosshair overlay"""
    
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None
        self._paused_overlay = False
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(1, 1)

    def setImage(self, image):
        self.image = image
        self.update()
    
    def setPaused(self, paused: bool):
        self._paused_overlay = bool(paused)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        try:
            rect = self.rect()
            if self._paused_overlay:
                qp.fillRect(rect, QColor(0, 0, 0))
                # Draw a pause symbol in the center
                bar_w = max(6, rect.width() // 30)
                bar_h = max(24, rect.height() // 6)
                gap = bar_w
                center = rect.center()
                x_left = center.x() - gap - bar_w
                x_right = center.x() + gap
                y_top = center.y() - (bar_h // 2)
                qp.setPen(QtCore.Qt.NoPen)
                qp.setBrush(QColor(220, 220, 220))
                qp.drawRect(x_left, y_top, bar_w, bar_h)
                qp.drawRect(x_right, y_top, bar_w, bar_h)
            elif self.image:
                # Draw the image scaled to fit
                qp.drawImage(rect, self.image)

                # Draw green crosshair overlay
                pen = QtGui.QPen(QColor(0, 255, 0))
                pen.setWidth(2)
                qp.setPen(pen)
                center = rect.center()
                # Horizontal line
                qp.drawLine(rect.left(), center.y(), rect.right(), center.y())
                # Vertical line
                qp.drawLine(center.x(), rect.top(), center.x(), rect.bottom())
        except Exception:
            pass
        qp.end()


class WheelSelector(QtWidgets.QFrame):
    """Lightweight wheel-style selector (iPhone-like picker)."""

    valueChanged = QtCore.pyqtSignal(str)

    def __init__(self, options=None, parent=None, visible_rows=3):
        super().__init__(parent)
        self.setObjectName("wheelSelector")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet(
            """
            QWidget#wheelSelector {
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 6px;
                background-color: rgba(0, 0, 0, 0.2);
            }
            """
        )

        self._options = options or []
        self._row_height = 28
        self._updating_selection = False
        self._current_value = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.list_view = QtWidgets.QListView(self)
        self.list_view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.list_view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.list_view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.list_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_view.setSpacing(4)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setStyleSheet(
            """
            QListView {
                background: transparent;
                color: #f0f0f0;
                font-size: 11pt;
            }
            QListView::item {
                height: 26px;
            }
            QListView::item:selected {
                color: #ffffff;
            }
            """
        )

        layout.addWidget(self.list_view)

        self.model = QtGui.QStandardItemModel(self.list_view)
        self.list_view.setModel(self.model)
        self.setOptions(self._options)

        self.list_view.selectionModel().currentChanged.connect(
            lambda cur, prev: self._emit_value(cur.row())
        )
        self.list_view.clicked.connect(lambda idx: self._emit_value(idx.row()))
        self.list_view.verticalScrollBar().valueChanged.connect(self._sync_selection_with_scroll)

        self.highlight = QtWidgets.QFrame(self)
        self.highlight.setStyleSheet(
            """
            background: transparent;
            border: none;
            """
        )
        self.highlight.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        row_height = self.list_view.sizeHintForRow(0)
        if row_height > 0:
            self._row_height = row_height + self.list_view.spacing()

        self._visible_rows = max(2, visible_rows)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        max_height = 250
        self.setMinimumHeight(max_height)
        self.setMaximumHeight(max_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        highlight_height = self._row_height
        top = (self.height() - highlight_height) // 2
        self.highlight.setGeometry(0, top, self.width(), highlight_height)

    def setOptions(self, options):
        """Populate wheel with new options."""
        self._options = [str(o) for o in options] if options else []
        self.model.clear()
        for opt in self._options:
            item = QtGui.QStandardItem(f"{opt}")
            item.setEditable(False)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.model.appendRow(item)
        if self._options:
            self.setValue(self._options[0])

    def value(self) -> str:
        return self._current_value or (self._options[0] if self._options else "")

    def setValue(self, value: str):
        """Set the current value (expects raw number as string)."""
        if not self._options:
            return
        value_str = str(value).strip().replace("µm", "").strip()
        if value_str not in self._options:
            value_str = self._options[0]
        row = self._options.index(value_str)
        model_index = self.model.index(row, 0)
        self._select_index(model_index, center=True)
        self._emit_value(row, force=True)

    def _select_index(self, model_index, center=False):
        if not model_index.isValid():
            return
        self._updating_selection = True
        self.list_view.setCurrentIndex(model_index)
        if center:
            self.list_view.scrollTo(model_index, QtWidgets.QAbstractItemView.PositionAtCenter)
        self._updating_selection = False

    def _emit_value(self, row: int, force: bool = False):
        if row < 0 or row >= len(self._options):
            return
        value = self._options[row]
        if not force and value == self._current_value:
            return
        self._current_value = value
        if not self._updating_selection:
            self.valueChanged.emit(value)

    def _sync_selection_with_scroll(self):
        if not self._options or self._updating_selection:
            return
        scrollbar = self.list_view.verticalScrollBar()
        if scrollbar.maximum() == 0:
            return
        row = int(round(scrollbar.value() / max(1, self._row_height)))
        row = max(0, min(row, len(self._options) - 1))
        model_index = self.model.index(row, 0)
        if model_index.isValid():
            self._select_index(model_index)
            self._emit_value(row)

def create_modern_splash(app_name: str) -> QtGui.QPixmap:
    """Create a modern splash QPixmap and persist it to splash.png."""
    width, height = 960, 540
    pix = QtGui.QPixmap(width, height)
    pix.fill(QtCore.Qt.transparent)

    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

    # Use splash.png from the project root as background if available
    splash_path = Path("splash.png")
    if splash_path.exists():
        bg = QtGui.QPixmap(str(splash_path))
        if not bg.isNull():
            bg = bg.scaled(width, height, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
            x = (bg.width() - width) // 2
            y = (bg.height() - height) // 2
            painter.drawPixmap(0, 0, bg, x, y, width, height)
    else:
        painter.fillRect(0, 0, width, height, QtGui.QColor(30, 30, 30))

    # Title
    title_font = QtGui.QFont('Segoe UI', 32, QtGui.QFont.DemiBold)
    painter.setFont(title_font)
    painter.setPen(QtGui.QColor(255, 255, 255))
    painter.drawText(QtCore.QRect(0, int(height * 0.34), width, 60), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, app_name)

    # Subtitle
    subtitle_font = QtGui.QFont('Segoe UI', 16)
    painter.setFont(subtitle_font)
    painter.setPen(QtGui.QColor(255, 255, 255))
    painter.drawText(
        QtCore.QRect(0, int(height * 0.34) + 64, width, 30),
        QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
        "february jungle version"
    )

    # Top-right year
    year_font = QtGui.QFont('Segoe UI', 24, QtGui.QFont.DemiBold)
    painter.setFont(year_font)
    painter.setPen(QtGui.QColor(255, 255, 255))
    painter.drawText(QtCore.QRect(0, 20, width-30, 40), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, "2026")

    # Build datetime (centered below subtitle)
    try:
        mtime = os.path.getmtime(__file__)
        from datetime import datetime
        build_str = f"Last modified {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')}"
    except Exception:
        build_str = ""
    if build_str:
        build_font = QtGui.QFont('Segoe UI', 16)
        painter.setFont(build_font)
        painter.setPen(QtGui.QColor(180, 180, 180))
        painter.drawText(QtCore.QRect(0, int(height * 0.34) + 98, width, 30),
                         QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, build_str)

    painter.end()

    return pix

def ensure_images_folder():
    """Ensure images/ exists and migrate legacy icon files from root if needed."""
    import shutil
    img_dir = Path('images')
    img_dir.mkdir(parents=True, exist_ok=True)

    legacy_files = [
        'arrow-up.svg', 'arrow-down.svg', 'arrow-left.svg', 'arrow-right.svg',
        'chevrons-up.svg', 'chevrons-down.svg', 'folder.svg', 'home.svg',
        'stop-circle.svg', 'x-octagon.svg', 'logo.png', 'icon_aconvert.ico', 'splash.png'
    ]
    for name in legacy_files:
        src = Path(name)
        dst = img_dir / name
        try:
            if src.exists() and not dst.exists():
                shutil.copyfile(str(src), str(dst))
        except Exception:
            # Non-fatal if copy fails
            pass

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Main application window"""
    
    text_update = QtCore.pyqtSignal(str)
    frame_update = QtCore.pyqtSignal(QtGui.QImage)
    
    def __init__(self):
        super().__init__()
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info("LIBS Application starting...")
        self.settings = QtCore.QSettings("RBINS", "BruniquelLIBS")
        
        # Load configuration (persist defaults only on first run)
        self.config_path = Path(__file__).parent.absolute() / "config.json"
        self.config = config
        if not self.config_path.exists():
            try:
                self.config.save_to_file(str(self.config_path))
            except Exception as e:
                self.logger.warning(f"Could not write default config: {e}")
        
        # Initialize device handlers
        self.spectrometer_handler = SpectrometerHandler(self.config)
        self.laser_handler = LaserHandler(self.config)
        self.serial_handler = SerialHandler(self.config)
        
        # Initialize data processors
        self.spectrum_processor = SpectrumProcessor()
        self.plot_generator = PlotGenerator(self.config.dark_mode)
        self.peak_identifier = PeakIdentifier()
        
        # Initialize data storage
        self.spectro1_data = pd.DataFrame(columns=['wavelength_(nm)', 'intensity_dark_corrected'])
        self.dark_data = pd.DataFrame(columns=['wavelength_(nm)', 'dark'])
        # One Shot: detect peaks only when user presses the button
        self._one_shot_detect_peaks_next = False
        # Latest camera frame buffer for saving
        self._latest_frame = None
        # Spectrometer selection state
        self._active_spectrometers = []
        self._spectrometer_selection_updating = False
        # Embedded mapping UI
        self.mapping_tab = None
        # Track whether the laser is currently configured for mapping mode
        self._laser_in_mapping_mode = False
        # Track camera state across mapping runs
        self._camera_was_running_before_mapping = False
        # Track microscope light state across mapping runs
        self._light_was_on_before_mapping = False
        
        # Initialize UI
        self._setup_ui()
        self._set_output_directory(getattr(self.config, 'folder_path', ''), source="init", persist=False, log_change=False)
        # Restore file stem from config
        if hasattr(self, 'one_shot_filename_box') and getattr(self.config, 'save_path', ''):
            self.one_shot_filename_box.setText(self.config.save_path)
        self._embed_mapping_tab()
        self._setup_connections()
        self._initialize_devices()
        
        self.logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Setup the user interface"""
        try:
            # Register Qt search path for icons before UI loads
            QtCore.QDir.addSearchPath('img', str(Path('images').resolve()))

            # Instantiate UI directly from inlined designer class
            self.setupUi(self)
            self.setWindowState(QtCore.Qt.WindowMaximized)
            # Remove legacy Spectrometers tab (controls will be embedded elsewhere)
            try:
                if hasattr(self, 'tab_spectrometers') and hasattr(self, 'tabWidget'):
                    idx = self.tabWidget.indexOf(self.tab_spectrometers)
                    if idx != -1:
                        self.tabWidget.removeTab(idx)
            except Exception:
                pass
            self._apply_responsive_layout()
            self._rebuild_acquisition_tab_layout()
            self._rebuild_micro_tab_layout()
            self._apply_modern_styles()
            self._apply_text_standards()
            self._apply_help_tooltips()
            self._modernize_manual_controls()

            # Setup image display
            self.disp = ImageWidget(self)
            if hasattr(self, 'display_micro') and isinstance(self.display_micro, QtWidgets.QHBoxLayout):
                self.display_micro.addWidget(self.disp)
            else:
                self.logger.warning("display_micro layout not found; camera view disabled")
            # Connect frame updates from camera thread
            self.frame_update.connect(self._handle_frame_update)
            
            # Setup matplotlib plots
            self._setup_plots()
            
            # Setup combo boxes
            self._setup_combo_boxes()
            # Setup graphical CNC step wheels
            self._setup_step_wheels()
            # Extend settings tab with mapping forms and connections
            self._setup_settings_tab_sections()
            # Setup spectrometer selection (moved into settings)
            self._setup_spectrometer_tab()

            # Setup icons (ensure relative SVGs are loaded)
            self._setup_icons()
            # Setup inline external laser toggle next to laser buttons
            self._setup_external_laser_checkbox_inline()
            # Setup PeakID tab (peak detection and identification)
            self._setup_peakid_tab()
            # Add help link to the status bar
            self._setup_help_link()
            
            self.logger.info("UI setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup UI: {e}")
            self._show_error_message("UI Setup Error", f"Failed to load UI: {e}")

    def _apply_responsive_layout(self):
        """Replace absolute positioning with splitters so sections become resizable."""
        try:
            if getattr(self, '_responsive_layout_applied', False):
                return
            central = self.centralWidget()
            if central is None:
                return

            outer_layout = QtWidgets.QHBoxLayout()
            outer_layout.setContentsMargins(12, 12, 12, 12)
            outer_layout.setSpacing(12)
            central.setLayout(outer_layout)

            outer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            outer_splitter.setChildrenCollapsible(False)
            outer_layout.addWidget(outer_splitter)

            for widget in (self.tabWidget, self.manual_controls, self.tabWidget_2):
                widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

            outer_splitter.addWidget(self.tabWidget)
            outer_splitter.addWidget(self.manual_controls)

            console_panel = QtWidgets.QWidget()
            console_layout = QtWidgets.QVBoxLayout(console_panel)
            console_layout.setContentsMargins(12, 12, 12, 12)
            console_layout.setSpacing(8)
            console_label = QtWidgets.QLabel("Application console")
            console_label.setObjectName("consolePanelTitle")
            console_layout.addWidget(console_label)
            self.console_box.setParent(console_panel)
            self.console_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            console_layout.addWidget(self.console_box)

            self._console_layout = console_layout

            right_column = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            right_column.setChildrenCollapsible(False)
            right_column.addWidget(self.tabWidget_2)
            right_column.addWidget(console_panel)
            right_column.setSizes([600, 250])
            outer_splitter.addWidget(right_column)

            outer_splitter.setStretchFactor(0, 3)
            outer_splitter.setStretchFactor(1, 1)
            outer_splitter.setStretchFactor(2, 3)

            self.outer_splitter = outer_splitter
            self.right_splitter = right_column
            self._restore_splitter_states()

            self._responsive_layout_applied = True
        except Exception as e:
            self.logger.warning(f"Unable to modernize window layout: {e}")

    def _rebuild_acquisition_tab_layout(self):
        """Rebuild the acquisition tab layout to make the plot adapt to available space."""
        try:
            if not hasattr(self, 'tab_acquisition'):
                return
            
            # Create main vertical layout for the tab
            main_layout = QtWidgets.QVBoxLayout()
            main_layout.setContentsMargins(10, 10, 10, 10)
            main_layout.setSpacing(10)
            self.tab_acquisition.setLayout(main_layout)
            
            # Top controls row (laser buttons, directory, filename, etc.)
            top_row = QtWidgets.QHBoxLayout()
            top_row.setSpacing(10)
            
            # Left side: Laser controls (connect/stop + status widgets)
            laser_group = QtWidgets.QHBoxLayout()
            laser_group.setSpacing(10)

            laser_buttons_column = QtWidgets.QVBoxLayout()
            laser_buttons_column.setSpacing(6)
            if hasattr(self, 'Connect_laser_button'):
                self.Connect_laser_button.setParent(self.tab_acquisition)
                self.Connect_laser_button.setMinimumWidth(180)
                self.Connect_laser_button.setSizePolicy(
                    QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                laser_buttons_column.addWidget(self.Connect_laser_button)
            if hasattr(self, 'laser_stop_button'):
                self.laser_stop_button.setParent(self.tab_acquisition)
                self.laser_stop_button.setMinimumWidth(180)
                self.laser_stop_button.setSizePolicy(
                    QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                laser_buttons_column.addWidget(self.laser_stop_button)
            if laser_buttons_column.count():
                laser_group.addLayout(laser_buttons_column)

            laser_status_column = QtWidgets.QVBoxLayout()
            laser_status_column.setSpacing(6)
            if hasattr(self, 'progressBar_laser'):
                self.progressBar_laser.setParent(self.tab_acquisition)
                self.progressBar_laser.setMaximumHeight(24)
                laser_status_column.addWidget(self.progressBar_laser)
            if hasattr(self, 'external_laser_checkbox'):
                self.external_laser_checkbox.setParent(self.tab_acquisition)
                laser_status_column.addWidget(self.external_laser_checkbox)
            if laser_status_column.count():
                laser_group.addLayout(laser_status_column)

            if laser_group.count():
                top_row.addLayout(laser_group)
            
            if laser_group.count():
                top_row.addStretch(1)
            main_layout.addLayout(top_row)

            # Hide legacy directory button / label (replaced by fields below)
            for _attr in ('choose_directory_button', 'current_directory_label', 'label_28'):
                _w = getattr(self, _attr, None)
                if _w:
                    _w.hide()

            # Output settings: save directory + browse, file stem (mirrors mapping tab)
            output_grid = QtWidgets.QGridLayout()
            output_grid.setSpacing(6)

            self.acq_save_dir = QtWidgets.QLineEdit(self.tab_acquisition)
            init_dir = (getattr(self.config, 'folder_path', '') or '').rstrip(os.sep)
            self.acq_save_dir.setText(init_dir)
            self.acq_browse_btn = QtWidgets.QPushButton("Browse...", self.tab_acquisition)
            save_dir_row = QtWidgets.QHBoxLayout()
            save_dir_row.addWidget(self.acq_save_dir)
            save_dir_row.addWidget(self.acq_browse_btn)
            output_grid.addWidget(QtWidgets.QLabel("Save directory:"), 0, 0)
            output_grid.addLayout(save_dir_row, 0, 1)

            output_grid.addWidget(QtWidgets.QLabel("File stem:"), 1, 0)
            if hasattr(self, 'one_shot_filename_box'):
                self.one_shot_filename_box.setParent(self.tab_acquisition)
                output_grid.addWidget(self.one_shot_filename_box, 1, 1)

            main_layout.addLayout(output_grid)
            
            # Middle controls row (action buttons, profile settings)
            middle_row = QtWidgets.QHBoxLayout()
            middle_row.setSpacing(10)
            
            if hasattr(self, 'go_button'):
                middle_row.addWidget(self.go_button)
            if hasattr(self, 'profile_go_button'):
                middle_row.addWidget(self.profile_go_button)
            
            # Profile settings
            if hasattr(self, 'label_27'):
                middle_row.addWidget(self.label_27)
            if hasattr(self, 'profile_steps_combobox'):
                middle_row.addWidget(self.profile_steps_combobox)
            if hasattr(self, 'label_29'):
                middle_row.addWidget(self.label_29)
            if hasattr(self, 'shots_delay_combobox'):
                middle_row.addWidget(self.shots_delay_combobox)
            
            middle_row.addStretch(1)
            
            if hasattr(self, 'one_shot_detect_btn'):
                middle_row.addWidget(self.one_shot_detect_btn)
            
            main_layout.addLayout(middle_row)
            
            # Graph layout - this should expand to fill remaining space
            if hasattr(self, 'graph_layout'):
                # Remove fixed geometry and set proper size policy
                self.graph_layout.setParent(self.tab_acquisition)
                self.graph_layout.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding, 
                    QtWidgets.QSizePolicy.Expanding
                )
                # Add with stretch factor so it takes all available space
                main_layout.addWidget(self.graph_layout, 1)
            self._synchronize_laser_button_sizes()
        except Exception as e:
            self.logger.warning(f"Unable to rebuild acquisition tab layout: {e}")

    def _synchronize_laser_button_sizes(self):
        """Keep the connect and stop laser buttons visually consistent."""
        try:
            buttons = [
                getattr(self, 'Connect_laser_button', None),
                getattr(self, 'laser_stop_button', None),
            ]
            buttons = [btn for btn in buttons if isinstance(btn, QtWidgets.QPushButton)]
            if len(buttons) < 2:
                return
            for btn in buttons:
                btn.setMinimumWidth(200)
                btn.setMinimumHeight(36)
                btn.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        except Exception as e:
            self.logger.warning(f"Unable to align laser button sizes: {e}")

    def _rebuild_micro_tab_layout(self):
        """Give the micro tab a modern, stretchable layout."""
        try:
            micro_tab = getattr(self, 'tabWidget_2', None)
            if micro_tab is None:
                return
            first_tab = micro_tab.widget(0)
            if first_tab is None:
                return

            layout = QtWidgets.QVBoxLayout()
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)
            first_tab.setLayout(layout)

            viewer_container = getattr(self, 'horizontalLayoutWidget_2', None)
            if viewer_container is not None:
                viewer_container.setParent(first_tab)
                viewer_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                layout.addWidget(viewer_container, 1)

            controls_row = QtWidgets.QHBoxLayout()
            controls_row.addWidget(self.save_micro_capture)
            controls_row.addStretch(1)
            controls_row.addWidget(self.micro_light_checkbox)
            layout.addLayout(controls_row)
        except Exception as e:
            self.logger.warning(f"Unable to rebuild micro tab layout: {e}")

    def _modernize_manual_controls(self):
        """Convert manual control group to a layout-driven design."""
        try:
            controls = getattr(self, 'manual_controls', None)
            if controls is None:
                return

            direction_buttons = [
                getattr(self, 'north_button', None),
                getattr(self, 'south_button', None),
                getattr(self, 'east_button', None),
                getattr(self, 'west_button', None),
                getattr(self, 'up_button', None),
                getattr(self, 'down_button', None),
            ]
            if not all(direction_buttons):
                return

            # Clear any existing layout and build a modern one
            for child_layout in controls.findChildren(QtWidgets.QLayout):
                QtWidgets.QWidget().setLayout(child_layout)

            layout = QtWidgets.QVBoxLayout()
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(16)
            controls.setLayout(layout)

            # Directional pad (XY + Z) inside a capped container
            axes_container = QtWidgets.QWidget(controls)
            axes_container.setObjectName("manualAxesContainer")
            axes_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            axes_container.setMaximumHeight(220)
            axes_layout = QtWidgets.QHBoxLayout(axes_container)
            axes_layout.setSpacing(24)
            axes_layout.setContentsMargins(8, 4, 8, 4)

            xy_stack = QtWidgets.QVBoxLayout()
            xy_grid = QtWidgets.QGridLayout()
            xy_grid.setSpacing(6)
            xy_label = getattr(self, 'label_6', None)
            if xy_label is not None:
                xy_label.setAlignment(QtCore.Qt.AlignCenter)
                xy_grid.addWidget(xy_label, 1, 1, QtCore.Qt.AlignCenter)
            xy_grid.addWidget(self.north_button, 0, 1)
            xy_grid.addWidget(self.west_button, 1, 0)
            xy_grid.addWidget(self.east_button, 1, 2)
            xy_grid.addWidget(self.south_button, 2, 1)
            xy_stack.addLayout(xy_grid)
            axes_layout.addLayout(xy_stack, 2)

            z_stack = QtWidgets.QVBoxLayout()
            z_stack.setSpacing(6)
            z_label = getattr(self, 'label_7', None)
            z_stack.addWidget(self.up_button, 0, QtCore.Qt.AlignCenter)
            if z_label is not None:
                z_label.setAlignment(QtCore.Qt.AlignCenter)
                z_stack.addWidget(z_label, 0, QtCore.Qt.AlignCenter)
            z_stack.addWidget(self.down_button, 0, QtCore.Qt.AlignCenter)
            axes_layout.addLayout(z_stack, 1)

            layout.addWidget(axes_container)

            # Compact step size selectors (placeholders for wheels)
            steps_layout = QtWidgets.QHBoxLayout()
            steps_layout.setSpacing(18)

            label_xy = getattr(self, 'label_4', None)
            label_z = getattr(self, 'label_5', None)

            if hasattr(self, 'steps_xy_spinbox'):
                xy_container = QtWidgets.QWidget(controls)
                xy_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

                xy_column = QtWidgets.QVBoxLayout(xy_container)
                xy_column.setContentsMargins(0, 0, 0, 0)
                xy_column.setSpacing(4)
                if label_xy is not None:
                    label_xy.setAlignment(QtCore.Qt.AlignCenter)
                    label_xy.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                    label_xy.setFixedHeight(18)
                    xy_column.addWidget(label_xy, 0, QtCore.Qt.AlignCenter)
                self.steps_xy_spinbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                xy_column.addWidget(self.steps_xy_spinbox, 0, QtCore.Qt.AlignCenter)
                steps_layout.addWidget(xy_container, 1)

            if hasattr(self, 'steps_z_spinbox'):
                z_container = QtWidgets.QWidget(controls)
                z_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

                z_column = QtWidgets.QVBoxLayout(z_container)
                z_column.setContentsMargins(0, 0, 0, 0)
                z_column.setSpacing(4)
                if label_z is not None:
                    label_z.setAlignment(QtCore.Qt.AlignCenter)
                    label_z.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                    label_z.setFixedHeight(18)
                    z_column.addWidget(label_z, 0, QtCore.Qt.AlignCenter)
                self.steps_z_spinbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                z_column.addWidget(self.steps_z_spinbox, 0, QtCore.Qt.AlignCenter)
                steps_layout.addWidget(z_container, 1)

            layout.addLayout(steps_layout)

            # STOP button at the bottom
            if hasattr(self, 'stop_button'):
                self.stop_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                self.stop_button.setMinimumHeight(160)
                self.stop_button.setMaximumHeight(200)
                layout.addWidget(self.stop_button)
                # Reset button appears after stop to re-enable CNC controls
                if not hasattr(self, 'reset_cnc_button') or self.reset_cnc_button is None:
                    self.reset_cnc_button = QtWidgets.QPushButton("Reset CNC")
                    self.reset_cnc_button.setEnabled(False)
                self.reset_cnc_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                self.reset_cnc_button.setMaximumHeight(48)
                layout.addWidget(self.reset_cnc_button)
                layout.addStretch(1)

        except Exception as e:
            self.logger.warning(f"Unable to modernize manual controls: {e}")
    
    def _embed_mapping_tab(self):
        """Embed the Mapping GUI as an additional tab within the main window."""
        try:
            if not hasattr(self, 'tabWidget') or self.tabWidget is None:
                self.logger.warning("tabWidget not found; cannot embed mapping UI")
                return
            self.mapping_tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(self.mapping_tab)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

            controls_layout = QtWidgets.QHBoxLayout()
            self.mapping_start_btn = QtWidgets.QPushButton("Start Mapping")
            self.mapping_start_btn.setMinimumWidth(160)
            self.mapping_start_btn.clicked.connect(self._on_mapping_start_clicked)
            self.mapping_stop_btn = QtWidgets.QPushButton("Stop Mapping")
            self.mapping_stop_btn.setMinimumWidth(160)
            self.mapping_stop_btn.setEnabled(False)
            self.mapping_stop_btn.clicked.connect(self._on_mapping_stop_clicked)
            green_style = (
                "QPushButton { background-color: #28a745; color: white; font-weight: bold; }"
                "QPushButton:enabled:hover { background-color: #218838; }"
                "QPushButton:disabled { background-color: #2a2a2a; color: #7f7f7f; }"
            )
            self.mapping_start_btn.setStyleSheet(green_style)
            self.mapping_stop_btn.setStyleSheet(green_style)
            # Mapping basic settings moved from Settings tab
            if not hasattr(self, 'mapping_basic_form') or self.mapping_basic_form is None:
                self.mapping_basic_form = BasicParamsTab()
            self.mapping_basic_form.setParent(self.mapping_tab)
            self.mapping_basic_form.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
            info = QtWidgets.QLabel("Adjust the basic mapping grid, timing, and output directory here. Settings are saved automatically.")
            info.setWordWrap(True)

            self.mapping_progress_plot = ProgressPlotWidget()

            self.mapping_content_tabs = QtWidgets.QTabWidget(self.mapping_tab)
            settings_tab = QtWidgets.QWidget()
            settings_layout = QtWidgets.QVBoxLayout(settings_tab)
            settings_layout.setContentsMargins(8, 8, 8, 8)
            settings_layout.setSpacing(8)
            settings_layout.addWidget(info)
            settings_layout.addWidget(self.mapping_basic_form, 1)
            self.mapping_content_tabs.addTab(settings_tab, "Settings")

            progress_tab = QtWidgets.QWidget()
            progress_layout = QtWidgets.QVBoxLayout(progress_tab)
            progress_layout.setContentsMargins(0, 0, 0, 0)
            progress_layout.setSpacing(0)
            self.mapping_progress_plot.setParent(progress_tab)
            self.mapping_progress_plot.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
            progress_layout.addWidget(self.mapping_progress_plot, 1)
            self.mapping_content_tabs.addTab(progress_tab, "Progress")

            layout.addWidget(self.mapping_content_tabs, 1)

            connections_group = QtWidgets.QGroupBox("Device Connections")
            grid = QtWidgets.QGridLayout(connections_group)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(8)

            grid.addWidget(QtWidgets.QLabel("CNC:"), 0, 0)
            self.mapping_cnc_status = QtWidgets.QLabel("Connecting…")
            self.mapping_cnc_status.setStyleSheet("color: orange;")
            grid.addWidget(self.mapping_cnc_status, 0, 1)

            grid.addWidget(QtWidgets.QLabel("Pulse controller:"), 1, 0)
            self.mapping_pulse_status = QtWidgets.QLabel("Connecting…")
            self.mapping_pulse_status.setStyleSheet("color: orange;")
            grid.addWidget(self.mapping_pulse_status, 1, 1)

            connections_group.setMaximumWidth(480)

            estimates_group = QtWidgets.QGroupBox("Estimates")
            estimates_layout = QtWidgets.QVBoxLayout(estimates_group)
            self.mapping_estimate_duration = QtWidgets.QLabel("Estimated duration: --")
            self.mapping_estimate_filesize = QtWidgets.QLabel("Estimated file size: --")
            estimates_layout.addWidget(self.mapping_estimate_duration)
            estimates_layout.addWidget(self.mapping_estimate_filesize)
            estimates_group.setMaximumWidth(480)
            info_row = QtWidgets.QHBoxLayout()
            info_row.addWidget(connections_group)
            info_row.addWidget(estimates_group)
            info_row.addStretch(1)
            layout.addLayout(info_row)

            controls_layout.addWidget(self.mapping_start_btn)
            controls_layout.addWidget(self.mapping_stop_btn)
            controls_layout.addStretch(1)
            layout.addLayout(controls_layout)

            self.mapping_progress_bar = QtWidgets.QProgressBar()
            self.mapping_progress_bar.setRange(0, 100)
            self.mapping_progress_bar.setValue(0)
            layout.addWidget(self.mapping_progress_bar)

            self.mapping_status_label = QtWidgets.QLabel("Ready")
            layout.addWidget(self.mapping_status_label)

            self.mapping_log_queue = queue.Queue()
            self.mapping_log_timer = QtCore.QTimer(self)
            self.mapping_log_timer.timeout.connect(self._drain_mapping_log_queue)
            self.mapping_log_timer.start(200)

            self.mapping_progress_png_path = None
            self.mapping_progress_timer = QtCore.QTimer(self)
            self.mapping_progress_timer.timeout.connect(self._refresh_mapping_progress_plot)
            self.mapping_progress_timer.start(2000)

            self.mapping_monitor_timer = QtCore.QTimer(self)
            self.mapping_monitor_timer.timeout.connect(self._check_mapping_thread)
            self.mapping_monitor_timer.start(500)

            self.mapping_thread = None
            self._mapping_run_had_error = False

            # Insert Mapping right after the "One shot" tab (index 1)
            tab_index = self.tabWidget.insertTab(1, self.mapping_tab, "Mapping")
            self.tabWidget.setTabToolTip(tab_index, "Automated mapping control (CNC, pulse, laser)")
            # Move Settings tab (tab_2) to the end
            if hasattr(self, 'tab_2'):
                settings_idx = self.tabWidget.indexOf(self.tab_2)
                if settings_idx != -1:
                    settings_text = self.tabWidget.tabText(settings_idx)
                    self.tabWidget.removeTab(settings_idx)
                    new_idx = self.tabWidget.addTab(self.tab_2, settings_text)
                    self.tabWidget.setTabToolTip(new_idx, "Spectrometer and application settings")
            self._setup_mapping_estimate_signals()
            self._update_mapping_estimates()
        except Exception as e:
            self.mapping_tab = None
            self.logger.warning(f"Failed to embed mapping UI: {e}")

    def _append_mapping_console(self, text: str):
        """Route mapping log messages to the main console."""
        message = f"[Mapping] {text}\n"
        try:
            self.append_text(message)
        except Exception:
            self.logger.info(message)

    def _drain_mapping_log_queue(self):
        """Pull log entries produced by the mapping thread (console + log file)."""
        if not hasattr(self, 'mapping_log_queue'):
            return
        try:
            while not self.mapping_log_queue.empty():
                msg = self.mapping_log_queue.get_nowait()
                if msg.startswith("ERROR"):
                    self._mapping_run_had_error = True
                self.logger.info(f"[Mapping] {msg}")
                self._append_mapping_console(msg)
        except Exception:
            pass

    def _refresh_mapping_progress_plot(self):
        """Refresh the mapping progress plot from the generated PNG."""
        if not getattr(self, 'mapping_progress_plot', None):
            return
        if not self.mapping_progress_png_path:
            return
        try:
            self.mapping_progress_plot.update_plot(str(self.mapping_progress_png_path))
        except Exception:
            pass

    def _check_mapping_thread(self):
        """Monitor the mapping background thread."""
        thread = getattr(self, 'mapping_thread', None)
        if thread is None:
            return
        if not thread.is_alive():
            self._finalize_mapping_run()

    def _set_mapping_lock(self, locked: bool):
        """Disable (or re-enable) controls that must not be used during mapping.

        This prevents accidental table movements or laser shots while the
        mapping thread owns the CNC, pulse controller, and laser.
        """
        enabled = not locked
        # Table direction buttons
        for attr in ('north_button', 'south_button', 'east_button', 'west_button',
                     'up_button', 'down_button'):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(enabled)
        # One-shot / multi-shot acquisition buttons
        for attr in ('go_button', 'profile_go_button'):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(enabled)

    def _finalize_mapping_run(self):
        """Clean up UI state after mapping completes."""
        if getattr(self, 'mapping_thread', None):
            self.mapping_thread = None

        # Re-enable table direction buttons only; shot buttons stay disabled
        # because the laser is fully disconnected below.
        for attr in ('north_button', 'south_button', 'east_button', 'west_button',
                     'up_button', 'down_button'):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(True)

        self.mapping_start_btn.setEnabled(True)
        self.mapping_stop_btn.setEnabled(False)
        if getattr(self, '_mapping_run_had_error', False):
            self.mapping_status_label.setText("Completed with errors")
        else:
            self.mapping_status_label.setText("Completed")

        # Fully disconnect the laser so the user must reconnect via
        # "Connect Laser" before firing again.
        try:
            self.laser_handler.disconnect()
            self._laser_in_mapping_mode = False
        except Exception as exc:
            self.logger.warning(f"Failed to disconnect laser after mapping: {exc}")
        # Disable shot buttons and reset the laser progress bar
        if hasattr(self, 'go_button'):
            self.go_button.setEnabled(False)
        if hasattr(self, 'profile_go_button'):
            self.profile_go_button.setEnabled(False)
        if hasattr(self, 'progressBar_laser'):
            self.progressBar_laser.setValue(0)

        self._refresh_mapping_connection_status()
        self._resume_light_after_mapping()
        self._resume_camera_after_mapping()

    def _on_mapping_start_clicked(self):
        """Validate configuration and start the mapping worker."""
        if getattr(self, 'mapping_thread', None) and self.mapping_thread.is_alive():
            self._show_warning_message("Mapping", "Mapping is already running.")
            return
        try:
            params = self._build_mapping_params()
        except Exception as exc:
            self._show_error_message("Mapping", f"Invalid mapping settings: {exc}")
            return

        if not self.serial_handler.cnc_serial:
            self._show_warning_message("Mapping", "Connect the CNC controller first.")
            return
        if not getattr(self.serial_handler, 'pulse_serial', None):
            self._show_warning_message("Mapping", "Connect the pulse controller first.")
            return
        if not self.laser_handler.get_mapping_controller() and not getattr(self.laser_handler, 'connected', False):
            self._show_warning_message("Mapping", "Connect the laser first.")
            return
        if not self._active_spectrometers:
            self._show_warning_message("Mapping", "Select at least one spectrometer before mapping.")
            return
        self.logger.info(self._build_spectrometer_selection_summary())

        if not self._ensure_mapping_laser_ready():
            return

        handles, wavelengths, identities = self._collect_spectrometer_data_for_mapping()
        if not handles:
            self._show_error_message("Mapping", "No spectrometers available for mapping.")
            self._restore_acquisition_laser_mode()
            return

        self.mapping_progress_png_path = Path(params['basic']['save_dir']) / "progress.png"
        # Delete stale progress PNG so the timer doesn't reload the old image
        try:
            if self.mapping_progress_png_path.exists():
                self.mapping_progress_png_path.unlink()
        except Exception:
            pass
        # Reset progress plot to black at the start of a new mapping
        if hasattr(self, 'mapping_progress_plot') and self.mapping_progress_plot:
            self.mapping_progress_plot.clear_plot()
        self.mapping_progress_bar.setValue(0)
        self.mapping_status_label.setText("Running")
        self._mapping_run_had_error = False

        try:
            self._pause_light_for_mapping()
            self._pause_camera_for_mapping()
            self.mapping_thread = MappingThread(
                params=params,
                log_queue=self.mapping_log_queue,
                progress_callback=self._update_mapping_progress,
                cnc_ser=self.serial_handler.cnc_serial,
                pulse_ser=self.serial_handler.pulse_serial,
                laser_obj=self.laser_handler.get_mapping_controller(),
                spectrometer_handles=handles,
                spectrometer_wavelengths=wavelengths,
                spectrometer_identities=identities,
            )
            self.mapping_thread.start()
            self.mapping_start_btn.setEnabled(False)
            self.mapping_stop_btn.setEnabled(True)
            self._set_mapping_lock(True)
        except Exception as exc:
            self.logger.error(f"Failed to start mapping thread: {exc}")
            self._show_error_message("Mapping", f"Failed to start mapping: {exc}")
            self._resume_light_after_mapping()
            self._resume_camera_after_mapping()
            self._restore_acquisition_laser_mode()

    def _update_mapping_progress(self, value: int):
        """Update progress bar from the mapping thread callback."""
        if hasattr(self, 'mapping_progress_bar'):
            self.mapping_progress_bar.setValue(int(value))

    def _on_mapping_stop_clicked(self):
        """Request stop for the mapping worker."""
        thread = getattr(self, 'mapping_thread', None)
        if not thread or not thread.is_alive():
            return
        try:
            thread.stop_event.set()
            self._append_mapping_console("Stop requested. Mapping will halt after the current line.")
        except Exception:
            pass

    def _set_output_directory(self, folder_path: str, source: str = "main", persist: bool = True, log_change: bool = True):
        """Synchronize output directory across acquisition and mapping tabs.

        *source* indicates who initiated the change so that we skip updating
        the widget that already holds the new value (avoids signal loops).
        Accepted values: ``"acquisition"``, ``"mapping"``, ``"main"`` (Browse
        button or programmatic), ``"init"`` (startup).
        """
        folder_path = (folder_path or "").strip()
        if not folder_path:
            return
        try:
            normalized = os.path.normpath(folder_path)
            if not normalized:
                return
            with_sep = normalized if normalized.endswith(os.sep) else normalized + os.sep
            display = normalized  # without trailing separator, for text fields

            # Skip if both config fields already match
            if (getattr(self.config, 'folder_path', '') == with_sep
                    and os.path.normpath(getattr(self.config.mapping, 'save_dir', '') or '') == normalized):
                return

            # Update config
            self.config.folder_path = with_sep
            self.config.mapping.save_dir = display

            # Update one-shot tab save-dir field
            if source != "acquisition" and hasattr(self, 'acq_save_dir'):
                blocker = QtCore.QSignalBlocker(self.acq_save_dir)
                self.acq_save_dir.setText(display)
                del blocker

            # Update mapping tab save-dir field
            if source != "mapping" and getattr(self, 'mapping_basic_form', None):
                save_dir_field = getattr(self.mapping_basic_form, 'save_dir', None)
                if save_dir_field is not None:
                    blocker = QtCore.QSignalBlocker(save_dir_field)
                    save_dir_field.setText(display)
                    del blocker

            if persist:
                try:
                    self._persist_config()
                except Exception as persist_error:
                    self.logger.warning(f"Failed to persist output directory: {persist_error}")
            if log_change:
                self.logger.info(f"Output directory set by {source}: {display}")
        except Exception as e:
            self.logger.warning(f"Unable to update output directory ({source}): {e}")

    def _handle_acq_save_dir_changed(self, new_path: str):
        """Handle save directory edits from the one-shot acquisition tab."""
        new_path = (new_path or "").strip()
        if not new_path:
            return
        # Only sync when the typed path points to an existing directory
        if os.path.isdir(new_path):
            self._set_output_directory(new_path, source="acquisition", persist=True)

    def _handle_mapping_save_dir_changed(self, new_path: str):
        """Handle save directory edits initiated from the embedded mapping tab."""
        new_path = (new_path or "").strip()
        if not new_path:
            return
        # Only sync when the typed path points to an existing directory
        if os.path.isdir(new_path):
            self._set_output_directory(new_path, source="mapping", persist=True)

    def _apply_modern_styles(self):
        """Application-level stylesheet: consistent fonts + modern look."""
        try:
            stylesheet = """
                /* ===== Font hierarchy (single source of truth) ===== */
                * {
                    font-family: "Segoe UI", sans-serif;
                    font-size: 11pt;
                }
                /* Tab labels */
                QTabBar::tab {
                    font-size: 11pt;
                }
                /* Group box titles */
                QGroupBox {
                    font-size: 11pt;
                    font-weight: bold;
                }
                QGroupBox > QWidget {
                    font-weight: normal;
                }
                /* Action buttons (One Shot, Multi Shots, Connect Laser, etc.) */
                QPushButton#go_button,
                QPushButton#profile_go_button,
                QPushButton#Connect_laser_button,
                QPushButton#laser_stop_button,
                QPushButton#mapping_start_btn,
                QPushButton#mapping_stop_btn {
                    font-size: 12pt;
                    font-weight: bold;
                }
                /* Emergency stop button */
                QPushButton#stop_button {
                    font-size: 14pt;
                    font-weight: bold;
                }
                /* CNC axis group labels (XY / Z) */
                QLabel#label_6, QLabel#label_7 {
                    font-size: 18pt;
                    font-weight: bold;
                }
                /* Console output */
                QTextEdit#console_box {
                    font-family: "Consolas", "Courier New", monospace;
                    font-size: 9pt;
                }
                /* Console panel title */
                QLabel#consolePanelTitle {
                    font-weight: 600;
                    letter-spacing: 0.5px;
                }

                /* ===== Visual styling ===== */
                QWidget#manual_controls, QGroupBox {
                    border: 1px solid #3a3a3a;
                    border-radius: 8px;
                    margin-top: 12px;
                    padding: 12px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 0 6px;
                    color: #d0d0d0;
                }
                QPushButton {
                    border-radius: 6px;
                    padding: 6px 12px;
                    border: 1px solid #444;
                    background-color: #2b2b2b;
                    color: #f0f0f0;
                }
                QPushButton:enabled:hover {
                    background-color: #3a7bd5;
                    border-color: #3a7bd5;
                    color: white;
                }
                QPushButton:disabled {
                    background-color: #2a2a2a;
                    color: #7f7f7f;
                    border-color: #2a2a2a;
                }
                QPushButton:flat {
                    background: transparent;
                    border: none;
                    padding: 4px;
                }
            """
            self.setStyleSheet(self.styleSheet() + stylesheet)
        except Exception as e:
            self.logger.warning(f"Unable to apply modern styles: {e}")

    def _apply_text_standards(self):
        """Ensure button labels follow a consistent style."""
        try:
            text_map = {
                'Connect_laser_button': "Connect Laser",
                'laser_stop_button': "Stop Laser",
                'go_button': "One Shot",
                'profile_go_button': "Multi Shots",
                'one_shot_detect_btn': "Detect Peaks",
                'save_micro_capture': "Save Micro Capture",
            }
            for name, text in text_map.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setText(text)
            
            # Style acquisition buttons green
            if hasattr(self, 'go_button'):
                self.go_button.setStyleSheet(
                    "QPushButton { background-color: #28a745; color: white; font-weight: bold; }"
                    "QPushButton:hover { background-color: #218838; }"
                    "QPushButton:disabled { background-color: #2a2a2a; color: #7f7f7f; }"
                )
            if hasattr(self, 'profile_go_button'):
                self.profile_go_button.setStyleSheet(
                    "QPushButton { background-color: #28a745; color: white; font-weight: bold; }"
                    "QPushButton:hover { background-color: #218838; }"
                    "QPushButton:disabled { background-color: #2a2a2a; color: #7f7f7f; }"
                )
        except Exception as e:
            self.logger.warning(f"Unable to harmonize button text: {e}")

    def _apply_help_tooltips(self):
        """Add concise hover help to key controls for discoverability."""
        try:
            tooltip_map = {
                # Acquisition tab
                'Connect_laser_button': "Connect to the configured laser controller and unlock acquisition controls.",
                'laser_stop_button': "Stop the laser immediately and abort any pending acquisition.",
                'acq_save_dir': "Working directory where spectra and micro captures will be saved.",
                'acq_browse_btn': "Browse for an output directory.",
                'one_shot_filename_box': "Base filename for saved spectra; timestamps are added automatically.",
                'go_button': "Trigger a single acquisition using the current settings.",
                'profile_go_button': "Start a multi-shot profile with the selected steps and delay.",
                'profile_steps_combobox': "Number of laser shots to execute in a Multi Shots run.",
                'shots_delay_combobox': "Pause between successive shots during Multi Shots (seconds).",
                'one_shot_detect_btn': "Detect peaks in the latest spectrum and list their positions.",
                'external_laser_checkbox': "Enable when the experiment relies on an external laser trigger.",
                'progressBar_laser': "Indicates the current status of the laser connection/setup.",
                # Settings tab
                'integration_time_combobox': "Spectrometer integration time applied to each exposure (µs).",
                'trigger_delay_combobox': "Delay between receiving the trigger and starting the exposure.",
                'prominence_spinbox': "Minimum prominence (counts) for the peak detection algorithm.",
                'distance_spinbox': "Minimum point distance between peaks to consider them separate.",
                'show_grid_option': "Toggle the grid overlay on the main spectrum plot.",
                # Spectrometers tab
                'spectrometer_info_label': "Overview of which devices participate in acquisitions.",
                'detect_spectrometers_button': "Reset spectrometer state and re-scan the USB bus.",
                'spectrometer_list_widget': "Check the spectrometers that should participate in acquisitions.",
                # Micro view tab
                'save_micro_capture': "Store the current micro-camera frame to the output directory.",
                'micro_light_checkbox': "Enable or disable illumination for the micro camera scene.",
                'console_box': "Live application console output for troubleshooting.",
                # Manual controls
                'north_button': "Jog the XY table forward (Y+).",
                'south_button': "Jog the XY table backward (Y-).",
                'east_button': "Jog the XY table to the right (X+).",
                'west_button': "Jog the XY table to the left (X-).",
                'up_button': "Raise the Z axis.",
                'down_button': "Lower the Z axis.",
                'stop_button': "Immediately stop any manual movement.",
                'steps_xy_spinbox': "Step distance (µm) used for XY jog commands.",
                'steps_z_spinbox': "Step distance (µm) used for Z jog commands.",
            }
            for attr, tip in tooltip_map.items():
                widget = getattr(self, attr, None)
                if widget is None:
                    continue
                widget.setToolTip(tip)
        except Exception as e:
            self.logger.warning(f"Unable to apply hover tooltips: {e}")

    def _setup_help_link(self):
        """Add a clickable help link below the console output."""
        try:
            help_path = Path(__file__).parent.absolute() / "help.html"
            help_url = QtCore.QUrl.fromLocalFile(str(help_path))

            help_btn = QtWidgets.QPushButton("📖  Open Help Documentation")
            help_btn.setObjectName("helpLinkButton")
            help_btn.setCursor(QtCore.Qt.PointingHandCursor)
            help_btn.setToolTip(f"Open help documentation ({help_path.name}) in your browser")
            help_btn.setStyleSheet(
                "QPushButton#helpLinkButton {"
                "  color: #4fc3f7; text-decoration: underline; border: none;"
                "  background: transparent; font-size: 12pt; padding: 6px 12px;"
                "}"
                "QPushButton#helpLinkButton:hover { color: #81d4fa; }"
            )
            help_btn.clicked.connect(
                lambda: QtGui.QDesktopServices.openUrl(help_url)
            )

            if hasattr(self, '_console_layout'):
                self._console_layout.addWidget(help_btn, 0, QtCore.Qt.AlignLeft)
            else:
                self.statusbar.addPermanentWidget(help_btn)
        except Exception as e:
            self.logger.warning(f"Unable to add help link: {e}")

    def _restore_splitter_states(self):
        """Restore splitter positions from previous session."""
        try:
            if not hasattr(self, 'settings'):
                return
            mappings = [
                ('outer_splitter', 'splitters/outer'),
                ('right_splitter', 'splitters/right'),
            ]
            for attr, key in mappings:
                splitter = getattr(self, attr, None)
                if splitter is None:
                    continue
                state = self.settings.value(key)
                if state is None:
                    continue
                if isinstance(state, QtCore.QByteArray):
                    splitter.restoreState(state)
                elif isinstance(state, str):
                    splitter.restoreState(QtCore.QByteArray.fromHex(state.encode('utf-8')))
        except Exception as e:
            self.logger.warning(f"Unable to restore splitter positions: {e}")

    def _save_splitter_states(self):
        """Persist current splitter positions."""
        try:
            if not hasattr(self, 'settings'):
                return
            mappings = [
                ('outer_splitter', 'splitters/outer'),
                ('right_splitter', 'splitters/right'),
            ]
            for attr, key in mappings:
                splitter = getattr(self, attr, None)
                if splitter is None:
                    continue
                self.settings.setValue(key, splitter.saveState())
            self.settings.sync()
        except Exception as e:
            self.logger.warning(f"Unable to save splitter positions: {e}")

    def _setup_icons(self):
        """Assign icons to buttons explicitly so SVGs load reliably"""
        try:
            # Base path
            base = Path('images')

            def set_icon(widget, filename, size=48):
                path = str((base / filename).resolve())
                icon = QIcon(path)
                widget.setIcon(icon)
                widget.setIconSize(QtCore.QSize(size, size))

            # Legacy folder icon button removed; directory is now a text field

            # Movement arrows
            if hasattr(self, 'north_button'):
                set_icon(self.north_button, 'arrow-up.svg', 48)
            if hasattr(self, 'south_button'):
                set_icon(self.south_button, 'arrow-down.svg', 48)
            if hasattr(self, 'west_button'):
                set_icon(self.west_button, 'arrow-left.svg', 48)
            if hasattr(self, 'east_button'):
                set_icon(self.east_button, 'arrow-right.svg', 48)
            if hasattr(self, 'up_button'):
                set_icon(self.up_button, 'chevrons-up.svg', 48)
            if hasattr(self, 'down_button'):
                set_icon(self.down_button, 'chevrons-down.svg', 48)

        except Exception as e:
            self.logger.warning(f"Failed to setup icons: {e}")
    
    def _setup_peakid_tab(self):
        """Create 'PeakID' tab for peak detection and identification."""
        try:
            if not hasattr(self, 'tabWidget') or not isinstance(self.tabWidget, QtWidgets.QTabWidget):
                self.logger.warning("tabWidget not found; cannot add PeakID tab")
                return
            
            # Create tab and layouts
            self.peakid_tab = QtWidgets.QWidget()
            peakid_layout = QtWidgets.QVBoxLayout(self.peakid_tab)
            peakid_layout.setContentsMargins(8, 8, 8, 8)
            peakid_layout.setSpacing(8)
            
            # Row 1: parameter controls
            params_row = QtWidgets.QHBoxLayout()
            params_row.setSpacing(10)

            # Min height
            params_row.addWidget(QtWidgets.QLabel("Min height:"))
            self.peakid_min_height = QtWidgets.QDoubleSpinBox()
            self.peakid_min_height.setDecimals(1)
            self.peakid_min_height.setRange(0.0, 1e9)
            self.peakid_min_height.setSingleStep(10.0)
            self.peakid_min_height.setValue(50.0)
            self.peakid_min_height.setToolTip("Minimum peak height (counts)")
            params_row.addWidget(self.peakid_min_height)

            # Min prominence
            params_row.addWidget(QtWidgets.QLabel("Min prominence:"))
            self.peakid_min_prom = QtWidgets.QDoubleSpinBox()
            self.peakid_min_prom.setDecimals(1)
            self.peakid_min_prom.setRange(0.0, 1e9)
            self.peakid_min_prom.setSingleStep(10.0)
            self.peakid_min_prom.setValue(50.0)
            self.peakid_min_prom.setToolTip("Minimum peak prominence (counts)")
            params_row.addWidget(self.peakid_min_prom)

            # Match tolerance
            params_row.addWidget(QtWidgets.QLabel("Match tol (nm):"))
            self.peakid_tol_nm = QtWidgets.QDoubleSpinBox()
            self.peakid_tol_nm.setDecimals(3)
            self.peakid_tol_nm.setRange(0.0, 5.0)
            self.peakid_tol_nm.setSingleStep(0.01)
            self.peakid_tol_nm.setValue(0.05)
            self.peakid_tol_nm.setToolTip("Wavelength tolerance for matching (nm)")
            params_row.addWidget(self.peakid_tol_nm)

            params_row.addStretch(1)
            peakid_layout.addLayout(params_row)

            # Row 2: action buttons
            buttons_row = QtWidgets.QHBoxLayout()
            buttons_row.setSpacing(10)

            # Load spectrum button
            self.peakid_load_btn = QtWidgets.QPushButton("Load spectrum...")
            self.peakid_load_btn.setToolTip("Load a previously saved spectrum CSV")
            self.peakid_load_btn.clicked.connect(self.load_spectrum_file)
            buttons_row.addWidget(self.peakid_load_btn)

            # Analyze button
            self.peakid_analyze_btn = QtWidgets.QPushButton("Detect peaks & match")
            self.peakid_analyze_btn.setToolTip("Detect peaks on the current spectrum and match them to reference lines.")
            self.peakid_analyze_btn.clicked.connect(self.run_peakid_analysis)
            buttons_row.addWidget(self.peakid_analyze_btn)

            buttons_row.addStretch(1)
            peakid_layout.addLayout(buttons_row)
            
            # Plot area
            self.peakid_fig = Figure()
            self.peakid_ax = self.peakid_fig.add_subplot(111)
            self.peakid_canvas = FigureCanvas(self.peakid_fig)
            self.peakid_canvas.setToolTip("Live spectrum view with detected peaks overlay. Use the toolbar below to zoom or pan.")
            self.peakid_toolbar = NavigationToolbar(self.peakid_canvas, self, coordinates=True)
            self.peakid_toolbar.setToolTip("Standard matplotlib navigation controls (zoom, pan, save).")
            
            peakid_layout.addWidget(self.peakid_canvas)
            peakid_layout.addWidget(self.peakid_toolbar)
            
            # Summary text
            self.peakid_summary = QtWidgets.QTextEdit()
            self.peakid_summary.setReadOnly(True)
            self.peakid_summary.setToolTip("Text summary of detected peaks and element matches.")
            peakid_layout.addWidget(self.peakid_summary)
            
            # Add tab
            self.tabWidget.addTab(self.peakid_tab, "PeakID")
            self.logger.info("PeakID tab added")
        except Exception as e:
            self.logger.warning(f"Failed to setup PeakID tab: {e}")
    
    def run_peakid_analysis(self):
        """Run peak detection and line matching on the acquired spectrum."""
        try:
            if self.spectro1_data is None or self.spectro1_data.empty:
                self._show_warning_message("No spectrum", "Acquire a spectrum first (One shot).")
                return
            
            min_height = float(self.peakid_min_height.value())
            min_prom = float(self.peakid_min_prom.value())
            tol_nm = float(self.peakid_tol_nm.value())
            
            # Detect peaks on the current spectrum
            peaks_df = self.peak_identifier.detect_peaks(
                self.spectro1_data, min_height=min_height, min_prominence=min_prom
            )
            
            # Match peaks to reference lines
            matches_df = self.peak_identifier.match_peaks_to_lines(peaks_df, tol_nm=tol_nm)
            summary_df = self.peak_identifier.summarize_elements(matches_df)
            
            # Plot spectrum with peaks
            self.peakid_ax.clear()
            self.peakid_ax.plot(
                self.spectro1_data["wavelength_(nm)"],
                self.spectro1_data["intensity_dark_corrected"],
                linewidth=0.9,
                label="Spectrum"
            )
            
            # Plot detected peaks
            if peaks_df is not None and not peaks_df.empty:
                self.peakid_ax.scatter(
                    peaks_df["wavelength_(nm)"],
                    peaks_df["intensity_dark_corrected"],
                    marker="o",
                    s=20,
                    edgecolors="black",
                    facecolors="none",
                    label="Detected peaks"
                )
            
            # Plot matched peaks
            if matches_df is not None and not matches_df.empty:
                self.peakid_ax.scatter(
                    matches_df["peak_wavelength_nm"],
                    matches_df["intensity"],
                    marker="x",
                    s=40,
                    label="Matched peaks"
                )
                # Add small labels above each identified (matched) peak
                try:
                    labeled_wls = set()
                    for _, row in matches_df.iterrows():
                        x = float(row["peak_wavelength_nm"])
                        if x in labeled_wls:
                            continue
                        labeled_wls.add(x)
                        y = float(row["intensity"])
                        label = f"{row['element']} {row['ion']}"
                        self.peakid_ax.annotate(
                            label,
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 6),
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="#202020",
                            clip_on=True,
                        )
                except Exception:
                    pass
            
            self.peakid_ax.set_xlabel("Wavelength (nm)")
            self.peakid_ax.set_ylabel("Intensity (counts)")
            self.peakid_ax.set_title("Spectrum with detected and matched peaks")
            self.peakid_ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
            self.peakid_ax.legend(loc="upper right", fontsize="small")
            self.peakid_canvas.draw_idle()
            
            # Update summary
            self.peakid_summary.clear()
            num_detected = 0 if peaks_df is None or peaks_df.empty else int(len(peaks_df))
            num_matched_unique = 0
            if matches_df is not None and not matches_df.empty:
                num_matched_unique = int(matches_df["peak_wavelength_nm"].nunique())
            pct = (100.0 * num_matched_unique / num_detected) if num_detected > 0 else 0.0
            self.peakid_summary.insertPlainText(f"Matched peaks: {num_matched_unique} / {num_detected} ({pct:.1f}%)\n\n")
            
            if summary_df is None or summary_df.empty:
                self.peakid_summary.insertPlainText("No matches found.\n")
            else:
                self.peakid_summary.insertPlainText("Element summary (sorted by total matched intensity):\n\n")
                for _, row in summary_df.iterrows():
                    line = (
                        f"{row['element']:>3s}  "
                        f"n_lines={int(row['n_lines'])}, "
                        f"n_matches={int(row['n_matches'])}, "
                        f"total_I={row['total_intensity']:.1f}, "
                        f"max_I={row['max_intensity']:.1f}\n"
                    )
                    self.peakid_summary.insertPlainText(line)
                self.peakid_summary.insertPlainText("\nNote: Intensities are qualitative/semi-quantitative.\n")
            
        except Exception as e:
            self.logger.error(f"PeakID analysis failed: {e}")
            self._show_error_message("PeakID Error", f"Analysis failed: {e}")
    
    def load_spectrum_file(self):
        """Load a previously saved spectrum CSV and set as current spectrum."""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select CSV spectrum", "", "CSV files (*.csv);;All files (*.*)"
            )
            if not filename:
                return
            # Read CSV
            try:
                df = pd.read_csv(filename)
            except Exception as e:
                self._show_error_message("Load Error", f"Failed to read CSV:\n{e}")
                return
            if df is None or df.empty:
                self._show_error_message("Load Error", "CSV file is empty.")
                return
            # Map columns to expected names
            wave_col = None
            int_col = None
            for c in df.columns:
                cl = str(c).lower()
                if wave_col is None and ("wave" in cl or "nm" in cl):
                    wave_col = c
                if int_col is None and ("intensity_dark_corrected" in cl or "int" in cl or "count" in cl):
                    int_col = c
            # If exact saved format exists, prefer it
            if 'wavelength_(nm)' in df.columns and 'intensity_dark_corrected' in df.columns:
                wave_col = 'wavelength_(nm)'
                int_col = 'intensity_dark_corrected'
            if wave_col is None or int_col is None:
                self._show_error_message(
                    "Load Error",
                    "Could not find wavelength and intensity columns.\n"
                    "Expected columns containing 'wave'/'nm' and 'int'/'count' (or 'intensity_dark_corrected')."
                )
                return
            # Build normalized DataFrame
            loaded = pd.DataFrame({
                'wavelength_(nm)': df[wave_col].astype(float),
                'intensity_dark_corrected': df[int_col].astype(float)
            })
            self.spectro1_data = loaded
            # Plot in PeakID area
            if hasattr(self, 'peakid_ax'):
                self.peakid_ax.clear()
                self.peakid_ax.plot(
                    loaded['wavelength_(nm)'], loaded['intensity_dark_corrected'],
                    linewidth=0.9, label="Spectrum"
                )
                self.peakid_ax.set_xlabel("Wavelength (nm)")
                self.peakid_ax.set_ylabel("Intensity (counts)")
                self.peakid_ax.set_title(f"Spectrum loaded: {Path(filename).name}")
                self.peakid_ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
                self.peakid_ax.legend(loc="upper right", fontsize="small")
                self.peakid_canvas.draw_idle()
            if hasattr(self, 'peakid_summary'):
                self.peakid_summary.clear()
                self.peakid_summary.insertPlainText(
                    f"Loaded: {Path(filename).name}  (x: {wave_col}, y: {int_col})\n"
                )
            self.logger.info(f"Spectrum loaded from {filename}")
        except Exception as e:
            self.logger.error(f"Failed to load spectrum file: {e}")
            self._show_error_message("Load Error", f"Failed to load spectrum:\n{e}")
    
    
    def _setup_external_laser_checkbox_inline(self):
        """Create a checkbox next to laser buttons for external mode."""
        try:
            # If the .ui already defines the checkbox, just wire it; otherwise create fallback
            if hasattr(self, 'external_laser_checkbox') and isinstance(self.external_laser_checkbox, QtWidgets.QCheckBox):
                self.external_laser_checkbox.setChecked(getattr(self.config.laser, 'use_external', False))
                self.external_laser_checkbox.toggled.connect(self.toggle_external_laser)
            else:
                # Create checkbox
                self.external_laser_checkbox = QtWidgets.QCheckBox("External laser")
                self.external_laser_checkbox.setToolTip("Use external laser (no LAN). Spectrometer waits for external trigger.")
                self.external_laser_checkbox.setChecked(getattr(self.config.laser, 'use_external', False))
                self.external_laser_checkbox.toggled.connect(self.toggle_external_laser)
                
                # Try to insert next to Connect_laser_button in its parent layout
                parent = self.Connect_laser_button.parentWidget() if hasattr(self, 'Connect_laser_button') else None
                layout = parent.layout() if parent else None
                if layout:
                    # Find index of Connect_laser_button
                    insert_at = None
                    for i in range(layout.count()):
                        item = layout.itemAt(i)
                        w = item.widget() if item else None
                        if w is self.Connect_laser_button:
                            insert_at = i + 1  # place right after Connect
                            break
                    if insert_at is None:
                        # Fallback: append at end of the layout
                        layout.addWidget(self.external_laser_checkbox)
                    else:
                        layout.insertWidget(insert_at, self.external_laser_checkbox)
                else:
                    # As a fallback, add to status bar area
                    try:
                        self.statusbar.addPermanentWidget(self.external_laser_checkbox)
                    except Exception:
                        # If status bar not present, add near top-level layout if exists
                        if hasattr(self, 'verticalLayout'):
                            self.verticalLayout.addWidget(self.external_laser_checkbox)
            
            # If already enabled by config, reflect enabled state of acquisition buttons
            if getattr(self.config.laser, 'use_external', False):
                if hasattr(self, 'go_button'):
                    self.go_button.setEnabled(True)
                if hasattr(self, 'profile_go_button'):
                    self.profile_go_button.setEnabled(True)
                if hasattr(self, 'laser_stop_button'):
                    self.laser_stop_button.setEnabled(True)
        except Exception as e:
            self.logger.warning(f"Failed to setup inline external laser checkbox: {e}")

  
    def _setup_plots(self):
        """Setup matplotlib plots"""
        try:
            # Main spectrum plot
            fig1 = Figure()
            self.canvas = FigureCanvas(fig1)
            # Set size policy to expand and fill available space
            self.canvas.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
            self.mplvl.addWidget(self.canvas, 1)  # Stretch factor for canvas
            self.canvas.draw()
            self.toolbar = NavigationToolbar(self.canvas, self.mplwindow, coordinates=True)
            self.mplvl.addWidget(self.toolbar)  # Toolbar doesn't stretch
            
            # Profile plot (removed with Mapping tab)
            self.canvas2 = None
            self.toolbar2 = None
            
        except Exception as e:
            self.logger.error(f"Failed to setup plots: {e}")
    
    def _setup_combo_boxes(self):
        """Setup combo box values"""
        try:
            spec_cfg = getattr(self.config, 'spectrometer', None)
            mapping_cfg = getattr(self.config, 'mapping', None)

            def _set_combo_to_value(combo, value, fallback_index):
                if combo is None:
                    return
                target_value = value or ""
                if target_value:
                    idx = combo.findText(target_value)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                        return
                combo.setCurrentIndex(fallback_index)
            
            # Integration time
            self.integration_time_combobox.addItems(self.config.spectrometer.integration_time_list)
            selected_integration = getattr(spec_cfg, 'selected_integration_time', None)
            _set_combo_to_value(self.integration_time_combobox, selected_integration, 0)
            
            # Trigger delay
            self.trigger_delay_combobox.addItems(self.config.spectrometer.trigger_delay_list)
            selected_trigger_delay = getattr(spec_cfg, 'selected_trigger_delay', None)
            _set_combo_to_value(self.trigger_delay_combobox, selected_trigger_delay, 3)
            
            # Profile steps
            self.profile_steps_combobox.addItems(self.config.spectrometer.profile_steps_list)
            selected_profile_steps = getattr(spec_cfg, 'selected_profile_steps', None)
            _set_combo_to_value(self.profile_steps_combobox, selected_profile_steps, 1)
            
            # Shots delay
            self.shots_delay_combobox.addItems(self.config.spectrometer.shots_delay)
            selected_shots_delay = getattr(spec_cfg, 'selected_shots_delay', None)
            _set_combo_to_value(self.shots_delay_combobox, selected_shots_delay, 2)
            
            # Table steps - now using spinboxes for easier adjustment
            if hasattr(self, 'steps_xy_spinbox'):
                selected_xy_step = getattr(mapping_cfg, 'selected_xy_step', None)
                if selected_xy_step:
                    try:
                        self.steps_xy_spinbox.setValue(int(selected_xy_step))
                    except (ValueError, TypeError):
                        self.steps_xy_spinbox.setValue(100)  # Default to 100 µm
                else:
                    self.steps_xy_spinbox.setValue(100)
            if hasattr(self, 'steps_z_spinbox'):
                selected_z_step = getattr(mapping_cfg, 'selected_z_step', None)
                if selected_z_step:
                    try:
                        self.steps_z_spinbox.setValue(int(selected_z_step))
                    except (ValueError, TypeError):
                        self.steps_z_spinbox.setValue(100)  # Default to 100 µm
                else:
                    self.steps_z_spinbox.setValue(100)
            
        except Exception as e:
            self.logger.error(f"Failed to setup combo boxes: {e}")

    def _build_mapping_params(self) -> dict:
        """Construct parameter payload for the mapping thread."""
        self._persist_mapping_settings()
        if not hasattr(self, 'mapping_basic_form'):
            raise RuntimeError("Mapping forms are not available")

        basic = self.mapping_basic_form.get_params()
        advanced = self.mapping_advanced_form.get_params() if hasattr(self, 'mapping_advanced_form') else {}

        # Ensure ports use current configuration
        basic['port_CNC'] = self.config.serial.cnc_port
        basic['port_pulse_controller'] = self.config.serial.pulse_port

        # Ensure save directory exists
        save_dir = basic.get('save_dir') or self.config.mapping.save_dir or self.config.folder_path or str(Path.cwd())
        save_dir = os.path.normpath(save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        basic['save_dir'] = save_dir

        advanced['laser_ip'] = self.config.laser.host
        advanced['laser_port'] = self.config.laser.port
        advanced['laser_login'] = self.config.laser.login_code
        advanced['ftp_path'] = advanced.get('ftp_path') or "."

        params = {
            'basic': basic,
            'advanced': advanced,
            'spectrometers_selected': self._gather_active_spectrometer_handles(),
        }
        return params

    def _persist_mapping_settings(self):
        """Persist mapping settings from forms into the configuration.

        All form-to-config syncing is handled by ``_sync_config_from_ui``
        (called automatically inside ``_persist_config``), so this method
        simply triggers a config save.
        """
        self._persist_config()

    def _setup_step_wheels(self):
        """Wrap CNC step selectors with wheel-style pickers."""
        try:
            options = getattr(self.config.mapping, 'table_steps', None) or ["25", "50", "100", "250", "500", "1000", "10000"]
            self._syncing_xy_step = False
            self._syncing_z_step = False

            if hasattr(self, 'steps_xy_spinbox'):
                parent_layout = self.steps_xy_spinbox.parentWidget().layout() if self.steps_xy_spinbox.parentWidget() else None
                self.xy_step_wheel = WheelSelector(options, parent=self.steps_xy_spinbox.parentWidget(), visible_rows=3)
                self.xy_step_wheel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                self.xy_step_wheel.setToolTip("Scroll to pick the XY jog distance (µm).")
                if parent_layout is not None:
                    parent_layout.replaceWidget(self.steps_xy_spinbox, self.xy_step_wheel)
                self.xy_step_wheel.valueChanged.connect(self._handle_xy_wheel_change)
                self.steps_xy_spinbox.valueChanged.connect(self._handle_xy_spinbox_change)
                self.steps_xy_spinbox.hide()
                self.xy_step_wheel.setValue(str(self.steps_xy_spinbox.value()))

            if hasattr(self, 'steps_z_spinbox'):
                parent_layout = self.steps_z_spinbox.parentWidget().layout() if self.steps_z_spinbox.parentWidget() else None
                self.z_step_wheel = WheelSelector(options, parent=self.steps_z_spinbox.parentWidget(), visible_rows=3)
                self.z_step_wheel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                self.z_step_wheel.setToolTip("Scroll to pick the Z jog distance (µm).")
                if parent_layout is not None:
                    parent_layout.replaceWidget(self.steps_z_spinbox, self.z_step_wheel)
                self.z_step_wheel.valueChanged.connect(self._handle_z_wheel_change)
                self.steps_z_spinbox.valueChanged.connect(self._handle_z_spinbox_change)
                self.steps_z_spinbox.hide()
                self.z_step_wheel.setValue(str(self.steps_z_spinbox.value()))

        except Exception as e:
            self.logger.warning(f"Unable to setup CNC step wheels: {e}")

    def _setup_spectrometer_tab(self):
        """Initialize spectrometer selection tab."""
        try:
            if not hasattr(self, 'spectrometer_list_widget'):
                return
            self.spectrometer_list_widget.setUniformItemSizes(False)
            self.spectrometer_list_widget.setAlternatingRowColors(True)
            self.spectrometer_list_widget.setSpacing(6)
            # Ensure signals are connected only once
            if hasattr(self, 'detect_spectrometers_button'):
                try:
                    self.detect_spectrometers_button.clicked.disconnect()
                except Exception:
                    pass
                self.detect_spectrometers_button.clicked.connect(self._on_detect_spectrometers_clicked)
            try:
                self.spectrometer_list_widget.itemChanged.disconnect()
            except Exception:
                pass
            self.spectrometer_list_widget.itemChanged.connect(self._handle_spectrometer_item_changed)
            self._refresh_spectrometer_list(preserve_selection=False)
        except Exception as e:
            self.logger.warning(f"Failed to setup spectrometer tab: {e}")
    def _load_mapping_forms_from_config(self):
        """Populate mapping forms with current configuration values."""
        try:
            if not hasattr(self, 'mapping_basic_form'):
                return
            mapping_cfg = self.config.mapping
            serial_cfg = self.config.serial
            laser_cfg = self.config.laser

            basic_params = {
                'step_size': mapping_cfg.step_size,
                'mapping_X_size': mapping_cfg.x_size,
                'mapping_Y_size': mapping_cfg.y_size,
                'laser_frequency': mapping_cfg.laser_frequency,
                'integration_time_ms': mapping_cfg.integration_time_ms,
                'integration_delay_ns': mapping_cfg.integration_delay_ns,
                'line_start_wait_s': getattr(mapping_cfg, 'line_start_wait_s', 10),
                'averages': mapping_cfg.averages,
                'port_CNC': serial_cfg.cnc_port,
                'port_pulse_controller': serial_cfg.pulse_port,
                'save_dir': mapping_cfg.save_dir or self.config.folder_path or self.config.save_path,
                'save_stem': mapping_cfg.save_stem,
                'progress_every_lines': mapping_cfg.progress_every_lines,
                'progress_selection': mapping_cfg.progress_selection,
                'progress_tol_nm': mapping_cfg.progress_tol_nm,
            }
            self.mapping_basic_form.set_params(basic_params)

            advanced_params = {
                'baud_CNC': serial_cfg.cnc_baudrate,
                'steps_per_mm': mapping_cfg.steps_per_mm,
                'return_feed_mm_min': mapping_cfg.return_feed_mm_min,
                'baud_pulse': serial_cfg.pulse_baudrate,
                'laser_ip': laser_cfg.host,
                'laser_port': laser_cfg.port,
                'laser_login': laser_cfg.login_code,
                'LASER_QSON_SINGLE_SHOT': mapping_cfg.laser_single_shot_mode,
                'scan_line_error_margin': mapping_cfg.scan_line_error_margin,
                'VERBOSE': mapping_cfg.verbose,
                'LOG_EVERY_X': mapping_cfg.log_every_x,
                'ftp_host': mapping_cfg.ftp_host or None,
                'ftp_user': mapping_cfg.ftp_user or None,
                'ftp_pass': mapping_cfg.ftp_pass or None,
                'ftp_path': mapping_cfg.ftp_path or ".",
            }
            self.mapping_advanced_form.set_params(advanced_params)
        except Exception as exc:
            self.logger.warning(f"Unable to load mapping config into forms: {exc}")

    def _setup_settings_tab_sections(self):
        """Add mapping forms and connection controls into the Settings tab."""
        try:
            if not hasattr(self, 'tab_2') or self.tab_2 is None:
                return
            if getattr(self, 'settings_inner_tabs', None):
                return

            container_layout = QtWidgets.QVBoxLayout(self.tab_2)
            container_layout.setContentsMargins(8, 8, 8, 8)
            container_layout.setSpacing(8)

            self.settings_inner_tabs = QtWidgets.QTabWidget(self.tab_2)
            container_layout.addWidget(self.settings_inner_tabs)

            # Acquisition tab (existing controls)
            acquisition_tab = QtWidgets.QWidget()
            acquisition_layout = QtWidgets.QVBoxLayout(acquisition_tab)
            acquisition_layout.setSpacing(10)

            # Integration row
            integration_row = QtWidgets.QHBoxLayout()
            for widget in (self.label_integration, self.integration_time_combobox):
                widget.setParent(acquisition_tab)
                integration_row.addWidget(widget)
            integration_row.addStretch(1)
            acquisition_layout.addLayout(integration_row)

            # Trigger delay row
            trigger_row = QtWidgets.QHBoxLayout()
            for widget in (self.label_trigger, self.trigger_delay_combobox):
                widget.setParent(acquisition_tab)
                trigger_row.addWidget(widget)
            trigger_row.addStretch(1)
            acquisition_layout.addLayout(trigger_row)

            # Embed spectrometer controls from former tab
            container = getattr(self, 'spectrometer_tab_widget', None)
            if container is not None:
                container.setParent(acquisition_tab)
                container.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding,
                    QtWidgets.QSizePolicy.Expanding
                )
                acquisition_layout.addWidget(container, 1)
            else:
                fallback = QtWidgets.QLabel("Spectrometer controls are unavailable.")
                fallback.setAlignment(QtCore.Qt.AlignCenter)
                acquisition_layout.addWidget(fallback, 1)

            self.settings_inner_tabs.addTab(acquisition_tab, "One Shot / Multi Shots Acquisition")

            # Mapping basic settings (instantiated here, embedded in Mapping tab)
            self.mapping_basic_form = BasicParamsTab()

            # Mapping advanced tab
            self.mapping_advanced_form = AdvancedParamsTab()
            self.mapping_advanced_form.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
            mapping_adv_tab = QtWidgets.QWidget()
            mapping_adv_layout = QtWidgets.QVBoxLayout(mapping_adv_tab)
            mapping_adv_layout.setContentsMargins(8, 8, 8, 8)
            mapping_adv_layout.setSpacing(8)
            mapping_adv_layout.addWidget(self.mapping_advanced_form, 1)
            mapping_adv_layout.addWidget(QtWidgets.QLabel("Advanced CNC, pulse, and FTP settings for mapping."))
            self.settings_inner_tabs.addTab(mapping_adv_tab, "Mapping Advanced")

            # Connections tab
            self._load_mapping_forms_from_config()
            self.mapping_basic_form.save_dir.textChanged.connect(self._handle_mapping_save_dir_changed)
            self.mapping_basic_form.browse_btn.clicked.connect(self.choose_directory)
            self._refresh_mapping_connection_status()
            self._update_mapping_spectrometer_summary()
        except Exception as exc:
            self.logger.warning(f"Unable to setup mapping settings tabs: {exc}")

    def _refresh_mapping_connection_status(self):
        """Update connection status labels in the Connections tab."""
        try:
            status = getattr(self, 'mapping_cnc_status', None)
            if status is not None:
                if self.serial_handler.cnc_serial:
                    status.setText("Connected")
                    status.setStyleSheet("color: green;")
                else:
                    status.setText("Disconnected")
                    status.setStyleSheet("color: red;")

            pulse_status = getattr(self, 'mapping_pulse_status', None)
            if pulse_status is not None:
                if getattr(self.serial_handler, 'pulse_serial', None):
                    pulse_status.setText("Connected")
                    pulse_status.setStyleSheet("color: green;")
                else:
                    pulse_status.setText("Disconnected")
                    pulse_status.setStyleSheet("color: red;")

            if hasattr(self, 'mapping_laser_status'):
                if self.laser_handler.get_mapping_controller():
                    self.mapping_laser_status.setText("Connected (use Stop Laser to release)")
                    self.mapping_laser_status.setStyleSheet("color: green;")
                else:
                    self.mapping_laser_status.setText("Disconnected")
                    self.mapping_laser_status.setStyleSheet("color: red;")
            self._update_mapping_spectrometer_summary()
        except Exception as exc:
            self.logger.warning(f"Unable to refresh mapping connection statuses: {exc}")

    def _on_detect_spectrometers_clicked(self):
        """Handle 'Detect Spectrometers' button."""
        try:
            self.detect_spectrometers_button.setEnabled(False)
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            success = self.spectrometer_handler.initialize()
            QApplication.restoreOverrideCursor()
            self.detect_spectrometers_button.setEnabled(True)
            if not success:
                self._show_error_message("Spectrometer Detection", "No spectrometers detected. Please check connections.")
                return
            self._refresh_spectrometer_list(preserve_selection=False)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            if hasattr(self, 'detect_spectrometers_button'):
                self.detect_spectrometers_button.setEnabled(True)
            self.logger.error(f"Failed to detect spectrometers: {e}")
            self._show_error_message("Spectrometer Detection", f"Failed to detect spectrometers:\n{e}")

    def _refresh_spectrometer_list(self, preserve_selection: bool = True):
        """Populate spectrometer list widget with detected devices."""
        if not hasattr(self, 'spectrometer_list_widget'):
            return
        try:
            info = self.spectrometer_handler.get_device_info()
            previous = set(self._active_spectrometers) if preserve_selection and self._active_spectrometers else set()
            default_selection = []
            self._spectrometer_selection_updating = True
            self.spectrometer_list_widget.clear()
            metrics = self.spectrometer_list_widget.fontMetrics()
            if not info:
                placeholder = QtWidgets.QListWidgetItem("No spectrometers detected.")
                placeholder.setFlags(QtCore.Qt.NoItemFlags)
                self.spectrometer_list_widget.addItem(placeholder)
                self._active_spectrometers = []
                self._spectrometer_selection_updating = False
                return
            if previous:
                default_selection = [spec['index'] for spec in info if spec.get('index') in previous]
            if not default_selection:
                default_selection = [spec['index'] for spec in info]
            for spec in info:
                label = spec.get('label') or "Spectrometer"
                serial = spec.get('serial') or "Unknown"
                range_text = spec.get('range_text') or "Range unknown"
                pixels = spec.get('pixels')
                friendly = spec.get('friendly_name')
                idx = spec.get('index')
                fpga_version = spec.get('fpga_version')
                firmware_version = spec.get('firmware_version')
                dll_version = spec.get('dll_version')
                handle_val = None
                handle = spec.get('handle')
                if handle is not None:
                    try:
                        handle_val = int(handle)
                    except Exception:
                        handle_val = getattr(handle, 'value', None)
                lines = [label]
                if friendly and friendly != label:
                    lines.append(f"Friendly name: {friendly}")

                range_line = f"Range: {range_text}"
                detail_bits = []
                if pixels:
                    detail_bits.append(f"{int(pixels)} px")
                if handle_val is not None:
                    detail_bits.append(f"Handle {handle_val}")
                if detail_bits:
                    range_line += " • " + " • ".join(detail_bits)
                lines.append(range_line)
                lines.append(f"Channel index: {idx}")
                display_text = "\n".join(lines)
                item = QtWidgets.QListWidgetItem(display_text)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                item.setData(QtCore.Qt.UserRole, idx)
                item.setData(QtCore.Qt.UserRole + 1, spec)
                item.setToolTip("\n".join(lines))
                height = metrics.lineSpacing() * len(lines) + 8
                item.setSizeHint(QtCore.QSize(0, height))
                state = QtCore.Qt.Checked if idx in default_selection else QtCore.Qt.Unchecked
                item.setCheckState(state)
                self.spectrometer_list_widget.addItem(item)
            self._spectrometer_selection_updating = False
            self._apply_selected_spectrometers(default_selection, force=True)
            self._update_mapping_spectrometer_summary()
        except Exception as e:
            self._spectrometer_selection_updating = False
            self.logger.error(f"Failed to refresh spectrometer list: {e}")

    def _build_spectrometer_selection_summary(self) -> str:
        """Return human-readable summary of selected spectrometers."""
        try:
            if not self._active_spectrometers:
                return "No spectrometers selected."
            info = self.spectrometer_handler.get_device_info()
            info_map = {spec.get('index'): spec for spec in info if isinstance(spec, dict)}
            labels = []
            for idx in self._active_spectrometers:
                spec = info_map.get(idx, {})
                label = spec.get('label') or f"Spectrometer {idx+1}"
                range_text = spec.get('range_text')
                if range_text:
                    labels.append(f"{label} ({range_text})")
                else:
                    labels.append(label)
            if not labels:
                return "No spectrometers selected."
            return f"Selected {len(labels)} spectrometer(s): " + ", ".join(labels)
        except Exception as exc:
            self.logger.warning(f"Unable to summarize spectrometer selection: {exc}")
            return "Unable to determine spectrometer selection."

    def _update_mapping_spectrometer_summary(self):
        """Update summary label in the mapping connections tab."""
        if not hasattr(self, 'mapping_spectrometer_summary'):
            return
        try:
            summary = self._build_spectrometer_selection_summary()
        except Exception:
            summary = "Unable to determine spectrometer selection."
        self.mapping_spectrometer_summary.setText(summary)

    def _gather_selected_spectrometers(self) -> list:
        """Return list of currently checked spectrometer indices."""
        indices = []
        if not hasattr(self, 'spectrometer_list_widget'):
            return indices
        for i in range(self.spectrometer_list_widget.count()):
            item = self.spectrometer_list_widget.item(i)
            if item is None:
                continue
            if not (item.flags() & QtCore.Qt.ItemIsUserCheckable):
                continue
            if item.checkState() == QtCore.Qt.Checked:
                idx = item.data(QtCore.Qt.UserRole)
                if idx is not None:
                    indices.append(int(idx))
        return indices

    def _handle_spectrometer_item_changed(self, item: QtWidgets.QListWidgetItem):
        """Sync handler selection when user toggles a spectrometer."""
        if getattr(self, '_spectrometer_selection_updating', False):
            return
        selected = self._gather_selected_spectrometers()
        if not selected:
            self._spectrometer_selection_updating = True
            item.setCheckState(QtCore.Qt.Checked)
            self._spectrometer_selection_updating = False
            self._show_warning_message("Spectrometers", "At least one spectrometer must remain selected.")
            return
        self._apply_selected_spectrometers(selected)

    def _apply_selected_spectrometers(self, indices: list, force: bool = False):
        """Apply user selection to spectrometer handler."""
        try:
            if not indices:
                return
            normalized = []
            for i in indices:
                try:
                    val = int(i)
                except (TypeError, ValueError):
                    continue
                if val not in normalized:
                    normalized.append(val)
            if not force and normalized == self._active_spectrometers:
                return
            success = self.spectrometer_handler.set_active_devices(normalized)
            if success:
                self._active_spectrometers = normalized
                self._update_mapping_spectrometer_summary()
                self._update_mapping_estimates()
            else:
                self.logger.warning("Failed to apply spectrometer selection")
        except Exception as e:
            self.logger.error(f"Failed to apply spectrometer selection: {e}")

    def _gather_active_spectrometer_handles(self) -> List[int]:
        """Return a list of spectrometer handles for active devices."""
        handles = []
        try:
            all_handles = getattr(self.spectrometer_handler, 'handles', [])
            for idx in self._active_spectrometers:
                if 0 <= idx < len(all_handles):
                    handles.append(all_handles[idx])
        except Exception as exc:
            self.logger.warning(f"Unable to gather spectrometer handles: {exc}")
        return handles

    def _collect_spectrometer_data_for_mapping(self):
        """Collect handle and wavelength information for the mapping thread."""
        handles = []
        wavelengths = {}
        identities = []
        try:
            all_handles = getattr(self.spectrometer_handler, 'handles', [])
            lambdas = getattr(self.spectrometer_handler, 'wavelengths_per_device', [])
            info = getattr(self.spectrometer_handler, 'device_info', {})
            active_indices = list(self._active_spectrometers) if self._active_spectrometers else []
            if not active_indices:
                try:
                    active_indices = self.spectrometer_handler.get_active_device_indices()
                except Exception:
                    active_indices = []
            if not active_indices:
                active_indices = list(range(len(all_handles)))
            for idx in active_indices:
                if idx >= len(all_handles):
                    continue
                handle = all_handles[idx]
                handles.append(handle)
                if idx < len(lambdas):
                    wavelengths[handle] = np.asarray(lambdas[idx], dtype=np.float64)
                else:
                    wavelengths[handle] = np.array([], dtype=np.float64)
                meta = info.get(idx, {})
                identities.append({
                    "serial": meta.get("serial", ""),
                    "name": meta.get("label", meta.get("friendly_name", "")),
                })
        except Exception as exc:
            self.logger.warning(f"Unable to pull spectrometer metadata: {exc}")
        return handles, wavelengths, identities

    def _ensure_spectrometer_selection(self) -> bool:
        """Verify that at least one spectrometer is available and selected."""
        if self._active_spectrometers:
            return True
        try:
            handler_active = self.spectrometer_handler.get_active_device_indices()
            if handler_active:
                self._active_spectrometers = handler_active
                return True
        except Exception:
            pass
        self._show_warning_message(
            "Spectrometers",
            "No spectrometers are selected. Please select at least one device in the Spectrometers tab."
        )
        return False

    def _setup_mapping_estimate_signals(self):
        """Wire mapping form inputs to update estimates."""
        try:
            if hasattr(self, 'mapping_basic_form') and self.mapping_basic_form:
                for line_edit in self.mapping_basic_form.findChildren(QtWidgets.QLineEdit):
                    line_edit.textChanged.connect(self._update_mapping_estimates)
                for combo in self.mapping_basic_form.findChildren(QtWidgets.QComboBox):
                    combo.currentTextChanged.connect(self._update_mapping_estimates)
            if hasattr(self, 'mapping_advanced_form') and self.mapping_advanced_form:
                for line_edit in self.mapping_advanced_form.findChildren(QtWidgets.QLineEdit):
                    line_edit.textChanged.connect(self._update_mapping_estimates)
                for combo in self.mapping_advanced_form.findChildren(QtWidgets.QComboBox):
                    combo.currentTextChanged.connect(self._update_mapping_estimates)
        except Exception as exc:
            self.logger.warning(f"Failed to wire mapping estimate signals: {exc}")

    def _estimate_mapping_channels(self):
        """Estimate total channels and device count from active spectrometers."""
        try:
            handles = list(getattr(self.spectrometer_handler, 'handles', []) or [])
            pixels = list(getattr(self.spectrometer_handler, 'num_pixels_per_device', []) or [])
        except Exception:
            return 0, 0

        if not handles or not pixels:
            return 0, 0

        indices = list(self._active_spectrometers) if self._active_spectrometers else []
        if not indices:
            try:
                indices = list(self.spectrometer_handler.get_active_device_indices())
            except Exception:
                indices = list(range(len(handles)))

        total_channels = 0
        num_devices = 0
        for idx in indices:
            if 0 <= idx < len(pixels):
                total_channels += int(pixels[idx])
                num_devices += 1
        return total_channels, num_devices

    def _update_mapping_estimates(self):
        """Update duration and file size estimates from current mapping settings."""
        if not hasattr(self, 'mapping_estimate_duration') or not hasattr(self, 'mapping_estimate_filesize'):
            return
        try:
            if not hasattr(self, 'mapping_basic_form') or self.mapping_basic_form is None:
                return
            basic = self.mapping_basic_form.get_params()
            advanced = self.mapping_advanced_form.get_params() if hasattr(self, 'mapping_advanced_form') else {}

            step_size = float(basic.get('step_size') or 0)
            mapping_X_size = float(basic.get('mapping_X_size') or 0)
            mapping_Y_size = float(basic.get('mapping_Y_size') or 0)
            laser_frequency = float(basic.get('laser_frequency') or 0)
            return_feed_mm_min = float(advanced.get('return_feed_mm_min') or 0)

            if step_size <= 0 or mapping_X_size <= 0 or mapping_Y_size <= 0 or laser_frequency <= 0:
                self.mapping_estimate_duration.setText("Estimated duration: --")
                self.mapping_estimate_filesize.setText("Estimated file size: --")
                return

            mapping_X_points = int(mapping_X_size / step_size)
            mapping_Y_points = int(mapping_Y_size / step_size)
            if mapping_X_points <= 0 or mapping_Y_points <= 0:
                self.mapping_estimate_duration.setText("Estimated duration: --")
                self.mapping_estimate_filesize.setText("Estimated file size: --")
                return

            x_travel_mm = mapping_X_size + (step_size * 10.0)
            x_speed_mm_s = step_size * laser_frequency
            if x_speed_mm_s <= 0:
                self.mapping_estimate_duration.setText("Estimated duration: --")
                self.mapping_estimate_filesize.setText("Estimated file size: --")
                return
            x_feed_mm_min = x_speed_mm_s * 60.0

            try:
                from mapping_engine import estimate_mapping_duration, estimate_file_size
            except Exception as exc:
                self.mapping_estimate_duration.setText("Estimated duration: --")
                self.mapping_estimate_filesize.setText("Estimated file size: --")
                self.logger.debug(f"Mapping estimate import failed: {exc}")
                return

            est = estimate_mapping_duration(
                mapping_X_points, mapping_Y_points, x_travel_mm, step_size,
                x_feed_mm_min, return_feed_mm_min=return_feed_mm_min,
                y_feed_mm_min=return_feed_mm_min
            )
            self.mapping_estimate_duration.setText(
                f"Estimated duration: {est['hours']:d}h {est['minutes']:d}m {est['seconds']:d}s"
            )

            total_channels, num_devices = self._estimate_mapping_channels()
            if total_channels <= 0 or num_devices <= 0:
                self.mapping_estimate_filesize.setText("Estimated file size: --")
                return

            file_est = estimate_file_size(
                total_channels, mapping_X_points, mapping_Y_points, num_devices=num_devices
            )
            compression_reduction = (1.0 - file_est['compression_ratio']) * 100
            if file_est['estimated_mb'] < 1:
                size_text = f"{file_est['estimated_bytes']/1024:.2f} KB"
            elif file_est['estimated_mb'] < 1024:
                size_text = f"{file_est['estimated_mb']:.2f} MB"
            else:
                size_text = f"{file_est['estimated_gb']:.2f} GB"
            self.mapping_estimate_filesize.setText(
                f"Estimated file size: {size_text} (compressed, ~{compression_reduction:.0f}% reduction)"
            )
        except Exception as exc:
            self.logger.debug(f"Mapping estimate update failed: {exc}")

    def _handle_xy_wheel_change(self, value: str):
        if getattr(self, '_syncing_xy_step', False):
            return
        try:
            self._syncing_xy_step = True
            if hasattr(self, 'steps_xy_spinbox'):
                self.steps_xy_spinbox.setValue(int(value))
        except Exception:
            pass
        finally:
            self._syncing_xy_step = False

    def _handle_xy_spinbox_change(self, value: int):
        if getattr(self, '_syncing_xy_step', False):
            return
        try:
            self._syncing_xy_step = True
            if hasattr(self, 'xy_step_wheel'):
                self.xy_step_wheel.setValue(str(value))
        finally:
            self._syncing_xy_step = False

    def _handle_z_wheel_change(self, value: str):
        if getattr(self, '_syncing_z_step', False):
            return
        try:
            self._syncing_z_step = True
            if hasattr(self, 'steps_z_spinbox'):
                self.steps_z_spinbox.setValue(int(value))
        except Exception:
            pass
        finally:
            self._syncing_z_step = False

    def _handle_z_spinbox_change(self, value: int):
        if getattr(self, '_syncing_z_step', False):
            return
        try:
            self._syncing_z_step = True
            if hasattr(self, 'z_step_wheel'):
                self.z_step_wheel.setValue(str(value))
        finally:
            self._syncing_z_step = False

    def _handle_frame_update(self, image: QtGui.QImage):
        """Store latest camera frame and forward it to the display widget."""
        try:
            if image is None:
                return
            # store a copy so downstream modifications don't mutate saved frame
            self._latest_frame = image.copy()
            self.disp.setImage(self._latest_frame)
        except Exception as e:
            self.logger.warning(f"Failed to handle frame update: {e}")
    
    def _setup_connections(self):
        """Setup signal connections"""
        try:
            # Text output
            self.text_update.connect(self.append_text)
            sys.stdout = self
            
            # Button connections
            self.go_button.clicked.connect(self.check_file_exist)
            self.profile_go_button.clicked.connect(self.check_file_exist_profile)
            self.Connect_laser_button.clicked.connect(self.laser_connect)
            self.laser_stop_button.clicked.connect(self.laser_stop)
            if hasattr(self, 'acq_browse_btn'):
                self.acq_browse_btn.clicked.connect(self.choose_directory)
            if hasattr(self, 'acq_save_dir'):
                self.acq_save_dir.textChanged.connect(self._handle_acq_save_dir_changed)
            self.save_micro_capture.clicked.connect(self.save_micro_image)
            if hasattr(self, 'reset_cnc_button') and self.reset_cnc_button:
                self.reset_cnc_button.clicked.connect(self.reset_cnc_after_stop)
            
            # Microscope light control
            self.micro_light_checkbox.stateChanged.connect(self.change_micro_light)
            
            # Graph options
            self.show_grid_option.stateChanged.connect(self.change_graph)
            # One Shot detect button from UI (if present)
            if hasattr(self, 'one_shot_detect_btn') and isinstance(self.one_shot_detect_btn, QtWidgets.QPushButton):
                try:
                    self.one_shot_detect_btn.clicked.connect(self.one_shot_detect_peaks)
                except Exception:
                    pass
            
            # Movement controls
            self.stop_button.clicked.connect(self.stop)
            
            # Directional movement
            speed = 125
            self.north_button.clicked.connect(
                lambda: self.move_north(str(self.steps_xy_spinbox.value()), speed)
            )
            self.east_button.clicked.connect(
                lambda: self.move_east(str(self.steps_xy_spinbox.value()), speed)
            )
            self.west_button.clicked.connect(
                lambda: self.move_west(str(self.steps_xy_spinbox.value()), speed)
            )
            self.south_button.clicked.connect(
                lambda: self.move_south(str(self.steps_xy_spinbox.value()), speed)
            )
            self.up_button.clicked.connect(
                lambda: self.move_up(str(self.steps_z_spinbox.value()), speed)
            )
            self.down_button.clicked.connect(
                lambda: self.move_down(str(self.steps_z_spinbox.value()), speed)
            )
            
            # Auto-save config when step sizes change
            if hasattr(self, 'steps_xy_spinbox'):
                self.steps_xy_spinbox.valueChanged.connect(self._persist_config)
            if hasattr(self, 'steps_z_spinbox'):
                self.steps_z_spinbox.valueChanged.connect(self._persist_config)
            
        except Exception as e:
            self.logger.error(f"Failed to setup connections: {e}")
    
    def _initialize_devices(self):
        """Initialize device connections"""
        try:
            # Initialize serial connections
            if self.serial_handler.connect_arduino():
                self.logger.info("Arduino connected")
                # Turn on microscope LED at startup (delayed to allow Arduino
                # bootloader to finish after the DTR-triggered reset)
                def _turn_on_led_after_boot():
                    try:
                        self.serial_handler.send_arduino_command("ON\r")
                        if hasattr(self, 'micro_light_checkbox'):
                            self.micro_light_checkbox.blockSignals(True)
                            self.micro_light_checkbox.setChecked(True)
                            self.micro_light_checkbox.blockSignals(False)
                        self.logger.info("Microscope LED turned on at startup")
                    except Exception:
                        pass
                QtCore.QTimer.singleShot(2000, _turn_on_led_after_boot)
            else:
                self.logger.warning("Arduino connection failed")
            
            if self.serial_handler.connect_cnc():
                self.logger.info("CNC connected")
                # Skip homing by default and enable controls
                try:
                    self.serial_handler.send_cnc_command("$X \n")
                    # Use QTimer to delay enabling controls slightly to ensure CNC command is processed
                    QtCore.QTimer.singleShot(500, lambda: self._enable_cnc_controls())
                    self.logger.info("Homing skipped by default - controls will be enabled")
                except Exception as e:
                    self.logger.warning(f"Failed to skip homing: {e}")
            else:
                self.logger.warning("CNC connection failed")
            if self.serial_handler.connect_pulse_controller():
                self.logger.info("Pulse controller connected")
            else:
                self.logger.warning("Pulse controller connection failed")
            
            # Initialize Basler camera (non-fatal if unavailable)
            self._start_camera()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize devices: {e}")
        finally:
            self._refresh_mapping_connection_status()

    def _sync_config_from_ui(self):
        """Update config dataclasses with the latest UI selections."""
        if not hasattr(self, 'config'):
            return
        try:
            # Sync one-shot acquisition output settings
            if hasattr(self, 'acq_save_dir'):
                val = self.acq_save_dir.text().strip()
                if val:
                    normalized = os.path.normpath(val)
                    self.config.folder_path = normalized if normalized.endswith(os.sep) else normalized + os.sep
            if hasattr(self, 'one_shot_filename_box'):
                val = self.one_shot_filename_box.text().strip()
                if val:
                    self.config.save_path = val

            spec_cfg = getattr(self.config, 'spectrometer', None)
            if spec_cfg:
                if hasattr(self, 'integration_time_combobox'):
                    current = self.integration_time_combobox.currentText()
                    if current:
                        spec_cfg.selected_integration_time = current
                if hasattr(self, 'trigger_delay_combobox'):
                    current = self.trigger_delay_combobox.currentText()
                    if current:
                        spec_cfg.selected_trigger_delay = current
                if hasattr(self, 'profile_steps_combobox'):
                    current = self.profile_steps_combobox.currentText()
                    if current:
                        spec_cfg.selected_profile_steps = current
                if hasattr(self, 'shots_delay_combobox'):
                    current = self.shots_delay_combobox.currentText()
                    if current:
                        spec_cfg.selected_shots_delay = current

            mapping_cfg = getattr(self.config, 'mapping', None)
            if mapping_cfg:
                if hasattr(self, 'steps_xy_spinbox'):
                    mapping_cfg.selected_xy_step = str(self.steps_xy_spinbox.value())
                if hasattr(self, 'steps_z_spinbox'):
                    mapping_cfg.selected_z_step = str(self.steps_z_spinbox.value())

                # Sync all mapping form values into config
                if hasattr(self, 'mapping_basic_form'):
                    basic = self.mapping_basic_form.get_params()
                    mapping_cfg.step_size = float(basic.get('step_size', mapping_cfg.step_size))
                    mapping_cfg.x_size = float(basic.get('mapping_X_size', mapping_cfg.x_size))
                    mapping_cfg.y_size = float(basic.get('mapping_Y_size', mapping_cfg.y_size))
                    mapping_cfg.laser_frequency = float(basic.get('laser_frequency', mapping_cfg.laser_frequency))
                    mapping_cfg.integration_time_ms = float(basic.get('integration_time_ms', mapping_cfg.integration_time_ms))
                    mapping_cfg.integration_delay_ns = int(basic.get('integration_delay_ns', mapping_cfg.integration_delay_ns))
                    mapping_cfg.line_start_wait_s = float(basic.get('line_start_wait_s', getattr(mapping_cfg, 'line_start_wait_s', 10)))
                    mapping_cfg.averages = int(basic.get('averages', mapping_cfg.averages))
                    mapping_cfg.save_dir = basic.get('save_dir') or mapping_cfg.save_dir
                    mapping_cfg.save_stem = basic.get('save_stem', mapping_cfg.save_stem)
                    mapping_cfg.progress_every_lines = max(1, int(basic.get('progress_every_lines', mapping_cfg.progress_every_lines or 1)))
                    mapping_cfg.progress_selection = basic.get('progress_selection', mapping_cfg.progress_selection)
                    mapping_cfg.progress_tol_nm = float(basic.get('progress_tol_nm', mapping_cfg.progress_tol_nm))

                if hasattr(self, 'mapping_advanced_form'):
                    advanced = self.mapping_advanced_form.get_params()
                    mapping_cfg.steps_per_mm = int(advanced.get('steps_per_mm', mapping_cfg.steps_per_mm))
                    mapping_cfg.return_feed_mm_min = float(advanced.get('return_feed_mm_min', mapping_cfg.return_feed_mm_min))
                    mapping_cfg.scan_line_error_margin = float(advanced.get('scan_line_error_margin', mapping_cfg.scan_line_error_margin))
                    mapping_cfg.verbose = bool(advanced.get('VERBOSE', mapping_cfg.verbose))
                    mapping_cfg.log_every_x = int(advanced.get('LOG_EVERY_X', mapping_cfg.log_every_x))
                    mapping_cfg.laser_single_shot_mode = bool(advanced.get('LASER_QSON_SINGLE_SHOT', mapping_cfg.laser_single_shot_mode))
                    mapping_cfg.ftp_host = advanced.get('ftp_host') or ""
                    mapping_cfg.ftp_user = advanced.get('ftp_user') or ""
                    mapping_cfg.ftp_pass = advanced.get('ftp_pass') or ""
                    mapping_cfg.ftp_path = advanced.get('ftp_path') or "."
                    # Sync serial and laser config from advanced form
                    self.config.serial.cnc_baudrate = int(advanced.get('baud_CNC', self.config.serial.cnc_baudrate))
                    self.config.serial.pulse_baudrate = int(advanced.get('baud_pulse', self.config.serial.pulse_baudrate))
                    self.config.laser.host = advanced.get('laser_ip', self.config.laser.host)
                    self.config.laser.port = int(advanced.get('laser_port', self.config.laser.port))
                    self.config.laser.login_code = advanced.get('laser_login', self.config.laser.login_code)
        except Exception as e:
            self.logger.warning(f"Unable to synchronize UI settings to config: {e}")

    def _persist_config(self):
        """Persist current configuration to disk."""
        try:
            self._sync_config_from_ui()
            if hasattr(self, 'config_path'):
                self.config.save_to_file(str(self.config_path))
            else:
                self.config.save_to_file("config.json")
        except Exception as e:
            self.logger.warning(f"Unable to save configuration: {e}")

    def _start_camera(self):
        """Start Basler camera acquisition using pypylon and emit frames to UI."""
        try:
            if pylon is None:
                self.logger.warning("pypylon not available; camera view disabled")
                self._camera_running = False
                return
            
            # Instantiate camera
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if len(devices) == 0:
                self.logger.warning("No Basler camera detected; camera view disabled")
                self._camera_running = False
                return
            
            self.cam_micro = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self.cam_micro.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
            # Converter to BGR8 (OpenCV-like)
            self._pylon_converter = pylon.ImageFormatConverter()
            self._pylon_converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._pylon_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            
            self._camera_running = True
            self._camera_thread = threading.Thread(target=self._camera_loop, name="CameraThread", daemon=True)
            self._camera_thread.start()
            self.logger.info("Basler camera started")
        except Exception as e:
            self._camera_running = False
            try:
                if hasattr(self, 'cam_micro') and self.cam_micro:
                    self.cam_micro.StopGrabbing()
            except Exception:
                pass
            self.logger.warning(f"Camera not started: {e}")

    def _camera_loop(self):
        """Background camera grab loop emitting QImage frames."""
        disp_interval = max(1, getattr(self.config.camera, 'disp_msec', 5)) / 1000.0
        consecutive_errors = 0
        max_consecutive_errors = 50  # restart camera after this many failures

        while getattr(self, '_camera_running', False):
            try:
                if not (hasattr(self, 'cam_micro') and self.cam_micro and self.cam_micro.IsGrabbing()):
                    time.sleep(0.1)
                    continue

                # Use Return handling so a timeout never throws
                grab_result = self.cam_micro.RetrieveResult(2000, pylon.TimeoutHandling_Return)
                if grab_result is None:
                    time.sleep(disp_interval)
                    continue

                try:
                    if not grab_result.GrabSucceeded():
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            self.logger.warning("Camera: too many consecutive bad frames, restarting grab")
                            try:
                                self.cam_micro.StopGrabbing()
                                time.sleep(0.2)
                                self.cam_micro.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                            except Exception:
                                pass
                            consecutive_errors = 0
                        continue

                    converted = self._pylon_converter.Convert(grab_result)
                    img = converted.GetArray()
                    if img is None or len(img.shape) < 2:
                        consecutive_errors += 1
                        continue
                finally:
                    # Always release immediately to free the buffer
                    try:
                        grab_result.Release()
                    except Exception:
                        pass

                # -- Frame processing (grab_result already released) --
                try:
                    # Crop
                    h, w = img.shape[:2]
                    t, b, l, r = 100, 980, 300, 1620
                    t, b = max(0, min(t, h)), max(0, min(b, h))
                    l, r = max(0, min(l, w)), max(0, min(r, w))
                    if b > t and r > l:
                        img = img[t:b, l:r]

                    # Scale based on config (disp_scale=2 -> 50%)
                    scale = 1.0 / max(1.0, float(getattr(self.config.camera, 'disp_scale', 1)))
                    if scale != 1.0:
                        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                    # Flip vertically (Y axis) for correct orientation
                    img = cv2.flip(img, 0)

                    height, width = img.shape[0], img.shape[1]
                    bytes_per_line = img.strides[0]
                    qimage = (QtGui.QImage(img.data, width, height, bytes_per_line,
                                           QtGui.QImage.Format_RGB888)
                              .rgbSwapped().copy())
                    self.frame_update.emit(qimage)
                    consecutive_errors = 0  # good frame resets counter
                except Exception:
                    # Bad frame during processing -- skip it silently
                    consecutive_errors += 1

                time.sleep(disp_interval)

            except Exception:
                # Unexpected error -- short pause and retry
                consecutive_errors += 1
                time.sleep(0.05)

    def _stop_camera(self):
        """Stop camera and release resources.

        Waits long enough for the camera loop to finish its current
        ``RetrieveResult`` call (up to 3 s) so that ``StopGrabbing`` /
        ``Close`` never race with the background thread.
        """
        try:
            self._camera_running = False
            # Stop grabbing first so RetrieveResult returns immediately
            if hasattr(self, 'cam_micro') and self.cam_micro:
                try:
                    if self.cam_micro.IsGrabbing():
                        self.cam_micro.StopGrabbing()
                except Exception:
                    pass
            # Now wait for the thread to notice and exit
            if hasattr(self, '_camera_thread') and self._camera_thread:
                try:
                    self._camera_thread.join(timeout=3.0)
                except Exception:
                    pass
                self._camera_thread = None
            # Close device after the thread has finished
            if hasattr(self, 'cam_micro') and self.cam_micro:
                try:
                    self.cam_micro.Close()
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(f"Error stopping camera: {e}")

    def _stop_laser_after_mapping(self):
        """Stop laser output after mapping completes without disconnecting."""
        try:
            if hasattr(self, 'laser_handler') and self.laser_handler:
                self.laser_handler.stop_only()
        except Exception as exc:
            self.logger.warning(f"Failed to stop laser after mapping: {exc}")

    def _pause_camera_for_mapping(self):
        """Stop camera capture while mapping is running."""
        self._camera_was_running_before_mapping = bool(getattr(self, '_camera_running', False))
        if hasattr(self, 'disp') and self.disp:
            self.disp.setPaused(True)
        if self._camera_was_running_before_mapping:
            self.logger.info("Stopping camera for mapping run")
            self._stop_camera()

    def _resume_camera_after_mapping(self):
        """Restart camera capture if it was running before mapping."""
        if hasattr(self, 'disp') and self.disp:
            self.disp.setPaused(False)
        if getattr(self, '_camera_was_running_before_mapping', False):
            self._camera_was_running_before_mapping = False
            self.logger.info("Restarting camera after mapping run")
            self._start_camera()
    
    def _pause_light_for_mapping(self):
        """Turn off microscope light during mapping, if it was on."""
        checkbox = getattr(self, 'micro_light_checkbox', None)
        self._light_was_on_before_mapping = bool(checkbox and checkbox.isChecked())
        if self._light_was_on_before_mapping:
            try:
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
            finally:
                checkbox.blockSignals(False)
            self.change_micro_light()

    def _resume_light_after_mapping(self):
        """Restore microscope light after mapping if it was previously on."""
        if not getattr(self, '_light_was_on_before_mapping', False):
            return
        checkbox = getattr(self, 'micro_light_checkbox', None)
        if not checkbox:
            self._light_was_on_before_mapping = False
            return
        try:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
        finally:
            checkbox.blockSignals(False)
        self.change_micro_light()
        self._light_was_on_before_mapping = False
    
    def change_micro_light(self):
        """Control microscope light"""
        try:
            if self.micro_light_checkbox.isChecked():
                self.serial_handler.send_arduino_command("ON\r")
                self.logger.info("Microscope light ON")
            else:
                self.serial_handler.send_arduino_command("OFF\r")
                self.logger.info("Microscope light OFF")
        except Exception as e:
            self.logger.error(f"Failed to control microscope light: {e}")
    
    def check_file_exist(self):
        """Check if output file exists before acquisition"""
        try:
            filename = self.one_shot_filename_box.text()
            if not filename:
                self._show_warning_message("No filename", "Please enter a filename")
                return
            
            full_path = self.config.folder_path + filename + ".csv"
            
            if os.path.exists(full_path):
                reply = QMessageBox.question(
                    self, 'Warning', 
                    'This file already exists, overwrite?',
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.acquire_spectrum()
            else:
                self.acquire_spectrum()
                
        except Exception as e:
            self.logger.error(f"Failed to check file existence: {e}")
    
    def check_file_exist_profile(self):
        """Check if profile file exists before acquisition"""
        try:
            filename = self.one_shot_filename_box.text()
            if not filename:
                self._show_warning_message("No filename", "Please enter a filename")
                return
            
            full_path = self.config.folder_path + filename + "_multishots.csv"
            
            if os.path.exists(full_path):
                reply = QMessageBox.question(
                    self, 'Warning', 
                    'This multishots file already exists, overwrite?',
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.acquire_spectrum_profile()
            else:
                self.acquire_spectrum_profile()
                
        except Exception as e:
            self.logger.error(f"Failed to check profile file existence: {e}")
    
    def acquire_spectrum(self):
        """Acquire single spectrum"""
        if not self._ensure_spectrometer_selection():
            return
        # Keep microscope light off during the whole acquisition
        prev_light_on = self.micro_light_checkbox.isChecked()
        if prev_light_on:
            self.serial_handler.send_arduino_command("OFF\r")
        try:
            integration_time = int(self.integration_time_combobox.currentText())
            trigger_delay = int(self.trigger_delay_combobox.currentText())
            filename = self.one_shot_filename_box.text()
            
            self.logger.info(f"Acquiring spectrum: {filename}")
            self.logger.info(f"Integration time: {integration_time}μs")
            self.logger.info(f"Trigger delay: {trigger_delay}ns")
            
            # Prepare spectrometer for dark measurement (no external trigger)
            if not self.spectrometer_handler.prepare_measurement(
                integration_time, trigger_delay, triggered=False
            ):
                self._show_error_message("Spectrometer Error", "Failed to prepare spectrometer")
                return
            
            # Arm and read dark without firing
            self.spectrometer_handler.start_measurement()
            dark_scopes = self.spectrometer_handler.read_measurement()
            if dark_scopes is None:
                self._show_error_message("Acquisition Error", "Failed to acquire dark spectrum")
                return
            # Process dark data into DataFrame with expected 'dark' column
            dark_df, _ = self.spectrometer_handler.process_spectrum_data(dark_scopes)
            self.dark_data = pd.DataFrame({
                'wavelength_(nm)': dark_df['wavelength_(nm)'],
                'dark': dark_df['intensity_dark_corrected']
            })
            
            # Prepare for actual measurement (external trigger)
            if not self.spectrometer_handler.prepare_measurement(
                integration_time, trigger_delay, triggered=True
            ):
                self._show_error_message("Spectrometer Error", "Failed to prepare spectrometer")
                return
            
            # Arm spectrometers, then fire a single laser shot
            if not self.spectrometer_handler.start_measurement():
                self._show_error_message("Acquisition Error", "Failed to arm spectrometers")
                return
            if not getattr(self.config.laser, 'use_external', False):
                if not self.laser_handler.fire():
                    self._show_error_message("Laser Error", "Failed to fire laser")
                    return
            else:
                self.logger.info("External laser mode: waiting for external trigger")
            
            # Read spectrum after trigger
            read_timeout = getattr(self.config.laser, 'external_trigger_timeout_s', 5.0) if getattr(self.config.laser, 'use_external', False) else 3.0
            scopes = self.spectrometer_handler.read_measurement(timeout_s=read_timeout)
            if scopes is None:
                self._show_error_message("Acquisition Error", "Failed to acquire spectrum")
                return
            
            # Process spectrum data and subtract dark
            measured_df, _ = self.spectrometer_handler.process_spectrum_data(scopes)
            self.spectro1_data = self.spectrum_processor.dark_correction(measured_df, self.dark_data)
            
            # Save spectrum
            output_filename = self.spectrum_processor.save_spectrum(
                self.spectro1_data, filename + ".csv", self.config.folder_path
            )
            
            if output_filename:
                self.logger.info(f"Spectrum saved: {output_filename}")
                self.trace_graph(self.spectro1_data)
            else:
                self._show_error_message("Save Error", "Failed to save spectrum")
                
        except Exception as e:
            self.logger.error(f"Failed to acquire spectrum: {e}")
            self._show_error_message("Acquisition Error", f"Failed to acquire spectrum: {e}")
        finally:
            # Restore microscope light state after the whole acquisition
            if prev_light_on:
                self.serial_handler.send_arduino_command("ON\r")
    
    def acquire_spectrum_profile(self):
        """Acquire spectrum profile (multiple shots)"""
        if not self._ensure_spectrometer_selection():
            return
        # Keep microscope light off during the whole multishot acquisition
        prev_light_on = self.micro_light_checkbox.isChecked()
        if prev_light_on:
            self.serial_handler.send_arduino_command("OFF\r")
        try:
            integration_time = int(self.integration_time_combobox.currentText())
            trigger_delay = int(self.trigger_delay_combobox.currentText())
            profile_steps = int(self.profile_steps_combobox.currentText())
            delay_between_shots = int(self.shots_delay_combobox.currentText())
            filename = self.one_shot_filename_box.text()
            
            self.logger.info(f"Acquiring profile: {filename}, {profile_steps} shots")
            
            # Prepare for dark measurement
            if not self.spectrometer_handler.prepare_measurement(
                integration_time, trigger_delay, triggered=False
            ):
                return
            
            # Acquire dark spectrum (arm + read, no fire)
            self.spectrometer_handler.start_measurement()
            dark_scopes = self.spectrometer_handler.read_measurement()
            if dark_scopes is None:
                return
            
            # Process dark data
            dark_measured, _ = self.spectrometer_handler.process_spectrum_data(dark_scopes)
            dark_df_profile = pd.DataFrame({
                'wavelength_(nm)': dark_measured['wavelength_(nm)'],
                'dark': dark_measured['intensity_dark_corrected']
            })
            
            # Initialize profile data
            profile_data = pd.DataFrame(columns=['wavelength_(nm)'])
            profile_data['wavelength_(nm)'] = dark_df_profile['wavelength_(nm)']
            
            # Acquire multiple shots
            for n in range(profile_steps):
                self.logger.info(f"Acquiring shot {n+1}/{profile_steps}")
                
                # Camera image saving removed - can be added back later
                
                # Prepare and arm for external trigger
                if not self.spectrometer_handler.prepare_measurement(
                    integration_time, trigger_delay, triggered=True
                ):
                    continue
                
                if not self.spectrometer_handler.start_measurement():
                    continue
                # Fire laser or wait for external trigger
                if not getattr(self.config.laser, 'use_external', False):
                    if not self.laser_handler.fire():
                        continue
                else:
                    self.logger.info("External laser mode: waiting for external trigger")
                
                # Read spectrum
                read_timeout = getattr(self.config.laser, 'external_trigger_timeout_s', 5.0) if getattr(self.config.laser, 'use_external', False) else 3.0
                scopes = self.spectrometer_handler.read_measurement(timeout_s=read_timeout)
                if scopes is not None:
                    shot_measured, _ = self.spectrometer_handler.process_spectrum_data(scopes)
                    shot_corrected = self.spectrum_processor.dark_correction(shot_measured, dark_df_profile)
                    profile_data[str(n+1)] = shot_corrected['intensity_dark_corrected']
                
                # Wait between shots
                if n < profile_steps - 1:  # Don't wait after last shot
                    QTest.qWait(delay_between_shots * 1000)
            
            # Save profile data
            profile_filename = f"{self.config.folder_path}{filename}_multishots.csv"
            profile_data.to_csv(profile_filename, index=False)
            
            self.logger.info(f"Profile saved: {profile_filename}")
            self.trace_graph_profile(profile_data)
            
        except Exception as e:
            self.logger.error(f"Failed to acquire spectrum profile: {e}")
            self._show_error_message("Profile Error", f"Failed to acquire profile: {e}")
        finally:
            # Restore microscope light state after the whole acquisition
            if prev_light_on:
                self.serial_handler.send_arduino_command("ON\r")
    
    def trace_graph(self, spectrum_data):
        """Update the spectrum plot"""
        try:
            # Create plot inline (avoid default numeric peak labels)
            from matplotlib.figure import Figure as _Figure
            fig = _Figure()
            ax = fig.add_subplot(111)
            ax.plot(
                spectrum_data["wavelength_(nm)"],
                spectrum_data["intensity_dark_corrected"],
                linewidth=0.9,
                label="Spectrum"
            )
            # If user has requested detection via the One Shot button, overlay detected/matched peaks
            if getattr(self, "_one_shot_detect_peaks_next", False):
                # Use PeakID tab parameters for detection/matching
                min_height = float(self.peakid_min_height.value())
                min_prom = float(self.peakid_min_prom.value())
                tol_nm = float(self.peakid_tol_nm.value())
                # Detect peaks and match lines using PeakIdentifier
                peaks_df = self.peak_identifier.detect_peaks(
                    spectrum_data, min_height=min_height, min_prominence=min_prom
                )
                matches_df = self.peak_identifier.match_peaks_to_lines(peaks_df, tol_nm=tol_nm)
                if peaks_df is not None and not peaks_df.empty:
                    ax.scatter(
                        peaks_df["wavelength_(nm)"],
                        peaks_df["intensity_dark_corrected"],
                        marker="o",
                        s=20,
                        edgecolors="black",
                        facecolors="none",
                        label="Detected peaks"
                    )
                if matches_df is not None and not matches_df.empty:
                    ax.scatter(
                        matches_df["peak_wavelength_nm"],
                        matches_df["intensity"],
                        marker="x",
                        s=40,
                        label="Matched peaks"
                    )
                    # Add element labels above matched peaks (de-duplicate same x)
                    try:
                        labeled_wls = set()
                        for _, row in matches_df.iterrows():
                            x = float(row["peak_wavelength_nm"])
                            if x in labeled_wls:
                                continue
                            labeled_wls.add(x)
                            y = float(row["intensity"])
                            label = f"{row['element']} {row['ion']}"
                            ax.annotate(
                                label,
                                (x, y),
                                textcoords="offset points",
                                xytext=(0, 6),
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color="#202020",
                                clip_on=True,
                            )
                    except Exception:
                        pass
                # Set title for detection case and reset the one-shot flag
                ax.set_title("Spectrum (One shot) with detected and matched peaks")
                self._one_shot_detect_peaks_next = False
            else:
                # No detection overlay; plain spectrum
                ax.set_title("Spectrum (One shot)")
            
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity (counts)")
            if self.show_grid_option.isChecked():
                ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
            ax.legend(loc="upper right", fontsize="small")
            
            # Update display
            self._update_plot_display(fig, self.canvas, self.toolbar, self.mplvl)
            
        except Exception as e:
            self.logger.error(f"Failed to update spectrum plot: {e}")

    def one_shot_detect_peaks(self):
        """Trigger peak detection overlay on the One Shot plot when the button is pressed."""
        try:
            if self.spectro1_data is None or self.spectro1_data.empty:
                self._show_warning_message("No spectrum", "Acquire a spectrum first (One shot).")
                return
            # Set flag so the next trace_graph overlays peaks, then refresh the plot
            self._one_shot_detect_peaks_next = True
            self.trace_graph(self.spectro1_data)
        except Exception as e:
            self.logger.error(f"Failed to detect peaks on One Shot: {e}")
    
    def trace_graph_profile(self, profile_data):
        """Update the profile plot"""
        try:
            profile_steps = int(self.profile_steps_combobox.currentText())
            fig = self.plot_generator.create_profile_plot(profile_data, profile_steps)
            # Without a second plot area, reuse main plot
            self._update_plot_display(fig, self.canvas, self.toolbar, self.mplvl)
            
        except Exception as e:
            self.logger.error(f"Failed to update profile plot: {e}")
    
    def _update_plot_display(self, fig, canvas, toolbar, layout):
        """Update plot display"""
        try:
            # Remove old plot widgets
            if canvas is not None:
                layout.removeWidget(canvas)
                canvas.close()
                canvas.deleteLater()
            if toolbar is not None:
                layout.removeWidget(toolbar)
                toolbar.close()
                toolbar.deleteLater()
            
            # Create new plot
            new_canvas = FigureCanvas(fig)
            # Set size policy to expand and fill available space
            new_canvas.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
            layout.addWidget(new_canvas, 1)  # Stretch factor for canvas
            new_canvas.draw()
            
            new_toolbar = NavigationToolbar(new_canvas, self, coordinates=True)
            layout.addWidget(new_toolbar)
            
            # Update instance variables if this is one of the main plots
            if layout == self.mplvl:
                self.canvas = new_canvas
                self.toolbar = new_toolbar
            # No secondary plot container anymore
            
        except Exception as e:
            self.logger.error(f"Failed to update plot display: {e}")
    
    def change_graph(self):
        """Update graph when options change"""
        if not self.spectro1_data.empty:
            self.trace_graph(self.spectro1_data)
    
    def choose_directory(self):
        """Choose output directory (shared by one-shot and mapping Browse buttons)."""
        try:
            # Prefer the current text in the one-shot field, fall back to config
            start_dir = ''
            if hasattr(self, 'acq_save_dir'):
                start_dir = self.acq_save_dir.text().strip()
            if not start_dir or not os.path.isdir(start_dir):
                start_dir = self.config.mapping.save_dir or self.config.folder_path or str(Path.cwd())
            folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', start_dir)
            if folder_path:
                self._set_output_directory(folder_path, source="main", persist=True)
        except Exception as e:
            self.logger.error(f"Failed to choose directory: {e}")
    
    def save_micro_image(self):
        """Save microscope image"""
        try:
            if self._latest_frame is None:
                self._show_warning_message("No Image", "No camera frame available to save.")
                return

            default_dir = self.config.folder_path or str(Path.cwd())
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Microscope Image",
                str(Path(default_dir) / "micro_capture.png"),
                "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All files (*.*)"
            )
            if not filename:
                return

            image_to_save = self._latest_frame
            # Determine format from suffix if provided
            suffix = Path(filename).suffix.lower()
            fmt = None
            if suffix in {'.png'}:
                fmt = 'PNG'
            elif suffix in {'.jpg', '.jpeg'}:
                fmt = 'JPEG'

            if not image_to_save.save(filename, fmt):
                self._show_error_message("Save Error", "Failed to save image.")
                return

            self.logger.info(f"Microscope image saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            self._show_error_message("Save Error", f"Failed to save image: {e}")
    
    def _resolve_mapping_laser_preferences(self):
        """Return mapping-specific laser preferences from the advanced form."""
        advanced = self.mapping_advanced_form.get_params() if hasattr(self, 'mapping_advanced_form') else {}
        # Per user request, mapping mode should use $QSON 1 / $TRIG EI.
        single_shot = False
        return single_shot
    
    def _ensure_mapping_laser_ready(self) -> bool:
        """Connect the mapping controller if it is not already active."""
        if self._laser_in_mapping_mode:
            return True
        single_shot = self._resolve_mapping_laser_preferences()
        if not self.laser_handler.connect_mapping_controller(single_shot=single_shot):
            self.logger.error("Failed to connect mapping laser controller automatically.")
            self._show_error_message("Mapping", "Failed to prepare the laser for mapping mode.")
            return False
        self._laser_in_mapping_mode = True
        self._refresh_mapping_connection_status()
        return True

    def _restore_acquisition_laser_mode(self):
        """Return the laser to its default one-shot / multi-shot configuration."""
        try:
            if self._laser_in_mapping_mode:
                self.laser_handler.disconnect_mapping_controller()
                self._laser_in_mapping_mode = False
        except Exception as exc:
            self.logger.warning(f"Failed to disconnect mapping laser controller: {exc}")
        try:
            self.laser_handler.apply_acquisition_defaults()
        except Exception as exc:
            self.logger.debug(f"Unable to reapply acquisition laser defaults: {exc}")
        self._refresh_mapping_connection_status()
    
    def laser_connect(self):
        """Connect to laser"""
        try:
            # If using external laser mode, skip LAN connection and just enable acquisitions
            if getattr(self.config.laser, 'use_external', False):
                self.logger.info("External laser mode active: skipping laser connection and enabling acquisition")
                self.go_button.setEnabled(True)
                self.profile_go_button.setEnabled(True)
                self.laser_stop_button.setEnabled(True)
                return
            # Progress callback to update progress bar
            def update_progress(value):
                if hasattr(self, 'progressBar_laser'):
                    self.progressBar_laser.setValue(value)
            if self.laser_handler.connect(progress_callback=update_progress):
                self.logger.info("Laser connected successfully")
                # Initialize spectrometer after laser connection
                if self.spectrometer_handler.initialize():
                    self.logger.info("Spectrometer initialized successfully after laser connection")
                else:
                    self.logger.warning("Spectrometer initialization failed after laser connection")
                if hasattr(self, 'go_button'):
                    self.go_button.setEnabled(True)
                if hasattr(self, 'profile_go_button'):
                    self.profile_go_button.setEnabled(True)
                if hasattr(self, 'laser_stop_button'):
                    self.laser_stop_button.setEnabled(True)
                # Ensure mapping controller is not lingering
                self.laser_handler.disconnect_mapping_controller()
                self._laser_in_mapping_mode = False
                self._refresh_mapping_connection_status()
            else:
                self._show_error_message("Laser Error", "Failed to connect to laser")
                if hasattr(self, 'progressBar_laser'):
                    self.progressBar_laser.setValue(0)
        except Exception as e:
            self.logger.error(f"Failed to connect laser: {e}")
            self._show_error_message("Laser Error", f"Failed to connect laser: {e}")
            if hasattr(self, 'progressBar_laser'):
                self.progressBar_laser.setValue(0)
    
    def laser_stop(self):
        """Stop laser"""
        try:
            # If in external mode without a LAN connection, treat stop as disabling acquisition
            if getattr(self.config.laser, 'use_external', False) and not self.laser_handler.connected:
                if hasattr(self, 'go_button'):
                    self.go_button.setEnabled(False)
                if hasattr(self, 'profile_go_button'):
                    self.profile_go_button.setEnabled(False)
                if hasattr(self, 'progressBar_laser'):
                    self.progressBar_laser.setValue(0)
                self.logger.info("External laser mode: acquisition disabled")
                return
            self.laser_handler.disconnect()
            self._laser_in_mapping_mode = False
            self._refresh_mapping_connection_status()
            if hasattr(self, 'progressBar_laser'):
                self.progressBar_laser.setValue(0)
            if hasattr(self, 'go_button'):
                self.go_button.setEnabled(False)
            if hasattr(self, 'profile_go_button'):
                self.profile_go_button.setEnabled(False)
            self.logger.info("Laser stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop laser: {e}")
    
    def toggle_external_laser(self, checked: bool):
        """Toggle external laser mode and update UI state accordingly."""
        try:
            self.config.laser.use_external = bool(checked)
            # Keep menu action and checkbox in sync without recursion
            if hasattr(self, 'external_laser_action'):
                try:
                    self.external_laser_action.blockSignals(True)
                    self.external_laser_action.setChecked(bool(checked))
                finally:
                    self.external_laser_action.blockSignals(False)
            if hasattr(self, 'external_laser_checkbox'):
                try:
                    self.external_laser_checkbox.blockSignals(True)
                    self.external_laser_checkbox.setChecked(bool(checked))
                finally:
                    self.external_laser_checkbox.blockSignals(False)
            if checked:
                self.logger.info("External laser mode enabled")
                if hasattr(self, 'go_button'):
                    self.go_button.setEnabled(True)
                if hasattr(self, 'profile_go_button'):
                    self.profile_go_button.setEnabled(True)
                if hasattr(self, 'laser_stop_button'):
                    self.laser_stop_button.setEnabled(True)
            else:
                self.logger.info("External laser mode disabled")
                # Only disable if not connected to LAN laser
                if not getattr(self.laser_handler, 'connected', False):
                    if hasattr(self, 'go_button'):
                        self.go_button.setEnabled(False)
                    if hasattr(self, 'profile_go_button'):
                        self.profile_go_button.setEnabled(False)
            self._persist_config()
        except Exception as e:
            self.logger.error(f"Failed to toggle external laser mode: {e}")
    
    # Movement control methods
    def move_north(self, step, speed):
        """Move table north"""
        self._move_table("Y", f"-{float(step)/1000}", speed, "north")
    
    def move_south(self, step, speed):
        """Move table south"""
        self._move_table("Y", f"{float(step)/1000}", speed, "south")
    
    def move_east(self, step, speed):
        """Move table east"""
        self._move_table("X", f"{float(step)/1000}", speed, "east")
    
    def move_west(self, step, speed):
        """Move table west"""
        self._move_table("X", f"-{float(step)/1000}", speed, "west")
    
    def move_up(self, step, speed):
        """Move table up"""
        self._move_table("Z", f"{float(step)/1000}", speed, "up")
    
    def move_down(self, step, speed):
        """Move table down"""
        self._move_table("Z", f"-{float(step)/1000}", speed, "down")
    
    def _move_table(self, axis, distance, speed, direction):
        """Generic table movement"""
        try:
            command = f"$J=G21G91{axis}{distance}F{speed}\n"
            if self.serial_handler.send_cnc_command(command):
                self.logger.info(f"Moving {direction} {distance}mm (Feed rate: {speed}mm/s)")
            else:
                self.logger.error(f"Failed to send movement command")
        except Exception as e:
            self.logger.error(f"Failed to move table {direction}: {e}")
    
    def _enable_cnc_controls(self):
        """Enable CNC movement controls after skipping homing"""
        try:
            self.show_controls(True)
            self.stop_button.setStyleSheet("background-color: red")
            self.logger.info("CNC controls enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable CNC controls: {e}")
    
    def stop(self):
        """Emergency stop"""
        try:
            self.serial_handler.send_cnc_command(serial.to_bytes([0x18]))
            self.stop_button.setStyleSheet("background-color: #DCDCDC")
            self.show_controls(False)
            if hasattr(self, 'reset_cnc_button') and self.reset_cnc_button:
                self.reset_cnc_button.setEnabled(True)
            self.logger.warning("Emergency stop activated")
        except Exception as e:
            self.logger.error(f"Failed to stop: {e}")

    def reset_cnc_after_stop(self):
        """Reset CNC after emergency stop and re-enable controls."""
        try:
            self.serial_handler.send_cnc_command("$X \n")
            self._enable_cnc_controls()
            if hasattr(self, 'reset_cnc_button') and self.reset_cnc_button:
                self.reset_cnc_button.setEnabled(False)
            self.logger.info("CNC reset after stop")
        except Exception as e:
            self.logger.error(f"Failed to reset CNC: {e}")
    
    def show_controls(self, control_state):
        """Enable/disable movement controls"""
        try:
            self.north_button.setEnabled(control_state)
            self.west_button.setEnabled(control_state)
            self.south_button.setEnabled(control_state)
            self.east_button.setEnabled(control_state)
            self.down_button.setEnabled(control_state)
            self.up_button.setEnabled(control_state)
            self.stop_button.setEnabled(control_state)
        except Exception as e:
            self.logger.error(f"Failed to update control state: {e}")
    
    def _show_error_message(self, title, message):
        """Show error message"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()
    
    def _show_warning_message(self, title, message):
        """Show warning message"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()
    
    # Text output handling
    def write(self, text):
        self.text_update.emit(str(text))
    
    def flush(self):
        pass
    
    def append_text(self, text):
        """Append text to console"""
        try:
            cur = self.console_box.textCursor()
            cur.movePosition(QtGui.QTextCursor.End)
            s = str(text)
            while s:
                head, sep, s = s.partition("\n")
                cur.insertText(head)
                if sep:
                    cur.insertBlock()
            self.console_box.setTextCursor(cur)
        except Exception as e:
            self.logger.error(f"Failed to append text: {e}")
    
    def closeEvent(self, event):
        """Handle application close"""
        try:
            reply = QMessageBox.question(
                self, 'Quit LIBS Application?',
                "Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.logger.info("Application closing...")
                    
                # Stop mapping thread if running
                if getattr(self, 'mapping_thread', None) and self.mapping_thread.is_alive():
                    try:
                        self.mapping_thread.stop_event.set()
                        self.mapping_thread.join(timeout=2.0)
                    except Exception:
                        pass

                # Turn off microscope LED before closing serial ports
                try:
                    self.serial_handler.send_arduino_command("OFF\r")
                except Exception:
                    pass

                # Disconnect devices
                self.spectrometer_handler.close()
                self.laser_handler.disconnect()
                self.serial_handler.close_all()
                
                # Stop camera
                self._stop_camera()

                # Save splitter states (resizable UI sections)
                self._save_splitter_states()

                # Persist configuration changes
                self._persist_config()
                
                event.accept()
                self.logger.info("Application closed")
            else:
                event.ignore()
                
        except Exception as e:
            self.logger.error(f"Error during close: {e}")
            event.accept()


def main():
    """Main application entry point"""
    try:
        # Ensure proper path setup
        current_dir = Path(__file__).parent.absolute()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        os.add_dll_directory(str(current_dir))
        
        # Create application
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")
        
        # Setup dark mode if enabled
        if config.dark_mode:
            app.setStyle("Fusion")
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, QtCore.Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(55, 55, 55))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, QtCore.Qt.white)
            dark_palette.setColor(QPalette.ToolTipText, QtCore.Qt.white)
            dark_palette.setColor(QPalette.Text, QtCore.Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, QtCore.Qt.white)
            dark_palette.setColor(QPalette.BrightText, QtCore.Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, QtCore.Qt.white)
            app.setPalette(dark_palette)
            app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
        
        # Ensure images folder and migrate legacy icons
        ensure_images_folder()

        # Show splash screen (generate a modern one and also persist it)
        splash_pix = create_modern_splash(config.app_name)
        splash = QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
        splash.setMask(splash_pix.mask())
        splash.show()
        time.sleep(2)
        
        # Create and show main window
        win = MainWindow()
        win.show()
        
        splash.finish(win)
        
        win.setWindowTitle(config.app_name)
        win.setWindowIcon(QIcon(str(Path('images') / 'icon_aconvert.ico')))
        
        # Start application
        sys.exit(app.exec_())
        
    except Exception as e:
        traceback.print_exc()
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
