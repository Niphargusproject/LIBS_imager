# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 22:45:33 2025

@author: Christian
"""

# Step-by-step LIBS hypercube processing: Load, Photo, Mask, Normalize, Map Explorer, Data Extraction, Composite, Export

import os

# joblib/loky (used by scikit-learn) counts physical cores via 'wmic', which
# was removed from recent Windows builds; without this it prints a warning +
# traceback on every use. Must be set before joblib is first imported, and must
# be strictly below the logical core count for loky to respect it (which also
# leaves one core free for the UI during heavy computations).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.gridspec import GridSpec

import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

from PyQt5.QtWidgets import (

    QInputDialog, QListWidget, QListWidgetItem, QDialog as QDlgBase
)


from dataclasses import dataclass, asdict, field
from PyQt5.QtWidgets import (

    QListWidget, QAbstractItemView
)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient, QPen, QPolygonF, QPainterPath, QImage, QIcon
from PyQt5.QtCore import QPointF, QEvent
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QAction, QMenuBar, QStatusBar,
    QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QComboBox, QSlider, QDoubleSpinBox, QSpinBox,
    QCheckBox, QToolButton, QPushButton, QGroupBox, QMessageBox,
    QDialog, QLineEdit, QDialogButtonBox, QTreeWidget, QTreeWidgetItem, QHeaderView, QSplashScreen,
    QProgressDialog, QSplitter, QScrollArea, QMenu, QSizePolicy, QColorDialog, QRadioButton,
    QStyle
)

import pandas as pd
from scipy.signal import find_peaks

class _NormCancelled(Exception):
    """Raised when user cancels a normalization via the progress dialog."""
    pass

# ---- App constants ----
APP_NAME = "LIBS hypercube explorer"
BUILD_VERSION = "2026-07-24 19:18"

# ---- Matplotlib defaults ----
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 8
})

def get_view(ax):
    try:
        return {'xlim': tuple(ax.get_xlim()), 'ylim': tuple(ax.get_ylim())}
    except Exception:
        return {'xlim': None, 'ylim': None}

def set_view(ax, view):
    if not view: return
    try:
        if view.get('xlim'): ax.set_xlim(*view['xlim'])
        if view.get('ylim'): ax.set_ylim(*view['ylim'])
    except Exception:
        pass

def ndarray_or_none_to_list(a):
    if a is None: return None
    return np.asarray(a, dtype=float).tolist()

def list_or_none_to_ndarray(lst):
    if lst is None: return None
    return np.asarray(lst, dtype=float)

# ---------- Homography helpers (DLT with normalization) ----------
def _normalize_points(pts):
    pts = np.asarray(pts, dtype=float)
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    d = np.sqrt(((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2).mean())
    s = np.sqrt(2) / d if d > 0 else 1.0
    T = np.array([[s, 0, -s*cx],
                  [0, s, -s*cy],
                  [0, 0, 1]])
    pts_h = np.c_[pts, np.ones(len(pts))]
    npts = (T @ pts_h.T).T
    return npts[:, :2], T

def compute_homography(src_pts, dst_pts):
    """Return H s.t. dst ~ H * src (homog)."""
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)
    assert src.shape == (4,2) and dst.shape == (4,2)
    ns, Ts = _normalize_points(src)
    nd, Td = _normalize_points(dst)
    A = []
    for (x,y), (u,v) in zip(ns, nd):
        A.append([-x, -y, -1,  0,  0,  0, x*u, y*u, u])
        A.append([ 0,  0,  0, -x, -y, -1, x*v, y*v, v])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    Hn = h.reshape(3,3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2,2]) < 1e-12:
        raise ValueError("Degenerate homography (H[2,2] is zero).")
    return H / H[2,2]

def apply_H(H, x, y):
    p = np.array([x, y, 1.0])
    q = H @ p
    if abs(q[2]) < 1e-12: return None
    return q[0]/q[2], q[1]/q[2]


def create_colormap(start_color, end_color, name='custom_colormap'):
    colors = [start_color, end_color]
    return plt.cm.colors.LinearSegmentedColormap.from_list(name, colors)

def create_high_colormap(start_color, end_color, name='custom_colormap'):
    colors = [start_color, end_color, (1, 1, 1)]
    return plt.cm.colors.LinearSegmentedColormap.from_list(name, colors)


element_colormaps = {
    'Viridis': plt.cm.viridis, 'Inferno': plt.cm.inferno,
    'Red': create_colormap((0, 0, 0), (1, 0, 0), 'high_contrast_black_red'),
    'Green': create_colormap((0, 0, 0), (0, 1, 0), 'high_contrast_black_green'),
    'Blue': create_colormap((0, 0, 0), (0, 0, 1), 'high_contrast_black_blue'),
    'Yellow': create_colormap((0, 0, 0), (1, 1, 0), 'high_contrast_black_yellow'),
    'Orange': create_colormap((0, 0, 0), (1, 0.5, 0), 'high_contrast_black_orange'),
    'Purple': create_colormap((0, 0, 0), (0.5, 0, 0.5), 'high_contrast_black_purple'),
    'Gray': create_colormap((0, 0, 0), (0.5, 0.5, 0.5), 'high_contrast_black_gray'),
    'Cyan': create_colormap((0, 0, 0), (0, 1, 1), 'high_contrast_black_cyan'),
    'Magenta': create_colormap((0, 0, 0), (1, 0, 1), 'high_contrast_black_magenta'),
    'White': create_colormap((0, 0, 0), (1, 1, 1), 'high_contrast_black_white'),
    'Brown': create_colormap((0, 0, 0), (0.6, 0.3, 0), 'high_contrast_black_brown'),
    'Olive': create_colormap((0, 0, 0), (0.5, 0.5, 0.2), 'high_contrast_black_olive'),
    'Teal': create_colormap((0, 0, 0), (0, 0.5, 0.5), 'high_contrast_black_teal'),
    'Pink': create_colormap((0, 0, 0), (1, 0.7, 0.8), 'high_contrast_black_pink'),
    'Gold': create_colormap((0, 0, 0), (1, 0.84, 0), 'high_contrast_black_gold')
}

# ---- Common element lines (including CRM = Critical Raw Materials) ----
element_wavelengths_common = {
    # --- Major / common elements ---
    'Ag I 328.068': 328.068, 'Ag I 338.289': 338.289,
    'Ag II 224.641': 224.641, 'Ag II 243.779': 243.779,
    'Al I 308.215': 308.215, 'Al I 309.271': 309.271,
    'Al I doublet 308.215+309.271': [308.215, 309.271],
    'Al I 394.401': 394.401, 'Al I 396.153': 396.153,
    'Al I doublet 394.401+396.153': [394.401, 396.153],
    'Al II 281.616': 281.616, 'Al II 466.305': 466.305, 'Al II 559.330': 559.330,
    'Ar I 763.511': 763.511, 'Ar I 811.531': 811.531,
    'As I 228.812': 228.812, 'As I 234.984': 234.984, 'As I 278.022': 278.022,
    'Au I 242.795': 242.795, 'Au I 267.595': 267.595,
    'B I 249.677': 249.677, 'B I 249.773': 249.773,
    'Ba II 455.403': 455.4033, 'Ba II 493.408': 493.4077,
    'Br I 827.244': 827.244,
    'C I 247.856': 247.856,
    'C II 283.671': 283.671, 'C II 426.726': 426.726,
    'Ca I 422.673': 422.673,
    'Ca II 393.366': 393.366, 'Ca II 396.847': 396.847,
    'Cl I 837.594': 837.594, 'Cl II 479.456': 479.456,
    'Cu I 324.754': 324.754, 'Cu II 327.396': 327.396,
    'F I 685.603': 685.603, 'F I 739.868': 739.868,
    'Fe I 248.327': 248.327, 'Fe I 371.994': 371.994, 'Fe I 374.556': 374.556,
    'Fe II 238.204': 238.204, 'Fe II 259.940': 259.940,
    'H I 656.279': 656.279, 'H I 486.135': 486.135,
    'K I 766.490': 766.490, 'K I 769.896': 769.896,
    'Li I 670.776': 670.776, 'Li II 610.362': 610.362,
    'Mg I 279.553': 279.553, 'Mg II 280.270': 280.270,
    'Mg II doublet 279.553+280.270': [279.553, 280.270],
    'Mn I 257.610': 257.610, 'Mn I 403.076': 403.076, 'Mn II 260.568': 260.568,
    'N I 742.364': 742.364, 'N I 744.229': 744.229, 'N I 746.831': 746.831,
    'N II 500.515': 500.515,
    'Na I 589.000': 589.000, 'Na I 589.592': 589.592,
    'O I 777.194': 777.194, 'O I 777.417': 777.417, 'O I 844.636': 844.636,
    'P I 213.618': 213.618, 'P I 214.914': 214.914, 'P I 253.560': 253.560,
    'S I 921.287': 921.287, 'S I 922.809': 922.809, 'S I 923.754': 923.754,
    'S II 545.386': 545.386,
    'Si I 251.611': 251.611, 'Si I 288.158': 288.158, 'Si II 413.089': 413.089,
    'Sr I 460.733': 460.733, 'Sr II 407.771': 407.771,
    'Ti I 334.941': 334.941, 'Ti I 337.280': 337.280,
    'Ti II 368.520': 368.520, 'Ti II 375.929': 375.929,
    'Zn I 213.856': 213.856, 'Zn I 307.590': 307.590, 'Zn I 330.258': 330.258,
    'Zn II 202.548': 202.548, 'Zn II 206.200': 206.200, 'Zn II 334.502': 334.502,
    # --- Critical Raw Materials (CRM) ---
    'Be I 234.861': 234.861, 'Be II 313.042': 313.042,
    'Bi I 306.772': 306.772,
    'Cd I 228.802': 228.802, 'Cd I 326.106': 326.106,
    'Cd II 214.441': 214.441, 'Cd II 226.502': 226.502,
    'Co I 340.512': 340.512, 'Co I 345.350': 345.350,
    'Cr I 357.869': 357.869, 'Cr I 425.435': 425.435,
    'Ga I 287.424': 287.424, 'Ga I 417.204': 417.204,
    'Ge I 265.118': 265.118, 'Ge I 303.906': 303.906,
    'Hf II 264.141': 264.141,
    'In I 303.936': 303.936, 'In I 410.176': 410.176,
    'Ir I 254.397': 254.397,
    'Mo I 313.259': 313.259, 'Mo I 317.035': 317.035,
    'Mo I 379.825': 379.825, 'Mo I 386.411': 386.411,
    'Mo II 281.615': 281.615, 'Mo II 284.823': 284.823,
    'Nb I 405.894': 405.894, 'Nb II 309.418': 309.418,
    'Ni I 341.476': 341.476, 'Ni I 352.454': 352.454,
    'Pb I 283.306': 283.306, 'Pb I 368.346': 368.346, 'Pb I 405.781': 405.781,
    'Pd I 340.458': 340.458,
    'Pt I 265.945': 265.945,
    'Rb I 780.027': 780.027,
    'Re I 346.046': 346.046,
    'Sb I 259.805': 259.805, 'Sb I 287.792': 287.792,
    'Sc II 361.384': 361.384, 'Sc II 424.683': 424.683,
    'Se I 203.985': 203.985, 'Se I 206.279': 206.279,
    'Sn I 283.999': 283.999, 'Sn I 317.505': 317.505,
    'Ta I 331.116': 331.116, 'Ta II 268.517': 268.517,
    'Te I 214.281': 214.281,
    'V I 318.540': 318.540, 'V II 311.071': 311.071,
    'W I 400.875': 400.875, 'W II 248.923': 248.923,
    'Y II 371.030': 371.030, 'Y II 377.433': 377.433,
    'Zr II 339.198': 339.198, 'Zr II 343.823': 343.823,
    # Rare earths (selected strong lines)
    'Ce II 413.765': 413.765, 'Ce II 418.660': 418.660,
    'Dy II 353.170': 353.170,
    'Er II 337.271': 337.271,
    'Eu II 381.967': 381.967, 'Eu II 420.505': 420.505,
    'Gd II 342.247': 342.247,
    'Ho II 345.600': 345.600,
    'La II 408.672': 408.672, 'La II 394.910': 394.910,
    'Lu II 261.542': 261.542,
    'Nd II 401.225': 401.225, 'Nd II 430.358': 430.358,
    'Pr II 422.293': 422.293,
    'Sm II 359.260': 359.260,
    'Tb II 350.917': 350.917,
    'Tm II 313.126': 313.126,
    'Yb II 328.937': 328.937,
}

# ---- Specialized Ca/Mg lines (speleothem / carbonate work) ----
element_wavelengths_specialized = {
    # Calcium — extended
    'Ca I 422.673': 422.673,
    'Ca I 428.973': 428.973, 'Ca I 430.252': 430.252, 'Ca I 442.543': 442.543,
    'Ca I 445.478': 445.478, 'Ca I 526.218': 526.218,
    'Ca II 315.887': 315.887, 'Ca II 317.933': 317.933,
    'Ca II 370.602': 370.602, 'Ca II 393.366': 393.366,
    'Ca II 396.847': 396.847, 'Ca II 854.209': 854.209,
    'Ca(I) 336.19 test': 336.19, 'Ca(I) 364.44 test': 364.44,
    'Ca(I) 428.30 test': 428.30, 'Ca(I) 429.89 test': 429.89,
    'Ca(I) 430.77 test': 430.77, 'Ca(I) 431.86 test': 431.86,
    'Ca(I) 610.27 test': 610.27, 'Ca(I) 643.90 test': 643.90,
    'Ca_calc_LTE 362.41': 362.41, 'Ca_calc_LTE 363.09': 363.09,
    # Magnesium — extended
    'Mg I 279.553': 279.553, 'Mg I 285.213': 285.213,
    'Mg I 382.935': 382.935, 'Mg I 383.829': 383.829,
    'Mg II 280.270': 280.270, 'Mg II 438.535': 438.535,
    'Mg II 448.112': 448.112, 'Mg II 517.268': 517.268,
    'Mg II doublet 279.553+280.270': [279.553, 280.270],
}

# Merged dict for look-ups (common takes priority)
element_wavelengths = {**element_wavelengths_common, **element_wavelengths_specialized}

# ---------- Element-symbol → LIBS-line mapping (for periodic-table widget) ----------
import re as _re

def _build_element_to_lines():
    result = {}
    for name in element_wavelengths:
        m = _re.match(r'([A-Z][a-z]?)', name)
        if m:
            result.setdefault(m.group(1), []).append(name)
    for sym in result:
        result[sym].sort()
    return result

ELEMENT_TO_LINES = _build_element_to_lines()

# ---------- Periodic-table layout: (symbol, row, col) ----------
PERIODIC_TABLE_LAYOUT = [
    ('H',  0, 0),                                                                                       ('He', 0, 17),
    ('Li', 1, 0),  ('Be', 1, 1),                                     ('B',  1, 12), ('C',  1, 13), ('N',  1, 14), ('O',  1, 15), ('F',  1, 16), ('Ne', 1, 17),
    ('Na', 2, 0),  ('Mg', 2, 1),                                     ('Al', 2, 12), ('Si', 2, 13), ('P',  2, 14), ('S',  2, 15), ('Cl', 2, 16), ('Ar', 2, 17),
    ('K',  3, 0),  ('Ca', 3, 1),  ('Sc', 3, 2),  ('Ti', 3, 3),  ('V',  3, 4),  ('Cr', 3, 5),  ('Mn', 3, 6),  ('Fe', 3, 7),
    ('Co', 3, 8),  ('Ni', 3, 9),  ('Cu', 3, 10), ('Zn', 3, 11), ('Ga', 3, 12), ('Ge', 3, 13), ('As', 3, 14),
    ('Se', 3, 15), ('Br', 3, 16), ('Kr', 3, 17),
    ('Rb', 4, 0),  ('Sr', 4, 1),  ('Y',  4, 2),  ('Zr', 4, 3),  ('Nb', 4, 4),  ('Mo', 4, 5),
    ('Ru', 4, 7),  ('Rh', 4, 8),  ('Pd', 4, 9),  ('Ag', 4, 10), ('Cd', 4, 11), ('In', 4, 12), ('Sn', 4, 13),
    ('Sb', 4, 14), ('Te', 4, 15), ('I',  4, 16), ('Xe', 4, 17),
    ('Cs', 5, 0),  ('Ba', 5, 1),                  ('Hf', 5, 3),  ('Ta', 5, 4),  ('W',  5, 5),  ('Re', 5, 6),
    ('Os', 5, 7),  ('Ir', 5, 8),  ('Pt', 5, 9),  ('Au', 5, 10), ('Hg', 5, 11), ('Tl', 5, 12), ('Pb', 5, 13),
    ('Bi', 5, 14),
    ('Fr', 6, 0),  ('Ra', 6, 1),
    # Lanthanides (separate row)
    ('La', 7, 2),  ('Ce', 7, 3),  ('Pr', 7, 4),  ('Nd', 7, 5),  ('Sm', 7, 7),  ('Eu', 7, 8),
    ('Gd', 7, 9),  ('Tb', 7, 10), ('Dy', 7, 11), ('Ho', 7, 12), ('Er', 7, 13), ('Tm', 7, 14), ('Yb', 7, 15), ('Lu', 7, 16),
]

COMPOSITE_DEFAULT_COLORS = [
    (255, 0, 0), (0, 200, 0), (60, 60, 255), (0, 220, 220), (255, 0, 255),
]

from PyQt5.QtCore import pyqtSignal

class PeriodicTableWidget(QWidget):
    """Compact interactive periodic table; emits *lineSelected(str)* with the
    chosen LIBS line name (e.g. 'Ca II 393.366')."""

    lineSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons = {}
        self._layer_indicators = {}
        grid = QGridLayout(self)
        grid.setSpacing(1)
        grid.setContentsMargins(0, 0, 0, 0)

        available = set(ELEMENT_TO_LINES.keys())

        for sym, row, col in PERIODIC_TABLE_LAYOUT:
            btn = QPushButton(sym)
            btn.setFixedSize(30, 28)
            has_lines = sym in available
            if has_lines:
                btn.setStyleSheet(
                    "QPushButton{background:#3a3a3a;color:white;border:1px solid #666;"
                    "border-radius:2px;padding:0px;margin:0px;font:bold 7pt 'Segoe UI';}"
                    "QPushButton:hover{background:#555;}"
                )
                btn.clicked.connect(lambda _c=False, s=sym: self._on_element_click(s))
                btn.setToolTip(f"{sym}  –  {len(ELEMENT_TO_LINES[sym])} line(s)")
            else:
                btn.setStyleSheet(
                    "QPushButton{background:#1e1e1e;color:#444;border:1px solid #2a2a2a;"
                    "border-radius:2px;padding:0px;margin:0px;font:7pt 'Segoe UI';}"
                )
                btn.setEnabled(False)
            grid.addWidget(btn, row, col)
            self._buttons[sym] = btn

        lbl = QLabel("* Lanthanides")
        lbl.setStyleSheet("color:#888;font-size:7pt;")
        grid.addWidget(lbl, 5, 2, 1, 1, Qt.AlignCenter)

    def _on_element_click(self, symbol):
        lines = ELEMENT_TO_LINES.get(symbol, [])
        if not lines:
            return
        if len(lines) == 1:
            self.lineSelected.emit(lines[0])
            return
        menu = QMenu(self)
        for ln in lines:
            menu.addAction(ln, lambda _ln=ln: self.lineSelected.emit(_ln))
        menu.exec_(self._buttons[symbol].mapToGlobal(self._buttons[symbol].rect().bottomLeft()))

    def set_layer_highlight(self, symbol, color):
        """Put a colored border on an element button to show layer assignment."""
        btn = self._buttons.get(symbol)
        if btn and btn.isEnabled():
            r, g, b = color
            btn.setStyleSheet(
                f"QPushButton{{background:#3a3a3a;color:white;border:2px solid rgb({r},{g},{b});"
                f"border-radius:2px;padding:0px;margin:0px;font:bold 7pt 'Segoe UI';}}"
                f"QPushButton:hover{{background:#555;}}"
            )

    def clear_highlights(self):
        available = set(ELEMENT_TO_LINES.keys())
        for sym, btn in self._buttons.items():
            if sym in available:
                btn.setStyleSheet(
                    "QPushButton{background:#3a3a3a;color:white;border:1px solid #666;"
                    "border-radius:2px;padding:0px;margin:0px;font:bold 7pt 'Segoe UI';}"
                    "QPushButton:hover{background:#555;}"
                )


# ---------- Unified Export dialog ----------
class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setModal(True)

        self.figure_combo = QComboBox()
        self.figure_combo.addItems([
            "Map Explorer image",
            "Data extraction image (LIBS)",
            "Data extraction image (Photo)",
            "Data extraction montage (LIBS+Photo)",
            "Data extraction plot (right)",
            "Composite overlay image",
            "Composite overlay raw (1:1, no axes/legend)",
            "Raw pixel map (1:1, no axes/legend)"
        ])

        self.width_cm = QDoubleSpinBox();  self.width_cm.setRange(0.5, 500.0); self.width_cm.setDecimals(2); self.width_cm.setValue(15.0)
        self.height_cm = QDoubleSpinBox(); self.height_cm.setRange(0.5, 500.0); self.height_cm.setDecimals(2); self.height_cm.setValue(10.0)
        self.dpi = QDoubleSpinBox();       self.dpi.setRange(50, 2400); self.dpi.setDecimals(0); self.dpi.setValue(300)
        self.title_fs = QDoubleSpinBox();  self.title_fs.setRange(4, 72); self.title_fs.setDecimals(0); self.title_fs.setValue(10)
        self.axis_fs  = QDoubleSpinBox();  self.axis_fs.setRange(4, 72); self.axis_fs.setDecimals(0); self.axis_fs.setValue(8)
        self.format_combo = QComboBox();   self.format_combo.addItems(["PNG", "PDF", "SVG", "TIFF"])

        self.path_edit = QLineEdit()
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse)

        form = QFormLayout()
        form.addRow("Figure:", self.figure_combo)
        form.addRow("Width (cm):", self.width_cm)
        form.addRow("Height (cm):", self.height_cm)
        form.addRow("DPI:", self.dpi)
        form.addRow("Title font size:", self.title_fs)
        form.addRow("Axis font size:", self.axis_fs)
        form.addRow("Format:", self.format_combo)

        hpath = QHBoxLayout(); hpath.addWidget(self.path_edit); hpath.addWidget(self.browse_btn)
        form.addRow("Save to:", hpath)

        self._figure_controls = [self.width_cm, self.height_cm, self.dpi, self.title_fs, self.axis_fs]
        self._figure_labels = []
        for row_idx in range(1, 6):
            lbl = form.itemAt(row_idx, QFormLayout.LabelRole)
            if lbl and lbl.widget():
                self._figure_labels.append(lbl.widget())

        self.figure_combo.currentTextChanged.connect(self._on_figure_changed)
        self._on_figure_changed(self.figure_combo.currentText())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self); root.addLayout(form); root.addWidget(buttons)

    def _on_figure_changed(self, text):
        is_raw = "1:1" in text
        for w in self._figure_controls:
            w.setEnabled(not is_raw)
        for lbl in self._figure_labels:
            lbl.setEnabled(not is_raw)

    def _browse(self):
        fmt = self.format_combo.currentText().lower()
        filters = {"png":"PNG (*.png)","pdf":"PDF (*.pdf)","svg":"SVG (*.svg)","tiff":"TIFF (*.tif *.tiff)"}
        path, _ = QFileDialog.getSaveFileName(self, "Save As", "", filters[fmt])
        if path:
            ext = {"png":".png","pdf":".pdf","svg":".svg","tiff":".tiff"}[fmt]
            if not path.lower().endswith((".png",".pdf",".svg",".tif",".tiff")):
                path += ext
            self.path_edit.setText(path)

    def values(self):
        return {
            "which": self.figure_combo.currentText(),
            "width_cm": float(self.width_cm.value()),
            "height_cm": float(self.height_cm.value()),
            "dpi": int(self.dpi.value()),
            "title_fs": int(self.title_fs.value()),
            "axis_fs": int(self.axis_fs.value()),
            "format": self.format_combo.currentText().lower(),
            "path": self.path_edit.text().strip()
        }


@dataclass
class ExperimentState:
    name: str

    # IO / selections
    cube_path: Optional[str]
    # imaging/science display
    element_name: str
    band_index: int
    divider_enabled: bool
    divider_wavelength: Optional[float]
    cmap_name: str
    axes_units: str
    mm_per_px: float
    autoscale: bool
    pmin: float
    pmax: float
    vmin: float
    vmax: float
    band_label: str

    # masking
    mask_enabled: bool
    mask_wavelength: Optional[float]
    mask_threshold: Optional[float]

    # tools
    line_coords: Optional[List[float]]   # [y0,x0,y1,x1] in LIBS
    parallel_buffer: int
    ignore_zeros: bool

    pixel_coords: Optional[List[int]]    # [y, x]
    peak_detection_enabled: bool
    peak_prominence: float
    peak_distance: int

    # figure views (zoom/pan)
    view_imaging_map: Dict[str, Tuple[float, float]]
    view_science_libs: Dict[str, Tuple[float, float]]
    view_right_plot: Dict[str, Tuple[float, float]]

    # which tool mode was on
    mode_line: bool
    mode_pixel: bool

    nan_tolerance: int = 5

    # photo
    photo_path: Optional[str] = None
    polygon_points: Optional[List[Tuple[float, float]]] = None
    calib_points: Optional[List[Tuple[float, float]]] = None
    drag_calib_enabled: bool = False
    H_photo_to_libs: Optional[List[List[float]]] = None
    H_libs_to_photo: Optional[List[List[float]]] = None
    view_photo_tab: Optional[Dict[str, Tuple[float, float]]] = None
    view_science_photo: Optional[Dict[str, Tuple[float, float]]] = None

    # Map Explorer baseline & display
    baseline_enabled: bool = False
    baseline_method: str = "Peak height"
    baseline_halfwidth: float = 0.10
    baseline_gap: float = 0.02
    show_axes: bool = True
    locked_view: Optional[Dict[str, Tuple[float, float]]] = None

    # normalization
    norm_method: str = "None"
    norm_cont_start: float = 350.0
    norm_cont_end: float = 355.0
    norm_kernel: int = 21

    # cube-wide baseline correction (preprocessing)
    cube_baseline_method: str = "None"
    cube_baseline_snip_iter: int = 40
    cube_baseline_window: int = 101
    cube_baseline_clip_negatives: bool = True
    cube_baseline_asls_log10_lam: float = 5.0
    cube_baseline_asls_p: float = 0.01
    cube_baseline_asls_iter: int = 10

    # Composite overlay (5 layers)
    composite_layers: Optional[List[Dict[str, Any]]] = None
    composite_pmin: float = 0.5
    composite_pmax: float = 99.5
    composite_bl_method: str = "Peak height"
    composite_bl_hw: float = 0.10
    composite_bl_gap: float = 0.02
    composite_bg: str = "Black"
    composite_overlay_pos: str = "Inside image"

    # Map Explorer ratio (Ca-normalised) settings
    divider_element: str = "None"
    divider_min: float = 0.0
    divider_scale: float = 1000.0

@dataclass
class ProjectState:
    path: Optional[str]
    experiments: List[ExperimentState]
    active_index: int = -1
    shared_lines: List[Dict[str, Any]] = field(default_factory=list)   # [{'name': str, 'coords':[y0,x0,y1,x1]}]
    shared_pixels: List[Dict[str, Any]] = field(default_factory=list)  # [{'name': str, 'coords':[y,x]}]


class _BaselinePreviewDialog(QDialog):
    """Non-modal dialog: plot a random-pixel spectrum with its computed baseline."""

    def __init__(self, parent, method: str):
        super().__init__(parent)
        self.setWindowTitle(f"Baseline preview — {method}")
        self.resize(900, 600)
        self._parent_app = parent

        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        top = QHBoxLayout()
        self.lbl_pixel = QLabel("")
        self.lbl_pixel.setStyleSheet("font-weight: bold;")
        self.btn_refresh = QPushButton("Pick another random pixel")
        self.btn_refresh.clicked.connect(self._pick_and_plot)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        top.addWidget(self.lbl_pixel, 1)
        top.addWidget(self.btn_refresh)
        top.addWidget(self.btn_close)

        v = QVBoxLayout(self)
        v.addLayout(top)
        v.addWidget(self.toolbar)
        v.addWidget(self.canvas, 1)

        # Two stacked axes: raw+baseline on top, corrected on bottom
        self.ax_raw = self.fig.add_subplot(211)
        self.ax_corr = self.fig.add_subplot(212, sharex=self.ax_raw)
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.09, hspace=0.25)

        self._cached_spectrum = None  # (y_idx, x_idx, bands, spectrum)
        self._pick_and_plot()

    def _compute_baseline(self, bands, spectrum):
        app = self._parent_app
        method = app.cube_baseline_combo.currentText()
        if method == "SNIP":
            niter = int(app.cube_bl_snip_iter_spin.value())
            return app._baseline_snip_spectrum(spectrum, niter), f"SNIP, {niter} iters"
        if method == "Rolling minimum":
            w = int(app.cube_bl_window_spin.value())
            return app._baseline_rolling_min_spectrum(spectrum, w), f"Rolling min, w={w}"
        if method == "AsLS":
            lam = float(10 ** app.cube_bl_asls_lam_spin.value())
            p = float(app.cube_bl_asls_p_spin.value())
            niter = int(app.cube_bl_asls_iter_spin.value())
            return (app._baseline_asls_spectrum(spectrum, lam, p, niter),
                    f"AsLS λ=1e{app.cube_bl_asls_lam_spin.value():.1f}, p={p:g}, {niter} iters")
        return np.zeros_like(spectrum), "none"

    def _pick_and_plot(self):
        app = self._parent_app
        picked = app._baseline_random_spectrum()
        if picked is None:
            return
        self._cached_spectrum = picked
        self._replot()

    def _replot(self):
        if self._cached_spectrum is None:
            return
        app = self._parent_app
        method = app.cube_baseline_combo.currentText()
        self.setWindowTitle(f"Baseline preview — {method}")
        y_idx, x_idx, bands, spec = self._cached_spectrum
        try:
            baseline, param_str = self._compute_baseline(bands, spec)
        except Exception as e:
            QMessageBox.critical(self, "Preview error", str(e))
            return
        corrected = spec - baseline
        if app.cube_bl_clip_neg_chk.isChecked():
            corrected = np.clip(corrected, 0.0, None)
        self.lbl_pixel.setText(f"Pixel (y={y_idx}, x={x_idx}) — {param_str}")

        self.ax_raw.clear()
        self.ax_raw.plot(bands, spec, color='#2c3e50', lw=0.7, label='raw spectrum')
        self.ax_raw.plot(bands, baseline, color='#e74c3c', lw=1.2, label='baseline')
        self.ax_raw.fill_between(bands, baseline, spec,
                                 where=(spec >= baseline), color='#3498db', alpha=0.15,
                                 label='peak area above baseline')
        self.ax_raw.set_ylabel('Intensity')
        self.ax_raw.set_title(f"Random pixel (y={y_idx}, x={x_idx})  —  {method}: {param_str}",
                              fontsize=10)
        self.ax_raw.legend(loc='upper right', fontsize=8)
        self.ax_raw.grid(alpha=0.3)

        self.ax_corr.clear()
        self.ax_corr.axhline(0.0, color='#888', lw=0.6, linestyle='--')
        self.ax_corr.plot(bands, corrected, color='#27ae60', lw=0.7,
                          label='baseline-corrected')
        self.ax_corr.set_xlabel('Wavelength (nm)')
        self.ax_corr.set_ylabel('Intensity')
        self.ax_corr.legend(loc='upper right', fontsize=8)
        self.ax_corr.grid(alpha=0.3)

        self.canvas.draw_idle()


class ExperimentPicker(QDialog):
    def __init__(self, parent, title: str, items: List[str], initial_index: int = 0, allow_rename: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.list = QListWidget(self)
        for s in items:
            QListWidgetItem(s, self.list)
        if 0 <= initial_index < len(items):
            self.list.setCurrentRow(initial_index)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        v = QVBoxLayout(self)
        v.addWidget(self.list)
        v.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def selected_row(self):
        return self.list.currentRow()


# ---------- Periodic Table Dialog ----------

# Standard periodic table layout: (symbol, name, atomic_number, row, col)
_PERIODIC_TABLE = [
    ("H",  "Hydrogen",      1,  0, 0),  ("He", "Helium",        2,  0, 17),
    ("Li", "Lithium",       3,  1, 0),  ("Be", "Beryllium",     4,  1, 1),
    ("B",  "Boron",         5,  1, 12), ("C",  "Carbon",        6,  1, 13),
    ("N",  "Nitrogen",      7,  1, 14), ("O",  "Oxygen",        8,  1, 15),
    ("F",  "Fluorine",      9,  1, 16), ("Ne", "Neon",         10,  1, 17),
    ("Na", "Sodium",       11,  2, 0),  ("Mg", "Magnesium",    12,  2, 1),
    ("Al", "Aluminium",    13,  2, 12), ("Si", "Silicon",      14,  2, 13),
    ("P",  "Phosphorus",   15,  2, 14), ("S",  "Sulfur",       16,  2, 15),
    ("Cl", "Chlorine",     17,  2, 16), ("Ar", "Argon",        18,  2, 17),
    ("K",  "Potassium",    19,  3, 0),  ("Ca", "Calcium",      20,  3, 1),
    ("Sc", "Scandium",     21,  3, 2),  ("Ti", "Titanium",     22,  3, 3),
    ("V",  "Vanadium",     23,  3, 4),  ("Cr", "Chromium",     24,  3, 5),
    ("Mn", "Manganese",    25,  3, 6),  ("Fe", "Iron",         26,  3, 7),
    ("Co", "Cobalt",       27,  3, 8),  ("Ni", "Nickel",       28,  3, 9),
    ("Cu", "Copper",       29,  3, 10), ("Zn", "Zinc",         30,  3, 11),
    ("Ga", "Gallium",      31,  3, 12), ("Ge", "Germanium",    32,  3, 13),
    ("As", "Arsenic",      33,  3, 14), ("Se", "Selenium",     34,  3, 15),
    ("Br", "Bromine",      35,  3, 16), ("Kr", "Krypton",      36,  3, 17),
    ("Rb", "Rubidium",     37,  4, 0),  ("Sr", "Strontium",    38,  4, 1),
    ("Y",  "Yttrium",      39,  4, 2),  ("Zr", "Zirconium",    40,  4, 3),
    ("Nb", "Niobium",      41,  4, 4),  ("Mo", "Molybdenum",   42,  4, 5),
    ("Tc", "Technetium",   43,  4, 6),  ("Ru", "Ruthenium",    44,  4, 7),
    ("Rh", "Rhodium",      45,  4, 8),  ("Pd", "Palladium",    46,  4, 9),
    ("Ag", "Silver",       47,  4, 10), ("Cd", "Cadmium",      48,  4, 11),
    ("In", "Indium",       49,  4, 12), ("Sn", "Tin",          50,  4, 13),
    ("Sb", "Antimony",     51,  4, 14), ("Te", "Tellurium",    52,  4, 15),
    ("I",  "Iodine",       53,  4, 16), ("Xe", "Xenon",        54,  4, 17),
    ("Cs", "Caesium",      55,  5, 0),  ("Ba", "Barium",       56,  5, 1),
    ("La", "Lanthanum",    57,  8, 2),
    ("Hf", "Hafnium",      72,  5, 3),  ("Ta", "Tantalum",     73,  5, 4),
    ("W",  "Tungsten",     74,  5, 5),  ("Re", "Rhenium",      75,  5, 6),
    ("Os", "Osmium",       76,  5, 7),  ("Ir", "Iridium",      77,  5, 8),
    ("Pt", "Platinum",     78,  5, 9),  ("Au", "Gold",         79,  5, 10),
    ("Hg", "Mercury",      80,  5, 11), ("Tl", "Thallium",     81,  5, 12),
    ("Pb", "Lead",         82,  5, 13), ("Bi", "Bismuth",      83,  5, 14),
    ("Po", "Polonium",     84,  5, 15), ("At", "Astatine",     85,  5, 16),
    ("Rn", "Radon",        86,  5, 17),
    ("Fr", "Francium",     87,  6, 0),  ("Ra", "Radium",       88,  6, 1),
    ("Ac", "Actinium",     89,  9, 2),
    ("Rf", "Rutherfordium",104, 6, 3),  ("Db", "Dubnium",     105,  6, 4),
    ("Sg", "Seaborgium",   106, 6, 5),  ("Bh", "Bohrium",     107,  6, 6),
    ("Hs", "Hassium",      108, 6, 7),  ("Mt", "Meitnerium",  109,  6, 8),
    ("Ds", "Darmstadtium", 110, 6, 9),  ("Rg", "Roentgenium", 111,  6, 10),
    ("Cn", "Copernicium",  112, 6, 11), ("Nh", "Nihonium",    113,  6, 12),
    ("Fl", "Flerovium",    114, 6, 13), ("Mc", "Moscovium",   115,  6, 14),
    ("Lv", "Livermorium",  116, 6, 15), ("Ts", "Tennessine",  117,  6, 16),
    ("Og", "Oganesson",    118, 6, 17),
    # Lanthanides (row 8)
    ("Ce", "Cerium",       58,  8, 3),  ("Pr", "Praseodymium", 59, 8, 4),
    ("Nd", "Neodymium",    60,  8, 5),  ("Pm", "Promethium",   61, 8, 6),
    ("Sm", "Samarium",     62,  8, 7),  ("Eu", "Europium",     63, 8, 8),
    ("Gd", "Gadolinium",   64,  8, 9),  ("Tb", "Terbium",      65, 8, 10),
    ("Dy", "Dysprosium",   66,  8, 11), ("Ho", "Holmium",      67, 8, 12),
    ("Er", "Erbium",       68,  8, 13), ("Tm", "Thulium",      69, 8, 14),
    ("Yb", "Ytterbium",    70,  8, 15), ("Lu", "Lutetium",     71, 8, 16),
    # Actinides (row 9)
    ("Th", "Thorium",      90,  9, 3),  ("Pa", "Protactinium", 91, 9, 4),
    ("U",  "Uranium",      92,  9, 5),  ("Np", "Neptunium",    93, 9, 6),
    ("Pu", "Plutonium",    94,  9, 7),  ("Am", "Americium",    95, 9, 8),
    ("Cm", "Curium",       96,  9, 9),  ("Bk", "Berkelium",    97, 9, 10),
    ("Cf", "Californium",  98,  9, 11), ("Es", "Einsteinium",  99, 9, 12),
    ("Fm", "Fermium",     100,  9, 13), ("Md", "Mendelevium", 101,  9, 14),
    ("No", "Nobelium",    102,  9, 15), ("Lr", "Lawrencium",  103,  9, 16),
]

def _group_lines_by_element(wl_dict):
    """Parse keys like 'Ca II 393.366' and group them by element symbol."""
    import re
    groups = {}
    for key, val in wl_dict.items():
        m = re.match(r'^([A-Z][a-z]?)', key)
        if m:
            sym = m.group(1)
            if sym not in groups:
                groups[sym] = []
            groups[sym].append((key, val))
    for sym in groups:
        groups[sym].sort(key=lambda kv: kv[0])
    return groups


class PeriodicTableDialog(QDialog):
    """A periodic table popup that lets the user pick a LIBS emission line."""

    def __init__(self, wavelength_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Periodic Table — LIBS Line Selection")
        self.setMinimumSize(950, 480)
        self.selected_line_key = None

        self._lines_by_element = _group_lines_by_element(wavelength_dict)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(2)

        # Lanthanide / Actinide labels
        lbl_ln = QLabel("Lanthanides")
        lbl_ln.setStyleSheet("font-size: 7pt; color: #666;")
        lbl_ln.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(lbl_ln, 8, 0, 1, 2)
        lbl_ac = QLabel("Actinides")
        lbl_ac.setStyleSheet("font-size: 7pt; color: #666;")
        lbl_ac.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(lbl_ac, 9, 0, 1, 2)

        # Spacer row between main table and lanthanides/actinides
        grid.setRowMinimumHeight(7, 12)

        for sym, name, z, row, col in _PERIODIC_TABLE:
            btn = QPushButton(sym)
            btn.setFixedSize(48, 38)
            btn.setToolTip(f"{z} {name}")

            lines = self._lines_by_element.get(sym)
            n = len(lines) if lines else 0

            if n > 0:
                btn.setStyleSheet(
                    "QPushButton { background-color: #2980b9; color: white; "
                    "font-weight: bold; font-size: 9pt; border: 1px solid #1a5276; border-radius: 3px; }"
                    "QPushButton:hover { background-color: #3498db; }"
                )
                btn.setToolTip(f"{z} {name}\n{n} LIBS line(s)")
                btn.clicked.connect(lambda checked, s=sym: self._show_lines_menu(s))
            else:
                btn.setStyleSheet(
                    "QPushButton { background-color: #d5d8dc; color: #666; "
                    "font-size: 9pt; border: 1px solid #bbb; border-radius: 3px; }"
                )
                btn.setEnabled(False)

            grid.addWidget(btn, row, col)

        layout.addLayout(grid)

        # Legend
        legend = QHBoxLayout()
        legend.addStretch()
        for color, text in [("#2980b9", "Has LIBS lines"), ("#d5d8dc", "No lines available")]:
            box = QLabel("  ")
            box.setFixedSize(14, 14)
            box.setStyleSheet(f"background-color: {color}; border: 1px solid #999; border-radius: 2px;")
            legend.addWidget(box)
            legend.addWidget(QLabel(text))
            legend.addSpacing(12)
        legend.addStretch()
        layout.addLayout(legend)

        # Warning
        warn = QLabel(
            "\u26A0  Always check that the selected LIBS line is clearly visible in the spectra. "
            "Interferences with nearby lines from other elements can produce incorrect maps."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet(
            "color: #b35900; background-color: #fff3e0; border: 1px solid #e6a817; "
            "border-radius: 4px; padding: 6px; font-size: 8pt;"
        )
        layout.addWidget(warn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _show_lines_menu(self, symbol):
        """Show a context menu listing available LIBS lines for this element."""
        lines = self._lines_by_element.get(symbol, [])
        if not lines:
            return

        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { font-size: 9pt; }"
            "QMenu::item { padding: 5px 24px; }"
            "QMenu::item:selected { background-color: #2980b9; color: white; }"
        )
        for key, val in lines:
            if isinstance(val, list):
                label = f"{key}  ({', '.join(f'{w:.3f}' for w in val)} nm)"
            else:
                label = f"{key}  ({val:.3f} nm)"
            action = menu.addAction(label)
            action.setData(key)

        chosen = menu.exec_(self.cursor().pos())
        if chosen:
            self.selected_line_key = chosen.data()
            self.accept()


class HypercubeExplorer(QMainWindow):
    # Path for persistent UI config
    _CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

    def _load_config(self) -> dict:
        try:
            with open(self._CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_config(self):
        cfg = {}
        # Window geometry
        cfg['window_width'] = self.width()
        cfg['window_height'] = self.height()
        # Splitter states
        if hasattr(self, 'imaging_splitter'):
            cfg['imaging_splitter'] = [int(s) for s in self.imaging_splitter.sizes()]
        if hasattr(self, 'science_splitter'):
            cfg['science_splitter'] = [int(s) for s in self.science_splitter.sizes()]
        if hasattr(self, 'science_left_splitter'):
            cfg['science_left_splitter'] = [int(s) for s in self.science_left_splitter.sizes()]
        try:
            with open(self._CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hypercube Explorer")
        self._cfg = self._load_config()
        self.resize(self._cfg.get('window_width', 1350), self._cfg.get('window_height', 880))

        # ----- state -----
        self.original_ds = None
        self.ds = None
        # Cube-wide baseline-corrected cache (None = not applied).
        # When non-None, it takes the place of self.original_ds as the source
        # for apply_normalization().
        self._baseline_ds = None
        self._baseline_method_applied = "None"

        # LIBS view state
        self.current_data_array = None
        self.colorbar = None        # Science LIBS colorbar
        self.map_colorbar = None    # Imaging colorbar
        self.subset_colorbar = None # Subset tab colorbar
        self.current_rgb_composite = None
        self.last_vmin = None
        self.last_vmax = None
        self.last_cmap_name = 'Viridis'
        self.last_band_label = ""

        # interactions (shared)
        self.shared_line = None
        self.is_drawing = False
        self.line_libs = None
        self.pixel_marker_libs = None
        self.current_spectrum = None
        self.current_pixel = None

        # Photo / reference image + homography
        self.photo_img = None
        self.photo_polygon = []
        self.calibration_active = False
        self.calib_pts_photo = []
        self.H_photo_to_libs = None
        self.H_libs_to_photo = None
        self.drag_calib_enabled = False
        self.dragging_idx = None
        self.line_img = None
        self.pixel_marker_img = None

        self.project = ProjectState(path=None, experiments=[], active_index=-1)
        self.loaded_cube_path = None
        self.loaded_photo_path = None
        self.active_mask = None
        self._mask_painting = False
        self._mask_poly_pts = []

        # Baseline inspector state
        self._bi_spectrum = None
        self._bi_wavelengths = None
        self._bi_pixel = None

        # Chemometrics (PCA/NMF/clustering) state
        self.chem_result = None
        self.chem_colorbar = None
        # SAM reference spectra: list of {'name', 'wl', 'sp'} dicts.
        # Kept across cube loads on purpose (cross-sample comparison).
        self.chem_sam_refs = []

        # ----- build UI -----
        self._build_menubar()
        self._build_central_ui()
        self._connect_ui()
        
        self._roi_refresh_lists()

    # ====== UI BUILDERS ======
    def _build_menubar(self):
        menubar = QMenuBar(self)
    
        # File
        file_menu = menubar.addMenu("File")
        self.actionOpen = QAction("Open LIBS cube…", self); file_menu.addAction(self.actionOpen)
        file_menu.addSeparator()
        self.actionExportUnified = QAction("Export…", self); file_menu.addAction(self.actionExportUnified)
    
        # Project
        proj_menu = menubar.addMenu("Project")
        self.actProjNew = QAction("New Project", self); proj_menu.addAction(self.actProjNew)
        self.actProjOpen = QAction("Open Project…", self); proj_menu.addAction(self.actProjOpen)
        proj_menu.addSeparator()
        self.actProjSave = QAction("Save Project", self); proj_menu.addAction(self.actProjSave)
        self.actProjSaveAs = QAction("Save Project As…", self); proj_menu.addAction(self.actProjSaveAs)
        proj_menu.addSeparator()
        self.actAddExperiment = QAction("Add Experiment (snapshot)…", self); proj_menu.addAction(self.actAddExperiment)

        self.actLoadExperiment = QAction("Load Experiment…", self); proj_menu.addAction(self.actLoadExperiment)
        proj_menu.addSeparator()
        self.actUpdateExperiment = QAction("Update Current Experiment", self); proj_menu.addAction(self.actUpdateExperiment)

        self.actRenameExperiment = QAction("Rename Experiment…", self); proj_menu.addAction(self.actRenameExperiment)
        self.actDeleteExperiment = QAction("Delete Experiment…", self); proj_menu.addAction(self.actDeleteExperiment)
    
        # Help
        help_menu = menubar.addMenu("Help")
        self.actHelp  = QAction("User Guide…", self); help_menu.addAction(self.actHelp)
        help_menu.addSeparator()
        self.actAbout = QAction("About…", self); help_menu.addAction(self.actAbout)

        self.setMenuBar(menubar); self.setStatusBar(QStatusBar(self))
        
        from PyQt5.QtWidgets import QLabel
        
        self.status_label = QLabel("Project: (no project) | Experiment: —")
        # Use a permanent widget so it stays visible while transient messages appear on the left
        self.statusBar().addPermanentWidget(self.status_label, 1)

    def _refresh_status_bar(self, transient_msg: str = None):
        proj = os.path.basename(self.project.path) if getattr(self, "project", None) and self.project.path else "(no project)"
        exp = "—"
        try:
            if self.project and 0 <= self.project.active_index < len(self.project.experiments):
                exp = self.project.experiments[self.project.active_index].name or "—"
        except Exception:
            pass
        self.status_label.setText(f"Project: {proj} | Experiment: {exp}")
        if transient_msg:
            self.statusBar().showMessage(transient_msg, 3000)
            

    def _build_central_ui(self):
        central = QWidget(self); self.setCentralWidget(central)
        main_v = QVBoxLayout(central)
        self.tabs = QTabWidget(self); main_v.addWidget(self.tabs)

        self._build_tab_load()
        self._build_tab_photo()
        self._build_tab_mask()
        self._build_tab_normalize()
        self._build_tab_mapexplorer()
        self._build_tab_extraction()
        self._build_tab_chemometrics()
        self._build_tab_rgb()
        self._build_tab_export()
        self._build_tab_cube_utils()

    # --- Load tab ---
    def _build_tab_load(self):
        self.tabLoad = QWidget(self)
        layout = QVBoxLayout(self.tabLoad)
        layout.setContentsMargins(8, 8, 8, 8)

        self.load_file_label = QLabel("No file loaded — use File → Open LIBS cube")
        self.load_file_label.setStyleSheet("color: #555; font-size: 9pt; padding: 4px;")
        layout.addWidget(self.load_file_label)

        layout.addWidget(QLabel("NetCDF metadata", alignment=Qt.AlignHCenter))
        self.metadata_tree = QTreeWidget()
        self.metadata_tree.setColumnCount(2)
        self.metadata_tree.setHeaderLabels(["Key", "Value"])
        header = self.metadata_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.metadata_tree.setAlternatingRowColors(True)
        layout.addWidget(self.metadata_tree, stretch=1)

        self.tabs.addTab(self.tabLoad, "1. Info")
        self._clear_metadata_tree("No NetCDF file loaded.")

    # --- Photo tab ---
    def _build_tab_photo(self):
        self.tabPhoto = QWidget(self)
        layout = QVBoxLayout(self.tabPhoto)

        row = QHBoxLayout()
        self.btn_load_photo = QPushButton("Load photo\u2026")
        row.addWidget(self.btn_load_photo)

        self.btn_poly_mode = QToolButton()
        self.btn_poly_mode.setText("Define scan polygon")
        self.btn_poly_mode.setCheckable(True)
        row.addWidget(self.btn_poly_mode)
        self.btn_poly_clear = QPushButton("Clear polygon")
        row.addWidget(self.btn_poly_clear)

        self.btn_calib_start = QPushButton("Start 4-point calibration")
        row.addWidget(self.btn_calib_start)
        self.btn_calib_clear = QPushButton("Clear calibration")
        row.addWidget(self.btn_calib_clear)

        self.chk_drag_calib = QCheckBox("Drag calib points")
        self.chk_drag_calib.setChecked(False)
        row.addWidget(self.chk_drag_calib)

        row.addStretch()
        layout.addLayout(row)

        self.photo_canvas_tab = FigureCanvas(plt.Figure(dpi=100))
        self.photo_ax_tab = self.photo_canvas_tab.figure.add_subplot(111)
        layout.addWidget(self.photo_canvas_tab)
        self.photo_toolbar_tab = NavigationToolbar(self.photo_canvas_tab, self)
        layout.addWidget(self.photo_toolbar_tab)

        tip = QLabel("Calibration order: Top-Left \u2192 Top-Right \u2192 Bottom-Right \u2192 Bottom-Left")
        tip.setStyleSheet("color: gray;")
        layout.addWidget(tip)

        self.tabs.addTab(self.tabPhoto, "2. Photo")

    # --- Mask tab ---
    def _build_tab_mask(self):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

        self.tabMask = QWidget(self)
        outer = QHBoxLayout(self.tabMask)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # --- Left: Map with mask overlay ---
        map_widget = QWidget()
        map_lay = QVBoxLayout(map_widget)
        map_lay.setContentsMargins(0, 0, 4, 0)
        map_lay.setSpacing(2)

        self.mask_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.mask_ax = self.mask_canvas.figure.add_subplot(111)
        map_lay.addWidget(self.mask_canvas, stretch=1)
        self.mask_toolbar = NavigationToolbar(self.mask_canvas, self)
        map_lay.addWidget(self.mask_toolbar)
        self.mask_colorbar = None

        # --- Right: Tool sidebar (scrollable) ---
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QScrollArea.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_widget = QWidget()
        sidebar = QVBoxLayout(sidebar_widget)
        sidebar.setContentsMargins(2, 2, 2, 2)
        sidebar.setSpacing(4)

        # Band selector for map preview
        grpBand = QGroupBox("Band Preview")
        fb = QFormLayout(grpBand)
        fb.setContentsMargins(6, 14, 6, 6)
        fb.setSpacing(4)
        self.mask_band_slider = QSlider(Qt.Horizontal)
        fb.addRow("Band:", self.mask_band_slider)
        self.mask_band_label = QLabel("Band: -")
        self.mask_band_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        fb.addRow(self.mask_band_label)
        sidebar.addWidget(grpBand)

        # Threshold tools
        grpThresh = QGroupBox("Threshold Masking")
        ft = QFormLayout(grpThresh)
        ft.setContentsMargins(6, 14, 6, 6)
        ft.setSpacing(4)

        self.mask_thresh_spin = QDoubleSpinBox()
        self.mask_thresh_spin.setDecimals(2)
        self.mask_thresh_spin.setRange(-1e12, 1e12)
        self.mask_thresh_spin.setSingleStep(1.0)
        self.mask_thresh_spin.setValue(0.0)
        ft.addRow("Threshold:", self.mask_thresh_spin)

        self.mask_thresh_dir = QComboBox()
        self.mask_thresh_dir.addItems(["Below threshold", "Above threshold"])
        ft.addRow("Mask pixels:", self.mask_thresh_dir)

        self.mask_thresh_apply_btn = QPushButton("Apply threshold")
        ft.addRow(self.mask_thresh_apply_btn)
        sidebar.addWidget(grpThresh)

        # Paint tools
        grpPaint = QGroupBox("Paint Tools")
        fp = QFormLayout(grpPaint)
        fp.setContentsMargins(6, 14, 6, 6)
        fp.setSpacing(4)

        tool_row = QHBoxLayout()
        self.mask_rect_btn = QToolButton(); self.mask_rect_btn.setText("Rect")
        self.mask_rect_btn.setCheckable(True)
        self.mask_brush_btn = QToolButton(); self.mask_brush_btn.setText("Brush")
        self.mask_brush_btn.setCheckable(True)
        self.mask_poly_btn = QToolButton(); self.mask_poly_btn.setText("Polygon")
        self.mask_poly_btn.setCheckable(True)
        tool_row.addWidget(self.mask_rect_btn)
        tool_row.addWidget(self.mask_brush_btn)
        tool_row.addWidget(self.mask_poly_btn)
        fp.addRow("Tool:", tool_row)

        self.mask_brush_size_spin = QSpinBox()
        self.mask_brush_size_spin.setRange(1, 50)
        self.mask_brush_size_spin.setValue(3)
        fp.addRow("Brush size (px):", self.mask_brush_size_spin)

        self.mask_mode_combo = QComboBox()
        self.mask_mode_combo.addItems(["Mask", "Unmask"])
        fp.addRow("Mode:", self.mask_mode_combo)
        sidebar.addWidget(grpPaint)

        # Actions
        grpActions = QGroupBox("Actions")
        fa = QFormLayout(grpActions)
        fa.setContentsMargins(6, 14, 6, 6)
        fa.setSpacing(4)

        self.mask_clear_btn = QPushButton("Clear all")
        fa.addRow(self.mask_clear_btn)
        self.mask_invert_btn = QPushButton("Invert mask")
        fa.addRow(self.mask_invert_btn)

        self.mask_count_label = QLabel("Masked: 0 px")
        self.mask_count_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        fa.addRow(self.mask_count_label)

        btn_row = QHBoxLayout()
        self.mask_load_btn = QPushButton("Load mask")
        self.mask_save_btn = QPushButton("Save mask")
        btn_row.addWidget(self.mask_load_btn)
        btn_row.addWidget(self.mask_save_btn)
        fa.addRow(btn_row)
        sidebar.addWidget(grpActions)

        # Apply to cube
        grpApply = QGroupBox("Apply to Cube")
        fa2 = QVBoxLayout(grpApply)
        fa2.setContentsMargins(6, 14, 6, 6)
        fa2.setSpacing(4)
        self.mask_dirty_label = QLabel("")
        self.mask_dirty_label.setWordWrap(True)
        self.mask_dirty_label.setVisible(False)
        self.mask_dirty_label.setStyleSheet(
            "background-color: #fff3cd; color: #856404; font-size: 8pt;"
            " padding: 3px 6px; border: 1px solid #ffc107; border-radius: 3px;"
            " font-family: 'Segoe UI';")
        fa2.addWidget(self.mask_dirty_label)
        self.mask_apply_cube_btn = QPushButton("Apply mask to cube")
        self.mask_apply_cube_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white;"
            " font-weight: bold; padding: 6px; border-radius: 3px; }"
            " QPushButton:hover { background-color: #3498db; }")
        fa2.addWidget(self.mask_apply_cube_btn)
        sidebar.addWidget(grpApply)

        sidebar.addStretch()
        sidebar_scroll.setWidget(sidebar_widget)
        sidebar_scroll.setMinimumWidth(200)

        splitter.addWidget(map_widget)
        splitter.addWidget(sidebar_scroll)
        splitter.setSizes([900, 280])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        outer.addWidget(splitter)

        # Internal state for paint tools
        self._mask_painting = False
        self._mask_poly_pts = []
        self._mask_is_dirty = False

        self.tabs.addTab(self.tabMask, "3. Mask")

    # --- Normalize tab ---
    def _build_tab_normalize(self):
        self.tabNormalize = QWidget(self)
        outer = QHBoxLayout(self.tabNormalize)
        outer.setContentsMargins(8, 8, 8, 8)

        ctrl = QVBoxLayout()
        ctrl.setSpacing(4)

        # ===================== Baseline correction =====================
        grpBaseline = QGroupBox("Baseline correction (cube-wide)")
        fbl = QFormLayout(grpBaseline)
        fbl.setContentsMargins(6, 14, 6, 6)
        fbl.setSpacing(4)

        self.cube_baseline_combo = QComboBox()
        self.cube_baseline_combo.addItems(["None", "SNIP", "Rolling minimum", "AsLS"])
        self.cube_baseline_combo.setToolTip(
            "Cube-wide spectral baseline removal applied before normalization.\n"
            "SNIP: iterative lower envelope (LIBS/XRF standard).\n"
            "Rolling minimum: minimum filter + smoothing (fast, simple).\n"
            "AsLS: Asymmetric Least Squares (smooth, flexible; slower).")
        fbl.addRow("Method:", self.cube_baseline_combo)

        self.cube_bl_snip_iter_spin = QSpinBox()
        self.cube_bl_snip_iter_spin.setRange(1, 500)
        self.cube_bl_snip_iter_spin.setValue(40)
        self.cube_bl_snip_iter_spin.setToolTip(
            "SNIP iteration count. Larger values produce smoother baselines "
            "but erode narrow peaks if too large. Typical: 30–80.")
        self.cube_bl_snip_iter_label = QLabel("SNIP iterations:")
        fbl.addRow(self.cube_bl_snip_iter_label, self.cube_bl_snip_iter_spin)

        self.cube_bl_window_spin = QSpinBox()
        self.cube_bl_window_spin.setRange(3, 9999)
        self.cube_bl_window_spin.setSingleStep(2)
        self.cube_bl_window_spin.setValue(101)
        self.cube_bl_window_spin.setToolTip(
            "Rolling-minimum window in bands (forced odd). "
            "Should be wider than the broadest peak but narrower than baseline curvature.")
        self.cube_bl_window_label = QLabel("Window (bands):")
        fbl.addRow(self.cube_bl_window_label, self.cube_bl_window_spin)

        # AsLS parameters
        self.cube_bl_asls_lam_spin = QDoubleSpinBox()
        self.cube_bl_asls_lam_spin.setDecimals(1)
        self.cube_bl_asls_lam_spin.setRange(1.0, 12.0)
        self.cube_bl_asls_lam_spin.setSingleStep(0.5)
        self.cube_bl_asls_lam_spin.setValue(6.0)
        self.cube_bl_asls_lam_spin.setToolTip(
            "AsLS smoothness (as log10 λ). Higher = smoother baseline. Typical: 4–9.")
        self.cube_bl_asls_lam_label = QLabel("AsLS log10(λ):")
        fbl.addRow(self.cube_bl_asls_lam_label, self.cube_bl_asls_lam_spin)

        self.cube_bl_asls_p_spin = QDoubleSpinBox()
        self.cube_bl_asls_p_spin.setDecimals(4)
        self.cube_bl_asls_p_spin.setRange(0.0001, 0.5)
        self.cube_bl_asls_p_spin.setSingleStep(0.001)
        self.cube_bl_asls_p_spin.setValue(0.01)
        self.cube_bl_asls_p_spin.setToolTip(
            "AsLS asymmetry weight p (for points above the baseline). "
            "Smaller p pushes the baseline further below peaks. Typical: 0.001–0.05.")
        self.cube_bl_asls_p_label = QLabel("AsLS p:")
        fbl.addRow(self.cube_bl_asls_p_label, self.cube_bl_asls_p_spin)

        self.cube_bl_asls_iter_spin = QSpinBox()
        self.cube_bl_asls_iter_spin.setRange(1, 100)
        self.cube_bl_asls_iter_spin.setValue(10)
        self.cube_bl_asls_iter_spin.setToolTip(
            "AsLS iteration count. 10 is typical; rarely needs more than 20.")
        self.cube_bl_asls_iter_label = QLabel("AsLS iterations:")
        fbl.addRow(self.cube_bl_asls_iter_label, self.cube_bl_asls_iter_spin)

        self.cube_bl_clip_neg_chk = QCheckBox("Clip negatives to 0 after subtraction")
        self.cube_bl_clip_neg_chk.setChecked(True)
        self.cube_bl_clip_neg_chk.setToolTip(
            "LIBS emission intensities should be non-negative. "
            "Uncheck to preserve residual noise around zero.")
        fbl.addRow(self.cube_bl_clip_neg_chk)

        bl_btns = QHBoxLayout()
        self.cube_bl_apply_btn = QPushButton("Apply baseline")
        self.cube_bl_apply_btn.setStyleSheet(
            "QPushButton { background-color: #8e44ad; color: white; font-weight: bold;"
            " padding: 4px; border-radius: 3px; }"
            " QPushButton:hover { background-color: #9b59b6; }")
        self.cube_bl_reset_btn = QPushButton("Reset")
        self.cube_bl_reset_btn.setToolTip("Discard baseline correction and restore the original cube.")
        self.cube_bl_preview_btn = QPushButton("Preview…")
        self.cube_bl_preview_btn.setToolTip(
            "Show the computed baseline on a random pixel's spectrum without "
            "modifying the cube.")
        self.cube_bl_save_btn = QPushButton("Save cube…")
        self.cube_bl_save_btn.setToolTip(
            "Export the baseline-corrected cube as a new NetCDF file "
            "(without any subsequent normalization).")
        bl_btns.addWidget(self.cube_bl_apply_btn)
        bl_btns.addWidget(self.cube_bl_reset_btn)
        bl_btns.addWidget(self.cube_bl_preview_btn)
        bl_btns.addWidget(self.cube_bl_save_btn)
        fbl.addRow(bl_btns)

        self.cube_bl_status_label = QLabel("")
        self.cube_bl_status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 8pt;")
        fbl.addRow(self.cube_bl_status_label)

        ctrl.addWidget(grpBaseline)

        grpNorm = QGroupBox("Normalization")
        fn = QFormLayout(grpNorm)
        self.norm_form = fn
        fn.setContentsMargins(6, 14, 6, 6)
        fn.setSpacing(4)

        self.norm_combo = QComboBox()
        self.norm_combo.addItems([
            "None",
            "Total Emission (TEN)",
            "Total Area (TAN)",
            "Continuum window",
            "SNV (Standard Normal Variate)",
            "Max-norm per pixel",
            "Spatial median filter"
        ])
        fn.addRow("Method:", self.norm_combo)

        self.norm_cont_start = QDoubleSpinBox()
        self.norm_cont_start.setDecimals(2); self.norm_cont_start.setRange(0, 1e6)
        self.norm_cont_start.setValue(350.0)
        fn.addRow("Cont. start (nm):", self.norm_cont_start)

        self.norm_cont_end = QDoubleSpinBox()
        self.norm_cont_end.setDecimals(2); self.norm_cont_end.setRange(0, 1e6)
        self.norm_cont_end.setValue(355.0)
        fn.addRow("Cont. end (nm):", self.norm_cont_end)

        self.norm_kernel_spin = QSpinBox()
        self.norm_kernel_spin.setRange(3, 201); self.norm_kernel_spin.setSingleStep(2)
        self.norm_kernel_spin.setValue(21)
        fn.addRow("Kernel (px):", self.norm_kernel_spin)

        self.norm_spectro_edit = QLineEdit()
        self.norm_spectro_edit.setPlaceholderText("all — e.g. 1-3,5")
        self.norm_spectro_edit.setToolTip(
            "Restrict the channels used to compute the normalization statistics\n"
            "(total emission / area, SNV mean & std, per-pixel max) to specific\n"
            "spectrometers. The resulting factor is applied to the whole spectrum.\n"
            "Syntax: '1-3', '1,3,5', 'all'. Detected ranges are listed below."
        )
        fn.addRow("Spectrometers:", self.norm_spectro_edit)

        self.norm_spectro_info = QLabel("")
        self.norm_spectro_info.setStyleSheet("color: #555; font-size: 8pt;")
        self.norm_spectro_info.setWordWrap(True)
        fn.addRow(self.norm_spectro_info)

        norm_btns = QHBoxLayout()
        self.norm_apply_btn = QPushButton("Apply")
        self.norm_save_btn = QPushButton("Save cube...")
        self.norm_save_btn.setToolTip("Export the current normalized dataset as a new NetCDF file")
        norm_btns.addWidget(self.norm_apply_btn)
        norm_btns.addWidget(self.norm_save_btn)
        fn.addRow(norm_btns)

        self.norm_status_label = QLabel("")
        self.norm_status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 8pt;")
        fn.addRow(self.norm_status_label)

        ctrl.addWidget(grpNorm)

        info = QLabel(
            "<b>Pipeline:</b> baseline correction → normalization → (masking).<br><br>"
            "<b>Baseline methods:</b><br>"
            "<b>SNIP</b> - Iterative lower-envelope clipping in LLS-transformed space. "
            "Preserves narrow peaks; good default for LIBS/XRF.<br>"
            "<b>Rolling minimum</b> - Minimum filter over a band window, then smoothed. "
            "Fast but tends to erode peaks if the window is too small.<br>"
            "<b>AsLS</b> - Asymmetric Least Squares (Eilers &amp; Boelens). "
            "Smooth, flexible baseline controlled by λ (smoothness) and p "
            "(asymmetry). Slower than SNIP on very large cubes.<br><br>"
            "Use <b>Preview…</b> to test parameters on a random pixel before applying.<br><br>"
            "<b>Normalization methods:</b><br>"
            "<b>TEN</b> - Divide each pixel by its total emission (simple sum)<br>"
            "<b>TAN</b> - Divide each pixel by its total area (trapezoidal integration, accounts for non-uniform channel spacing)<br>"
            "<b>Continuum</b> - Divide by mean of a featureless window<br>"
            "<b>SNV</b> - Subtract mean, divide by std per pixel<br>"
            "<b>Max-norm</b> - Divide each pixel by its maximum<br>"
            "<b>Spatial median</b> - Divide each band by spatial median filter<br><br>"
            "<b>Spectrometers</b> - For TEN / TAN / SNV / Max-norm, the field restricts "
            "which spectrometers are used to <i>compute</i> the normalization factor "
            "(e.g. '1-3,5'); the factor is then applied to the whole spectrum. "
            "Useful to exclude a noisy or saturated spectrometer."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-size: 9pt; padding: 8px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 4px;")
        ctrl.addWidget(info)
        ctrl.addStretch()

        outer.addLayout(ctrl)
        self.tabs.addTab(self.tabNormalize, "4. Normalize")

    # --- Map Explorer tab ---
    def _build_tab_mapexplorer(self):
        self.tabMapExplorer = QWidget(self)
        outer = QHBoxLayout(self.tabMapExplorer)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        # ===================== QSplitter (resizable sidebar | map area) =====================
        self.imaging_splitter = QSplitter(Qt.Horizontal)
        self.imaging_splitter.setChildrenCollapsible(False)

        # --- Sidebar container (scrollable) ---
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QScrollArea.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_widget = QWidget()
        sidebar = QVBoxLayout(sidebar_widget)
        sidebar.setContentsMargins(2, 2, 2, 2)
        sidebar.setSpacing(4)

        # --- Group 1: Band Selection ---
        grpBand = QGroupBox("Band Selection")
        fb = QFormLayout(grpBand)
        fb.setContentsMargins(6, 14, 6, 6)
        fb.setSpacing(4)

        self.element_combo = QComboBox()
        self.element_combo.addItems(['None'] + list(element_wavelengths_common.keys()))
        fb.addRow("Element:", self.element_combo)

        self.element_combo_spec = QComboBox()
        self.element_combo_spec.addItems(['None'] + list(element_wavelengths_specialized.keys()))
        fb.addRow("Ca/Mg specialist:", self.element_combo_spec)

        self.periodic_table_btn = QPushButton("Periodic Table…")
        self.periodic_table_btn.setToolTip("Open the periodic table to browse and select LIBS lines")
        fb.addRow(self.periodic_table_btn)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(element_colormaps.keys())
        fb.addRow("Colormap:", self.colormap_combo)

        self.slider = QSlider(Qt.Horizontal)
        fb.addRow("Band index:", self.slider)

        self.wl_input_spin = QDoubleSpinBox()
        self.wl_input_spin.setDecimals(3)
        self.wl_input_spin.setRange(0, 1e6)
        self.wl_input_spin.setSingleStep(0.1)
        self.wl_input_spin.setToolTip("Enter a wavelength (nm) and press Enter to jump to the nearest band")
        fb.addRow("Go to λ (nm):", self.wl_input_spin)

        self.wavelength_label = QLabel("Wavelength: -")
        self.wavelength_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        fb.addRow(self.wavelength_label)

        self.index_label = QLabel("Index: -")
        self.index_label.setStyleSheet("color: #2c3e50;")
        fb.addRow(self.index_label)

        sidebar.addWidget(grpBand)

        # --- Group 2: Display ---
        grpDisp = QGroupBox("Display")
        fd = QFormLayout(grpDisp)
        fd.setContentsMargins(6, 14, 6, 6)
        fd.setSpacing(4)

        self.axes_units_combo = QComboBox()
        self.axes_units_combo.addItems(["mm", "µm"])
        fd.addRow("Axis units:", self.axes_units_combo)

        self.mm_per_px_spin = QDoubleSpinBox()
        self.mm_per_px_spin.setDecimals(5); self.mm_per_px_spin.setRange(1e-6, 1e6)
        self.mm_per_px_spin.setValue(0.1)
        fd.addRow("Pixel size (mm/px):", self.mm_per_px_spin)

        self.um_axes_checkbox = QCheckBox("Show axes")
        self.um_axes_checkbox.setChecked(True)
        fd.addRow(self.um_axes_checkbox)

        self.autoscale_checkbox = QCheckBox("Autoscale vmin/vmax")
        fd.addRow(self.autoscale_checkbox)

        self.pmin_spin = QDoubleSpinBox()
        self.pmin_spin.setDecimals(2); self.pmin_spin.setRange(0.0, 100.0)
        self.pmin_spin.setValue(0.5)
        fd.addRow("Low %:", self.pmin_spin)

        self.pmax_spin = QDoubleSpinBox()
        self.pmax_spin.setDecimals(2); self.pmax_spin.setRange(0.0, 100.0)
        self.pmax_spin.setValue(99.5)
        fd.addRow("High %:", self.pmax_spin)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setDecimals(2); self.vmin_spin.setRange(0, 65536)
        self.vmin_spin.setSingleStep(1); self.vmin_spin.setPrefix("vmin: ")
        self.vmin_spin.setValue(0)
        fd.addRow(self.vmin_spin)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setDecimals(2); self.vmax_spin.setRange(0, 65536)
        self.vmax_spin.setSingleStep(1); self.vmax_spin.setPrefix("vmax: ")
        self.vmax_spin.setValue(65535)
        fd.addRow(self.vmax_spin)

        sidebar.addWidget(grpDisp)

        # --- Group 3: Processing ---
        grpProc = QGroupBox("Processing")
        fp = QFormLayout(grpProc)
        fp.setContentsMargins(6, 14, 6, 6)
        fp.setSpacing(4)

        self.divide_checkbox = QCheckBox("Divide by denominator (ratio map)")
        self.divide_checkbox.setToolTip(
            "Compute a ratio map: numerator / denominator.\n"
            "Useful for Ca-normalised Mg/Ca, Sr/Ca, Ba/Ca etc. in speleothem LIBS.\n"
            "Local baseline settings (below) are applied to BOTH numerator and denominator.")
        fp.addRow(self.divide_checkbox)

        self.div_element_combo = QComboBox()
        self.div_element_combo.addItems(['None'] + list(element_wavelengths_common.keys()))
        self.div_element_combo.setToolTip("Denominator element / line")
        fp.addRow("Denominator:", self.div_element_combo)

        self.div_element_combo_spec = QComboBox()
        self.div_element_combo_spec.addItems(['None'] + list(element_wavelengths_specialized.keys()))
        self.div_element_combo_spec.setToolTip("Denominator — Ca/Mg specialist lines")
        fp.addRow("Ca/Mg specialist:", self.div_element_combo_spec)

        self.divider_spin = QDoubleSpinBox()
        self.divider_spin.setDecimals(3); self.divider_spin.setRange(0, 1e6)
        self.divider_spin.setSingleStep(0.1)
        self.divider_spin.setToolTip("Denominator wavelength (nm). Updated automatically when an element is selected above.")
        fp.addRow("Denominator λ (nm):", self.divider_spin)

        self.div_min_spin = QDoubleSpinBox()
        self.div_min_spin.setDecimals(2)
        self.div_min_spin.setRange(0.0, 1e9)
        self.div_min_spin.setValue(0.0)
        self.div_min_spin.setSingleStep(10.0)
        self.div_min_spin.setToolTip(
            "Mask pixels where the denominator is below this value (ratio becomes NaN).\n"
            "Use 0 to disable masking. Recommended: a small fraction of the median Ca signal "
            "to avoid divide-by-noise artefacts.")
        fp.addRow("Mask denom. <:", self.div_min_spin)

        self.div_scale_spin = QDoubleSpinBox()
        self.div_scale_spin.setDecimals(1)
        self.div_scale_spin.setRange(0.0001, 1e9)
        self.div_scale_spin.setValue(1000.0)
        self.div_scale_spin.setSingleStep(10.0)
        self.div_scale_spin.setToolTip(
            "Multiplicative scale applied to the ratio (handy for displaying mmol/mol).\n"
            "Set to 1 for a raw ratio.")
        fp.addRow("Scale factor ×:", self.div_scale_spin)

        self.baseline_checkbox = QCheckBox("Local baseline subtraction")
        fp.addRow(self.baseline_checkbox)

        self.baseline_method_combo = QComboBox()
        self.baseline_method_combo.addItems(["Peak height", "Peak area"])
        fp.addRow("Method:", self.baseline_method_combo)

        self.baseline_halfwidth_spin = QDoubleSpinBox()
        self.baseline_halfwidth_spin.setDecimals(3)
        self.baseline_halfwidth_spin.setRange(0.01, 5.00)
        self.baseline_halfwidth_spin.setSingleStep(0.05)
        self.baseline_halfwidth_spin.setValue(0.10)
        fp.addRow("Half-width (nm):", self.baseline_halfwidth_spin)

        self.baseline_gap_spin = QDoubleSpinBox()
        self.baseline_gap_spin.setDecimals(3)
        self.baseline_gap_spin.setRange(0.00, 1.00)
        self.baseline_gap_spin.setSingleStep(0.01)
        self.baseline_gap_spin.setValue(0.02)
        fp.addRow("Excl. ± (nm):", self.baseline_gap_spin)

        self.baseline_apply_btn = QPushButton("Apply baseline")
        fp.addRow(self.baseline_apply_btn)

        sidebar.addWidget(grpProc)

        self._locked_view = None

        sidebar.addStretch()
        sidebar_scroll.setWidget(sidebar_widget)
        sidebar_scroll.setMinimumWidth(200)

        # Hidden vmin/vmax sliders (kept for signal compatibility)
        self.vmin_slider = QSlider(Qt.Vertical)
        self.vmin_slider.setRange(0, 65535); self.vmin_slider.setValue(0)
        self.vmin_slider.setVisible(False)
        self.vmax_slider = QSlider(Qt.Vertical)
        self.vmax_slider.setRange(0, 65535); self.vmax_slider.setValue(65535)
        self.vmax_slider.setVisible(False)

        # ===================== RIGHT: MAP + HISTOGRAM =====================
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(4, 0, 0, 0)
        right.setSpacing(4)

        self.map_low_signal_label = QLabel("")
        self.map_low_signal_label.setWordWrap(True)
        self.map_low_signal_label.setStyleSheet(
            "background-color: #ffcccc; color: #990000; font-weight: bold;"
            " font-size: 8pt; padding: 4px 8px; border: 1px solid #cc0000;"
            " border-radius: 3px; font-family: 'Segoe UI';")
        self.map_low_signal_label.setVisible(False)
        right.addWidget(self.map_low_signal_label)

        self.map_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.map_ax = self.map_canvas.figure.add_subplot(111)
        right.addWidget(self.map_canvas, stretch=1)
        self.map_toolbar = NavigationToolbar(self.map_canvas, self)
        right.addWidget(self.map_toolbar)

        # Histogram
        self.hist_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.hist_ax = self.hist_canvas.figure.add_subplot(111)
        self.hist_canvas.setFixedHeight(100)
        right.addWidget(self.hist_canvas)

        # Add both sides to the splitter
        self.imaging_splitter.addWidget(sidebar_scroll)
        self.imaging_splitter.addWidget(right_widget)

        # Restore saved splitter position, or default (280 | rest)
        saved = self._cfg.get('imaging_splitter')
        if saved and len(saved) == 2:
            self.imaging_splitter.setSizes(saved)
        else:
            self.imaging_splitter.setSizes([280, 1070])

        self.imaging_splitter.setStretchFactor(0, 0)  # sidebar: fixed feel
        self.imaging_splitter.setStretchFactor(1, 1)  # map area: stretches

        outer.addWidget(self.imaging_splitter)

        self.tabs.addTab(self.tabMapExplorer, "5. Map Explorer")

    
    def _build_tab_extraction(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

        self.tabExtraction = QWidget(self)
        outer = QHBoxLayout(self.tabExtraction)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        # ===================== QSplitter =====================
        self.science_splitter = QSplitter(Qt.Horizontal)
        self.science_splitter.setChildrenCollapsible(False)

        # -------- LEFT: LIBS image in vertical splitter --------
        left_widget = QWidget()
        left_col = QVBoxLayout(left_widget)
        left_col.setContentsMargins(0, 0, 4, 0)
        left_col.setSpacing(0)

        self.science_left_splitter = QSplitter(Qt.Vertical)
        self.science_left_splitter.setChildrenCollapsible(False)

        # LIBS image panel
        libs_panel = QWidget()
        libs_lay = QVBoxLayout(libs_panel)
        libs_lay.setContentsMargins(0, 0, 0, 0)
        libs_lay.setSpacing(2)
        lbl_libs = QLabel("LIBS image")
        lbl_libs.setAlignment(Qt.AlignHCenter)
        lbl_libs.setObjectName("sectionLabel")
        libs_lay.addWidget(lbl_libs)
        self.canvas = FigureCanvas(plt.Figure(dpi=100))
        self.ax = self.canvas.figure.add_subplot(111)
        libs_lay.addWidget(self.canvas, stretch=1)
        self.toolbar = NavigationToolbar(self.canvas, self)
        libs_lay.addWidget(self.toolbar)

        # Coordinate readouts anchored below the LIBS image
        coord_row = QHBoxLayout()
        coord_row.setContentsMargins(4, 0, 4, 2)
        self.lbl_line_coords = QLabel("Line: \u2014")
        self.lbl_line_coords.setStyleSheet("color: #2c3e50; font-size: 8pt;")
        self.lbl_pixel_coords = QLabel("Pixel: \u2014")
        self.lbl_pixel_coords.setStyleSheet("color: #2c3e50; font-size: 8pt;")
        coord_row.addWidget(self.lbl_line_coords)
        coord_row.addWidget(self.lbl_pixel_coords)
        libs_lay.addLayout(coord_row)

        self.science_left_splitter.addWidget(libs_panel)
        self.science_left_splitter.setStretchFactor(0, 1)

        # Photo image panel (optional, shown when photo is loaded)
        photo_panel = QWidget()
        photo_lay = QVBoxLayout(photo_panel)
        photo_lay.setContentsMargins(0, 0, 0, 0)
        photo_lay.setSpacing(2)
        lbl_photo = QLabel("Photo (reference)")
        lbl_photo.setAlignment(Qt.AlignHCenter)
        lbl_photo.setObjectName("sectionLabel")
        photo_lay.addWidget(lbl_photo)
        self.photo_canvas_sci = FigureCanvas(plt.Figure(dpi=100))
        self.photo_ax_sci = self.photo_canvas_sci.figure.add_subplot(111)
        photo_lay.addWidget(self.photo_canvas_sci, stretch=1)
        self.photo_toolbar_sci = NavigationToolbar(self.photo_canvas_sci, self)
        photo_lay.addWidget(self.photo_toolbar_sci)

        self.science_left_splitter.addWidget(photo_panel)

        left_col.addWidget(self.science_left_splitter)

        # -------- RIGHT: scrollable sidebar (Tools + ROIs) + plot --------
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)

        # -- Mini tab widget: Tools | ROIs --
        self.science_settings_tabs = QTabWidget()
        self.science_settings_tabs.setMinimumWidth(200)

        # ===== Tab 1: Tools (scrollable) =====
        tools_scroll = QScrollArea()
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setFrameShape(QScrollArea.NoFrame)
        tools_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tools_inner = QWidget()
        tools_lay = QVBoxLayout(tools_inner)
        tools_lay.setContentsMargins(2, 2, 2, 2)
        tools_lay.setSpacing(4)

        # --- Group: Drawing Tools ---
        grpDraw = QGroupBox("Drawing Tools")
        fd = QFormLayout(grpDraw)
        fd.setContentsMargins(6, 14, 6, 6)
        fd.setSpacing(4)

        mode_row = QHBoxLayout()
        self.line_mode_button = QToolButton()
        self.line_mode_button.setText("Line mode")
        self.line_mode_button.setCheckable(True)
        mode_row.addWidget(self.line_mode_button)
        self.pixel_mode_button = QToolButton()
        self.pixel_mode_button.setText("Pixel mode")
        self.pixel_mode_button.setCheckable(True)
        mode_row.addWidget(self.pixel_mode_button)
        fd.addRow(mode_row)

        self.ignore_null_checkbox = QCheckBox("Ignore zeros")
        fd.addRow(self.ignore_null_checkbox)

        self.parallel_line_spinbox = QSpinBox()
        self.parallel_line_spinbox.setRange(0, 50)
        fd.addRow("Parallel buffer:", self.parallel_line_spinbox)

        self.nan_tolerance_spin = QSpinBox()
        self.nan_tolerance_spin.setRange(0, 100)
        self.nan_tolerance_spin.setValue(5)
        self.nan_tolerance_spin.valueChanged.connect(self.update_line_plot)
        fd.addRow("NaN tolerance:", self.nan_tolerance_spin)

        tools_lay.addWidget(grpDraw)

        # --- Group: Peak Detection ---
        grpPeak = QGroupBox("Peak Detection")
        fp = QFormLayout(grpPeak)
        fp.setContentsMargins(6, 14, 6, 6)
        fp.setSpacing(4)

        self.peak_detection_checkbox = QCheckBox("Detect peaks (pixel spectra)")
        fp.addRow(self.peak_detection_checkbox)

        self.prominence_spinbox = QDoubleSpinBox()
        self.prominence_spinbox.setDecimals(3)
        self.prominence_spinbox.setRange(0, 1e9)
        self.prominence_spinbox.setSingleStep(0.1)
        fp.addRow("Prominence:", self.prominence_spinbox)

        self.distance_spinbox = QSpinBox()
        self.distance_spinbox.setRange(1, 1_000_000)
        fp.addRow("Min distance (pts):", self.distance_spinbox)

        tools_lay.addWidget(grpPeak)

        tools_lay.addStretch()

        tools_scroll.setWidget(tools_inner)
        self.science_settings_tabs.addTab(tools_scroll, "Tools")

        # ===== Tab 2: ROIs (scrollable) =====
        roi_scroll = QScrollArea()
        roi_scroll.setWidgetResizable(True)
        roi_scroll.setFrameShape(QScrollArea.NoFrame)
        roi_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        roi_inner = QWidget()
        roi_lay = QVBoxLayout(roi_inner)
        roi_lay.setContentsMargins(2, 2, 2, 2)
        roi_lay.setSpacing(4)

        self.chk_use_project_rois = QCheckBox("Use project-shared ROIs")
        self.chk_use_project_rois.setChecked(True)
        roi_lay.addWidget(self.chk_use_project_rois)

        # Lines
        roi_lay.addWidget(QLabel("Lines:"))
        self.list_lines = QListWidget()
        self.list_lines.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_lines.setMinimumHeight(60)
        roi_lay.addWidget(self.list_lines)

        line_btns = QHBoxLayout()
        self.btn_add_line = QPushButton("Add")
        self.btn_add_line.setToolTip("Add current line as ROI")
        self.btn_use_line = QPushButton("Use")
        self.btn_use_line.setToolTip("Restore selected line")
        self.btn_rename_line = QPushButton("Rename")
        self.btn_remove_line = QPushButton("Remove")
        for b in (self.btn_add_line, self.btn_use_line, self.btn_rename_line, self.btn_remove_line):
            line_btns.addWidget(b)
        roi_lay.addLayout(line_btns)

        custom_grp = QGroupBox("Custom line position")
        custom_form = QFormLayout(custom_grp)
        custom_form.setContentsMargins(6, 14, 6, 6)
        custom_form.setSpacing(3)
        coord_style = "font-size: 8pt;"
        self.roi_x0_spin = QSpinBox()
        self.roi_x0_spin.setRange(0, 99999); self.roi_x0_spin.setStyleSheet(coord_style)
        custom_form.addRow("X start (px):", self.roi_x0_spin)
        self.roi_y0_spin = QSpinBox()
        self.roi_y0_spin.setRange(0, 99999); self.roi_y0_spin.setStyleSheet(coord_style)
        custom_form.addRow("Y start (px):", self.roi_y0_spin)
        self.roi_x1_spin = QSpinBox()
        self.roi_x1_spin.setRange(0, 99999); self.roi_x1_spin.setStyleSheet(coord_style)
        custom_form.addRow("X end (px):", self.roi_x1_spin)
        self.roi_y1_spin = QSpinBox()
        self.roi_y1_spin.setRange(0, 99999); self.roi_y1_spin.setStyleSheet(coord_style)
        custom_form.addRow("Y end (px):", self.roi_y1_spin)
        custom_btn_row = QHBoxLayout()
        self.btn_add_custom_line = QPushButton("Add custom line")
        self.btn_add_custom_line.setToolTip("Add a line ROI from the typed coordinates")
        self.btn_draw_custom_line = QPushButton("Draw")
        self.btn_draw_custom_line.setToolTip("Draw the custom line on the map without saving as ROI")
        custom_btn_row.addWidget(self.btn_add_custom_line)
        custom_btn_row.addWidget(self.btn_draw_custom_line)
        custom_form.addRow(custom_btn_row)
        roi_lay.addWidget(custom_grp)

        # Pixels
        roi_lay.addWidget(QLabel("Pixels:"))
        self.list_pixels = QListWidget()
        self.list_pixels.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_pixels.setMinimumHeight(60)
        roi_lay.addWidget(self.list_pixels)

        px_btns = QHBoxLayout()
        self.btn_add_pixel = QPushButton("Add")
        self.btn_add_pixel.setToolTip("Add current pixel as ROI")
        self.btn_use_pixel = QPushButton("Use")
        self.btn_use_pixel.setToolTip("Restore selected pixel")
        self.btn_rename_pixel = QPushButton("Rename")
        self.btn_remove_pixel = QPushButton("Remove")
        for b in (self.btn_add_pixel, self.btn_use_pixel, self.btn_rename_pixel, self.btn_remove_pixel):
            px_btns.addWidget(b)
        roi_lay.addLayout(px_btns)

        roi_lay.addStretch()
        roi_scroll.setWidget(roi_inner)
        self.science_settings_tabs.addTab(roi_scroll, "ROIs")

        # ===== Tab 3: Baseline Inspector (scrollable) =====
        bi_scroll = QScrollArea()
        bi_scroll.setWidgetResizable(True)
        bi_scroll.setFrameShape(QScrollArea.NoFrame)
        bi_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        bi_inner = QWidget()
        bi_lay = QVBoxLayout(bi_inner)
        bi_lay.setContentsMargins(2, 2, 2, 2)
        bi_lay.setSpacing(4)

        grpBILoad = QGroupBox("Spectrum")
        fbi = QFormLayout(grpBILoad)
        fbi.setContentsMargins(6, 14, 6, 6)
        fbi.setSpacing(4)

        self.bi_load_btn = QPushButton("Load current pixel spectrum")
        self.bi_load_btn.setToolTip("Copy the currently selected pixel spectrum into the baseline inspector")
        fbi.addRow(self.bi_load_btn)

        self.bi_pixel_label = QLabel("No spectrum loaded")
        self.bi_pixel_label.setStyleSheet("color: #555; font-size: 8pt;")
        fbi.addRow(self.bi_pixel_label)
        bi_lay.addWidget(grpBILoad)

        grpBISettings = QGroupBox("Local Baseline Settings")
        fbs = QFormLayout(grpBISettings)
        fbs.setContentsMargins(6, 14, 6, 6)
        fbs.setSpacing(4)

        self.bi_center_spin = QDoubleSpinBox()
        self.bi_center_spin.setDecimals(3)
        self.bi_center_spin.setRange(0, 1e6)
        self.bi_center_spin.setSingleStep(0.1)
        self.bi_center_spin.setToolTip("Center wavelength of the peak of interest")
        fbs.addRow("Center λ (nm):", self.bi_center_spin)

        self.bi_hw_spin = QDoubleSpinBox()
        self.bi_hw_spin.setDecimals(3)
        self.bi_hw_spin.setRange(0.01, 5.0)
        self.bi_hw_spin.setSingleStep(0.01)
        self.bi_hw_spin.setValue(0.10)
        fbs.addRow("Half-width (nm):", self.bi_hw_spin)

        self.bi_gap_spin = QDoubleSpinBox()
        self.bi_gap_spin.setDecimals(3)
        self.bi_gap_spin.setRange(0.0, 1.0)
        self.bi_gap_spin.setSingleStep(0.01)
        self.bi_gap_spin.setValue(0.02)
        fbs.addRow("Excl. ± (nm):", self.bi_gap_spin)

        bi_lay.addWidget(grpBISettings)

        grpBIZoom = QGroupBox("View Range")
        fbz = QFormLayout(grpBIZoom)
        fbz.setContentsMargins(6, 14, 6, 6)
        fbz.setSpacing(4)

        self.bi_view_range_spin = QDoubleSpinBox()
        self.bi_view_range_spin.setDecimals(2)
        self.bi_view_range_spin.setRange(0.1, 50.0)
        self.bi_view_range_spin.setSingleStep(0.1)
        self.bi_view_range_spin.setValue(1.0)
        self.bi_view_range_spin.setToolTip("Spectral window ± around center wavelength (nm)")
        fbz.addRow("± window (nm):", self.bi_view_range_spin)

        bi_lay.addWidget(grpBIZoom)

        self.bi_update_btn = QPushButton("Update")
        self.bi_update_btn.setStyleSheet("font-weight:bold; padding:4px;")
        bi_lay.addWidget(self.bi_update_btn)

        grpBIResults = QGroupBox("Results")
        fbr = QFormLayout(grpBIResults)
        fbr.setContentsMargins(6, 14, 6, 6)
        fbr.setSpacing(4)

        self.bi_height_label = QLabel("Peak height: —")
        self.bi_height_label.setStyleSheet("font-size: 9pt;")
        fbr.addRow(self.bi_height_label)

        self.bi_area_label = QLabel("Peak area: —")
        self.bi_area_label.setStyleSheet("font-size: 9pt;")
        fbr.addRow(self.bi_area_label)

        self.bi_ratio_label = QLabel("Area / Height: —")
        self.bi_ratio_label.setStyleSheet("font-size: 9pt; color: #555;")
        fbr.addRow(self.bi_ratio_label)

        bi_lay.addWidget(grpBIResults)

        bi_lay.addStretch()
        bi_scroll.setWidget(bi_inner)
        self.science_settings_tabs.addTab(bi_scroll, "Baseline Inspector")

        # -- Plot area: tabbed (Spectrum/Line + Peak Extraction) --
        self.plot_tabs = QTabWidget()

        # Tab 1: existing spectrum / line plot
        plot_widget = QWidget()
        plot_lay = QVBoxLayout(plot_widget)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.setSpacing(2)

        self.line_plot_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.line_ax = self.line_plot_canvas.figure.add_subplot(111)
        plot_lay.addWidget(self.line_plot_canvas, stretch=1)

        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(4)
        self.line_toolbar = NavigationToolbar(self.line_plot_canvas, self)
        toolbar_row.addWidget(self.line_toolbar)
        toolbar_row.addStretch()
        self.save_button = QPushButton("Save Plot Data")
        self.erase_button = QPushButton("Erase Line")
        toolbar_row.addWidget(self.save_button)
        toolbar_row.addWidget(self.erase_button)
        plot_lay.addLayout(toolbar_row)

        self.plot_tabs.addTab(plot_widget, "Spectrum / Line")

        # Tab 2: Baseline Inspector plot
        bi_plot_widget = QWidget()
        bi_plot_lay = QVBoxLayout(bi_plot_widget)
        bi_plot_lay.setContentsMargins(0, 0, 0, 0)
        bi_plot_lay.setSpacing(2)

        self.bi_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.bi_ax = self.bi_canvas.figure.add_subplot(111)
        bi_plot_lay.addWidget(self.bi_canvas, stretch=1)

        bi_toolbar_row = QHBoxLayout()
        bi_toolbar_row.setSpacing(4)
        self.bi_toolbar = NavigationToolbar(self.bi_canvas, self)
        bi_toolbar_row.addWidget(self.bi_toolbar)
        bi_plot_lay.addLayout(bi_toolbar_row)

        self.plot_tabs.addTab(bi_plot_widget, "Baseline Inspector")

        # Vertical splitter: settings tabs above, plot tabs below
        right_vsplit = QSplitter(Qt.Vertical)
        right_vsplit.setChildrenCollapsible(False)
        right_vsplit.addWidget(self.science_settings_tabs)
        right_vsplit.addWidget(self.plot_tabs)
        right_vsplit.setSizes([300, 450])
        right_vsplit.setStretchFactor(0, 1)
        right_vsplit.setStretchFactor(1, 0)

        right.addWidget(right_vsplit)

        # -------- Assemble into horizontal splitter --------
        self.science_splitter.addWidget(left_widget)
        self.science_splitter.addWidget(right_widget)

        # Restore saved splitter position, or default (images 60% | sidebar+plot 40%)
        saved = self._cfg.get('science_splitter')
        if saved and len(saved) == 2:
            self.science_splitter.setSizes(saved)
        else:
            self.science_splitter.setSizes([750, 600])

        self.science_splitter.setStretchFactor(0, 1)
        self.science_splitter.setStretchFactor(1, 1)

        outer.addWidget(self.science_splitter)

        self.tabs.addTab(self.tabExtraction, "6. Data Extraction")

    # --- Chemometrics tab (PCA / NMF / clustering) ---
    def _build_tab_chemometrics(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

        self.tabChem = QWidget(self)
        outer = QHBoxLayout(self.tabChem)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # -------- LEFT: component / cluster map --------
        map_widget = QWidget()
        map_lay = QVBoxLayout(map_widget)
        map_lay.setContentsMargins(0, 0, 4, 0)
        map_lay.setSpacing(2)

        lbl_map = QLabel("Component / cluster map")
        lbl_map.setAlignment(Qt.AlignHCenter)
        lbl_map.setObjectName("sectionLabel")
        map_lay.addWidget(lbl_map)

        self.chem_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.chem_ax = self.chem_canvas.figure.add_subplot(111)
        # Scroll area so a tall multi-map grid can extend beyond the screen
        self.chem_map_scroll = QScrollArea()
        self.chem_map_scroll.setWidgetResizable(True)
        self.chem_map_scroll.setFrameShape(QScrollArea.NoFrame)
        self.chem_map_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chem_map_scroll.setWidget(self.chem_canvas)
        map_lay.addWidget(self.chem_map_scroll, stretch=1)
        self.chem_toolbar = NavigationToolbar(self.chem_canvas, self)
        map_lay.addWidget(self.chem_toolbar)

        # -------- RIGHT: controls (top) + loadings plot (bottom) --------
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)

        # -- Mini tab widget: Analysis | SAM refs | Results (declutters the sidebar) --
        self.chem_settings_tabs = QTabWidget()
        self.chem_settings_tabs.setMinimumWidth(200)

        def _make_settings_tab(group):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            inner = QWidget()
            lay = QVBoxLayout(inner)
            lay.setContentsMargins(2, 2, 2, 2)
            lay.setSpacing(4)
            lay.addWidget(group)
            lay.addStretch()
            scroll.setWidget(inner)
            return scroll

        # --- Group: Analysis setup ---
        grpA = QGroupBox("Analysis")
        fa = QFormLayout(grpA)
        fa.setContentsMargins(6, 14, 6, 6)
        fa.setSpacing(4)

        self.chem_method_combo = QComboBox()
        self.chem_method_combo.addItems(["PCA", "NMF", "K-Means clustering",
                                         "SAM (Spectral Angle Mapper)",
                                         "Unmixing (NNLS)"])
        self.chem_method_combo.setToolTip(
            "PCA: orthogonal components, loadings show which peaks vary together (can be negative).\n"
            "NMF: additive non-negative 'endmember-like' spectra, often easier to read chemically.\n"
            "K-Means: hard segmentation into chemical domains; centroid spectra show what defines each domain.\n"
            "SAM: supervised — matches each pixel to user-provided reference spectra by spectral\n"
            "angle (intensity-invariant). Add references in the 'References' tab.\n"
            "Unmixing: supervised — models each pixel as a non-negative mixture of the reference\n"
            "spectra (NNLS) and maps the abundance fraction of each, plus a residual map."
        )
        fa.addRow("Method:", self.chem_method_combo)

        self.chem_ncomp_spin = QSpinBox()
        self.chem_ncomp_spin.setRange(2, 20)
        self.chem_ncomp_spin.setValue(5)
        fa.addRow("Components / clusters:", self.chem_ncomp_spin)

        self.chem_wl_min_spin = QDoubleSpinBox()
        self.chem_wl_min_spin.setDecimals(2); self.chem_wl_min_spin.setRange(0.0, 1e6)
        self.chem_wl_min_spin.setValue(0.0)
        fa.addRow("λ min (nm):", self.chem_wl_min_spin)

        self.chem_wl_max_spin = QDoubleSpinBox()
        self.chem_wl_max_spin.setDecimals(2); self.chem_wl_max_spin.setRange(0.0, 1e6)
        self.chem_wl_max_spin.setValue(1e6)
        fa.addRow("λ max (nm):", self.chem_wl_max_spin)

        self.chem_spectro_edit = QLineEdit()
        self.chem_spectro_edit.setPlaceholderText("all — e.g. 1-3,5")
        self.chem_spectro_edit.setToolTip(
            "Restrict the analysis to specific spectrometers (combined with the\n"
            "λ range above). Syntax: '1-3', '1,3,5', 'all'.\n"
            "Detected ranges are listed below."
        )
        fa.addRow("Spectrometers:", self.chem_spectro_edit)

        self.chem_spectro_info = QLabel("")
        self.chem_spectro_info.setStyleSheet("color: #555; font-size: 8pt;")
        self.chem_spectro_info.setWordWrap(True)
        fa.addRow(self.chem_spectro_info)

        self.chem_exclude_mask_cb = QCheckBox("Exclude masked pixels")
        self.chem_exclude_mask_cb.setChecked(True)
        fa.addRow(self.chem_exclude_mask_cb)

        self.chem_standardize_cb = QCheckBox("Standardize bands (z-score)")
        self.chem_standardize_cb.setToolTip(
            "Scale each band to unit variance before PCA/K-Means so weak lines\n"
            "can contribute as much as intense ones. Ignored for NMF."
        )
        fa.addRow(self.chem_standardize_cb)

        self.chem_max_fit_spin = QSpinBox()
        self.chem_max_fit_spin.setRange(1000, 10_000_000)
        self.chem_max_fit_spin.setSingleStep(10000)
        self.chem_max_fit_spin.setValue(50000)
        self.chem_max_fit_spin.setToolTip(
            "Model is fitted on a random subset of at most this many pixels\n"
            "(then applied to all pixels). Lower = faster on large maps."
        )
        fa.addRow("Max pixels for fit:", self.chem_max_fit_spin)

        self.chem_run_btn = QPushButton("Run analysis")
        self.chem_run_btn.setStyleSheet(self._CHEM_BTN_STYLE_IDLE)
        fa.addRow(self.chem_run_btn)

        sil_row = QHBoxLayout()
        self.chem_kmax_spin = QSpinBox()
        self.chem_kmax_spin.setRange(3, 20)
        self.chem_kmax_spin.setValue(10)
        self.chem_kmax_spin.setToolTip("Largest cluster count tested by the silhouette scan (k = 2 … this value)")
        sil_row.addWidget(QLabel("k ≤"))
        sil_row.addWidget(self.chem_kmax_spin)
        self.chem_sil_btn = QPushButton("Silhouette scan (suggest k)")
        self.chem_sil_btn.setToolTip(
            "Run K-Means for k = 2 … k max and score each clustering with the\n"
            "mean silhouette coefficient (computed on a pixel subsample).\n"
            "The k with the highest score is the suggested number of chemical domains.\n"
            "Results appear in the 'Silhouette' plot tab."
        )
        sil_row.addWidget(self.chem_sil_btn, stretch=1)
        fa.addRow("Suggest clusters:", sil_row)

        self.chem_status_label = QLabel("No analysis run yet.")
        self.chem_status_label.setStyleSheet("color: #555; font-size: 8pt;")
        self.chem_status_label.setWordWrap(True)
        fa.addRow(self.chem_status_label)

        self.chem_settings_tabs.addTab(_make_settings_tab(grpA), "Analysis")

        # --- Group: reference spectra (shared by SAM and NNLS unmixing) ---
        self.grpSAM = QGroupBox("Reference spectra (SAM / Unmixing)")
        fsam = QFormLayout(self.grpSAM)
        fsam.setContentsMargins(6, 14, 6, 6)
        fsam.setSpacing(4)

        self.chem_sam_list = QListWidget()
        self.chem_sam_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.chem_sam_list.setMaximumHeight(90)
        self.chem_sam_list.setToolTip(
            "Reference (endmember) spectra used by SAM and NNLS unmixing")
        fsam.addRow(self.chem_sam_list)

        sam_btns1 = QHBoxLayout()
        self.chem_sam_add_pixel_btn = QPushButton("Add pixel")
        self.chem_sam_add_pixel_btn.setToolTip(
            "Add the pixel spectrum currently selected in the Data Extraction tab")
        self.chem_sam_add_result_btn = QPushButton("Add last run")
        self.chem_sam_add_result_btn.setToolTip(
            "Add the centroids / components of the last K-Means or NMF run as references")
        sam_btns1.addWidget(self.chem_sam_add_pixel_btn)
        sam_btns1.addWidget(self.chem_sam_add_result_btn)
        fsam.addRow(sam_btns1)

        sam_btns2 = QHBoxLayout()
        self.chem_sam_load_btn = QPushButton("Load CSV…")
        self.chem_sam_load_btn.setToolTip(
            "Load references from CSV: first column wavelength (nm),\n"
            "one column per reference spectrum (same format as\n"
            "'Save component spectra (CSV)')")
        self.chem_sam_rename_btn = QPushButton("Rename")
        self.chem_sam_rename_btn.setToolTip("Rename the selected reference (or double-click it)")
        sam_btns2.addWidget(self.chem_sam_load_btn)
        sam_btns2.addWidget(self.chem_sam_rename_btn)
        fsam.addRow(sam_btns2)

        sam_btns3 = QHBoxLayout()
        self.chem_sam_remove_btn = QPushButton("Remove")
        self.chem_sam_clear_btn = QPushButton("Clear")
        sam_btns3.addWidget(self.chem_sam_remove_btn)
        sam_btns3.addWidget(self.chem_sam_clear_btn)
        fsam.addRow(sam_btns3)

        thresh_row = QHBoxLayout()
        self.chem_sam_thresh_spin = QDoubleSpinBox()
        self.chem_sam_thresh_spin.setDecimals(1)
        self.chem_sam_thresh_spin.setRange(0.1, 90.0)
        self.chem_sam_thresh_spin.setSingleStep(0.5)
        self.chem_sam_thresh_spin.setValue(10.0)
        self.chem_sam_thresh_spin.setToolTip(
            "Classification cutoff: pixels whose best spectral angle exceeds this\n"
            "value stay unclassified. 90° effectively disables the cutoff.\n"
            "After a SAM run, changing this re-applies instantly (no re-run needed).")
        thresh_row.addWidget(self.chem_sam_thresh_spin)
        self.chem_sam_auto_btn = QPushButton("Suggest (Otsu)")
        self.chem_sam_auto_btn.setToolTip(
            "Automatically suggest the cutoff from the best-angle histogram of the\n"
            "last SAM run (Otsu's method separates the 'matched' and 'unmatched'\n"
            "populations). Also plots the diagnostics in the 'Silhouette' tab.")
        thresh_row.addWidget(self.chem_sam_auto_btn)
        fsam.addRow("Max angle (°):", thresh_row)

        self.chem_settings_tabs.addTab(_make_settings_tab(self.grpSAM), "References")

        # --- Group: Results ---
        grpR = QGroupBox("Results")
        fr = QFormLayout(grpR)
        fr.setContentsMargins(6, 14, 6, 6)
        fr.setSpacing(4)

        self.chem_comp_combo = QComboBox()
        fr.addRow("Component:", self.chem_comp_combo)

        self.chem_grid_cb = QCheckBox("Show all component maps (grid)")
        self.chem_grid_cb.setToolTip(
            "Display every component/abundance map at once in a 2-column grid\n"
            "(PCA, NMF, SAM angles, unmixing). Click a map to select it.\n"
            "Scroll the map panel if the grid is taller than the window."
        )
        fr.addRow(self.chem_grid_cb)

        self.chem_npeaks_spin = QSpinBox()
        self.chem_npeaks_spin.setRange(0, 30)
        self.chem_npeaks_spin.setValue(8)
        self.chem_npeaks_spin.setToolTip("Number of strongest peaks to label on the loading spectrum")
        fr.addRow("Peaks to label:", self.chem_npeaks_spin)

        self.chem_peaks_label = QLabel("Top peaks: —")
        self.chem_peaks_label.setStyleSheet("color: #2c3e50; font-size: 8pt;")
        self.chem_peaks_label.setWordWrap(True)
        self.chem_peaks_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fr.addRow(self.chem_peaks_label)

        self.chem_export_btn = QPushButton("Save component spectra (CSV)")
        fr.addRow(self.chem_export_btn)

        self.chem_settings_tabs.addTab(_make_settings_tab(grpR), "Results")

        # -- Plot area: tabbed (Loadings | Score scatter) --
        self.chem_plot_tabs = QTabWidget()

        # Tab 1: loadings / centroid plot
        plot_widget = QWidget()
        plot_lay = QVBoxLayout(plot_widget)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.setSpacing(2)
        self.chem_plot_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.chem_plot_ax = self.chem_plot_canvas.figure.add_subplot(111)
        plot_lay.addWidget(self.chem_plot_canvas, stretch=1)
        self.chem_plot_toolbar = NavigationToolbar(self.chem_plot_canvas, self)
        plot_lay.addWidget(self.chem_plot_toolbar)
        self.chem_plot_tabs.addTab(plot_widget, "Loadings / centroids")

        # Tab 2: user-defined component-vs-component scatter
        scatter_widget = QWidget()
        scatter_lay = QVBoxLayout(scatter_widget)
        scatter_lay.setContentsMargins(0, 0, 0, 0)
        scatter_lay.setSpacing(2)

        sc_row = QHBoxLayout()
        sc_row.setContentsMargins(4, 2, 4, 0)
        sc_row.addWidget(QLabel("X:"))
        self.chem_scatter_x_combo = QComboBox()
        sc_row.addWidget(self.chem_scatter_x_combo, stretch=1)
        sc_row.addWidget(QLabel("Y:"))
        self.chem_scatter_y_combo = QComboBox()
        sc_row.addWidget(self.chem_scatter_y_combo, stretch=1)
        scatter_lay.addLayout(sc_row)

        self.chem_scatter_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.chem_scatter_ax = self.chem_scatter_canvas.figure.add_subplot(111)
        scatter_lay.addWidget(self.chem_scatter_canvas, stretch=1)
        self.chem_scatter_toolbar = NavigationToolbar(self.chem_scatter_canvas, self)
        scatter_lay.addWidget(self.chem_scatter_toolbar)
        self.chem_plot_tabs.addTab(scatter_widget, "Score scatter")

        # Tab 3: silhouette scan (K-Means cluster-count suggestion)
        sil_widget = QWidget()
        sil_lay = QVBoxLayout(sil_widget)
        sil_lay.setContentsMargins(0, 0, 0, 0)
        sil_lay.setSpacing(2)
        self.chem_sil_canvas = FigureCanvas(plt.Figure(dpi=100))
        sil_lay.addWidget(self.chem_sil_canvas, stretch=1)
        self.chem_sil_toolbar = NavigationToolbar(self.chem_sil_canvas, self)
        sil_lay.addWidget(self.chem_sil_toolbar)
        self.chem_plot_tabs.addTab(sil_widget, "Silhouette")

        right_vsplit = QSplitter(Qt.Vertical)
        right_vsplit.setChildrenCollapsible(False)
        right_vsplit.addWidget(self.chem_settings_tabs)
        right_vsplit.addWidget(self.chem_plot_tabs)
        right_vsplit.setSizes([320, 430])
        right.addWidget(right_vsplit)

        splitter.addWidget(map_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 650])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)

        # Signals
        self.chem_run_btn.clicked.connect(self._chem_run)
        self.chem_comp_combo.currentIndexChanged.connect(self._chem_on_component_changed)
        self.chem_npeaks_spin.valueChanged.connect(self._chem_on_component_changed)
        self.chem_export_btn.clicked.connect(self._chem_export_spectra)
        self.chem_grid_cb.toggled.connect(self._chem_update_map)
        self.chem_scatter_x_combo.currentIndexChanged.connect(self._chem_update_scatter)
        self.chem_scatter_y_combo.currentIndexChanged.connect(self._chem_update_scatter)
        self.chem_canvas.mpl_connect('button_press_event', self._chem_on_map_click)
        self.chem_sil_btn.clicked.connect(self._chem_silhouette_scan)
        self.chem_method_combo.currentTextChanged.connect(self._chem_on_method_changed)
        self.chem_sam_add_pixel_btn.clicked.connect(self._sam_add_pixel)
        self.chem_sam_add_result_btn.clicked.connect(self._sam_add_from_result)
        self.chem_sam_load_btn.clicked.connect(self._sam_load_csv)
        self.chem_sam_rename_btn.clicked.connect(self._sam_rename_selected)
        self.chem_sam_list.itemDoubleClicked.connect(lambda _item: self._sam_rename_selected())
        self.chem_sam_remove_btn.clicked.connect(self._sam_remove_selected)
        self.chem_sam_clear_btn.clicked.connect(self._sam_clear)
        self.chem_sam_thresh_spin.valueChanged.connect(self._sam_apply_threshold)
        self.chem_sam_auto_btn.clicked.connect(self._sam_auto_threshold)

        # Icons (drawn in code; standard style icons for the action buttons)
        style = self.style()
        self.chem_method_combo.setItemIcon(0, self._chem_icon('pca'))
        self.chem_method_combo.setItemIcon(1, self._chem_icon('nmf'))
        self.chem_method_combo.setItemIcon(2, self._chem_icon('kmeans'))
        self.chem_method_combo.setItemIcon(3, self._chem_icon('sam'))
        self.chem_method_combo.setItemIcon(4, self._chem_icon('unmix'))
        self.chem_run_btn.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.chem_sil_btn.setIcon(style.standardIcon(QStyle.SP_MessageBoxQuestion))
        self.chem_export_btn.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        self.chem_plot_tabs.setTabIcon(0, self._chem_icon('spectrum'))
        self.chem_plot_tabs.setTabIcon(1, self._chem_icon('scatter'))
        self.chem_plot_tabs.setTabIcon(2, self._chem_icon('silhouette'))
        self.chem_settings_tabs.setTabIcon(0, style.standardIcon(QStyle.SP_FileDialogDetailedView))
        self.chem_settings_tabs.setTabIcon(1, self._chem_icon('sam'))
        self.chem_settings_tabs.setTabIcon(2, self._chem_icon('kmeans'))

        self.tabs.addTab(self.tabChem, "7. Chemometrics")

    # --- Composite Overlay tab ---
    def _build_tab_rgb(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

        NUM_LAYERS = 5
        self.tabRGB = QWidget(self)
        outer = QHBoxLayout(self.tabRGB)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # ---- Left panel (scrollable) ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_inner = QWidget()
        ctrl = QVBoxLayout(left_inner)
        ctrl.setContentsMargins(2, 2, 2, 2)
        ctrl.setSpacing(4)

        ctrl.addWidget(QLabel("<b>Element Composite Overlay</b>", alignment=Qt.AlignHCenter))

        # Periodic table
        self.composite_ptable = PeriodicTableWidget(self)
        ctrl.addWidget(self.composite_ptable)

        # Active-layer selector + layer controls
        self.comp_active_layer = 0
        self.comp_layer_radios = []
        self.comp_layer_enables = []
        self.comp_layer_color_btns = []
        self.comp_layer_colors = [list(c) for c in COMPOSITE_DEFAULT_COLORS]
        self.comp_layer_labels = []
        self.comp_layer_lines = [None] * NUM_LAYERS
        self.comp_layer_opacity_sliders = []
        self.comp_layer_gain_spins = []
        self.comp_layer_auto_cbs = []
        self.comp_layer_min_spins = []
        self.comp_layer_max_spins = []
        self.comp_layer_baseline_cbs = []
        self.comp_layer_cbar_cbs = []
        self.comp_layer_clear_btns = []

        layers_grp = QGroupBox("Layers (click periodic table to assign)")
        layers_lay = QVBoxLayout(layers_grp)
        layers_lay.setSpacing(2)
        layers_lay.setContentsMargins(4, 14, 4, 4)

        from PyQt5.QtWidgets import QButtonGroup
        self.comp_layer_btn_group = QButtonGroup(self)
        self.comp_layer_btn_group.setExclusive(True)

        for i in range(NUM_LAYERS):
            r, g, b = COMPOSITE_DEFAULT_COLORS[i]
            row_w = QWidget()
            row_top = QHBoxLayout()
            row_top.setContentsMargins(0, 0, 0, 0)
            row_top.setSpacing(3)

            radio = QRadioButton()
            radio.setChecked(i == 0)
            radio.toggled.connect(lambda checked, idx=i: self._comp_set_active(idx) if checked else None)
            self.comp_layer_btn_group.addButton(radio, i)
            row_top.addWidget(radio)
            self.comp_layer_radios.append(radio)

            en_cb = QCheckBox()
            en_cb.setChecked(True)
            en_cb.setToolTip("Enable/disable this layer")
            row_top.addWidget(en_cb)
            self.comp_layer_enables.append(en_cb)

            col_btn = QPushButton()
            col_btn.setFixedSize(20, 20)
            col_btn.setStyleSheet(f"background:rgb({r},{g},{b});border:1px solid #888;border-radius:3px;")
            col_btn.setToolTip("Click to change colour")
            col_btn.clicked.connect(lambda _c=False, idx=i: self._comp_pick_color(idx))
            row_top.addWidget(col_btn)
            self.comp_layer_color_btns.append(col_btn)

            lbl = QLabel(f"Layer {i+1}: <i>none</i>")
            lbl.setStyleSheet("font-size:8pt;")
            lbl.setMinimumWidth(120)
            row_top.addWidget(lbl, stretch=1)
            self.comp_layer_labels.append(lbl)

            clear_btn = QToolButton()
            clear_btn.setText("\u2715")
            clear_btn.setFixedSize(18, 18)
            clear_btn.setToolTip("Clear this layer")
            clear_btn.clicked.connect(lambda _c=False, idx=i: self._comp_clear_layer(idx))
            row_top.addWidget(clear_btn)
            self.comp_layer_clear_btns.append(clear_btn)

            row_bot = QHBoxLayout()
            row_bot.setContentsMargins(24, 0, 0, 0)
            row_bot.setSpacing(3)

            row_bot.addWidget(QLabel("Op:"))
            opacity = QSlider(Qt.Horizontal)
            opacity.setRange(0, 100)
            opacity.setValue(80)
            opacity.setFixedWidth(70)
            opacity.setToolTip("Opacity 0–100 %")
            row_bot.addWidget(opacity)
            self.comp_layer_opacity_sliders.append(opacity)

            row_bot.addWidget(QLabel("G:"))
            gain = QDoubleSpinBox()
            gain.setDecimals(2)
            gain.setRange(0.0, 20.0)
            gain.setSingleStep(0.1)
            gain.setValue(1.0)
            gain.setFixedWidth(55)
            row_bot.addWidget(gain)
            self.comp_layer_gain_spins.append(gain)

            auto_cb = QCheckBox("Auto")
            auto_cb.setChecked(True)
            auto_cb.setToolTip("Auto-scale min/max from percentiles")
            row_bot.addWidget(auto_cb)
            self.comp_layer_auto_cbs.append(auto_cb)

            mn = QDoubleSpinBox()
            mn.setDecimals(1); mn.setRange(-1e9, 1e9); mn.setValue(0); mn.setFixedWidth(60)
            mn.setEnabled(False); mn.setPrefix("mn:")
            row_bot.addWidget(mn)
            self.comp_layer_min_spins.append(mn)

            mx = QDoubleSpinBox()
            mx.setDecimals(1); mx.setRange(-1e9, 1e9); mx.setValue(65535); mx.setFixedWidth(60)
            mx.setEnabled(False); mx.setPrefix("mx:")
            row_bot.addWidget(mx)
            self.comp_layer_max_spins.append(mx)

            auto_cb.toggled.connect(lambda checked, _mn=mn, _mx=mx: (
                _mn.setEnabled(not checked), _mx.setEnabled(not checked)
            ))

            bl_cb = QCheckBox("BL")
            bl_cb.setToolTip("Apply local baseline subtraction")
            row_bot.addWidget(bl_cb)
            self.comp_layer_baseline_cbs.append(bl_cb)

            cbar_cb = QCheckBox("CB")
            cbar_cb.setToolTip("Show colorbar for this layer")
            row_bot.addWidget(cbar_cb)
            self.comp_layer_cbar_cbs.append(cbar_cb)

            col_lay = QVBoxLayout(row_w)
            col_lay.setContentsMargins(0, 2, 0, 2)
            col_lay.setSpacing(0)
            col_lay.addLayout(row_top)
            col_lay.addLayout(row_bot)

            sep = QWidget()
            sep.setFixedHeight(1)
            sep.setStyleSheet("background:#444;")

            layers_lay.addWidget(row_w)
            if i < NUM_LAYERS - 1:
                layers_lay.addWidget(sep)

        ctrl.addWidget(layers_grp)

        # Shared settings
        settings_grp = QGroupBox("Settings")
        sg = QFormLayout(settings_grp)
        sg.setContentsMargins(6, 14, 6, 6)
        sg.setSpacing(3)

        self.comp_pmin_spin = QDoubleSpinBox()
        self.comp_pmin_spin.setDecimals(2); self.comp_pmin_spin.setRange(0, 100); self.comp_pmin_spin.setValue(0.5)
        sg.addRow("Auto low %:", self.comp_pmin_spin)

        self.comp_pmax_spin = QDoubleSpinBox()
        self.comp_pmax_spin.setDecimals(2); self.comp_pmax_spin.setRange(0, 100); self.comp_pmax_spin.setValue(99.5)
        sg.addRow("Auto high %:", self.comp_pmax_spin)

        self.comp_bl_method = QComboBox()
        self.comp_bl_method.addItems(["Peak height", "Peak area"])
        sg.addRow("Baseline method:", self.comp_bl_method)

        self.comp_bl_halfwidth = QDoubleSpinBox()
        self.comp_bl_halfwidth.setDecimals(3); self.comp_bl_halfwidth.setRange(0.01, 5.0)
        self.comp_bl_halfwidth.setSingleStep(0.05); self.comp_bl_halfwidth.setValue(0.10)
        sg.addRow("Baseline ±hw (nm):", self.comp_bl_halfwidth)

        self.comp_bl_gap = QDoubleSpinBox()
        self.comp_bl_gap.setDecimals(3); self.comp_bl_gap.setRange(0.0, 1.0)
        self.comp_bl_gap.setSingleStep(0.01); self.comp_bl_gap.setValue(0.02)
        sg.addRow("Baseline excl (nm):", self.comp_bl_gap)

        bg_row = QHBoxLayout()
        self.comp_bg_combo = QComboBox()
        self.comp_bg_combo.addItems(["Black", "White", "Gray"])
        bg_row.addWidget(QLabel("Background:"))
        bg_row.addWidget(self.comp_bg_combo)
        sg.addRow(bg_row)

        # Legend & scale-bar placement (inside / below / above the image)
        ov_row = QHBoxLayout()
        self.comp_overlay_pos_combo = QComboBox()
        self.comp_overlay_pos_combo.addItems(["Inside image", "Below image", "Above image"])
        self.comp_overlay_pos_combo.setToolTip(
            "Where to draw the element legend and scale bar.\n"
            "• Inside image  - overlaid on top of the composite (default)\n"
            "• Below image  - in the margin under the composite\n"
            "• Above image  - in the margin above the composite (under the title)")
        ov_row.addWidget(QLabel("Legend / scale:"))
        ov_row.addWidget(self.comp_overlay_pos_combo)
        sg.addRow(ov_row)

        ctrl.addWidget(settings_grp)

        self.comp_update_btn = QPushButton("Update composite")
        self.comp_update_btn.setStyleSheet("font-weight:bold;padding:6px;")
        ctrl.addWidget(self.comp_update_btn)

        # View position recording
        view_grp = QGroupBox("View Position")
        view_lay = QVBoxLayout(view_grp)
        view_lay.setContentsMargins(6, 14, 6, 6)
        view_lay.setSpacing(4)

        btn_row = QHBoxLayout()
        self.record_view_btn = QPushButton("Record view position")
        self.record_view_btn.setToolTip(
            "Capture current zoom/pan from the composite map.\n"
            "Applied to Map Explorer, Data Extraction, and Composite."
        )
        btn_row.addWidget(self.record_view_btn)

        self.clear_view_btn = QPushButton("Clear")
        self.clear_view_btn.setFixedWidth(70)
        self.clear_view_btn.setToolTip("Clear the recorded view position")
        self.clear_view_btn.setEnabled(False)
        btn_row.addWidget(self.clear_view_btn)
        view_lay.addLayout(btn_row)

        self.view_info_label = QLabel("No view recorded.")
        self.view_info_label.setStyleSheet("font-size:8pt; color:gray;")
        self.view_info_label.setWordWrap(True)
        view_lay.addWidget(self.view_info_label)
        ctrl.addWidget(view_grp)

        ctrl.addStretch()
        left_scroll.setWidget(left_inner)
        left_scroll.setMinimumWidth(420)

        outer.addWidget(left_scroll, stretch=2)

        # ---- Right: image ----
        right = QVBoxLayout()
        self.comp_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.comp_ax = self.comp_canvas.figure.add_subplot(111)
        self.comp_ax.set_title("Element Composite", fontsize=8)
        self.comp_ax.set_xticks([]); self.comp_ax.set_yticks([])
        self.comp_canvas.figure.patch.set_facecolor('black')
        self.comp_ax.set_facecolor('black')
        self._comp_legend_entries = []
        self._comp_overlay_artists = []
        self._comp_current_bg_value = 0.0
        self.comp_canvas.mpl_connect('draw_event', self._comp_on_draw)
        right.addWidget(self.comp_canvas, stretch=1)
        self.comp_toolbar = NavigationToolbar(self.comp_canvas, self)
        right.addWidget(self.comp_toolbar)

        self.comp_info_label = QLabel("Select elements from the periodic table, assign to layers, then Update.")
        self.comp_info_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.comp_info_label.setWordWrap(True)
        right.addWidget(self.comp_info_label)

        outer.addLayout(right, stretch=3)
        self.tabs.addTab(self.tabRGB, "8. Composite")

    # -- Composite helpers --

    def _comp_on_draw(self, event):
        """Called after every canvas draw — schedule an overlay refresh."""
        if getattr(self, '_comp_overlays_busy', False):
            return
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._comp_refresh_overlays)

    def _comp_refresh_overlays(self):
        """Remove old overlay artists, recompute, and blit new ones."""
        if getattr(self, '_comp_overlays_busy', False):
            return
        self._comp_overlays_busy = True
        try:
            self._comp_draw_overlays_inner()
        finally:
            self._comp_overlays_busy = False

    def _comp_draw_overlays(self):
        """Explicit call from _update_rgb_composite to draw overlays."""
        self._comp_overlays_busy = True
        try:
            self._comp_draw_overlays_inner()
        finally:
            self._comp_overlays_busy = False

    def _comp_draw_overlays_inner(self):
        from matplotlib.patches import Rectangle
        ax = self.comp_ax

        for art in getattr(self, '_comp_overlay_artists', []):
            try:
                art.remove()
            except Exception:
                pass
        self._comp_overlay_artists = []

        legend_entries = getattr(self, '_comp_legend_entries', [])

        renderer = self.comp_canvas.get_renderer()
        bbox_disp = ax.get_window_extent(renderer=renderer)
        ax_w_px = max(bbox_disp.width, 1)
        ax_h_px = max(bbox_disp.height, 1)

        _cfont = {'size': 12, 'weight': 'bold'}
        sq_px = 32.0
        box_w = sq_px / ax_w_px
        box_h = sq_px / ax_h_px

        # Choose overlay placement: inside (default), below or above the image.
        pos = self.comp_overlay_pos_combo.currentText() if hasattr(self, 'comp_overlay_pos_combo') else "Inside image"
        # Background-aware foreground colour for text/bars in the margin
        bg_val_curr = float(getattr(self, '_comp_current_bg_value', 0.0))
        margin_fg = 'black' if bg_val_curr > 0.5 else 'white'
        if pos == "Below image":
            outside = True
            y_legend = -0.02 - box_h
            y_scale_center = -0.02 - (5.0 / ax_h_px) / 2
        elif pos == "Above image":
            outside = True
            y_legend = 1.02
            y_scale_center = 1.02 + (5.0 / ax_h_px) / 2
        else:  # Inside image
            outside = False
            y_legend = 0.02
            y_scale_center = 0.03

        if legend_entries:
            gap_px = 6.0
            gap_x = gap_px / ax_w_px
            x0 = 0.02
            for idx, (sym, col) in enumerate(legend_entries):
                xx = x0 + idx * (box_w + gap_x)
                rc = [col[0] / 255.0, col[1] / 255.0, col[2] / 255.0]
                # The symbol text is always drawn ON TOP of the coloured swatch,
                # whether the legend is inside the image or in the figure margin.
                # So contrast must be computed against the swatch colour, never
                # against the figure background.
                lum = 0.299 * rc[0] + 0.587 * rc[1] + 0.114 * rc[2]
                txt_color = 'black' if lum > 0.5 else 'white'
                rect = Rectangle((xx, y_legend), box_w, box_h,
                    facecolor=rc, edgecolor='none',
                    transform=ax.transAxes, clip_on=False)
                ax.add_patch(rect)
                self._comp_overlay_artists.append(rect)
                t = ax.text(xx + box_w / 2, y_legend + box_h / 2, sym,
                    transform=ax.transAxes, ha='center', va='center',
                    fontdict=_cfont, color=txt_color)
                self._comp_overlay_artists.append(t)

        mm_per_px = float(self.mm_per_px_spin.value()) if hasattr(self, 'mm_per_px_spin') else 0
        if mm_per_px > 0:
            xlim = ax.get_xlim()
            visible_width_px = abs(xlim[1] - xlim[0])
            visible_width_mm = visible_width_px * mm_per_px

            nice_steps = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                          1, 2, 5, 10, 20, 50, 100, 200, 500]
            target = visible_width_mm * 0.2
            bar_mm = min(nice_steps, key=lambda s: abs(s - target))
            bar_frac = bar_mm / visible_width_mm

            bar_h_px = 5.0
            bar_h = bar_h_px / ax_h_px
            text_gap = 3.0 / ax_h_px

            x1 = 0.98
            x0_bar = x1 - bar_frac
            y_bar = y_scale_center

            # Scale-bar colour: in the margin it follows the background
            # foreground; inside the image it stays white (overlaid on dark
            # signal regions, which is the LIBS convention).
            bar_color = margin_fg if outside else 'white'
            text_color = margin_fg if outside else 'white'
            bar_rect = Rectangle((x0_bar, y_bar - bar_h / 2), bar_frac, bar_h,
                facecolor=bar_color, edgecolor='none',
                transform=ax.transAxes, clip_on=False)
            ax.add_patch(bar_rect)
            self._comp_overlay_artists.append(bar_rect)

            if bar_mm >= 1:
                label = f"{bar_mm:.0f} mm"
            else:
                label = f"{bar_mm * 1000:.0f} µm"
            t = ax.text((x0_bar + x1) / 2, y_bar + bar_h / 2 + text_gap, label,
                transform=ax.transAxes, ha='center', va='bottom',
                fontdict=_cfont, color=text_color)
            self._comp_overlay_artists.append(t)

        self.comp_canvas.draw_idle()

    def _comp_on_bg_changed(self, _idx=None):
        """Recompute the composite when the background-colour choice changes.

        The compositing math depends on bg_val (alpha-blend over the chosen
        colour), so a redraw is required, not just an overlay refresh.
        """
        if getattr(self, 'current_rgb_composite', None) is None:
            return
        self._update_rgb_composite()

    def _comp_on_overlay_pos_changed(self, _idx=None):
        """Called when the overlay-position combo changes.

        Adjusts figure subplot margins to make room for legend/scale bar in the
        margin (Below/Above) or restore the default tight layout (Inside),
        bumps title pad when overlays are above so they don't collide with the
        title, then triggers an overlay refresh.
        """
        if not hasattr(self, 'comp_overlay_pos_combo'):
            return
        pos = self.comp_overlay_pos_combo.currentText()
        fig = self.comp_canvas.figure
        if pos == "Below image":
            fig.subplots_adjust(top=0.93, bottom=0.10, left=0.04, right=0.98)
            self.comp_ax.title.set_y(1.0)
            self.comp_ax.title.set_in_layout(True)
            try:
                self.comp_ax.set_title(self.comp_ax.get_title(), fontsize=12, color='white', pad=6)
            except Exception:
                pass
        elif pos == "Above image":
            fig.subplots_adjust(top=0.85, bottom=0.04, left=0.04, right=0.98)
            try:
                self.comp_ax.set_title(self.comp_ax.get_title(), fontsize=12, color='white', pad=40)
            except Exception:
                pass
        else:  # Inside
            fig.subplots_adjust(top=0.93, bottom=0.04, left=0.04, right=0.98)
            try:
                self.comp_ax.set_title(self.comp_ax.get_title(), fontsize=12, color='white', pad=6)
            except Exception:
                pass
        self._comp_refresh_overlays()

    def _comp_set_active(self, idx):
        self.comp_active_layer = idx

    def _comp_pick_color(self, idx):
        r, g, b = self.comp_layer_colors[idx]
        initial = QColor(r, g, b)
        color = QColorDialog.getColor(initial, self, f"Layer {idx+1} colour")
        if color.isValid():
            self.comp_layer_colors[idx] = [color.red(), color.green(), color.blue()]
            btn = self.comp_layer_color_btns[idx]
            btn.setStyleSheet(
                f"background:rgb({color.red()},{color.green()},{color.blue()});"
                "border:1px solid #888;border-radius:3px;"
            )
            self._comp_refresh_ptable_highlights()

    def _comp_clear_layer(self, idx):
        self.comp_layer_lines[idx] = None
        self.comp_layer_labels[idx].setText(f"Layer {idx+1}: <i>none</i>")
        self._comp_refresh_ptable_highlights()

    def _comp_on_line_selected(self, line_name):
        """Called when user picks a LIBS line from the periodic table."""
        idx = self.comp_active_layer
        self.comp_layer_lines[idx] = line_name
        r, g, b = self.comp_layer_colors[idx]
        self.comp_layer_labels[idx].setText(
            f"Layer {idx+1}: <b>{line_name}</b>"
        )
        self._comp_refresh_ptable_highlights()

    def _comp_refresh_ptable_highlights(self):
        self.composite_ptable.clear_highlights()
        for i, line in enumerate(self.comp_layer_lines):
            if line is None:
                continue
            m = _re.match(r'([A-Z][a-z]?)', line)
            if m:
                self.composite_ptable.set_layer_highlight(m.group(1), self.comp_layer_colors[i])

    def _connect_rgb_signals(self):
        self.composite_ptable.lineSelected.connect(self._comp_on_line_selected)
        self.comp_update_btn.clicked.connect(self._update_rgb_composite)
        self.comp_overlay_pos_combo.currentIndexChanged.connect(self._comp_on_overlay_pos_changed)
        self.comp_bg_combo.currentIndexChanged.connect(self._comp_on_bg_changed)

    def _update_rgb_composite(self):
        if self.ds is None:
            return
        bands = self._get_bands_values(warn=False)
        if bands is None:
            return

        active_count = sum(
            1 for i in range(5)
            if self.comp_layer_lines[i] is not None and self.comp_layer_enables[i].isChecked()
            and element_wavelengths.get(self.comp_layer_lines[i]) is not None
        )
        if active_count == 0:
            self.comp_ax.clear()
            self.comp_ax.text(0.5, 0.5, "Assign at least one element\nand click Update",
                              ha='center', va='center', fontsize=10, color='white',
                              transform=self.comp_ax.transAxes)
            self.comp_canvas.draw_idle()
            return

        progress = QProgressDialog("Building composite…", "Cancel", 0, active_count + 1, self)
        progress.setWindowTitle("Composite")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        QApplication.processEvents()

        shape = None
        layer_images = []
        layer_vmins = [0.0] * 5
        layer_vmaxs = [1.0] * 5

        hw = float(self.comp_bl_halfwidth.value())
        gap = float(self.comp_bl_gap.value())
        comp_bl_method = self.comp_bl_method.currentText()
        plo = float(self.comp_pmin_spin.value())
        phi = float(self.comp_pmax_spin.value())

        info_parts = []
        step = 0

        for i in range(5):
            line_name = self.comp_layer_lines[i]
            if line_name is None or not self.comp_layer_enables[i].isChecked():
                layer_images.append(None)
                continue

            target = element_wavelengths.get(line_name)
            if target is None:
                layer_images.append(None)
                continue

            if progress.wasCanceled():
                return

            step += 1
            progress.setLabelText(f"Layer {i+1}: {line_name}…")
            progress.setValue(step)
            QApplication.processEvents()

            use_bl = self.comp_layer_baseline_cbs[i].isChecked()
            img = self._line_or_doublet_map(target, use_bl, hw, gap, comp_bl_method)

            if shape is None:
                shape = img.shape

            gain = float(self.comp_layer_gain_spins[i].value())
            if self.comp_layer_auto_cbs[i].isChecked():
                valid = img[np.isfinite(img) & (img > 0)]
                if valid.size > 0:
                    vmin = float(np.nanpercentile(valid, plo))
                    vmax = float(np.nanpercentile(valid, phi))
                else:
                    vmin, vmax = 0.0, 1.0
                self.comp_layer_min_spins[i].blockSignals(True)
                self.comp_layer_min_spins[i].setValue(vmin)
                self.comp_layer_min_spins[i].blockSignals(False)
                self.comp_layer_max_spins[i].blockSignals(True)
                self.comp_layer_max_spins[i].setValue(vmax)
                self.comp_layer_max_spins[i].blockSignals(False)
            else:
                vmin = float(self.comp_layer_min_spins[i].value())
                vmax = float(self.comp_layer_max_spins[i].value())

            if vmax <= vmin:
                vmax = vmin + 1.0

            layer_vmins[i] = vmin
            layer_vmaxs[i] = vmax

            scaled = np.clip((img - vmin) / (vmax - vmin) * gain, 0, 1)
            scaled = np.nan_to_num(scaled, nan=0.0)
            layer_images.append(scaled)

            r, g, b = self.comp_layer_colors[i]
            opacity = self.comp_layer_opacity_sliders[i].value()
            info_parts.append(
                f"<span style='color:rgb({r},{g},{b})'>{line_name} ({opacity}%)</span>"
            )

        if progress.wasCanceled():
            return

        progress.setLabelText("Blending layers…")
        progress.setValue(active_count)
        QApplication.processEvents()

        bg_name = self.comp_bg_combo.currentText()
        bg_val = {"Black": 0.0, "White": 1.0, "Gray": 0.3}.get(bg_name, 0.0)

        # Build the additive element signal on a black canvas (this preserves
        # the standard "fluorescence" mixing: red+green = yellow, etc.)
        signal = np.zeros((*shape, 3), dtype=float)
        for i, scaled in enumerate(layer_images):
            if scaled is None:
                continue
            r, g, b = [c / 255.0 for c in self.comp_layer_colors[i]]
            alpha = self.comp_layer_opacity_sliders[i].value() / 100.0
            color_vec = np.array([r, g, b])
            signal += scaled[..., np.newaxis] * color_vec * alpha
        signal = np.clip(signal, 0.0, 1.0)

        # Alpha-blend signal over the selected background colour.
        # Per-pixel "signal alpha" = max channel of the additive signal.
        # This way:
        #   • pure-signal pixels keep their colour (red stays red on white),
        #   • zero-signal pixels show the background colour,
        #   • on a black background the result is identical to the old additive formula.
        sig_alpha = signal.max(axis=-1, keepdims=True)
        composite = signal + bg_val * (1.0 - sig_alpha)
        composite = np.clip(composite, 0.0, 1.0)
        self.current_rgb_composite = composite

        self.comp_ax.clear()
        self.comp_ax.imshow(composite, interpolation='nearest')
        self.comp_ax.set_xticks([]); self.comp_ax.set_yticks([])
        # Match figure/axes background to the selected composite background
        # so the surrounding margins/legend area look consistent.
        bg_color = (bg_val, bg_val, bg_val)
        self.comp_canvas.figure.patch.set_facecolor(bg_color)
        self.comp_ax.set_facecolor(bg_color)
        # Track for overlay-text colour decisions
        self._comp_current_bg_value = bg_val

        # Apply locked view early so overlays use zoomed extent
        if self._locked_view is not None:
            set_view(self.comp_ax, self._locked_view)

        # Build title with wavelength info
        from matplotlib.patches import Rectangle
        legend_entries = []
        title_parts = []
        for i in range(5):
            ln = self.comp_layer_lines[i]
            if ln is None or not self.comp_layer_enables[i].isChecked():
                continue
            if element_wavelengths.get(ln) is None:
                continue
            m = _re.match(r'([A-Z][a-z]?)', ln)
            sym = m.group(1) if m else ln[:2]
            legend_entries.append((sym, self.comp_layer_colors[i]))
            title_parts.append(ln)

        title = " | ".join(title_parts) if title_parts else "Element Composite"
        ov_pos = self.comp_overlay_pos_combo.currentText() if hasattr(self, 'comp_overlay_pos_combo') else "Inside image"
        title_pad = 40 if ov_pos == "Above image" else 6
        title_color = 'black' if bg_val > 0.5 else 'white'
        self.comp_ax.set_title(title, fontsize=12, color=title_color, pad=title_pad)
        # Apply margin choice consistent with current overlay placement
        if ov_pos == "Below image":
            self.comp_canvas.figure.subplots_adjust(top=0.93, bottom=0.10, left=0.04, right=0.98)
        elif ov_pos == "Above image":
            self.comp_canvas.figure.subplots_adjust(top=0.85, bottom=0.04, left=0.04, right=0.98)
        else:
            self.comp_canvas.figure.subplots_adjust(top=0.93, bottom=0.04, left=0.04, right=0.98)

        self._comp_legend_entries = legend_entries
        self._comp_overlay_artists = []
        self._comp_draw_overlays()

        # Per-layer colorbars on the right edge
        cbar_entries = []
        for i in range(5):
            if layer_images[i] is None:
                continue
            if not self.comp_layer_cbar_cbs[i].isChecked():
                continue
            m = _re.match(r'([A-Z][a-z]?)', self.comp_layer_lines[i] or '')
            sym = m.group(1) if m else '?'
            cbar_entries.append((sym, self.comp_layer_colors[i],
                                 layer_vmins[i], layer_vmaxs[i]))

        # Remove previous colorbar axes
        for old_ax in getattr(self, '_comp_cbar_axes', []):
            try:
                old_ax.remove()
            except Exception:
                pass
        self._comp_cbar_axes = []

        if cbar_entries:
            fig = self.comp_canvas.figure
            n_cb = len(cbar_entries)
            cb_w = 0.025
            cb_h = 0.50
            cb_gap = 0.045
            cb_x_start = 0.92 - n_cb * (cb_w + cb_gap)
            cb_y0 = 0.25
            res_cb = 256
            gradient = np.linspace(0, 1, res_cb).reshape(-1, 1)

            for idx_cb, (sym, col, vlo, vhi) in enumerate(cbar_entries):
                fx = cb_x_start + idx_cb * (cb_w + cb_gap)
                rc = [col[0] / 255.0, col[1] / 255.0, col[2] / 255.0]

                cb_img = np.zeros((res_cb, 1, 3))
                for ch in range(3):
                    cb_img[:, 0, ch] = gradient[:, 0] * rc[ch]

                cb_ax = fig.add_axes([fx, cb_y0, cb_w, cb_h])
                cb_ax.imshow(cb_img, aspect='auto', origin='lower')
                cb_ax.set_xticks([])
                cb_ax.set_yticks([0, res_cb - 1])
                fg_color = 'black' if bg_val > 0.5 else 'white'
                cb_ax.set_yticklabels([f"{vlo:.0f}", f"{vhi:.0f}"],
                    fontsize=7, fontfamily='Segoe UI', fontweight='bold', color=fg_color)
                cb_ax.tick_params(axis='y', length=0, pad=2)
                cb_ax.set_title(sym, fontsize=8, fontfamily='Segoe UI',
                    fontweight='bold', color=rc, pad=3)
                for spine in cb_ax.spines.values():
                    spine.set_edgecolor(fg_color)
                    spine.set_linewidth(0.5)
                cb_ax.patch.set_alpha(0)
                self._comp_cbar_axes.append(cb_ax)

        self.comp_info_label.setText(" | ".join(info_parts) if info_parts else "No active layers")

        if self._locked_view is not None:
            set_view(self.comp_ax, self._locked_view)
        self.comp_canvas.draw_idle()

        progress.setValue(active_count + 1)
        progress.close()

    # --- Export tab ---
    def _build_tab_export(self):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.widgets import RectangleSelector

        self.tabExport = QWidget(self)
        outer = QHBoxLayout(self.tabExport)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # --- Left: controls ---
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFrameShape(QScrollArea.NoFrame)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ctrl_inner = QWidget()
        ctrl_lay = QVBoxLayout(ctrl_inner)
        ctrl_lay.setContentsMargins(2, 2, 2, 2)
        ctrl_lay.setSpacing(4)

        # -- Subset group --
        grpSel = QGroupBox("Spatial Subset Extraction")
        fs = QFormLayout(grpSel)
        fs.setContentsMargins(6, 14, 6, 6)
        fs.setSpacing(4)

        tip = QLabel("Drag a rectangle on the map to define\nthe region to extract.")
        tip.setStyleSheet("color: gray; font-size: 8pt;")
        tip.setWordWrap(True)
        fs.addRow(tip)

        self.subset_x0_spin = QSpinBox(); self.subset_x0_spin.setRange(0, 99999)
        fs.addRow("X start (px):", self.subset_x0_spin)
        self.subset_x1_spin = QSpinBox(); self.subset_x1_spin.setRange(0, 99999)
        fs.addRow("X end (px):", self.subset_x1_spin)
        self.subset_y0_spin = QSpinBox(); self.subset_y0_spin.setRange(0, 99999)
        fs.addRow("Y start (px):", self.subset_y0_spin)
        self.subset_y1_spin = QSpinBox(); self.subset_y1_spin.setRange(0, 99999)
        fs.addRow("Y end (px):", self.subset_y1_spin)

        self.subset_size_label = QLabel("Selection: — x — px")
        self.subset_size_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        fs.addRow(self.subset_size_label)

        self.subset_save_btn = QPushButton("Extract && Save...")
        self.subset_save_btn.setToolTip("Slice the current cube to the selected region and save as a new NetCDF file")
        fs.addRow(self.subset_save_btn)

        self.subset_status_label = QLabel("")
        self.subset_status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 8pt;")
        self.subset_status_label.setWordWrap(True)
        fs.addRow(self.subset_status_label)

        ctrl_lay.addWidget(grpSel)

        grpReset = QGroupBox("Tools")
        fr = QFormLayout(grpReset)
        fr.setContentsMargins(6, 14, 6, 6)
        fr.setSpacing(4)
        self.subset_clear_btn = QPushButton("Clear selection")
        fr.addRow(self.subset_clear_btn)
        self.subset_refresh_btn = QPushButton("Refresh map")
        fr.addRow(self.subset_refresh_btn)
        ctrl_lay.addWidget(grpReset)

        ctrl_lay.addStretch()
        ctrl_scroll.setWidget(ctrl_inner)
        ctrl_scroll.setMinimumWidth(200)

        # --- Right: map canvas ---
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(4, 0, 0, 0)
        right_lay.setSpacing(4)

        self.subset_canvas = FigureCanvas(plt.Figure(dpi=100))
        self.subset_ax = self.subset_canvas.figure.add_subplot(111)
        right_lay.addWidget(self.subset_canvas, stretch=1)
        self.subset_toolbar = NavigationToolbar(self.subset_canvas, self)
        right_lay.addWidget(self.subset_toolbar)
        self.subset_colorbar = None

        self._subset_rect_selector = RectangleSelector(
            self.subset_ax, self._on_subset_rect_select, useblit=True, button=[1],
            interactive=True,
            props=dict(edgecolor='cyan', linewidth=1.5, linestyle='--', facecolor='cyan', alpha=0.15),
        )

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(ctrl_scroll)
        splitter.addWidget(right_widget)
        splitter.setSizes([280, 1070])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter)
        self.tabs.addTab(self.tabExport, "9. Cube subset")

    # --- Cube Utils tab ---
    def _build_tab_cube_utils(self):
        self.tabCubeUtils = QWidget()
        outer = QVBoxLayout(self.tabCubeUtils)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        info_label = QLabel(
            "Utilities for fixing and re-encoding LIBS hypercube files.\n"
            "Operations apply to the <b>original</b> loaded cube (before normalisation).")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 9pt; font-family: 'Segoe UI';")
        outer.addWidget(info_label)

        # --- Channel reorder ---
        grpReorder = QGroupBox("Spectrometer channel reorder")
        fr = QFormLayout(grpReorder)
        fr.setContentsMargins(6, 14, 6, 6)
        fr.setSpacing(4)

        reorder_info = QLabel(
            "Some cubes have the channels of one spectrometer recorded in the wrong "
            "position.  Use this tool to move a block of channels from the beginning "
            "of the spectral axis to the end (or vice-versa).\n"
            "This does <b>not</b> sort all channels by wavelength — it preserves "
            "intra-spectrometer order.")
        reorder_info.setWordWrap(True)
        reorder_info.setStyleSheet("font-size: 8pt; color: #555; font-family: 'Segoe UI';")
        fr.addRow(reorder_info)

        self.cu_n_channels_label = QLabel("Total channels: —")
        self.cu_n_channels_label.setStyleSheet("font-size: 8pt; font-family: 'Segoe UI';")
        fr.addRow(self.cu_n_channels_label)

        self.cu_block_size_spin = QSpinBox()
        self.cu_block_size_spin.setRange(1, 99999)
        self.cu_block_size_spin.setValue(2048)
        self.cu_block_size_spin.setToolTip("Number of channels in the spectrometer block to move")
        fr.addRow("Block size (channels):", self.cu_block_size_spin)

        self.cu_direction_combo = QComboBox()
        self.cu_direction_combo.addItems(["Move first N channels to end", "Move last N channels to start"])
        fr.addRow("Direction:", self.cu_direction_combo)

        self.cu_preview_label = QLabel("")
        self.cu_preview_label.setWordWrap(True)
        self.cu_preview_label.setStyleSheet(
            "font-size: 8pt; font-family: 'Segoe UI'; color: #333; "
            "background-color: #f0f0f0; padding: 4px; border-radius: 3px;")
        fr.addRow(self.cu_preview_label)

        self.cu_preview_btn = QPushButton("Preview reorder")
        self.cu_preview_btn.setToolTip("Show the wavelength ranges before and after reordering")
        fr.addRow(self.cu_preview_btn)

        self.cu_apply_reorder_btn = QPushButton("Apply reorder (in memory)")
        self.cu_apply_reorder_btn.setToolTip(
            "Reorder channels in the loaded dataset. "
            "Use 'Save re-encoded cube' below to write the result to disk.")
        self.cu_apply_reorder_btn.setStyleSheet(
            "QPushButton { background-color: #e67e22; color: white;"
            " font-weight: bold; padding: 6px; border-radius: 3px; }"
            " QPushButton:hover { background-color: #f39c12; }")
        fr.addRow(self.cu_apply_reorder_btn)

        self.cu_reorder_status = QLabel("")
        self.cu_reorder_status.setWordWrap(True)
        self.cu_reorder_status.setStyleSheet("font-size: 8pt; font-family: 'Segoe UI';")
        fr.addRow(self.cu_reorder_status)

        outer.addWidget(grpReorder)

        # --- Channel range removal ---
        grpRemove = QGroupBox("Channel range removal")
        fx = QFormLayout(grpRemove)
        fx.setContentsMargins(6, 14, 6, 6)
        fx.setSpacing(4)

        remove_info = QLabel(
            "Remove a contiguous range of channels (by <b>channel number</b>, "
            "not wavelength) from the cube. Useful for excising dead pixels, "
            "saturated regions, or overlap between spectrometers.\n"
            "Channel indices are <b>1-based</b>, inclusive.")
        remove_info.setWordWrap(True)
        remove_info.setStyleSheet("font-size: 8pt; color: #555; font-family: 'Segoe UI';")
        fx.addRow(remove_info)

        self.cu_remove_from_spin = QSpinBox()
        self.cu_remove_from_spin.setRange(1, 99999)
        self.cu_remove_from_spin.setValue(1)
        self.cu_remove_from_spin.setToolTip("First channel to remove (1-based, inclusive)")
        fx.addRow("From channel:", self.cu_remove_from_spin)

        self.cu_remove_to_spin = QSpinBox()
        self.cu_remove_to_spin.setRange(1, 99999)
        self.cu_remove_to_spin.setValue(2048)
        self.cu_remove_to_spin.setToolTip("Last channel to remove (1-based, inclusive)")
        fx.addRow("To channel:", self.cu_remove_to_spin)

        self.cu_remove_preview_label = QLabel("")
        self.cu_remove_preview_label.setWordWrap(True)
        self.cu_remove_preview_label.setStyleSheet(
            "font-size: 8pt; font-family: 'Segoe UI'; color: #333; "
            "background-color: #f0f0f0; padding: 4px; border-radius: 3px;")
        fx.addRow(self.cu_remove_preview_label)

        self.cu_remove_preview_btn = QPushButton("Preview removal")
        self.cu_remove_preview_btn.setToolTip(
            "Show the wavelengths, channel count and estimated memory that "
            "would be removed")
        fx.addRow(self.cu_remove_preview_btn)

        self.cu_remove_apply_btn = QPushButton("Apply removal (in memory)")
        self.cu_remove_apply_btn.setToolTip(
            "Remove the selected channel range from the loaded dataset. "
            "Use 'Save re-encoded cube' below to write the result to disk.")
        self.cu_remove_apply_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white;"
            " font-weight: bold; padding: 6px; border-radius: 3px; }"
            " QPushButton:hover { background-color: #e74c3c; }")
        fx.addRow(self.cu_remove_apply_btn)

        self.cu_remove_status = QLabel("")
        self.cu_remove_status.setWordWrap(True)
        self.cu_remove_status.setStyleSheet("font-size: 8pt; font-family: 'Segoe UI';")
        fx.addRow(self.cu_remove_status)

        outer.addWidget(grpRemove)

        # --- Re-encode / save ---
        grpSave = QGroupBox("Save re-encoded cube")
        fs = QFormLayout(grpSave)
        fs.setContentsMargins(6, 14, 6, 6)
        fs.setSpacing(4)

        save_info = QLabel(
            "Save the current (possibly reordered) cube as a new NetCDF file "
            "with optional dtype conversion and lossless compression.")
        save_info.setWordWrap(True)
        save_info.setStyleSheet("font-size: 8pt; color: #555; font-family: 'Segoe UI';")
        fs.addRow(save_info)

        self.cu_orig_dtype_label = QLabel("Original dtype: —")
        self.cu_orig_dtype_label.setStyleSheet(
            "font-size: 8pt; font-family: 'Segoe UI'; font-weight: bold;")
        fs.addRow(self.cu_orig_dtype_label)

        self.cu_target_dtype_combo = QComboBox()
        self.cu_target_dtype_combo.addItems([
            "Keep original", "uint16", "int16", "uint32", "int32", "float32", "float64"])
        self.cu_target_dtype_combo.setCurrentIndex(1)
        self.cu_target_dtype_combo.setToolTip("Data type for intensity values in the saved file")
        fs.addRow("Target dtype:", self.cu_target_dtype_combo)

        self.cu_dtype_note_label = QLabel("")
        self.cu_dtype_note_label.setWordWrap(True)
        self.cu_dtype_note_label.setStyleSheet(
            "font-size: 7.5pt; color: #666; font-family: 'Segoe UI';")
        fs.addRow(self.cu_dtype_note_label)
        self.cu_target_dtype_combo.currentTextChanged.connect(self._cubeutils_update_dtype_note)

        self.cu_compress_chk = QCheckBox("Enable compression")
        self.cu_compress_chk.setChecked(True)
        fs.addRow(self.cu_compress_chk)

        self.cu_compress_combo = QComboBox()
        self.cu_compress_combo.addItems(["zlib", "lzf"])
        self.cu_compress_combo.setToolTip("Compression algorithm (zlib is universal, lzf is faster)")
        fs.addRow("Algorithm:", self.cu_compress_combo)

        self.cu_compress_level_spin = QSpinBox()
        self.cu_compress_level_spin.setRange(1, 9)
        self.cu_compress_level_spin.setValue(4)
        self.cu_compress_level_spin.setToolTip("zlib compression level (1=fast, 9=best)")
        fs.addRow("Compression level:", self.cu_compress_level_spin)

        self.cu_compress_chk.toggled.connect(self._cubeutils_compress_toggled)
        self.cu_compress_combo.currentTextChanged.connect(self._cubeutils_compress_toggled)

        self.cu_save_btn = QPushButton("Save re-encoded cube…")
        self.cu_save_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white;"
            " font-weight: bold; padding: 6px; border-radius: 3px; }"
            " QPushButton:hover { background-color: #3498db; }")
        fs.addRow(self.cu_save_btn)

        self.cu_save_status = QLabel("")
        self.cu_save_status.setWordWrap(True)
        self.cu_save_status.setStyleSheet("font-size: 8pt; font-family: 'Segoe UI';")
        fs.addRow(self.cu_save_status)

        outer.addWidget(grpSave)
        outer.addStretch()

        self.tabs.addTab(self.tabCubeUtils, "10. Cube utils")

    # ====== Cube Utils logic ======
    def _cubeutils_on_tab_changed(self, index):
        """Update channel count and dtype labels when switching to the Cube Utils tab."""
        if not hasattr(self, 'tabCubeUtils') or self.tabs.widget(index) is not self.tabCubeUtils:
            return
        bands = self._get_bands_values(warn=False)
        if bands is not None:
            n_total = len(bands)
            self.cu_n_channels_label.setText(f"Total channels: {n_total}")
            self.cu_block_size_spin.setMaximum(n_total - 1)
            self.cu_remove_from_spin.setMaximum(n_total)
            self.cu_remove_to_spin.setMaximum(n_total)
            if self.cu_remove_to_spin.value() > n_total:
                self.cu_remove_to_spin.setValue(n_total)
        else:
            self.cu_n_channels_label.setText("Total channels: — (no cube loaded)")
        self._cubeutils_refresh_dtype_label()
        self._cubeutils_update_dtype_note()

    def _cubeutils_refresh_dtype_label(self):
        """Show the original cube's data type(s), compression, and file size."""
        if self.original_ds is None:
            self.cu_orig_dtype_label.setText("Original dtype: — (no cube loaded)")
            return
        dtypes = {}
        for var in self.original_ds.data_vars:
            dt = str(self.original_ds[var].dtype)
            dtypes.setdefault(dt, []).append(var)
        parts = []
        for dt, vars_list in dtypes.items():
            if len(vars_list) == len(list(self.original_ds.data_vars)):
                parts.append(f"<b>{dt}</b>")
            else:
                parts.append(f"<b>{dt}</b> ({', '.join(vars_list)})")

        comp_parts = set()
        for var in self.original_ds.data_vars:
            enc = self.original_ds[var].encoding
            if enc.get('zlib'):
                lvl = enc.get('complevel', '?')
                comp_parts.add(f"zlib (level {lvl})")
            elif enc.get('compression'):
                comp_parts.add(str(enc['compression']))
            # netCDF4 filters stored under 'filters' key by some backends
            elif enc.get('filters'):
                for f in enc['filters']:
                    comp_parts.add(f.get('id', str(f)))
        comp_str = ", ".join(sorted(comp_parts)) if comp_parts else "none"

        size_str = ""
        if self.loaded_cube_path and os.path.isfile(self.loaded_cube_path):
            sz = os.path.getsize(self.loaded_cube_path) / (1024 * 1024)
            size_str = f"  —  file: <b>{sz:.1f} MB</b>"
        self.cu_orig_dtype_label.setText(
            f"Original dtype: {', '.join(parts)}  —  compression: <b>{comp_str}</b>{size_str}")

    def _cubeutils_update_dtype_note(self, _text=None):
        """Update the info note below the target dtype combo."""
        target = self.cu_target_dtype_combo.currentText()
        notes = {
            "Keep original": "Data will be saved as-is, with compression only.",
            "uint16": "Unsigned 16-bit (0–65535). Values >65535 will be scaled; "
                      "negatives clipped to 0. Good for raw LIBS counts.",
            "int16": "Signed 16-bit (−32768 to 32767). Values outside range will be clipped.",
            "uint32": "Unsigned 32-bit (0–4 294 967 295). Larger but lossless for most integer data.",
            "int32": "Signed 32-bit. Larger but lossless for most integer data.",
            "float32": "32-bit float. Preserves fractional values; ~4 bytes/pixel.",
            "float64": "64-bit float. Full precision; ~8 bytes/pixel. Largest output.",
        }
        self.cu_dtype_note_label.setText(notes.get(target, ""))

    def _cubeutils_compress_toggled(self, *_args):
        """Enable/disable compression widgets based on checkbox and algorithm."""
        enabled = self.cu_compress_chk.isChecked()
        self.cu_compress_combo.setEnabled(enabled)
        use_zlib = self.cu_compress_combo.currentText() == "zlib"
        self.cu_compress_level_spin.setEnabled(enabled and use_zlib)

    def _cubeutils_preview_reorder(self):
        """Show what the reorder will look like without applying it."""
        bands = self._get_bands_values(ds=self.original_ds)
        if bands is None:
            self.cu_preview_label.setText("No cube loaded.")
            return

        n = self.cu_block_size_spin.value()
        total = len(bands)
        if n >= total:
            self.cu_preview_label.setText(f"Block size ({n}) must be < total channels ({total}).")
            return

        direction = self.cu_direction_combo.currentText()
        if "first" in direction.lower():
            block_a = bands[:n]
            block_b = bands[n:]
            new_order = np.concatenate([block_b, block_a])
            desc = f"First {n} channels → moved to end"
        else:
            block_a = bands[:-n]
            block_b = bands[-n:]
            new_order = np.concatenate([block_b, block_a])
            desc = f"Last {n} channels → moved to start"

        txt = (
            f"<b>{desc}</b><br>"
            f"<b>Current order:</b>  {bands[0]:.2f} … {bands[n-1]:.2f}  |  "
            f"{bands[n]:.2f} … {bands[-1]:.2f} nm<br>"
            f"<b>After reorder:</b>  {new_order[0]:.2f} … {new_order[total-n-1]:.2f}  |  "
            f"{new_order[total-n]:.2f} … {new_order[-1]:.2f} nm"
        )
        self.cu_preview_label.setText(txt)

    def _cubeutils_apply_reorder(self):
        """Reorder the spectral channels in both original_ds and ds."""
        if self.original_ds is None:
            QMessageBox.information(self, "Cube Utils", "No cube loaded.")
            return

        bands = self._get_bands_values(ds=self.original_ds)
        if bands is None:
            return

        n = self.cu_block_size_spin.value()
        total = len(bands)
        if n >= total:
            QMessageBox.warning(self, "Cube Utils",
                                f"Block size ({n}) must be smaller than total channels ({total}).")
            return

        direction = self.cu_direction_combo.currentText()
        if "first" in direction.lower():
            idx_order = list(range(n, total)) + list(range(0, n))
            desc = f"first {n} → end"
        else:
            idx_order = list(range(total - n, total)) + list(range(0, total - n))
            desc = f"last {n} → start"

        reply = QMessageBox.question(
            self, "Confirm reorder",
            f"This will reorder channels ({desc}) in memory.\n"
            f"Total channels: {total}\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        prog = QProgressDialog("Reordering channels…", None, 0, 4, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        QApplication.processEvents()

        first_var = next(iter(self.original_ds.data_vars))
        band_dim = 'bands'
        dims = list(self.original_ds[first_var].dims)
        if band_dim not in dims:
            for d in dims:
                if 'band' in d.lower() or 'wav' in d.lower() or 'spectral' in d.lower():
                    band_dim = d
                    break

        prog.setLabelText("Reordering original dataset…")
        prog.setValue(1)
        QApplication.processEvents()

        self.original_ds = self.original_ds.isel({band_dim: idx_order})

        prog.setLabelText("Rebuilding working dataset…")
        prog.setValue(2)
        QApplication.processEvents()

        self.ds = self.original_ds.copy()
        if self.active_mask is not None and np.any(self.active_mask):
            self.ds = self.ds.where(~self.active_mask, other=0)

        prog.setLabelText("Updating UI…")
        prog.setValue(3)
        QApplication.processEvents()

        new_bands = self._get_bands_values()
        if new_bands is not None:
            self.slider.setMaximum(len(new_bands) - 1)
            if hasattr(self, 'mask_band_slider'):
                self.mask_band_slider.setMaximum(len(new_bands) - 1)
            self.cu_n_channels_label.setText(f"Total channels: {len(new_bands)}")
            self.cu_block_size_spin.setMaximum(len(new_bands) - 1)

        self.cu_reorder_status.setText(
            f"<span style='color:green;'>✓ Reorder applied ({desc}). "
            f"Use <b>Save re-encoded cube</b> to write to disk.</span>")
        self.cu_reorder_status.setStyleSheet(
            "font-size: 8pt; font-family: 'Segoe UI'; color: green;")

        self.update_plot()
        prog.setValue(4)
        prog.close()
        self.statusBar().showMessage(f"Channels reordered ({desc})", 3000)

    # --- Channel range removal ---
    def _cubeutils_resolve_remove_range(self, bands):
        """Return (start0, stop0, n_total, n_remove) for the currently selected range.

        Indices are converted from 1-based UI values to 0-based Python slice
        bounds (start inclusive, stop exclusive).
        Returns None and shows a message box if the range is invalid.
        """
        n_total = len(bands)
        ch_from = int(self.cu_remove_from_spin.value())
        ch_to = int(self.cu_remove_to_spin.value())
        if ch_from > ch_to:
            ch_from, ch_to = ch_to, ch_from
        ch_from = max(1, min(n_total, ch_from))
        ch_to = max(1, min(n_total, ch_to))
        start0 = ch_from - 1
        stop0 = ch_to  # exclusive
        n_remove = stop0 - start0
        if n_remove <= 0:
            QMessageBox.warning(self, "Cube Utils",
                                "The selected removal range is empty.")
            return None
        if n_remove >= n_total:
            QMessageBox.warning(self, "Cube Utils",
                                "Cannot remove all channels; at least one must remain.")
            return None
        return start0, stop0, n_total, n_remove

    def _cubeutils_preview_remove(self):
        """Show the wavelengths, channel count and memory that would be removed."""
        if self.original_ds is None:
            self.cu_remove_preview_label.setText("No cube loaded.")
            return
        bands = self._get_bands_values(ds=self.original_ds)
        if bands is None:
            return
        res = self._cubeutils_resolve_remove_range(bands)
        if res is None:
            return
        start0, stop0, n_total, n_remove = res

        wl_from = float(bands[start0])
        wl_to = float(bands[stop0 - 1])
        n_keep = n_total - n_remove

        # Rough memory estimate using primary variable
        try:
            var_name = next(iter(self.original_ds.data_vars))
            ref = self.original_ds[var_name]
            spatial = 1
            for d in ref.dims:
                if d != 'bands':
                    spatial *= int(ref.sizes[d])
            itemsize = np.dtype(ref.dtype).itemsize
            bytes_removed = n_remove * spatial * itemsize
            bytes_kept = n_keep * spatial * itemsize
            def _fmt(nbytes):
                if nbytes >= 1 << 30:
                    return f"{nbytes / (1 << 30):.2f} GB"
                if nbytes >= 1 << 20:
                    return f"{nbytes / (1 << 20):.1f} MB"
                return f"{nbytes / (1 << 10):.1f} KB"
            mem_line = (f"<br>Removed: ~{_fmt(bytes_removed)} — "
                        f"Kept: ~{_fmt(bytes_kept)} "
                        f"(dtype {ref.dtype}, {spatial} spatial px)")
        except Exception:
            mem_line = ""

        self.cu_remove_preview_label.setText(
            f"<b>Removing channels {start0 + 1}–{stop0} "
            f"({n_remove} channel{'s' if n_remove != 1 else ''}).</b><br>"
            f"Wavelength range removed: {wl_from:.3f} nm — {wl_to:.3f} nm<br>"
            f"Remaining channels: {n_keep} / {n_total}"
            f"{mem_line}"
        )

    def _cubeutils_apply_remove(self):
        """Remove the selected channel range in both original_ds and ds."""
        if self.original_ds is None:
            QMessageBox.information(self, "Cube Utils", "No cube loaded.")
            return
        bands = self._get_bands_values(ds=self.original_ds)
        if bands is None:
            return
        res = self._cubeutils_resolve_remove_range(bands)
        if res is None:
            return
        start0, stop0, n_total, n_remove = res

        wl_from = float(bands[start0])
        wl_to = float(bands[stop0 - 1])
        reply = QMessageBox.question(
            self, "Confirm channel removal",
            f"This will permanently remove {n_remove} channel"
            f"{'s' if n_remove != 1 else ''} from the loaded cube:\n\n"
            f"  Channels: {start0 + 1} – {stop0}\n"
            f"  Wavelengths: {wl_from:.3f} nm – {wl_to:.3f} nm\n"
            f"  Remaining: {n_total - n_remove} / {n_total} channels\n\n"
            "Masks and any applied baseline/normalization will be reset.\n"
            "The operation is in memory only — use 'Save re-encoded cube' "
            "to write the result to disk.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        prog = QProgressDialog("Removing channels…", None, 0, 4, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        QApplication.processEvents()

        # Determine the band dimension name (usually 'bands')
        first_var = next(iter(self.original_ds.data_vars))
        band_dim = 'bands'
        dims = list(self.original_ds[first_var].dims)
        if band_dim not in dims:
            for d in dims:
                if 'band' in d.lower() or 'wav' in d.lower() or 'spectral' in d.lower():
                    band_dim = d
                    break

        keep_idx = list(range(0, start0)) + list(range(stop0, n_total))

        prog.setLabelText("Slicing original dataset…")
        prog.setValue(1)
        QApplication.processEvents()

        try:
            self.original_ds = self.original_ds.isel({band_dim: keep_idx})
        except Exception as e:
            prog.close()
            QMessageBox.critical(self, "Cube Utils", f"Could not remove channels:\n{e}")
            return

        # Keep global metadata consistent with the new band axis (provenance
        # for cubes saved via 'Save re-encoded cube').
        if "bands_total" in self.original_ds.attrs:
            self.original_ds.attrs["bands_total"] = len(keep_idx)
        prev_removed = str(self.original_ds.attrs.get("channels_removed", "")).strip()
        removal_note = (f"{start0 + 1}-{stop0} ({wl_from:.3f}-{wl_to:.3f} nm)")
        self.original_ds.attrs["channels_removed"] = (
            f"{prev_removed}; {removal_note}" if prev_removed else removal_note)

        prog.setLabelText("Rebuilding working dataset…")
        prog.setValue(2)
        QApplication.processEvents()

        # Invalidate downstream state that depended on the previous band axis
        self._baseline_ds = None
        self._baseline_method_applied = "None"
        if hasattr(self, 'cube_bl_status_label'):
            self.cube_bl_status_label.setText("")
        if hasattr(self, 'norm_combo'):
            self.norm_combo.blockSignals(True)
            self.norm_combo.setCurrentIndex(0)
            self.norm_combo.blockSignals(False)
            self.norm_status_label.setText("")

        self.ds = self.original_ds.copy()
        if self.active_mask is not None and np.any(self.active_mask):
            self.ds = self.ds.where(~self.active_mask, other=0)

        prog.setLabelText("Updating UI…")
        prog.setValue(3)
        QApplication.processEvents()

        new_bands = self._get_bands_values()
        if new_bands is not None:
            n_new = len(new_bands)
            self.slider.setMaximum(n_new - 1)
            if hasattr(self, 'mask_band_slider'):
                self.mask_band_slider.setMaximum(n_new - 1)
            self.cu_n_channels_label.setText(f"Total channels: {n_new}")
            self.cu_block_size_spin.setMaximum(max(1, n_new - 1))
            self.cu_remove_from_spin.setMaximum(n_new)
            self.cu_remove_to_spin.setMaximum(n_new)
            if self.cu_remove_to_spin.value() > n_new:
                self.cu_remove_to_spin.setValue(n_new)

        self.cu_remove_status.setText(
            f"<span style='color:green;'>\u2713 Removed {n_remove} channels "
            f"({wl_from:.3f}–{wl_to:.3f} nm). "
            f"Use <b>Save re-encoded cube</b> to write to disk.</span>")
        self.cu_remove_status.setStyleSheet(
            "font-size: 8pt; font-family: 'Segoe UI'; color: green;")
        self.cu_remove_preview_label.setText("")

        self.update_plot()
        prog.setValue(4)
        prog.close()
        self.statusBar().showMessage(
            f"Removed {n_remove} channels ({wl_from:.3f}–{wl_to:.3f} nm)", 3000)

    def _cubeutils_save_cube(self):
        """Save the current original_ds as a new NetCDF with chosen dtype + optional compression.

        Memory-efficient: works chunk-by-chunk along the bands axis and never promotes
        the whole cube to float64. The output array is allocated in the target dtype.
        """
        if self.original_ds is None:
            QMessageBox.information(self, "Cube Utils", "No cube loaded.")
            return

        target_dtype_str = self.cu_target_dtype_combo.currentText()
        default_name = ""
        if self.loaded_cube_path:
            base, ext = os.path.splitext(self.loaded_cube_path)
            default_name = f"{base}_reencoded{ext}"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Re-encoded Cube", default_name,
            "NetCDF Files (*.nc);;All Files (*)")
        if not path:
            return

        use_compression = self.cu_compress_chk.isChecked()
        comp_algo = self.cu_compress_combo.currentText() if use_compression else None
        comp_level = self.cu_compress_level_spin.value() if (use_compression and comp_algo == "zlib") else 0

        keep_original = target_dtype_str == "Keep original"
        target_np = None if keep_original else np.dtype(target_dtype_str)
        is_unsigned_int = target_np is not None and np.issubdtype(target_np, np.unsignedinteger)
        is_int = target_np is not None and np.issubdtype(target_np, np.integer)
        type_max = float(np.iinfo(target_np).max) if is_int else None
        type_min = float(np.iinfo(target_np).min) if is_int else None

        ds_out = self.original_ds.copy()  # shallow copy; we'll replace variables one at a time

        n_vars = len(list(ds_out.data_vars))
        total_steps = n_vars * 2 + 2  # scan pass + convert pass per var
        prog = QProgressDialog("Preparing cube for saving…", "Cancel", 0, total_steps, self)
        prog.setWindowTitle("Save Re-encoded Cube")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        QApplication.processEvents()

        encoding = {}
        step = 0

        CHUNK_BANDS = 512  # chunk size along the bands axis

        for i, var in enumerate(list(ds_out.data_vars)):
            if prog.wasCanceled():
                return

            da = ds_out[var]
            dims = list(da.dims)
            shape = tuple(da.shape)

            # Locate the bands axis (fall back to first axis if not found)
            band_axis = 0
            for ax, d in enumerate(dims):
                if d == 'bands' or 'band' in d.lower() or 'wav' in d.lower():
                    band_axis = ax
                    break
            n_bands = shape[band_axis]

            def make_slice(start, stop):
                sl = [slice(None)] * len(shape)
                sl[band_axis] = slice(start, stop)
                return tuple(sl)

            src_dtype = da.dtype
            src_is_float = np.issubdtype(src_dtype, np.floating)

            # -------- Pass 1: determine global max if we need to scale for unsigned int
            scale = 1.0
            if (not keep_original) and is_unsigned_int:
                vmax_val = 0.0
                for cstart in range(0, n_bands, CHUNK_BANDS):
                    if prog.wasCanceled():
                        return
                    cstop = min(cstart + CHUNK_BANDS, n_bands)
                    chunk = da.isel({dims[band_axis]: slice(cstart, cstop)}).values
                    if chunk.size:
                        cmax = float(np.nanmax(chunk)) if src_is_float else float(chunk.max())
                        if cmax > vmax_val:
                            vmax_val = cmax
                    prog.setLabelText(
                        f"Scanning '{var}' ({i+1}/{n_vars})  "
                        f"[{cstop}/{n_bands} bands]{self._norm_mem_str()}")
                    QApplication.processEvents()
                if vmax_val > type_max:
                    scale = type_max / vmax_val
                    ds_out.attrs[f"{var}_{target_dtype_str}_scale_factor"] = float(1.0 / scale)
            step += 1
            prog.setValue(step)
            QApplication.processEvents()

            # -------- Pass 2: allocate output buffer and convert chunk-by-chunk
            if not keep_original:
                try:
                    out_arr = np.empty(shape, dtype=target_np)
                except MemoryError:
                    prog.close()
                    QMessageBox.critical(
                        self, "Memory error",
                        f"Could not allocate output array of shape {shape} as {target_dtype_str}.\n"
                        "Try a smaller target dtype or free up system memory.")
                    return

                for cstart in range(0, n_bands, CHUNK_BANDS):
                    if prog.wasCanceled():
                        return
                    cstop = min(cstart + CHUNK_BANDS, n_bands)
                    t = make_slice(cstart, cstop)
                    src_chunk = da.isel({dims[band_axis]: slice(cstart, cstop)}).values

                    if is_unsigned_int:
                        if src_is_float:
                            work = np.nan_to_num(src_chunk, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                            np.clip(work, 0.0, None, out=work)
                            if scale != 1.0:
                                work *= scale
                            out_arr[t] = work.astype(target_np, copy=False)
                        else:
                            if scale != 1.0:
                                # Need float temporary only for the chunk
                                work = src_chunk.astype(np.float32, copy=True)
                                np.clip(work, 0.0, None, out=work)
                                work *= scale
                                out_arr[t] = work.astype(target_np, copy=False)
                            else:
                                np.clip(src_chunk, 0, int(type_max), out=out_arr[t],
                                        casting='unsafe')
                    elif is_int:
                        if src_is_float:
                            work = np.nan_to_num(src_chunk, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
                            np.clip(work, type_min, type_max, out=work)
                            out_arr[t] = work.astype(target_np, copy=False)
                        else:
                            np.clip(src_chunk, int(type_min), int(type_max),
                                    out=out_arr[t], casting='unsafe')
                    else:
                        out_arr[t] = src_chunk.astype(target_np, copy=False)

                    prog.setLabelText(
                        f"Converting '{var}' → {target_dtype_str}  "
                        f"[{cstop}/{n_bands} bands]{self._norm_mem_str()}")
                    QApplication.processEvents()

                ds_out[var] = xr.DataArray(
                    out_arr, dims=da.dims, coords=da.coords, attrs=da.attrs, name=var)
            step += 1
            prog.setValue(step)
            QApplication.processEvents()

            ds_out[var].encoding.clear()
            out_dtype = str(target_np) if target_np is not None else str(ds_out[var].dtype)
            var_enc = {'dtype': out_dtype}
            if use_compression:
                if comp_algo == "zlib":
                    var_enc['zlib'] = True
                    var_enc['complevel'] = comp_level
                elif comp_algo == "lzf":
                    var_enc['compression'] = 'lzf'
            encoding[var] = var_enc

        if prog.wasCanceled():
            return
        prog.setLabelText(f"Writing NetCDF file…{self._norm_mem_str()}")
        prog.setValue(total_steps - 1)
        QApplication.processEvents()

        try:
            ds_out.to_netcdf(path, encoding=encoding)
        except Exception as e:
            prog.close()
            QMessageBox.critical(self, "Save error", str(e))
            return

        prog.setValue(total_steps)
        QApplication.processEvents()

        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        orig_size_str = ""
        if self.loaded_cube_path and os.path.isfile(self.loaded_cube_path):
            orig_mb = os.path.getsize(self.loaded_cube_path) / (1024 * 1024)
            ratio = (file_size_mb / orig_mb * 100) if orig_mb > 0 else 0
            orig_size_str = f"  ({ratio:.0f}% of original {orig_mb:.1f} MB)"
        basename = os.path.basename(path)
        dtype_display = target_dtype_str if not keep_original else "original"
        if use_compression:
            comp_display = f"{comp_algo}" + (f" level {comp_level}" if comp_algo == "zlib" else "")
        else:
            comp_display = "none"
        self.cu_save_status.setText(
            f"<span style='color:green;'>✓ Saved: {basename}<br>"
            f"Size: {file_size_mb:.1f} MB{orig_size_str}<br>"
            f"Compression: {comp_display}  |  dtype: {dtype_display}</span>")
        prog.close()
        self.statusBar().showMessage(f"Re-encoded cube saved: {basename} ({file_size_mb:.1f} MB)", 4000)

    def _on_subset_rect_select(self, eclick, erelease):
        """Called when the user drags a rectangle on the subset map."""
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        x0, x1 = int(max(0, round(x0))), int(round(x1))
        y0, y1 = int(max(0, round(y0))), int(round(y1))

        if self.ds is not None:
            first_var = next(iter(self.ds.data_vars))
            da = self.ds[first_var]
            dims = da.dims
            ny = da.shape[dims.index(dims[-2])] if len(dims) >= 2 else 1
            nx = da.shape[dims.index(dims[-1])] if len(dims) >= 1 else 1
            x1 = min(x1, nx - 1)
            y1 = min(y1, ny - 1)

        for sp, val in [(self.subset_x0_spin, x0), (self.subset_x1_spin, x1),
                        (self.subset_y0_spin, y0), (self.subset_y1_spin, y1)]:
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)

        w = x1 - x0 + 1
        h = y1 - y0 + 1
        self.subset_size_label.setText(f"Selection: {w} x {h} px")

    def _subset_refresh_map(self):
        """Redraw the subset tab's map with the current band image."""
        if self.ds is None:
            return
        if not hasattr(self, 'current_data_array') or self.current_data_array is None:
            return

        img = self.current_data_array
        cmap_name = self.colormap_combo.currentText()
        chosen_cmap = element_colormaps.get(cmap_name, 'viridis')
        vmin = float(self.vmin_spin.value())
        vmax = float(self.vmax_spin.value())

        self._draw_image_on(
            self.subset_ax, self.subset_canvas, 'subset_colorbar',
            getattr(self, 'last_band_label', ''), img, chosen_cmap, vmin, vmax
        )

        ny, nx = img.shape[:2]
        for sp, mx in [(self.subset_x0_spin, nx - 1), (self.subset_x1_spin, nx - 1),
                       (self.subset_y0_spin, ny - 1), (self.subset_y1_spin, ny - 1)]:
            sp.setMaximum(mx)

    def _subset_clear_selection(self):
        """Reset the rectangle selection."""
        for sp in (self.subset_x0_spin, self.subset_x1_spin,
                   self.subset_y0_spin, self.subset_y1_spin):
            sp.blockSignals(True); sp.setValue(0); sp.blockSignals(False)
        self.subset_size_label.setText("Selection: — x — px")
        self.subset_status_label.setText("")
        if hasattr(self, '_subset_rect_selector'):
            self._subset_rect_selector.set_visible(False)
            self.subset_canvas.draw_idle()

    def _on_tab_changed_subset(self, index):
        """Auto-refresh the subset map when the Export tab is selected."""
        if hasattr(self, 'tabExport') and self.tabs.widget(index) is self.tabExport:
            self._subset_refresh_map()

    def _subset_extract_and_save(self):
        """Slice the current dataset to the selected rectangle and save as NetCDF."""
        if self.ds is None:
            QMessageBox.information(self, "Subset", "No dataset loaded.")
            return

        x0 = self.subset_x0_spin.value()
        x1 = self.subset_x1_spin.value()
        y0 = self.subset_y0_spin.value()
        y1 = self.subset_y1_spin.value()

        if x0 >= x1 or y0 >= y1:
            QMessageBox.warning(self, "Subset", "Invalid selection — make sure the rectangle has non-zero area.")
            return

        # Determine spatial dimension names from the first data variable
        first_var = next(iter(self.ds.data_vars))
        dims = list(self.ds[first_var].dims)
        if len(dims) < 2:
            QMessageBox.warning(self, "Subset", "Dataset does not have at least 2 spatial dimensions.")
            return

        y_dim = dims[-2]
        x_dim = dims[-1]

        # Slice (x1/y1 are inclusive pixel indices, isel uses exclusive end)
        ds_sub = self.ds.isel({y_dim: slice(y0, y1 + 1), x_dim: slice(x0, x1 + 1)})

        # Suggest filename
        default_name = ""
        if self.loaded_cube_path:
            base, ext = os.path.splitext(self.loaded_cube_path)
            default_name = f"{base}_subset_x{x0}-{x1}_y{y0}-{y1}{ext}"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Subset Cube", default_name,
            "NetCDF Files (*.nc);;All Files (*)"
        )
        if not path:
            return

        try:
            prog = QProgressDialog("Extracting subset…", "Cancel", 0, 3, self)
            prog.setWindowTitle("Cube Subset Extraction")
            prog.setWindowModality(Qt.WindowModal)
            prog.setMinimumDuration(0)
            prog.setValue(0)
            QApplication.processEvents()

            ds_out = ds_sub.copy(deep=True)
            if self.norm_combo.currentText() != "None":
                for var in ds_out.data_vars:
                    ds_out[var].encoding.clear()
            if prog.wasCanceled():
                return
            prog.setLabelText("Adding provenance attributes…")
            prog.setValue(1)
            QApplication.processEvents()

            w = x1 - x0 + 1
            h = y1 - y0 + 1
            bands_arr = self._get_bands_values(ds=ds_out, warn=False)
            n_bands = len(bands_arr) if bands_arr is not None else 0
            ds_out.attrs["subset_x_range"] = f"{x0}:{x1}"
            ds_out.attrs["subset_y_range"] = f"{y0}:{y1}"
            ds_out.attrs["subset_x_pixels"] = w
            ds_out.attrs["subset_y_pixels"] = h
            ds_out.attrs["subset_n_bands"] = n_bands
            ds_out.attrs["subset_shape"] = f"({n_bands}, {h}, {w})"
            if self.loaded_cube_path:
                ds_out.attrs["subset_source_file"] = os.path.basename(self.loaded_cube_path)
                first_var = next(iter(self.ds.data_vars))
                orig_dims = self.ds[first_var].shape
                ds_out.attrs["subset_original_shape"] = str(orig_dims)

            if prog.wasCanceled():
                return
            prog.setLabelText("Writing NetCDF file…")
            prog.setValue(2)
            QApplication.processEvents()

            ds_out.to_netcdf(path)

            prog.setValue(3)
            basename = os.path.basename(path)
            self.subset_status_label.setText(f"Saved: {basename}\n({w} x {h} px, {n_bands} bands)")
            self.statusBar().showMessage(f"Subset saved: {basename}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _clear_metadata_tree(self, message: str):
        if not hasattr(self, "metadata_tree"):
            return
        self.metadata_tree.clear()
        root = QTreeWidgetItem(self.metadata_tree, ["Metadata", ""])
        QTreeWidgetItem(root, [message, ""])
        self.metadata_tree.expandAll()

    @staticmethod
    def _format_meta_value(value) -> str:
        try:
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                if len(value) > 20:
                    return f"{value[:20]} ... (len={len(value)})"
                return ", ".join(str(v) for v in value)
        except Exception:
            pass
        return str(value)

    def _populate_metadata_tab(self, file_path: Optional[str] = None):
        if not hasattr(self, "metadata_tree"):
            return
        if not file_path:
            file_path = self.loaded_cube_path
        if not file_path or self.original_ds is None:
            self._clear_metadata_tree("No NetCDF file loaded.")
            return

        self.metadata_tree.clear()

        # Global attributes (preferred order, then extras)
        global_root = QTreeWidgetItem(self.metadata_tree, ["Global attributes", ""])
        attrs = dict(self.original_ds.attrs) if self.original_ds is not None else {}
        preferred = [
            "created_utc",
            "integration_time_ms", "integration_delay_ns", "laser_delay_ns",
            "averages",
            "laser_frequency_hz",
            "step_size_mm", "steps_per_mm", "step_trigger_steps",
            "x_travel_mm", "x_feed_mm_min",
            "mapping_X_points", "mapping_Y_points",
            "mapping_X_size_mm", "mapping_Y_size_mm",
            "bands_total", "devices_total",
            "progress_selection", "progress_tol_nm"
        ]

        # Keys to highlight in bold (acquisition time, scan dimensions, step size)
        highlight_keys = {
            "created_utc",
            "mapping_X_size_mm", "mapping_Y_size_mm",
            "step_size_mm",
        }

        bold_font = self.metadata_tree.font()
        bold_font.setBold(True)

        for key in preferred:
            if key in attrs:
                item = QTreeWidgetItem(global_root, [key, self._format_meta_value(attrs[key])])
                if key in highlight_keys:
                    item.setFont(0, bold_font)
                    item.setFont(1, bold_font)

        other_keys = sorted(k for k in attrs.keys() if k not in preferred)
        if other_keys:
            other_root = QTreeWidgetItem(global_root, ["Other attributes", ""])
            for key in other_keys:
                QTreeWidgetItem(other_root, [key, self._format_meta_value(attrs[key])])
        elif not attrs:
            QTreeWidgetItem(global_root, ["(none)", ""])

        # Device groups (if present in the NetCDF structure)
        devices_root = QTreeWidgetItem(self.metadata_tree, ["Spectrometers used", ""])
        devices_found = False
        try:
            import netCDF4  # optional dependency
            with netCDF4.Dataset(file_path, "r") as nc:
                if "devices" in nc.groups:
                    devs = nc.groups["devices"]
                    for dev_name in sorted(devs.groups.keys()):
                        devices_found = True
                        dev_grp = devs.groups[dev_name]
                        dev_item = QTreeWidgetItem(devices_root, [dev_name, ""])
                        if dev_grp.__dict__:
                            for k in sorted(dev_grp.__dict__.keys()):
                                QTreeWidgetItem(dev_item, [k, self._format_meta_value(dev_grp.__dict__[k])])
                        else:
                            QTreeWidgetItem(dev_item, ["(no attributes)", ""])
        except ImportError:
            QTreeWidgetItem(devices_root, ["(netCDF4 not installed)", ""])
        except Exception as e:
            QTreeWidgetItem(devices_root, ["(error reading devices)", str(e)])

        if not devices_found:
            QTreeWidgetItem(devices_root, ["(no device groups)", ""])

        # Expand global attributes, but keep spectrometers collapsed
        global_root.setExpanded(True)
        devices_root.setExpanded(False)

    # ====== SIGNALS ======
    def _connect_ui(self):
        # Menubar
        self.actionOpen.triggered.connect(self.open_file)
        self.actionExportUnified.triggered.connect(self.export_dialog)
        self.actHelp.triggered.connect(self._open_help)
        self.actAbout.triggered.connect(self._show_about)

        # Imaging controls / autoscale
        self.slider.valueChanged.connect(self.update_plot)
        self.wl_input_spin.editingFinished.connect(self._on_wl_input_go)
        self.element_combo.currentIndexChanged.connect(self._on_common_combo_changed)
        self.element_combo_spec.currentIndexChanged.connect(self._on_spec_combo_changed)
        self.colormap_combo.currentIndexChanged.connect(self.update_plot)
        self.divide_checkbox.toggled.connect(self.update_plot)
        self.divider_spin.valueChanged.connect(self.update_plot)
        self.div_element_combo.currentIndexChanged.connect(self._on_div_common_combo_changed)
        self.div_element_combo_spec.currentIndexChanged.connect(self._on_div_spec_combo_changed)
        self.div_min_spin.valueChanged.connect(self.update_plot)
        self.div_scale_spin.valueChanged.connect(self.update_plot)
        self.um_axes_checkbox.toggled.connect(self.update_plot)
        self.mm_per_px_spin.valueChanged.connect(self.update_plot)
        self.axes_units_combo.currentIndexChanged.connect(self.update_plot)
        self.periodic_table_btn.clicked.connect(self._open_periodic_table)
        self.norm_combo.currentIndexChanged.connect(self._update_norm_fields)
        self.norm_apply_btn.clicked.connect(self.apply_normalization)
        self.norm_save_btn.clicked.connect(self.save_normalized_cube)
        self._update_norm_fields()
        # Cube-wide baseline correction
        self.cube_baseline_combo.currentIndexChanged.connect(self._cube_baseline_on_method_changed)
        self.cube_bl_apply_btn.clicked.connect(self._cube_baseline_apply)
        self.cube_bl_reset_btn.clicked.connect(self._cube_baseline_reset)
        self.cube_bl_preview_btn.clicked.connect(self._cube_baseline_preview)
        self.cube_bl_save_btn.clicked.connect(self._cube_baseline_save)
        self._cube_baseline_on_method_changed()
        self.autoscale_checkbox.toggled.connect(self.on_autoscale_toggle)
        self.pmin_spin.valueChanged.connect(self.update_plot)
        self.pmax_spin.valueChanged.connect(self.update_plot)
        self.vmin_slider.valueChanged.connect(self.on_vmin_slider)
        self.vmax_slider.valueChanged.connect(self.on_vmax_slider)
        self.vmin_spin.valueChanged.connect(self.on_vmin_spin)
        self.vmax_spin.valueChanged.connect(self.on_vmax_spin)

        # Science tools + masking
        self.line_mode_button.toggled.connect(self._ensure_mode_exclusive)
        self.pixel_mode_button.toggled.connect(self._ensure_mode_exclusive)
        self.ignore_null_checkbox.stateChanged.connect(self.update_line_plot)
        self.parallel_line_spinbox.valueChanged.connect(self.update_line_plot)
        self.peak_detection_checkbox.stateChanged.connect(self.update_pixel_plot)
        self.prominence_spinbox.valueChanged.connect(self.update_pixel_plot)
        self.distance_spinbox.valueChanged.connect(self.update_pixel_plot)
        self.save_button.clicked.connect(self.save_line_plot_data)
        self.erase_button.clicked.connect(self.erase_line)

        # Mouse events — LIBS view (Science top)
        self.canvas.mpl_connect('button_press_event', self.on_click_libs)
        self.canvas.mpl_connect('button_release_event', self.on_release_shared)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_shared)

        # Photo tab events
        self.btn_load_photo.clicked.connect(self.load_photo)
        self.btn_poly_clear.clicked.connect(self.clear_photo_polygon)
        self.photo_canvas_tab.mpl_connect('button_press_event', self.on_photo_tab_press)
        self.photo_canvas_tab.mpl_connect('button_release_event', self.on_photo_tab_release)
        self.photo_canvas_tab.mpl_connect('motion_notify_event', self.on_photo_tab_motion)
        self.photo_canvas_tab.mpl_connect('key_press_event', self.on_photo_tab_key)
        self.btn_calib_start.clicked.connect(self.start_calibration)
        self.btn_calib_clear.clicked.connect(self.clear_calibration)
        self.chk_drag_calib.toggled.connect(self.toggle_drag_calib)

        # Photo panel in Data Extraction
        self.photo_canvas_sci.mpl_connect('button_press_event', self.on_click_img)
        self.photo_canvas_sci.mpl_connect('button_release_event', self.on_release_shared)
        self.photo_canvas_sci.mpl_connect('motion_notify_event', self.on_motion_shared)

        self.actProjNew.triggered.connect(self._proj_new)
        self.actProjOpen.triggered.connect(self._proj_open)
        self.actProjSave.triggered.connect(self._proj_save)
        self.actProjSaveAs.triggered.connect(self._proj_save_as)
        self.actAddExperiment.triggered.connect(self._proj_add_experiment)
        self.actLoadExperiment.triggered.connect(self._proj_load_experiment)
        self.actRenameExperiment.triggered.connect(self._proj_rename_experiment)
        self.actDeleteExperiment.triggered.connect(self._proj_delete_experiment)
        self.actUpdateExperiment.triggered.connect(self._proj_update_current_experiment)

        self.chk_use_project_rois.toggled.connect(self._roi_refresh_lists)
        
        self.btn_add_line.clicked.connect(self._roi_add_line_from_current)
        self.btn_use_line.clicked.connect(self._roi_use_selected_line)
        self.btn_rename_line.clicked.connect(self._roi_rename_selected_line)
        self.btn_remove_line.clicked.connect(self._roi_remove_selected_line)
        self.btn_add_custom_line.clicked.connect(self._roi_add_custom_line)
        self.btn_draw_custom_line.clicked.connect(self._roi_draw_custom_line)
        
        self.btn_add_pixel.clicked.connect(self._roi_add_pixel_from_current)
        self.btn_use_pixel.clicked.connect(self._roi_use_selected_pixel)
        self.btn_rename_pixel.clicked.connect(self._roi_rename_selected_pixel)
        self.btn_remove_pixel.clicked.connect(self._roi_remove_selected_pixel)
        
        self.list_lines.itemDoubleClicked.connect(lambda _: self._roi_use_selected_line())
        self.list_pixels.itemDoubleClicked.connect(lambda _: self._roi_use_selected_pixel())
        self.list_lines.itemClicked.connect(lambda _: self._roi_use_selected_line())
        self.list_pixels.itemClicked.connect(lambda _: self._roi_use_selected_pixel())

        self.nan_tolerance_spin.valueChanged.connect(self.update_line_plot)

        # Subset tab
        self.subset_save_btn.clicked.connect(self._subset_extract_and_save)
        self.subset_clear_btn.clicked.connect(self._subset_clear_selection)
        self.subset_refresh_btn.clicked.connect(self._subset_refresh_map)
        self.tabs.currentChanged.connect(self._on_tab_changed_subset)
        self.tabs.currentChanged.connect(self._mask_on_tab_changed)

        # Cube Utils tab
        self.cu_preview_btn.clicked.connect(self._cubeutils_preview_reorder)
        self.cu_apply_reorder_btn.clicked.connect(self._cubeutils_apply_reorder)
        self.cu_remove_preview_btn.clicked.connect(self._cubeutils_preview_remove)
        self.cu_remove_apply_btn.clicked.connect(self._cubeutils_apply_remove)
        self.cu_save_btn.clicked.connect(self._cubeutils_save_cube)
        self.tabs.currentChanged.connect(self._cubeutils_on_tab_changed)

        # Mask tab connections
        self.mask_band_slider.valueChanged.connect(self._mask_update_map)
        self.mask_thresh_apply_btn.clicked.connect(self._mask_on_threshold_apply)
        self.mask_rect_btn.toggled.connect(self._mask_tool_changed)
        self.mask_brush_btn.toggled.connect(self._mask_tool_changed)
        self.mask_poly_btn.toggled.connect(self._mask_tool_changed)
        self.mask_clear_btn.clicked.connect(self._mask_clear_all)
        self.mask_invert_btn.clicked.connect(self._mask_invert)
        self.mask_load_btn.clicked.connect(self._mask_load)
        self.mask_save_btn.clicked.connect(self._mask_save)
        self.mask_apply_cube_btn.clicked.connect(self._mask_apply_to_cube)
        self.mask_canvas.mpl_connect('button_press_event', self._mask_on_press)
        self.mask_canvas.mpl_connect('motion_notify_event', self._mask_on_motion)
        self.mask_canvas.mpl_connect('button_release_event', self._mask_on_release)

        self.baseline_apply_btn.clicked.connect(self._apply_baseline_once)
        self.record_view_btn.clicked.connect(self._on_record_view)
        self.clear_view_btn.clicked.connect(self._on_clear_view)

        # Baseline Inspector
        self.bi_load_btn.clicked.connect(self._bi_load_spectrum)
        self.bi_update_btn.clicked.connect(self._bi_update_plot)

        # Composite Overlay tab
        self._connect_rgb_signals()

    # ====== Element combo mutual exclusion ======
    def _on_wl_input_go(self):
        """Jump the band slider to the nearest band for the entered wavelength."""
        if self.ds is None:
            return
        bands = self._get_bands_values()
        if bands is None:
            return
        target = float(self.wl_input_spin.value())
        _, idx = self.find_closest_band(bands, target)
        self.element_combo.blockSignals(True)
        self.element_combo.setCurrentIndex(0)
        self.element_combo.blockSignals(False)
        self.element_combo_spec.blockSignals(True)
        self.element_combo_spec.setCurrentIndex(0)
        self.element_combo_spec.blockSignals(False)
        self.slider.setValue(idx)

    def _on_common_combo_changed(self, index):
        if self.element_combo.currentText() != 'None':
            self.element_combo_spec.blockSignals(True)
            self.element_combo_spec.setCurrentIndex(0)  # reset to 'None'
            self.element_combo_spec.blockSignals(False)
        self.update_plot()

    def _on_spec_combo_changed(self, index):
        if self.element_combo_spec.currentText() != 'None':
            self.element_combo.blockSignals(True)
            self.element_combo.setCurrentIndex(0)  # reset to 'None'
            self.element_combo.blockSignals(False)
        self.update_plot()

    def _active_element(self) -> str:
        """Return the currently active element name from either combo, or 'None'."""
        common = self.element_combo.currentText()
        if common != 'None':
            return common
        spec = self.element_combo_spec.currentText()
        if spec != 'None':
            return spec
        return 'None'

    def _active_div_element(self) -> str:
        """Return the currently active denominator element name, or 'None'."""
        common = self.div_element_combo.currentText()
        if common != 'None':
            return common
        spec = self.div_element_combo_spec.currentText()
        if spec != 'None':
            return spec
        return 'None'

    def _sync_divider_spin_from_element(self):
        """Update the denominator wavelength spinbox from the currently selected element."""
        element = self._active_div_element()
        if element == 'None':
            return
        target = element_wavelengths.get(element)
        if target is None:
            return
        wl = float(np.mean(target)) if isinstance(target, (list, tuple)) else float(target)
        self.divider_spin.blockSignals(True)
        self.divider_spin.setValue(wl)
        self.divider_spin.blockSignals(False)
        if not self.divide_checkbox.isChecked():
            self.divide_checkbox.blockSignals(True)
            self.divide_checkbox.setChecked(True)
            self.divide_checkbox.blockSignals(False)

    def _on_div_common_combo_changed(self, _index):
        if self.div_element_combo.currentText() != 'None':
            self.div_element_combo_spec.blockSignals(True)
            self.div_element_combo_spec.setCurrentIndex(0)
            self.div_element_combo_spec.blockSignals(False)
        self._sync_divider_spin_from_element()
        self.update_plot()

    def _on_div_spec_combo_changed(self, _index):
        if self.div_element_combo_spec.currentText() != 'None':
            self.div_element_combo.blockSignals(True)
            self.div_element_combo.setCurrentIndex(0)
            self.div_element_combo.blockSignals(False)
        self._sync_divider_spin_from_element()
        self.update_plot()

    def _open_periodic_table(self):
        """Open the periodic table dialog and apply the selected line."""
        dlg = PeriodicTableDialog(element_wavelengths, self)
        if dlg.exec_() == QDialog.Accepted and dlg.selected_line_key:
            key = dlg.selected_line_key
            # Try common combo first, then specialized
            idx = self.element_combo.findText(key)
            if idx >= 0:
                self.element_combo.setCurrentIndex(idx)
                return
            idx = self.element_combo_spec.findText(key)
            if idx >= 0:
                self.element_combo_spec.setCurrentIndex(idx)
                return
            # Line exists in the merged dict but not in either combo —
            # set slider to the nearest band directly
            target = element_wavelengths.get(key)
            if target is not None:
                bands = self._get_bands_values(warn=False)
                if bands is not None:
                    if isinstance(target, (list, tuple)):
                        wl = float(target[0])
                    else:
                        wl = float(target)
                    _, band_idx = self.find_closest_band(bands, wl)
                    self.element_combo.setCurrentIndex(0)
                    self.element_combo_spec.setCurrentIndex(0)
                    self.slider.setValue(band_idx)

    # ====== Percentile autoscale toggle ======
    def on_autoscale_toggle(self, checked: bool):
        for w in (self.vmin_spin, self.vmax_spin, self.vmin_slider, self.vmax_slider):
            w.setEnabled(not checked)
        self.update_plot()

    # ====== vmin/vmax sync (manual mode) ======
    def on_vmin_slider(self, val: int):
        if self.autoscale_checkbox.isChecked(): return
        if val >= self.vmax_slider.value():
            new_max = min(val + 1, self.vmax_slider.maximum())
            if new_max != self.vmax_slider.value():
                self.vmax_slider.blockSignals(True); self.vmax_slider.setValue(new_max); self.vmax_slider.blockSignals(False)
            self.vmax_spin.blockSignals(True); self.vmax_spin.setValue(new_max); self.vmax_spin.blockSignals(False)
        self.vmin_spin.blockSignals(True); self.vmin_spin.setValue(val); self.vmin_spin.blockSignals(False)
        self.update_plot()

    def on_vmax_slider(self, val: int):
        if self.autoscale_checkbox.isChecked(): return
        if val <= self.vmin_slider.value():
            new_min = max(val - 1, self.vmin_slider.minimum())
            if new_min != self.vmin_slider.value():
                self.vmin_slider.blockSignals(True); self.vmin_slider.setValue(new_min); self.vmin_slider.blockSignals(False)
            self.vmin_spin.blockSignals(True); self.vmin_spin.setValue(new_min); self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(True); self.vmax_spin.setValue(val); self.vmax_spin.blockSignals(False)
        self.update_plot()

    def on_vmin_spin(self, val: float):
        if self.autoscale_checkbox.isChecked(): return
        val_i = int(val)
        if val_i >= self.vmax_slider.value():
            new_max = min(val_i + 1, self.vmax_slider.maximum())
            self.vmax_slider.blockSignals(True); self.vmax_slider.setValue(new_max); self.vmax_slider.blockSignals(False)
            self.vmax_spin.blockSignals(True); self.vmax_spin.setValue(new_max); self.vmax_spin.blockSignals(False)
        self.vmin_slider.blockSignals(True); self.vmin_slider.setValue(val_i); self.vmin_slider.blockSignals(False)
        self.update_plot()

    def on_vmax_spin(self, val: float):
        if self.autoscale_checkbox.isChecked(): return
        val_i = int(val)
        if val_i <= self.vmin_slider.value():
            new_min = max(val_i - 1, self.vmin_slider.minimum())
            self.vmin_slider.blockSignals(True); self.vmin_slider.setValue(new_min); self.vmin_slider.blockSignals(False)
            self.vmin_spin.blockSignals(True); self.vmin_spin.setValue(new_min); self.vmin_spin.blockSignals(False)
        self.vmax_slider.blockSignals(True); self.vmax_slider.setValue(val_i); self.vmax_slider.blockSignals(False)
        self.update_plot()

    # ====== Helpers ======
    @staticmethod
    def find_closest_band(available_bands, desired_band):
        closest_idx = np.abs(available_bands - desired_band).argmin()
        return available_bands[closest_idx], closest_idx

    def _get_bands_values(self, ds=None, warn=True):
        target = ds if ds is not None else self.ds
        if target is None:
            if warn:
                QMessageBox.warning(self, "Dataset issue", "No dataset loaded.")
            return None
        if 'bands' not in target.coords:
            if warn:
                QMessageBox.warning(self, "Dataset issue", "Expected coordinate 'bands' not found.")
            return None
        return np.asarray(target.coords['bands'].values, dtype=float)

    # ====== Spectrometer segments (multi-spectrometer cubes) ======
    def _read_device_groups(self):
        """Best-effort read of the 'devices' groups from the NetCDF file."""
        if not self.loaded_cube_path:
            return []
        try:
            import netCDF4  # optional dependency
            with netCDF4.Dataset(self.loaded_cube_path, "r") as nc:
                if "devices" not in nc.groups:
                    return []
                devs = nc.groups["devices"]
                return [{'name': n, 'attrs': dict(devs.groups[n].__dict__)}
                        for n in sorted(devs.groups.keys())]
        except Exception:
            return []

    def _get_spectrometer_segments(self):
        """Split the band axis into per-spectrometer segments.

        Primary detection uses the wavelength axis itself: in a stitched
        multi-spectrometer cube the wavelength restarts at a lower value at
        each spectrometer boundary (works even for old cubes without device
        metadata). If the axis is monotonic, per-device channel counts from
        the NetCDF 'devices' metadata are used as a fallback.

        Returns a list of dicts {'name', 'i0', 'i1', 'wl0', 'wl1'} in band
        order (i0/i1 are inclusive band indices).
        """
        bands = self._get_bands_values(warn=False)
        if bands is None or bands.size == 0:
            return []
        n = int(bands.size)
        starts = [0] + [i for i in range(1, n) if bands[i] <= bands[i - 1]]
        segs = []
        for j, s in enumerate(starts):
            e = (starts[j + 1] - 1) if j + 1 < len(starts) else n - 1
            segs.append({'i0': int(s), 'i1': int(e),
                         'wl0': float(bands[s]), 'wl1': float(bands[e])})

        devices = self._read_device_groups()
        if len(segs) == 1 and len(devices) > 1:
            # Monotonic axis (e.g. re-sorted cube): try per-device channel counts.
            counts = []
            for d in devices:
                cnt = None
                for k, v in d['attrs'].items():
                    kl = k.lower()
                    if any(t in kl for t in ('bands', 'channels', 'points', 'pixels')):
                        try:
                            iv = int(np.asarray(v).ravel()[0])
                            if iv > 1:
                                cnt = iv
                                break
                        except Exception:
                            pass
                counts.append(cnt)
            if all(c is not None for c in counts) and sum(counts) == n:
                segs = []
                off = 0
                for c in counts:
                    segs.append({'i0': off, 'i1': off + c - 1,
                                 'wl0': float(bands[off]), 'wl1': float(bands[off + c - 1])})
                    off += c

        names = [d['name'] for d in devices] if len(devices) == len(segs) else [None] * len(segs)
        for s, nm in zip(segs, names):
            s['name'] = nm
        return segs

    @staticmethod
    def _parse_spectro_selection(text, n_seg):
        """Parse a '1-3,5' style spectrometer selection into 0-based indices.

        Accepts commas/semicolons/spaces as separators and '-' or ':' for
        ranges. Empty text or 'all' selects every spectrometer.
        """
        text = (text or "").strip()
        if not text or text.lower() in ('all', '*'):
            return list(range(n_seg))
        picked = set()
        for token in text.replace(';', ',').replace(' ', ',').split(','):
            token = token.strip()
            if not token:
                continue
            sep = '-' if '-' in token else (':' if ':' in token else None)
            try:
                if sep:
                    a, b = token.split(sep, 1)
                    a, b = int(a), int(b)
                    if a > b:
                        a, b = b, a
                    values = range(a, b + 1)
                else:
                    values = [int(token)]
            except ValueError:
                raise ValueError(
                    f"Could not parse spectrometer selection '{token}'.\n"
                    "Use e.g. '1-3', '1,3,5' or 'all'.")
            for v in values:
                if not (1 <= v <= n_seg):
                    raise ValueError(f"Spectrometer {v} is out of range (1–{n_seg}).")
                picked.add(v - 1)
        if not picked:
            raise ValueError("Empty spectrometer selection.")
        return sorted(picked)

    def _spectro_selected_band_indices(self, text):
        """Band indices covered by the spectrometer selection, or None for 'all'.

        Raises ValueError for an invalid selection string."""
        segs = self._get_spectrometer_segments()
        if not segs:
            return None
        sel = self._parse_spectro_selection(text, len(segs))
        if len(sel) == len(segs):
            return None
        return np.concatenate([np.arange(segs[i]['i0'], segs[i]['i1'] + 1)
                               for i in sel]).astype(int)

    def _spectro_info_text(self):
        segs = self._get_spectrometer_segments()
        if not segs:
            return ""
        if len(segs) == 1:
            return "Single spectrometer detected — selection has no effect."
        parts = []
        for i, s in enumerate(segs):
            nm = f" ({s['name']})" if s.get('name') else ""
            parts.append(f"{i+1}: {s['wl0']:.0f}–{s['wl1']:.0f} nm{nm}")
        return f"{len(segs)} spectrometers detected — " + " | ".join(parts)

    def _update_spectrometer_info(self):
        """Refresh the spectrometer summary labels (Normalize + Chemometrics tabs)."""
        txt = self._spectro_info_text()
        if hasattr(self, 'norm_spectro_info'):
            self.norm_spectro_info.setText(txt)
        if hasattr(self, 'chem_spectro_info'):
            self.chem_spectro_info.setText(txt)

    # ====== Help / About ======
    def _open_help(self):
        """Open the HTML user guide in the system default browser."""
        import webbrowser
        help_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "help.html")
        if os.path.isfile(help_path):
            webbrowser.open(f"file:///{help_path.replace(os.sep, '/')}")
        else:
            QMessageBox.warning(self, "Help", f"User guide not found:\n{help_path}")

    def _show_about(self):
        """Display the About dialog with version and credits."""
        dlg = QDialog(self)
        dlg.setWindowTitle("About LIBS Hypercube Explorer")
        dlg.setFixedWidth(420)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(28, 28, 28, 20)
        lay.setSpacing(10)

        title = QLabel(APP_NAME)
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #38bdf8;")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        build = QLabel(f"Version &nbsp;<b>{BUILD_VERSION}</b>")
        build.setStyleSheet("font-size: 10pt; color: #000000;")
        build.setAlignment(Qt.AlignCenter)
        build.setTextFormat(Qt.RichText)
        lay.addWidget(build)

        lay.addSpacing(8)

        info = QLabel(
            "Interactive visualization and processing of LIBS hypercubes.<br><br>"
            "<b>Developed at:</b><br>"
            "RBINS-GSB<br>"
            "<b>Dependencies:</b> Python · PyQt5 · xarray · numpy<br>"
            "matplotlib · scipy · pandas · Pillow · scikit-learn"
        )
        info.setStyleSheet("font-size: 9pt; color: #00000; line-height: 1.6;")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignCenter)
        info.setTextFormat(Qt.RichText)
        lay.addWidget(info)

        lay.addSpacing(6)

        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec_()

    # ====== IO ======
    def open_file(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open NetCDF File", "", "NetCDF Files (*.nc);;All Files (*)")
        if not fileName: return
        try:
            self.original_ds = xr.open_dataset(fileName)
            self.ds = self.original_ds.copy()
        except Exception as e:
            QMessageBox.critical(self, "Error opening file", str(e)); return
        bands = self._get_bands_values()
        if bands is None:
            return
        self.slider.setMaximum(len(bands) - 1)
        if hasattr(self, 'mask_band_slider'):
            self.mask_band_slider.setMaximum(len(bands) - 1)
 
        self.loaded_cube_path = fileName
        base = os.path.basename(fileName)
        self.setWindowTitle(f"Hypercube Explorer — {base}")
        self.load_file_label.setText(f"Loaded: {base}")
        self._populate_metadata_tab(fileName)

        self._update_roi_spinbox_ranges()

        # Reset normalization state
        self.norm_combo.setCurrentIndex(0)
        self.norm_status_label.setText("")

        # Reset baseline-correction state
        self._baseline_ds = None
        self._baseline_method_applied = "None"
        if hasattr(self, 'cube_baseline_combo'):
            self.cube_baseline_combo.blockSignals(True)
            self.cube_baseline_combo.setCurrentIndex(0)
            self.cube_baseline_combo.blockSignals(False)
            self._cube_baseline_on_method_changed()
            self.cube_bl_status_label.setText("")

        # Reset chemometrics results (band range defaults to the new cube)
        self._chem_reset()

        self.update_plot()

    # ====== Export (unified) ======
    def export_dialog(self):
        dlg = ExportDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        opts = dlg.values()
        if not opts["path"]:
            QMessageBox.warning(self, "Export", "Please choose an output path.")
            return

        which = opts["which"]
        if which.startswith("Map Explorer"):
            fig = self.map_canvas.figure; cbar = self.map_colorbar
            self._export_figure(fig, cbar, opts)
        elif which.startswith("Data extraction image (LIBS"):
            fig = self.canvas.figure; cbar = self.colorbar
            self._export_figure(fig, cbar, opts)
        elif which.startswith("Data extraction image (Photo"):
            if hasattr(self, 'photo_canvas_sci'):
                fig = self.photo_canvas_sci.figure; cbar = None
                self._export_figure(fig, cbar, opts)
        elif which.startswith("Data extraction montage"):
            if self.photo_img is None or self.current_data_array is None:
                QMessageBox.warning(self, "Export montage", "Load both a LIBS cube and a Photo first.")
                return
            self._export_montage(opts)
        elif which == "Composite overlay image":
            if not hasattr(self, 'current_rgb_composite') or self.current_rgb_composite is None:
                QMessageBox.warning(self, "Export", "No composite available. Generate one first.")
                return
            fig = self.comp_canvas.figure; cbar = None
            self._export_figure(fig, cbar, opts)
        elif which.startswith("Composite overlay raw"):
            if not hasattr(self, 'current_rgb_composite') or self.current_rgb_composite is None:
                QMessageBox.warning(self, "Export", "No composite available. Generate one first.")
                return
            self._export_raw_rgb_composite(opts)
        elif which.startswith("Raw pixel map"):
            if self.current_data_array is None:
                QMessageBox.warning(self, "Export", "No LIBS map data available.")
                return
            self._export_raw_pixel_map(opts)
        else:
            fig = self.line_plot_canvas.figure; cbar = None
            self._export_figure(fig, cbar, opts)

    def _export_figure(self, fig, cbar, opts):
        old_size = fig.get_size_inches().copy()
        old_fonts = []
        for ax in fig.axes:
            old_fonts.append((
                ax.title.get_fontsize(), ax.xaxis.label.get_size(), ax.yaxis.label.get_size(),
                [t.get_fontsize() for t in ax.get_xticklabels()],
                [t.get_fontsize() for t in ax.get_yticklabels()]
            ))

        w_in = opts["width_cm"] / 2.54; h_in = opts["height_cm"] / 2.54
        try:
            fig.set_size_inches((w_in, h_in), forward=False)
            for ax in fig.axes:
                ax.title.set_fontsize(opts["title_fs"])
                ax.xaxis.label.set_size(opts["axis_fs"]); ax.yaxis.label.set_size(opts["axis_fs"])
                for t in ax.get_xticklabels(): t.set_fontsize(opts["axis_fs"])
                for t in ax.get_yticklabels(): t.set_fontsize(opts["axis_fs"])
            if cbar is not None:
                try:
                    cbar.ax.yaxis.label.set_size(opts["axis_fs"])
                    for t in cbar.ax.get_yticklabels(): t.set_fontsize(opts["axis_fs"])
                except Exception:
                    pass
            try: fig.tight_layout()
            except Exception: pass
            fig.savefig(opts["path"], dpi=int(opts["dpi"]), bbox_inches='tight')
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
        finally:
            for ax, (tfs, xfs, yfs, xts, yts) in zip(fig.axes, old_fonts):
                ax.title.set_fontsize(tfs); ax.xaxis.label.set_size(xfs); ax.yaxis.label.set_size(yfs)
                for lbl, sz in zip(ax.get_xticklabels(), xts): lbl.set_fontsize(sz)
                for lbl, sz in zip(ax.get_yticklabels(), yts): lbl.set_fontsize(sz)
            fig.set_size_inches(old_size, forward=False)
            try: fig.tight_layout()
            except Exception: pass
            fig.canvas.draw_idle()

    def _export_montage(self, opts):
        cmap = element_colormaps.get(self.last_cmap_name, plt.cm.viridis)
        vmin = self.last_vmin if self.last_vmin is not None else np.nanmin(self.current_data_array)
        vmax = self.last_vmax if self.last_vmax is not None else np.nanmax(self.current_data_array)
        band_label = self.last_band_label or "LIBS map"

        fig = plt.figure(figsize=(opts["width_cm"]/2.54, opts["height_cm"]/2.54), dpi=int(opts["dpi"]))
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.25, figure=fig)

        ax1 = fig.add_subplot(gs[0])
        im = ax1.imshow(self.current_data_array, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title(band_label, fontsize=opts["title_fs"])
        ax1.set_xticks([]); ax1.set_yticks([])
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("LIBS Intensity (a.u.)", fontsize=opts["axis_fs"])
        for t in cbar.ax.get_yticklabels(): t.set_fontsize(opts["axis_fs"])

        if self.shared_line is not None:
            y0, x0, y1, x1 = self.shared_line
            ax1.plot([x0, x1], [y0, y1], '-r', lw=1.2)
        if self.pixel_marker_libs is not None and self.current_pixel is not None:
            yl, xl = self.current_pixel
            ax1.plot([xl], [yl], 'o', ms=4, mfc='none', mec='red', mew=1.2)

        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(self.photo_img)
        ax2.set_title("Photo", fontsize=opts["title_fs"])
        ax2.set_xticks([]); ax2.set_yticks([])

        if len(self.photo_polygon) >= 2:
            xs, ys = zip(*self.photo_polygon)
            ax2.plot(xs, ys, '-r', lw=1.2)

        self._draw_libs_footprint(ax2)

        if self.shared_line is not None:
            y0, x0, y1, x1 = self.shared_line
            p0 = self._libs_to_img(x0, y0)
            p1 = self._libs_to_img(x1, y1)
            if p0 is not None and p1 is not None:
                ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], '-r', lw=1.0)
        if self.current_pixel is not None:
            yl, xl = self.current_pixel
            p = self._libs_to_img(xl, yl)
            if p is not None:
                ax2.plot([p[0]], [p[1]], 'o', ms=4, mfc='none', mec='red', mew=1.2)

        try:
            fig.tight_layout()
        except Exception:
            pass

        try:
            fig.savefig(opts["path"], dpi=int(opts["dpi"]), bbox_inches='tight')
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
        finally:
            plt.close(fig)

    def _export_raw_pixel_map(self, opts):
        """Export the current map as a 1:1 pixel image (no axes, colorbar, or title)."""
        from matplotlib.colors import Normalize
        import PIL.Image

        arr = self.current_data_array
        cmap_name = self.last_cmap_name or 'Viridis'
        cmap = element_colormaps.get(cmap_name, plt.cm.viridis)
        vmin = self.last_vmin if self.last_vmin is not None else float(np.nanmin(arr))
        vmax = self.last_vmax if self.last_vmax is not None else float(np.nanmax(arr))

        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = cmap(norm(arr))  # (H, W, 4) float in [0,1]
        rgb8 = (rgba[:, :, :3] * 255).astype(np.uint8)

        path = opts["path"]
        fmt = opts["format"]
        if fmt == "tiff" and not path.lower().endswith((".tif", ".tiff")):
            path += ".tiff"

        try:
            img = PIL.Image.fromarray(rgb8, mode="RGB")
            img.save(path)
            self.statusBar().showMessage(
                f"Raw pixel map saved: {os.path.basename(path)}  ({rgb8.shape[1]}x{rgb8.shape[0]} px)", 4000
            )
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def _export_raw_rgb_composite(self, opts):
        """Export the current RGB composite as a 1:1 pixel image (no axes or legend)."""
        import PIL.Image

        rgb = self.current_rgb_composite
        rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        path = opts["path"]
        try:
            img = PIL.Image.fromarray(rgb8, mode="RGB")
            img.save(path)
            self.statusBar().showMessage(
                f"Raw RGB composite saved: {os.path.basename(path)}  ({rgb8.shape[1]}x{rgb8.shape[0]} px)", 4000
            )
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    # ====== Axes (µm/mm) ======
    def _apply_spatial_axes(self, ax, arr):
        if not self.um_axes_checkbox.isChecked():
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel(""); return
        mm_per_px = float(self.mm_per_px_spin.value())
        if mm_per_px <= 0:
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel(""); return
        ny, nx = arr.shape[:2]
        Nticks = 6 if min(nx, ny) >= 5 else max(2, min(nx, ny))
        xpix = np.linspace(0, nx - 1, Nticks); ypix = np.linspace(0, ny - 1, Nticks)
        unit = self.axes_units_combo.currentText()
        if unit == "µm":
            factor = mm_per_px * 1000.0; xlabel, ylabel = "x (µm)", "y (µm)"; fmt = lambda v: f"{v:.0f}"
        else:
            factor = mm_per_px; xlabel, ylabel = "x (mm)", "y (mm)"; fmt = lambda v: f"{v:.3g}"
        ax.set_xticks(xpix); ax.set_yticks(ypix)
        ax.set_xticklabels([fmt(px * factor) for px in xpix])
        ax.set_yticklabels([fmt(py * factor) for py in ypix])
        ax.set_xlabel(xlabel, fontsize=7); ax.set_ylabel(ylabel, fontsize=7)

    # ====== Core plotting (LIBS) ======
    def _draw_image_on(self, ax, canvas, colorbar_attr, band_label, arr, cmap, vmin, vmax):
        prev_cbar = getattr(self, colorbar_attr)
        if prev_cbar is not None:
            try: prev_cbar.remove()
            except Exception: pass
            setattr(self, colorbar_attr, None)

        ax.clear()
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(band_label, fontsize=8)
        self._apply_spatial_axes(ax, arr)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = canvas.figure.colorbar(im, cax=cax)
        cbar.set_label('LIBS Intensity (a.u.)', fontsize=8)
        cbar.ax.tick_params(labelsize=6)
        setattr(self, colorbar_attr, cbar)

        if ax is self.ax:
            if self.line_libs is not None: ax.add_line(self.line_libs); self.line_libs.set_zorder(10)
            if self.pixel_marker_libs is not None: self.pixel_marker_libs.set_zorder(11)
        canvas.draw_idle()

    # ====== Update LIBS map + histogram + mirror to Science top ======
    def _apply_baseline_once(self):
        if not self.baseline_checkbox.isChecked():
            QMessageBox.information(self, "Baseline",
                                    "Check the 'Local baseline subtraction' box first.")
            return
        progress = QProgressDialog("Computing local baseline…", None, 0, 0, self)
        progress.setWindowTitle("Baseline")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()
        self._baseline_once = True
        self.update_plot()
        progress.close()

    def _on_record_view(self):
        self._locked_view = get_view(self.comp_ax)
        self._update_view_info_label()
        self.clear_view_btn.setEnabled(True)
        self._apply_locked_view()

    def _on_clear_view(self):
        self._locked_view = None
        self._update_view_info_label()
        self.clear_view_btn.setEnabled(False)

    def _update_view_info_label(self):
        v = self._locked_view
        if v is None:
            self.view_info_label.setText("No view recorded.")
            self.view_info_label.setStyleSheet("font-size:8pt; color:gray;")
            return
        xlim = v.get('xlim')
        ylim = v.get('ylim')
        mm = float(self.mm_per_px_spin.value()) if hasattr(self, 'mm_per_px_spin') else 0
        parts = []
        if xlim:
            w_px = abs(xlim[1] - xlim[0])
            parts.append(f"X: {xlim[0]:.0f} – {xlim[1]:.0f} px")
            if mm > 0:
                parts[-1] += f"  ({w_px * mm:.2f} mm)"
        if ylim:
            h_px = abs(ylim[1] - ylim[0])
            parts.append(f"Y: {min(ylim):.0f} – {max(ylim):.0f} px")
            if mm > 0:
                parts[-1] += f"  ({h_px * mm:.2f} mm)"
        self.view_info_label.setText("\n".join(parts))
        self.view_info_label.setStyleSheet("font-size:8pt; color:#ff6600; font-weight:bold;")

    def _apply_locked_view(self):
        """Apply locked view to Map Explorer, Data Extraction, and Composite maps."""
        v = self._locked_view
        if v is None:
            return
        for ax_name in ('map_ax', 'ax', 'comp_ax'):
            ax = getattr(self, ax_name, None)
            if ax is not None:
                set_view(ax, v)
        for canvas_name in ('map_canvas', 'canvas', 'comp_canvas'):
            c = getattr(self, canvas_name, None)
            if c is not None:
                c.draw_idle()

    def update_plot(self):
        if self.ds is None: return
        bands = self._get_bands_values()
        if bands is None:
            return

        element = self._active_element()
        colormap_name = self.colormap_combo.currentText()
        self.last_cmap_name = colormap_name
        chosen_colormap = element_colormaps[colormap_name]
        
        apply_baseline = getattr(self, '_baseline_once', False)
        self._baseline_once = False
        halfwidth_nm = float(self.baseline_halfwidth_spin.value())
        gap_nm = float(self.baseline_gap_spin.value())
        bl_method = self.baseline_method_combo.currentText()
        use_area = apply_baseline and bl_method == "Peak area"
        
        if element == 'None':
            index = self.slider.value()
            self.index_label.setText(f'Index: {index}')
            selected_band = self.ds.isel(bands=index)
            current_wavelength = float(bands[index])
            if use_area:
                img = self._local_peak_area_map(current_wavelength, halfwidth_nm, gap_nm)
            else:
                img = self._get_band_image(current_wavelength)
                if apply_baseline:
                    base = self._local_baseline_map(current_wavelength, halfwidth_nm, gap_nm)
                    img = img - base
            band_label = f'Band index: {index}, Wavelength: {current_wavelength:.3f} nm'
        else:
            target = element_wavelengths[element]  # can be float or list (doublet)
            available_bands = bands
        
            # Find nearest band(s)
            if isinstance(target, (list, tuple)):
                snapped = []
                idx_text = []
                for wl in target:
                    closest, idx = self.find_closest_band(available_bands, float(wl))
                    snapped.append(float(closest))
                    idx_text.append(f'{closest:.3f}')
                img = self._line_or_doublet_map(snapped, apply_baseline, halfwidth_nm, gap_nm, bl_method)
                self.index_label.setText('Index: ' + ','.join(str(np.where(available_bands==s)[0][0]) for s in snapped if s in available_bands))
                band_label = f'{element} map at {", ".join(idx_text)} nm'
                current_wavelength = float(np.mean(snapped))
            else:
                wl = float(target)
                closest_band, band_index = self.find_closest_band(available_bands, wl)
                self.slider.setValue(band_index)
                self.index_label.setText(f'Index: {band_index}')
                if use_area:
                    img = self._local_peak_area_map(float(closest_band), halfwidth_nm, gap_nm)
                else:
                    img = self._get_band_image(band_index)
                    if apply_baseline:
                        base = self._local_baseline_map(float(closest_band), halfwidth_nm, gap_nm)
                        img = img - base
                band_label = f'{element} map at {float(closest_band):.3f} nm (closest to {wl:.3f} nm)'
                current_wavelength = float(closest_band)
        
        self.wavelength_label.setText(f'Wavelength: {current_wavelength:.3f} nm')
        self.last_band_label = band_label
        if apply_baseline:
            band_label += f"\nBaseline ({bl_method}): ±{halfwidth_nm:.2f} nm (excl. ±{gap_nm:.2f} nm)"
            self.last_band_label = band_label
                
        # Divider (ratio) — baseline-subtracted / peak-area denominator,
        # with optional low-value masking and scale factor.
        if self.divide_checkbox.isChecked():
            try:
                div_element = self._active_div_element()
                avail = bands
                div_target = None
                if div_element != 'None':
                    div_target = element_wavelengths.get(div_element)
                if div_target is None:
                    div_target = float(self.divider_spin.value())

                if isinstance(div_target, (list, tuple)):
                    snapped = []
                    for wl in div_target:
                        closest, _ = self.find_closest_band(avail, float(wl))
                        snapped.append(float(closest))
                    divider = self._line_or_doublet_map(
                        snapped, apply_baseline, halfwidth_nm, gap_nm, bl_method)
                    div_wl_str = ", ".join(f"{w:.3f}" for w in snapped)
                    div_closest = float(np.mean(snapped))
                else:
                    div_closest, div_idx = self.find_closest_band(avail, float(div_target))
                    if use_area:
                        divider = self._local_peak_area_map(float(div_closest), halfwidth_nm, gap_nm)
                    else:
                        divider = self._get_band_image(int(div_idx))
                        if apply_baseline:
                            base_d = self._local_baseline_map(float(div_closest), halfwidth_nm, gap_nm)
                            divider = divider - base_d
                    div_wl_str = f"{float(div_closest):.3f} nm"

                divider = np.asarray(divider, dtype=float)
                num = np.asarray(img, dtype=float)

                # Mask low denominator values (avoid divide-by-noise)
                min_div = float(self.div_min_spin.value())
                valid = np.isfinite(divider) & (divider > max(min_div, 0.0))
                ratio = np.full_like(num, np.nan, dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio[valid] = num[valid] / divider[valid]

                scale = float(self.div_scale_spin.value())
                if scale != 1.0:
                    ratio = ratio * scale

                img = ratio
                num_label = div_element if div_element != 'None' else div_wl_str
                scale_txt = f" ×{scale:g}" if scale != 1.0 else ""
                band_label += f"\nRatio: numerator / {num_label}{scale_txt}"
                if min_div > 0:
                    band_label += f"  |  masked denom. < {min_div:g}"
                self.last_band_label = band_label
            except Exception as e:
                QMessageBox.warning(self, "Ratio error", f"Could not compute ratio:\n{e}")
        
        self.current_data_array = img

        # Histogram data
        data = img.flatten(); data = data[np.isfinite(data)]; data_pos = data[data > 0]

        # vmin/vmax
        if self.autoscale_checkbox.isChecked() and data_pos.size:
            pmin = float(self.pmin_spin.value()); pmax = float(self.pmax_spin.value())
            if pmin >= pmax: pmax = min(100.0, pmin + 0.1)
            vmin = float(np.percentile(data_pos, pmin)); vmax = float(np.percentile(data_pos, pmax))
            if vmin == vmax: vmax = vmin + 1.0
            for w, v in ((self.vmin_spin, vmin), (self.vmax_spin, vmax), (self.vmin_slider, int(vmin)), (self.vmax_slider, int(vmax))):
                w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        else:
            vmin = float(self.vmin_spin.value()); vmax = float(self.vmax_spin.value())
            if vmin >= vmax:
                vmin = float(np.nanmin(data_pos)) if data_pos.size else 0.0
                vmax = float(np.nanmax(data_pos)) if data_pos.size else 1.0
                if vmin == vmax: vmax = vmin + 1.0
                self.vmin_spin.setValue(vmin); self.vmax_spin.setValue(vmax)
                self.vmin_slider.setValue(int(vmin)); self.vmax_slider.setValue(int(vmax))

        self.last_vmin, self.last_vmax = vmin, vmax

        # Draw Imaging + Science LIBS
        self._draw_image_on(self.map_ax, self.map_canvas, 'map_colorbar', band_label, img, chosen_colormap, vmin, vmax)
        self._draw_image_on(self.ax, self.canvas, 'colorbar', band_label, img, chosen_colormap, vmin, vmax)

        # Low-signal warning (skip when no baseline, peak area, or ratio mode)
        LOW_SIGNAL_THRESHOLD = 100.0
        max_val = float(np.nanmax(img)) if img.size else 0.0
        is_ratio_mode = self.divide_checkbox.isChecked()
        if apply_baseline and not use_area and not is_ratio_mode and max_val < LOW_SIGNAL_THRESHOLD:
            self.map_low_signal_label.setText(
                f"\u26a0  Low signal — max pixel intensity is {max_val:.1f} (threshold: "
                f"{LOW_SIGNAL_THRESHOLD:.0f}). Values may represent noise only.")
            self.map_low_signal_label.setVisible(True)
        else:
            self.map_low_signal_label.setVisible(False)

        # Histogram
        self.hist_ax.clear()
        bins = 128 if data_pos.size > 0 else 10
        self.hist_ax.hist(data_pos if data_pos.size else data, bins=bins, log=True, color='gray', alpha=0.7)
        self.hist_ax.axvline(vmin, color='red', linestyle='--')
        self.hist_ax.axvline(vmax, color='blue', linestyle='--')
        self.hist_ax.set_title("LIBS Intensity histogram", fontsize=8, pad=6)
        self.hist_ax.set_xlabel("LIBS Intensity (a.u.)", fontsize=7)
        self.hist_ax.set_ylabel("Count (log)", fontsize=7)
        self.hist_canvas.figure.subplots_adjust(top=0.88, bottom=0.28, left=0.10, right=0.98)
        try: self.hist_canvas.figure.tight_layout(pad=0.4)
        except Exception: pass
        self.hist_canvas.draw_idle()

        # Apply locked view to map axes
        if self._locked_view is not None:
            set_view(self.map_ax, self._locked_view)
            self.map_canvas.draw_idle()
            set_view(self.ax, self._locked_view)
            self.canvas.draw_idle()

        self.redraw_photo_sci()
        
        if self.shared_line is not None:
            self.update_line_plot()

    # ====== Normalization ======
    def _make_norm_progress(self, title: str, maximum: int = 0) -> QProgressDialog:
        """Create a modal progress dialog for normalization."""
        prog = QProgressDialog(title, "Cancel", 0, maximum, self)
        prog.setWindowTitle("Normalization")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        QApplication.processEvents()
        return prog

    def _update_norm_fields(self):
        """Enable only the controls relevant to the currently selected normalization method."""
        method = self.norm_combo.currentText()
        show_cont   = method == "Continuum window"
        show_kernel = method == "Spatial median filter"
        show_spectro = method in ("Total Emission (TEN)", "Total Area (TAN)",
                                  "SNV (Standard Normal Variate)", "Max-norm per pixel")
        for widget, enabled in [
            (self.norm_cont_start,  show_cont),
            (self.norm_cont_end,    show_cont),
            (self.norm_kernel_spin, show_kernel),
            (self.norm_spectro_edit, show_spectro),
        ]:
            widget.setEnabled(enabled)
            lbl = self.norm_form.labelForField(widget)
            if lbl:
                lbl.setEnabled(enabled)

    def apply_normalization(self):
        """Apply selected normalization to self.ds from self.original_ds, then re-apply mask.

        If a cube-wide baseline correction has been applied (self._baseline_ds
        is not None), it is used as the source instead of self.original_ds so
        that normalization sees baseline-corrected spectra.
        """
        if self.original_ds is None:
            return
        method = self.norm_combo.currentText()

        # Validate the spectrometer selection up front (stats-based methods only)
        spectro_note = ""
        if method in ("Total Emission (TEN)", "Total Area (TAN)",
                      "SNV (Standard Normal Variate)", "Max-norm per pixel"):
            try:
                idx = self._spectro_selected_band_indices(self.norm_spectro_edit.text())
                if idx is not None:
                    spectro_note = f" (spectros {self.norm_spectro_edit.text().strip()})"
            except ValueError as e:
                QMessageBox.warning(self, "Normalization", str(e))
                return

        self.norm_status_label.setText("Computing…")
        QApplication.processEvents()

        # Pipeline: baseline-corrected cube (if any) feeds normalization.
        # _norm_* helpers all read from self.original_ds, so temporarily swap it.
        _orig_backup = self.original_ds
        if self._baseline_ds is not None:
            self.original_ds = self._baseline_ds

        try:
            if method == "None":
                self.ds = self.original_ds.copy()
                self.norm_status_label.setText("")
            elif method == "Total Emission (TEN)":
                self.ds = self._norm_ten()
                self.norm_status_label.setText("TEN applied" + spectro_note)
            elif method == "Total Area (TAN)":
                self.ds = self._norm_tan()
                self.norm_status_label.setText("TAN applied" + spectro_note)
            elif method == "Continuum window":
                self.ds = self._norm_continuum()
                self.norm_status_label.setText("Continuum norm applied")
            elif method == "SNV (Standard Normal Variate)":
                self.ds = self._norm_snv()
                self.norm_status_label.setText("SNV applied" + spectro_note)
            elif method == "Max-norm per pixel":
                self.ds = self._norm_max()
                self.norm_status_label.setText("Max-norm applied" + spectro_note)
            elif method == "Spatial median filter":
                self.ds = self._norm_spatial_median()
                self.norm_status_label.setText("Spatial median applied")
            else:
                self.ds = self.original_ds.copy()
                self.norm_status_label.setText("")

            # Re-apply mask if active
            if self.active_mask is not None and np.any(self.active_mask):
                self.ds = self.ds.where(~self.active_mask, other=0)

            self.update_plot()
            self.statusBar().showMessage(f"Normalization: {method}", 3000)
        except _NormCancelled:
            self.norm_status_label.setText("Cancelled")
            self.statusBar().showMessage("Normalization cancelled", 2000)
        except Exception as e:
            self.norm_status_label.setText("Error!")
            QMessageBox.critical(self, "Normalization error", str(e))
        finally:
            self.original_ds = _orig_backup

    def save_normalized_cube(self):
        """Export the current (normalized) dataset to a new NetCDF file."""
        if self.ds is None:
            QMessageBox.information(self, "Save", "No dataset loaded.")
            return
        method = self.norm_combo.currentText()
        if method == "None":
            QMessageBox.information(self, "Save", "No normalization is currently applied.\nApply a normalization first.")
            return

        # Suggest a filename based on the source cube
        default_name = ""
        if self.loaded_cube_path:
            base, ext = os.path.splitext(self.loaded_cube_path)
            suffix = method.replace(" ", "_").replace("(", "").replace(")", "")
            default_name = f"{base}_{suffix}{ext}"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Normalized Cube", default_name,
            "NetCDF Files (*.nc);;All Files (*)"
        )
        if not path:
            return

        try:
            prog = self._make_norm_progress("Saving normalized cube…", 2)
            # Copy the normalized ds and add provenance attributes
            ds_out = self.ds.copy(deep=True)
            # Strip original encoding so normalized float64 values are not
            # truncated by the source file's dtype/scale_factor/add_offset.
            for var in ds_out.data_vars:
                ds_out[var].encoding.clear()
            ds_out.attrs["normalization_method"] = method
            if method == "Continuum window":
                ds_out.attrs["normalization_cont_start_nm"] = float(self.norm_cont_start.value())
                ds_out.attrs["normalization_cont_end_nm"] = float(self.norm_cont_end.value())
            elif method == "Spatial median filter":
                ds_out.attrs["normalization_kernel_px"] = int(self.norm_kernel_spin.value())
            spectro_txt = self.norm_spectro_edit.text().strip()
            if spectro_txt and method in ("Total Emission (TEN)", "Total Area (TAN)",
                                          "SNV (Standard Normal Variate)", "Max-norm per pixel"):
                ds_out.attrs["normalization_spectrometers"] = spectro_txt
            if self.loaded_cube_path:
                ds_out.attrs["normalization_source_file"] = os.path.basename(self.loaded_cube_path)
            prog.setValue(1); QApplication.processEvents()

            ds_out.to_netcdf(path)
            prog.setValue(2)
            self.statusBar().showMessage(f"Saved normalized cube: {os.path.basename(path)}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    # ---- Band-by-band normalization helpers ----
    @staticmethod
    def _norm_should_update(i, n, interval=20):
        """Return True when the progress UI should be refreshed."""
        return i == n - 1 or i % interval == 0

    @staticmethod
    def _norm_mem_str():
        """Return system memory usage as a human-readable string."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            used = vm.total - vm.available
            if used >= 1 << 30:
                used_s = f"{used / (1 << 30):.1f} GB"
            else:
                used_s = f"{used / (1 << 20):.0f} MB"
            total_s = f"{vm.total / (1 << 30):.1f} GB"
            return f"\nMemory: {used_s} used / {total_s} total  ({vm.percent}%)"
        except Exception:
            return ""

    def _norm_progress_update(self, prog, base_label, i, n):
        """Update progress value and label with memory info."""
        prog.setValue(i + 1)
        prog.setLabelText(f"{base_label}  ({i+1}/{n}){self._norm_mem_str()}")
        QApplication.processEvents()
        if prog.wasCanceled():
            raise _NormCancelled()

    def _norm_var_and_meta(self):
        """Return (var_name, DataArray ref, n_bands, bands_axis) for the primary variable."""
        var_name = list(self.original_ds.data_vars)[0]
        ref = self.original_ds[var_name]
        bands_ax = list(ref.dims).index('bands')
        n_bands = ref.sizes['bands']
        return var_name, ref, n_bands, bands_ax

    def _norm_read_band(self, ref, i):
        """Read band *i* from a DataArray as a 2D float32 array (lazy-safe)."""
        return np.asarray(ref.isel(bands=i).values, dtype=np.float32)

    def _norm_stat_indices(self, n_bands):
        """Band indices used to compute the normalization statistics.

        Restricted to the spectrometers selected in the Normalize tab
        (the normalization factor is still applied to all bands).
        Raises ValueError for an invalid selection string."""
        text = self.norm_spectro_edit.text() if hasattr(self, 'norm_spectro_edit') else ""
        idx = self._spectro_selected_band_indices(text)
        if idx is None:
            return np.arange(n_bands, dtype=int)
        return idx

    def _norm_write_band(self, output, bands_ax, i, data):
        """Write a 2D array into band *i* of the pre-allocated output cube."""
        idx = [slice(None)] * output.ndim
        idx[bands_ax] = i
        output[tuple(idx)] = data

    def _norm_build_ds(self, var_name, ref, output):
        """Build a Dataset from the output array using original_ds as template."""
        return xr.Dataset(
            {var_name: (ref.dims, output, ref.attrs)},
            coords=self.original_ds.coords,
            attrs=self.original_ds.attrs,
        )

    # --- TEN: divide each pixel spectrum by its total emission ---
    def _norm_ten(self):
        var_name, ref, n_bands, bands_ax = self._norm_var_and_meta()
        stat_idx = self._norm_stat_indices(n_bands)
        n_stat = len(stat_idx)
        lbl1 = "TEN — pass 1/2: computing sum"
        prog = self._make_norm_progress(lbl1, n_stat)

        band0 = self._norm_read_band(ref, int(stat_idx[0]))
        spatial_shape = band0.shape
        total = np.zeros(spatial_shape, dtype=np.float64)

        for step, i in enumerate(stat_idx):
            band = self._norm_read_band(ref, int(i)) if step > 0 else band0
            np.add(total, np.where(np.isnan(band), 0.0, band), out=total)
            if self._norm_should_update(step, n_stat):
                self._norm_progress_update(prog, lbl1, step, n_stat)
        del band0

        total[total == 0] = 1.0
        output = np.empty(ref.shape, dtype=np.float32)

        lbl2 = "TEN — pass 2/2: normalizing"
        prog.setLabelText(lbl2)
        prog.setMaximum(n_bands); prog.setValue(0)
        for i in range(n_bands):
            band = self._norm_read_band(ref, i).astype(np.float64)
            band /= total
            self._norm_write_band(output, bands_ax, i, band.astype(np.float32))
            if self._norm_should_update(i, n_bands):
                self._norm_progress_update(prog, lbl2, i, n_bands)
        del total

        return self._norm_build_ds(var_name, ref, output)

    # --- TAN: divide each pixel spectrum by its total area (trapezoidal) ---
    def _norm_tan(self):
        var_name, ref, n_bands, bands_ax = self._norm_var_and_meta()
        bands = np.asarray(self._get_bands_values(ds=self.original_ds), dtype=np.float64)
        stat_idx = self._norm_stat_indices(n_bands)
        n_stat = len(stat_idx)
        lbl1 = "TAN — pass 1/2: computing area"
        prog = self._make_norm_progress(lbl1, n_stat)

        band0 = self._norm_read_band(ref, int(stat_idx[0]))
        spatial_shape = band0.shape
        total_area = np.zeros(spatial_shape, dtype=np.float64)
        prev_i = int(stat_idx[0])
        prev_band = np.where(np.isnan(band0), 0.0, band0.astype(np.float64))

        for step in range(1, n_stat):
            i = int(stat_idx[step])
            cur = self._norm_read_band(ref, i)
            cur_clean = np.where(np.isnan(cur), 0.0, cur.astype(np.float64))
            dwl = bands[i] - bands[prev_i]
            # Integrate only within contiguous, increasing-wavelength runs
            # (skips spectrometer boundaries and selection gaps).
            if i == prev_i + 1 and dwl > 0:
                total_area += 0.5 * dwl * (prev_band + cur_clean)
            prev_i = i
            prev_band = cur_clean
            if self._norm_should_update(step, n_stat):
                self._norm_progress_update(prog, lbl1, step, n_stat)
        del prev_band

        total_area[total_area == 0] = 1.0
        output = np.empty(ref.shape, dtype=np.float32)

        lbl2 = "TAN — pass 2/2: normalizing"
        prog.setLabelText(lbl2)
        prog.setMaximum(n_bands); prog.setValue(0)
        for i in range(n_bands):
            band = self._norm_read_band(ref, i).astype(np.float64)
            band /= total_area
            self._norm_write_band(output, bands_ax, i, band.astype(np.float32))
            if self._norm_should_update(i, n_bands):
                self._norm_progress_update(prog, lbl2, i, n_bands)
        del total_area

        return self._norm_build_ds(var_name, ref, output)

    # --- Continuum: divide by mean of a featureless spectral window ---
    def _norm_continuum(self):
        var_name, ref, n_bands, bands_ax = self._norm_var_and_meta()
        bands = self._get_bands_values(ds=self.original_ds)
        wl_start = float(self.norm_cont_start.value())
        wl_end = float(self.norm_cont_end.value())
        if wl_start > wl_end:
            wl_start, wl_end = wl_end, wl_start
        win_mask = (bands >= wl_start) & (bands <= wl_end)
        if not np.any(win_mask):
            raise ValueError(f"No bands in continuum window [{wl_start:.1f}, {wl_end:.1f}] nm")
        win_idxs = np.where(win_mask)[0]

        band0 = self._norm_read_band(ref, 0)
        spatial_shape = band0.shape
        cont_sum = np.zeros(spatial_shape, dtype=np.float64)
        cont_count = np.zeros(spatial_shape, dtype=np.int32)

        n_win = len(win_idxs)
        lbl1 = "Continuum — pass 1/2: computing baseline"
        prog = self._make_norm_progress(lbl1, n_win)
        for step, bi in enumerate(win_idxs):
            band = self._norm_read_band(ref, int(bi))
            valid = ~np.isnan(band)
            cont_sum[valid] += band[valid]
            cont_count[valid] += 1
            if self._norm_should_update(step, n_win):
                self._norm_progress_update(prog, lbl1, step, n_win)

        cont_count[cont_count == 0] = 1
        cont = (cont_sum / cont_count).astype(np.float64)
        cont[cont == 0] = 1.0
        del cont_sum, cont_count

        output = np.empty(ref.shape, dtype=np.float32)
        lbl2 = "Continuum — pass 2/2: normalizing"
        prog.setLabelText(lbl2)
        prog.setMaximum(n_bands); prog.setValue(0)
        for i in range(n_bands):
            band = self._norm_read_band(ref, i).astype(np.float64)
            band /= cont
            self._norm_write_band(output, bands_ax, i, band.astype(np.float32))
            if self._norm_should_update(i, n_bands):
                self._norm_progress_update(prog, lbl2, i, n_bands)
        del cont

        return self._norm_build_ds(var_name, ref, output)

    # --- SNV: per-pixel subtract mean, divide by std ---
    def _norm_snv(self):
        var_name, ref, n_bands, bands_ax = self._norm_var_and_meta()
        stat_idx = self._norm_stat_indices(n_bands)
        n_stat = len(stat_idx)

        band0 = self._norm_read_band(ref, int(stat_idx[0]))
        spatial_shape = band0.shape
        sum_vals = np.zeros(spatial_shape, dtype=np.float64)
        sum_sq = np.zeros(spatial_shape, dtype=np.float64)
        count = np.zeros(spatial_shape, dtype=np.int32)

        lbl1 = "SNV — pass 1/2: computing stats"
        prog = self._make_norm_progress(lbl1, n_stat)
        for step, i in enumerate(stat_idx):
            band = self._norm_read_band(ref, int(i)) if step > 0 else band0
            valid = ~np.isnan(band)
            bv = band[valid].astype(np.float64)
            sum_vals[valid] += bv
            sum_sq[valid] += bv * bv
            count[valid] += 1
            if self._norm_should_update(step, n_stat):
                self._norm_progress_update(prog, lbl1, step, n_stat)
        del band0

        safe_count = np.maximum(count, 1).astype(np.float64)
        mean = sum_vals / safe_count
        variance = sum_sq / safe_count - mean * mean
        variance[variance < 0] = 0.0
        std = np.sqrt(variance)
        std[std == 0] = 1.0
        del sum_vals, sum_sq, count, safe_count, variance

        output = np.empty(ref.shape, dtype=np.float32)
        lbl2 = "SNV — pass 2/2: normalizing"
        prog.setLabelText(lbl2)
        prog.setMaximum(n_bands); prog.setValue(0)
        for i in range(n_bands):
            band = self._norm_read_band(ref, i).astype(np.float64)
            band = (band - mean) / std
            self._norm_write_band(output, bands_ax, i, band.astype(np.float32))
            if self._norm_should_update(i, n_bands):
                self._norm_progress_update(prog, lbl2, i, n_bands)
        del mean, std

        return self._norm_build_ds(var_name, ref, output)

    # --- Max-norm: divide each pixel spectrum by its maximum value ---
    def _norm_max(self):
        var_name, ref, n_bands, bands_ax = self._norm_var_and_meta()
        stat_idx = self._norm_stat_indices(n_bands)
        n_stat = len(stat_idx)

        band0 = self._norm_read_band(ref, int(stat_idx[0]))
        spatial_shape = band0.shape
        mx = np.full(spatial_shape, -np.inf, dtype=np.float64)

        lbl1 = "Max-norm — pass 1/2: finding max"
        prog = self._make_norm_progress(lbl1, n_stat)
        for step, i in enumerate(stat_idx):
            band = self._norm_read_band(ref, int(i)) if step > 0 else band0
            np.fmax(mx, np.where(np.isnan(band), -np.inf, band), out=mx)
            if self._norm_should_update(step, n_stat):
                self._norm_progress_update(prog, lbl1, step, n_stat)
        del band0

        mx[mx == -np.inf] = 1.0
        mx[mx == 0] = 1.0
        output = np.empty(ref.shape, dtype=np.float32)

        lbl2 = "Max-norm — pass 2/2: normalizing"
        prog.setLabelText(lbl2)
        prog.setMaximum(n_bands); prog.setValue(0)
        for i in range(n_bands):
            band = self._norm_read_band(ref, i).astype(np.float64)
            band /= mx
            self._norm_write_band(output, bands_ax, i, band.astype(np.float32))
            if self._norm_should_update(i, n_bands):
                self._norm_progress_update(prog, lbl2, i, n_bands)
        del mx

        return self._norm_build_ds(var_name, ref, output)

    # --- Spatial median: divide each band image by its spatial median filter ---
    def _norm_spatial_median(self):
        from scipy.ndimage import median_filter
        var_name, ref, n_bands, bands_ax = self._norm_var_and_meta()
        kernel = int(self.norm_kernel_spin.value())
        if kernel % 2 == 0:
            kernel += 1
        output = np.empty(ref.shape, dtype=np.float32)
        lbl = "Spatial median normalization"
        prog = self._make_norm_progress(lbl, n_bands)
        for i in range(n_bands):
            band = self._norm_read_band(ref, i)
            smoothed = median_filter(band, size=kernel)
            smoothed[smoothed == 0] = 1.0
            self._norm_write_band(output, bands_ax, i, band / smoothed)
            if self._norm_should_update(i, n_bands):
                self._norm_progress_update(prog, lbl, i, n_bands)

        return self._norm_build_ds(var_name, ref, output)

    # =====================================================================
    # Cube-wide baseline correction (preprocessing step before normalization)
    # =====================================================================
    def _cube_baseline_on_method_changed(self):
        """Show/hide parameter fields based on the selected baseline method."""
        m = self.cube_baseline_combo.currentText()
        is_snip = (m == "SNIP")
        is_rmin = (m == "Rolling minimum")
        is_asls = (m == "AsLS")
        active = is_snip or is_rmin or is_asls
        self.cube_bl_snip_iter_label.setVisible(is_snip)
        self.cube_bl_snip_iter_spin.setVisible(is_snip)
        self.cube_bl_window_label.setVisible(is_rmin)
        self.cube_bl_window_spin.setVisible(is_rmin)
        self.cube_bl_asls_lam_label.setVisible(is_asls)
        self.cube_bl_asls_lam_spin.setVisible(is_asls)
        self.cube_bl_asls_p_label.setVisible(is_asls)
        self.cube_bl_asls_p_spin.setVisible(is_asls)
        self.cube_bl_asls_iter_label.setVisible(is_asls)
        self.cube_bl_asls_iter_spin.setVisible(is_asls)
        self.cube_bl_clip_neg_chk.setVisible(active)
        self.cube_bl_apply_btn.setEnabled(active)
        self.cube_bl_preview_btn.setEnabled(active)

    def _cube_baseline_apply(self):
        """Run the selected baseline method on original_ds and cache it.

        The result is stored in self._baseline_ds and exposed as self.ds so the
        user can immediately inspect the corrected cube in Map Explorer.
        apply_normalization() uses self._baseline_ds as its source if present.
        """
        if self.original_ds is None:
            QMessageBox.information(self, "Baseline", "No cube loaded.")
            return
        method = self.cube_baseline_combo.currentText()
        if method == "None":
            self._cube_baseline_reset()
            return

        self.cube_bl_status_label.setText("Computing…")
        QApplication.processEvents()
        try:
            if method == "SNIP":
                corrected = self._baseline_snip()
            elif method == "Rolling minimum":
                corrected = self._baseline_rolling_min()
            elif method == "AsLS":
                corrected = self._baseline_asls()
            else:
                return

            self._baseline_ds = corrected
            self._baseline_method_applied = method
            # Expose corrected cube as the working dataset (re-apply mask if any)
            self.ds = corrected.copy()
            if self.active_mask is not None and np.any(self.active_mask):
                self.ds = self.ds.where(~self.active_mask, other=0)

            # Any previously computed normalization is stale now
            try:
                self.norm_combo.blockSignals(True)
                self.norm_combo.setCurrentText("None")
            finally:
                self.norm_combo.blockSignals(False)
            self.norm_status_label.setText("")

            self.cube_bl_status_label.setText(f"{method} applied")
            self.update_plot()
            self.statusBar().showMessage(f"Baseline: {method} applied", 3000)
        except _NormCancelled:
            self.cube_bl_status_label.setText("Cancelled")
            self.statusBar().showMessage("Baseline cancelled", 2000)
        except MemoryError as e:
            self.cube_bl_status_label.setText("Memory error")
            QMessageBox.critical(self, "Baseline — out of memory",
                f"Not enough memory to apply baseline correction.\n\n{e}\n\n"
                "Try reducing SNIP iterations or use Rolling minimum.")
        except Exception as e:
            self.cube_bl_status_label.setText("Error!")
            QMessageBox.critical(self, "Baseline error", str(e))

    def _cube_baseline_reset(self):
        """Discard any baseline correction and restore the original cube."""
        if self._baseline_ds is None and self.cube_baseline_combo.currentText() == "None":
            return
        self._baseline_ds = None
        self._baseline_method_applied = "None"
        if self.original_ds is not None:
            self.ds = self.original_ds.copy()
            if self.active_mask is not None and np.any(self.active_mask):
                self.ds = self.ds.where(~self.active_mask, other=0)
        self.cube_bl_status_label.setText("")
        # Also reset downstream normalization since it was computed from baseline
        try:
            self.norm_combo.blockSignals(True)
            self.norm_combo.setCurrentText("None")
        finally:
            self.norm_combo.blockSignals(False)
        self.norm_status_label.setText("")
        if self.original_ds is not None:
            self.update_plot()
        self.statusBar().showMessage("Baseline reset", 2000)

    def _baseline_pixelwise_driver(self, process_chunk_fn, label):
        """Common driver that iterates over spatial chunks and calls process_chunk_fn.

        process_chunk_fn(block_float32) must return a float32 array of the same
        shape (n_bands, n_pixels_chunk) containing the baseline-corrected block.
        """
        var_name, ref, n_bands, bands_ax = self._norm_var_and_meta()
        src_full = np.asarray(ref.values)
        src_moved = np.moveaxis(src_full, bands_ax, 0)
        nb = src_moved.shape[0]
        spatial = src_moved.shape[1:]
        n_pix = int(np.prod(spatial)) if spatial else 1
        flat = src_moved.reshape(nb, n_pix)

        out_flat = np.empty((nb, n_pix), dtype=np.float32)
        clip_neg = bool(self.cube_bl_clip_neg_chk.isChecked())

        CHUNK = 4096
        total_chunks = max(1, (n_pix + CHUNK - 1) // CHUNK)
        prog = self._make_norm_progress(label, total_chunks)

        for i, pstart in enumerate(range(0, n_pix, CHUNK)):
            pstop = min(pstart + CHUNK, n_pix)
            block = flat[:, pstart:pstop].astype(np.float32, copy=True)
            np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            corrected = process_chunk_fn(block)

            if clip_neg:
                np.clip(corrected, 0.0, None, out=corrected)
            out_flat[:, pstart:pstop] = corrected

            if self._norm_should_update(i, total_chunks):
                self._norm_progress_update(prog, label, i, total_chunks)

        output_moved = out_flat.reshape(nb, *spatial)
        output = np.moveaxis(output_moved, 0, bands_ax)
        return self._norm_build_ds(var_name, ref, output)

    def _baseline_snip(self):
        """SNIP (Sensitive Nonlinear Iterative Peak-clipping) baseline.

        Vectorized over pixels: for each iteration p = 1..niter,
            y[i] <- min(y[i], 0.5 * (y[i-p] + y[i+p]))
        applied in LLS-transformed space: y = log(log(sqrt(x+1)+1)+1).
        """
        niter = int(self.cube_bl_snip_iter_spin.value())

        def _process(block):
            pos = np.clip(block, 0.0, None)
            y = np.log(np.log(np.sqrt(pos + 1.0) + 1.0) + 1.0)
            nb_ = y.shape[0]
            for p in range(1, niter + 1):
                if 2 * p >= nb_:
                    break
                left = y[:-2 * p]
                right = y[2 * p:]
                avg = 0.5 * (left + right)
                np.minimum(y[p:-p], avg, out=y[p:-p])
            baseline = (np.exp(np.exp(y) - 1.0) - 1.0) ** 2 - 1.0
            return block - baseline.astype(block.dtype, copy=False)

        return self._baseline_pixelwise_driver(
            _process, f"SNIP baseline ({niter} iters)")

    def _baseline_rolling_min(self):
        """Rolling-minimum baseline with a smoothing pass.

        baseline[i] = smooth_avg(rolling_min(x, w), w), w = window.
        """
        from scipy.ndimage import minimum_filter1d, uniform_filter1d
        window = int(self.cube_bl_window_spin.value())
        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1

        def _process(block):
            mn = minimum_filter1d(block, size=window, axis=0, mode='reflect')
            baseline = uniform_filter1d(mn, size=window, axis=0, mode='reflect')
            return block - baseline

        return self._baseline_pixelwise_driver(
            _process, f"Rolling min baseline (w={window})")

    def _baseline_asls(self):
        """AsLS (Asymmetric Least Squares, Eilers & Boelens 2005) baseline.

        Each spectrum is fit by minimising
            Σ w_i (y_i - z_i)² + λ Σ (Δ² z_i)²
        with w_i = p if y_i > z_i else (1 - p), iteratively updated.

        Implementation uses a banded Cholesky solve per pixel via
        scipy.linalg.solveh_banded for speed. The matrix sparsity pattern
        (pentadiagonal) is constant; only the main diagonal changes as w
        changes, so the banded form is rebuilt in-place each iteration.
        """
        from scipy.linalg import solveh_banded

        lam = float(10 ** self.cube_bl_asls_lam_spin.value())
        p = float(self.cube_bl_asls_p_spin.value())
        niter = int(self.cube_bl_asls_iter_spin.value())

        var_name, ref, n_bands, _bands_ax = self._norm_var_and_meta()
        # Pre-compute the upper banded form of D^T D (second-difference penalty):
        #   main diagonal:   [1, 5, 6, 6, ..., 6, 5, 1]
        #   1st superdiag:   [-2, -4, -4, ..., -4, -2]
        #   2nd superdiag:   [1, 1, ..., 1]
        # See Eilers (2003) "A perfect smoother".
        L = n_bands
        DtD_main = np.full(L, 6.0)
        DtD_main[0] = DtD_main[-1] = 1.0
        DtD_main[1] = DtD_main[-2] = 5.0
        DtD_sup1 = np.full(L - 1, -4.0)
        DtD_sup1[0] = DtD_sup1[-1] = -2.0
        DtD_sup2 = np.ones(L - 2)

        def _asls_spectrum(y):
            """Fit AsLS baseline to a single 1-D spectrum y."""
            y = np.asarray(y, dtype=np.float64)
            w = np.ones(L, dtype=np.float64)
            ab = np.zeros((3, L), dtype=np.float64)
            # Upper-banded layout for solveh_banded(lower=False, uband=2):
            #   ab[0, 2:  ] = 2nd superdiag
            #   ab[1, 1:  ] = 1st superdiag
            #   ab[2,  :  ] = main diagonal
            ab[0, 2:] = lam * DtD_sup2
            ab[1, 1:] = lam * DtD_sup1
            z = y.copy()
            for _ in range(niter):
                ab[2, :] = w + lam * DtD_main
                z = solveh_banded(ab, w * y, lower=False, overwrite_ab=False,
                                  overwrite_b=False, check_finite=False)
                w = np.where(y > z, p, 1.0 - p)
            return z

        # Chunk-wise processing: loop over pixels in each chunk.
        def _process(block):
            # block shape = (n_bands, n_pix_chunk)
            nb_, npx = block.shape
            out = np.empty_like(block)
            for k in range(npx):
                spec = block[:, k]
                # Replace negatives/NaN with 0 for numerical stability
                s = np.where(np.isfinite(spec) & (spec > 0), spec, 0.0)
                baseline = _asls_spectrum(s).astype(block.dtype, copy=False)
                out[:, k] = block[:, k] - baseline
            return out

        return self._baseline_pixelwise_driver(
            _process, f"AsLS baseline (λ=1e{self.cube_bl_asls_lam_spin.value():.1f}, p={p:g})")

    # ---- Single-spectrum variants used by the preview dialog ----
    def _baseline_snip_spectrum(self, spectrum, niter):
        """Apply SNIP to a single 1-D spectrum; return the baseline."""
        y_full = np.asarray(spectrum, dtype=np.float64).copy()
        pos = np.clip(y_full, 0.0, None)
        y = np.log(np.log(np.sqrt(pos + 1.0) + 1.0) + 1.0)
        nb_ = y.shape[0]
        for p in range(1, int(niter) + 1):
            if 2 * p >= nb_:
                break
            avg = 0.5 * (y[: -2 * p] + y[2 * p:])
            np.minimum(y[p:-p], avg, out=y[p:-p])
        return (np.exp(np.exp(y) - 1.0) - 1.0) ** 2 - 1.0

    def _baseline_rolling_min_spectrum(self, spectrum, window):
        """Apply Rolling-minimum baseline to a single 1-D spectrum."""
        from scipy.ndimage import minimum_filter1d, uniform_filter1d
        w = int(window)
        if w < 3:
            w = 3
        if w % 2 == 0:
            w += 1
        arr = np.asarray(spectrum, dtype=np.float64)
        mn = minimum_filter1d(arr, size=w, mode='reflect')
        return uniform_filter1d(mn, size=w, mode='reflect')

    def _baseline_asls_spectrum(self, spectrum, lam, p, niter):
        """Apply AsLS baseline to a single 1-D spectrum."""
        from scipy.linalg import solveh_banded
        y = np.asarray(spectrum, dtype=np.float64)
        y = np.where(np.isfinite(y) & (y > 0), y, 0.0)
        L = y.shape[0]
        DtD_main = np.full(L, 6.0)
        DtD_main[0] = DtD_main[-1] = 1.0
        DtD_main[1] = DtD_main[-2] = 5.0
        DtD_sup1 = np.full(L - 1, -4.0)
        DtD_sup1[0] = DtD_sup1[-1] = -2.0
        DtD_sup2 = np.ones(L - 2)

        w = np.ones(L, dtype=np.float64)
        ab = np.zeros((3, L), dtype=np.float64)
        ab[0, 2:] = lam * DtD_sup2
        ab[1, 1:] = lam * DtD_sup1
        z = y.copy()
        for _ in range(int(niter)):
            ab[2, :] = w + lam * DtD_main
            z = solveh_banded(ab, w * y, lower=False, overwrite_ab=False,
                              overwrite_b=False, check_finite=False)
            w = np.where(y > z, p, 1.0 - p)
        return z

    def _baseline_random_spectrum(self):
        """Pick a random non-masked pixel and return (y_idx, x_idx, wavelengths, spectrum)."""
        if self.original_ds is None:
            return None
        var_name = list(self.original_ds.data_vars)[0]
        ref = self.original_ds[var_name]
        ny = ref.sizes.get('y', ref.shape[ref.dims.index('y')]) if 'y' in ref.dims else ref.shape[1]
        nx = ref.sizes.get('x', ref.shape[ref.dims.index('x')]) if 'x' in ref.dims else ref.shape[2]
        bands = np.asarray(self._get_bands_values())

        import random
        attempts = 0
        y_idx = x_idx = 0
        while attempts < 50:
            y_idx = random.randrange(ny)
            x_idx = random.randrange(nx)
            if self.active_mask is not None and self.active_mask[y_idx, x_idx]:
                attempts += 1
                continue
            break
        spec = np.asarray(ref.isel(y=y_idx, x=x_idx).values, dtype=np.float64)
        return y_idx, x_idx, bands, spec

    def _cube_baseline_preview(self):
        """Open a non-modal dialog showing a random-pixel spectrum and its baseline."""
        if self.original_ds is None:
            QMessageBox.information(self, "Baseline preview", "No cube loaded.")
            return
        method = self.cube_baseline_combo.currentText()
        if method == "None":
            QMessageBox.information(self, "Baseline preview",
                                    "Choose a baseline method first.")
            return
        dlg = _BaselinePreviewDialog(self, method)
        dlg.show()

    def _cube_baseline_save(self):
        """Export the baseline-corrected cube to a new NetCDF file.

        Saves self._baseline_ds (i.e. the baseline correction only, without any
        normalization applied on top of it). Provenance attributes record the
        method and parameters used.
        """
        if self._baseline_ds is None:
            QMessageBox.information(
                self, "Save baseline-corrected cube",
                "No baseline correction has been applied yet.\n\n"
                "Select a method and click 'Apply baseline' first.")
            return

        method = str(self._baseline_method_applied or "None")

        default_name = ""
        if self.loaded_cube_path:
            base, ext = os.path.splitext(self.loaded_cube_path)
            tag = {"SNIP": "snip", "Rolling minimum": "rollmin", "AsLS": "asls"}.get(method, "baseline")
            default_name = f"{base}_bl-{tag}{ext}"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Baseline-corrected Cube", default_name,
            "NetCDF Files (*.nc);;All Files (*)"
        )
        if not path:
            return

        try:
            prog = self._make_norm_progress("Saving baseline-corrected cube…", 2)
            ds_out = self._baseline_ds.copy(deep=True)
            # Strip original encoding so corrected float32 values are not
            # truncated by the source file's dtype/scale_factor/add_offset.
            for var in ds_out.data_vars:
                ds_out[var].encoding.clear()

            ds_out.attrs["baseline_method"] = method
            ds_out.attrs["baseline_clip_negatives"] = int(self.cube_bl_clip_neg_chk.isChecked())
            if method == "SNIP":
                ds_out.attrs["baseline_snip_iterations"] = int(self.cube_bl_snip_iter_spin.value())
            elif method == "Rolling minimum":
                ds_out.attrs["baseline_window_bands"] = int(self.cube_bl_window_spin.value())
            elif method == "AsLS":
                ds_out.attrs["baseline_asls_log10_lambda"] = float(self.cube_bl_asls_lam_spin.value())
                ds_out.attrs["baseline_asls_p"] = float(self.cube_bl_asls_p_spin.value())
                ds_out.attrs["baseline_asls_iterations"] = int(self.cube_bl_asls_iter_spin.value())
            if self.loaded_cube_path:
                ds_out.attrs["baseline_source_file"] = os.path.basename(self.loaded_cube_path)
            prog.setValue(1)
            QApplication.processEvents()

            ds_out.to_netcdf(path)
            prog.setValue(2)
            self.cube_bl_status_label.setText(f"{method} applied — saved")
            self.statusBar().showMessage(
                f"Saved baseline-corrected cube: {os.path.basename(path)}", 4000)
        except _NormCancelled:
            self.statusBar().showMessage("Save cancelled", 2000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    # ====== Photo tab logic ======
    def load_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load photo", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not path: return
        try:
            img = plt.imread(path)
            if img.dtype.kind == 'f' and img.max() <= 1.0: img = (img * 255).astype(np.uint8)
            self.photo_img = img
        except Exception as e:
            QMessageBox.critical(self, "Photo error", str(e)); return
        self.loaded_photo_path = path
        self.photo_polygon = []
        self.clear_calibration(redraw=False)
        self.redraw_photo_tab()
        self.redraw_photo_sci()

    def redraw_photo_tab(self):
        self.photo_ax_tab.clear()
        if self.photo_img is not None:
            self.photo_ax_tab.imshow(self.photo_img)
            t = "Photo \u2014 "
            if self.calibration_active:
                t += "click 4 points: TL\u2192TR\u2192BR\u2192BL"
            else:
                t += "start calibration or drag points (if enabled)"
            self.photo_ax_tab.set_title(t, fontsize=8)
            self.photo_ax_tab.set_xticks([]); self.photo_ax_tab.set_yticks([])
            if len(self.photo_polygon) >= 2:
                xs, ys = zip(*self.photo_polygon)
                self.photo_ax_tab.plot(xs, ys, '-r', lw=1.2)
            if len(self.calib_pts_photo) > 0:
                xs, ys = zip(*self.calib_pts_photo)
                self.photo_ax_tab.plot(xs, ys, 'oy', ms=6, mfc='none', mew=1.4)
                for i, (x, y) in enumerate(self.calib_pts_photo, 1):
                    self.photo_ax_tab.text(x+3, y+3, str(i), color='y', fontsize=8)
            self._draw_libs_footprint(self.photo_ax_tab)
        else:
            self.photo_ax_tab.text(0.5, 0.5, "No photo loaded", ha='center', va='center')
        self.photo_canvas_tab.draw_idle()

    def _draw_libs_footprint(self, ax):
        if self.current_data_array is None or self.H_libs_to_photo is None:
            return
        Hlib, Wlib = self.current_data_array.shape[:2]
        corners = np.array([[0,0],[Wlib-1,0],[Wlib-1,Hlib-1],[0,Hlib-1]], dtype=float)
        pts = []
        for (x,y) in corners:
            out = apply_H(self.H_libs_to_photo, x, y)
            if out is None: return
            pts.append(out)
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, '--', color='lime', lw=1.2)

    def redraw_photo_sci(self):
        if not hasattr(self, 'photo_ax_sci'):
            return
        self.photo_ax_sci.clear()
        if self.photo_img is not None:
            self.photo_ax_sci.imshow(self.photo_img)
            self.photo_ax_sci.set_title("Photo", fontsize=8)
            self.photo_ax_sci.set_xticks([]); self.photo_ax_sci.set_yticks([])
            if len(self.photo_polygon) >= 2:
                xs, ys = zip(*self.photo_polygon)
                self.photo_ax_sci.plot(xs, ys, '-r', lw=1.2)
            self._draw_libs_footprint(self.photo_ax_sci)
            if self.line_img is not None:
                self.photo_ax_sci.add_line(self.line_img); self.line_img.set_zorder(10)
            if self.pixel_marker_img is not None:
                self.pixel_marker_img.set_zorder(11)
        else:
            self.photo_ax_sci.text(0.5, 0.5, "Load a photo in the Photo tab", ha='center', va='center')
        self.photo_canvas_sci.draw_idle()

    def on_photo_tab_key(self, event):
        if event.key in ('enter', 'return'):
            self.btn_poly_mode.setChecked(False)
            self.redraw_photo_tab()

    def on_photo_tab_press(self, event):
        if event.inaxes != self.photo_ax_tab or event.xdata is None or event.ydata is None:
            return
        if self.calibration_active:
            if len(self.calib_pts_photo) < 4:
                self.calib_pts_photo.append((event.xdata, event.ydata))
                self.redraw_photo_tab()
            if len(self.calib_pts_photo) == 4:
                self.finish_calibration()
            return
        if self.drag_calib_enabled and len(self.calib_pts_photo) == 4:
            idx = self._nearest_calib_point(event.xdata, event.ydata, tol_px=12)
            if idx is not None:
                self.dragging_idx = idx
                return
        if self.btn_poly_mode.isChecked():
            self.photo_polygon.append((event.xdata, event.ydata))
            self.redraw_photo_tab()

    def on_photo_tab_motion(self, event):
        if self.dragging_idx is None or event.inaxes != self.photo_ax_tab:
            return
        if event.xdata is None or event.ydata is None: return
        self.calib_pts_photo[self.dragging_idx] = (event.xdata, event.ydata)
        self._recompute_homography_from_points(silent=True)
        self.redraw_photo_tab()
        self.redraw_photo_sci()

    def on_photo_tab_release(self, event):
        if self.dragging_idx is not None:
            self.dragging_idx = None
            self._recompute_homography_from_points(silent=False)

    def _nearest_calib_point(self, x, y, tol_px=10):
        if len(self.calib_pts_photo) != 4: return None
        pts = np.asarray(self.calib_pts_photo)
        d2 = (pts[:,0]-x)**2 + (pts[:,1]-y)**2
        i = int(np.argmin(d2))
        if np.sqrt(d2[i]) <= tol_px:
            return i
        return None

    def clear_photo_polygon(self):
        self.photo_polygon = []
        self.redraw_photo_tab(); self.redraw_photo_sci()

    def start_calibration(self):
        if self.photo_img is None:
            QMessageBox.information(self, "Calibration", "Load a photo first.")
            return
        self.calibration_active = True
        self.calib_pts_photo = []
        self.redraw_photo_tab()

    def clear_calibration(self, redraw=True):
        self.calibration_active = False
        self.calib_pts_photo = []
        self.H_photo_to_libs = None
        self.H_libs_to_photo = None
        if redraw:
            self.redraw_photo_tab(); self.redraw_photo_sci()

    def finish_calibration(self):
        self.calibration_active = False
        if self.current_data_array is None:
            QMessageBox.warning(self, "Calibration", "Open a LIBS cube first, then calibrate.")
            self.calib_pts_photo = []
            self.redraw_photo_tab()
            return
        self._recompute_homography_from_points(silent=False)

    def toggle_drag_calib(self, checked):
        self.drag_calib_enabled = bool(checked)

    def _recompute_homography_from_points(self, silent=True):
        if len(self.calib_pts_photo) != 4 or self.current_data_array is None:
            return False
        Hlib, Wlib = self.current_data_array.shape[:2]
        dst = np.array([[0,0],[Wlib-1,0],[Wlib-1,Hlib-1],[0,Hlib-1]], dtype=float)
        src = np.array(self.calib_pts_photo, dtype=float)
        try:
            H = compute_homography(src, dst)
            self.H_photo_to_libs = H
            self.H_libs_to_photo = np.linalg.inv(H)
            if not silent:
                self.statusBar().showMessage("Homography updated", 3000)
            return True
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "Calibration error", str(e))
            return False

    def _libs_shape(self):
        if self.current_data_array is None: return (0, 0)
        return self.current_data_array.shape[0], self.current_data_array.shape[1]

    def _img_shape(self):
        if self.photo_img is None: return (0, 0)
        return self.photo_img.shape[0], self.photo_img.shape[1]

    def _img_to_libs(self, xi, yi):
        if self.H_photo_to_libs is not None:
            out = apply_H(self.H_photo_to_libs, xi, yi)
            if out is None: return None
            xl, yl = out
            Hlib, Wlib = self._libs_shape()
            if Hlib == 0: return None
            xl = int(np.clip(round(xl), 0, Wlib-1))
            yl = int(np.clip(round(yl), 0, Hlib-1))
            return xl, yl
        Hlib, Wlib = self._libs_shape(); Himg, Wimg = self._img_shape()
        if Hlib == 0 or Himg == 0: return None
        xl = int(np.clip(round(xi * Wlib / max(1, Wimg)), 0, Wlib - 1))
        yl = int(np.clip(round(yi * Hlib / max(1, Himg)), 0, Hlib - 1))
        return xl, yl

    def _libs_to_img(self, xl, yl):
        if self.H_libs_to_photo is not None:
            out = apply_H(self.H_libs_to_photo, xl, yl)
            if out is None: return None
            xi, yi = out
            return xi, yi
        Hlib, Wlib = self._libs_shape(); Himg, Wimg = self._img_shape()
        if Himg == 0 or Hlib == 0: return None
        xi = xl * Wimg / max(1, Wlib)
        yi = yl * Himg / max(1, Hlib)
        return xi, yi

    # ====== Interactive Mask Tab ======
    def _mask_ensure_mask(self):
        """Ensure active_mask exists with correct shape."""
        if self.ds is None:
            return False
        first_var = next(iter(self.ds.data_vars))
        da = self.ds[first_var]
        spatial_dims = [d for d in da.dims if d != 'bands']
        shape = tuple(da.sizes[d] for d in spatial_dims)
        if self.active_mask is None or self.active_mask.shape != shape:
            self.active_mask = np.zeros(shape, dtype=bool)
        return True

    def _mask_update_map(self):
        """Redraw the mask tab map with the current band and overlay."""
        if self.ds is None:
            self.mask_ax.clear()
            self.mask_ax.text(0.5, 0.5, "Load a dataset first (Tab 1)", ha='center', va='center', fontsize=10)
            self.mask_canvas.draw_idle()
            return

        bands = self._get_bands_values(warn=False)
        if bands is None:
            return
        idx = self.mask_band_slider.value()
        if idx >= len(bands):
            idx = len(bands) - 1
        wl = float(bands[idx])
        self.mask_band_label.setText(f"Band: {idx} — {wl:.3f} nm")

        img = self._get_band_image(idx)

        prev_cbar = getattr(self, 'mask_colorbar', None)
        if prev_cbar is not None:
            try:
                prev_cbar.remove()
            except Exception:
                pass
            self.mask_colorbar = None

        self.mask_ax.clear()
        data = img.flatten()
        data = data[np.isfinite(data)]
        data_pos = data[data > 0]
        if data_pos.size:
            vmin = float(np.percentile(data_pos, 1))
            vmax = float(np.percentile(data_pos, 99))
            if vmin == vmax:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

        im = self.mask_ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        self.mask_ax.set_title(f"Band {idx}: {wl:.3f} nm", fontsize=8)
        self.mask_ax.set_xticks([])
        self.mask_ax.set_yticks([])

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.mask_ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        self.mask_colorbar = self.mask_canvas.figure.colorbar(im, cax=cax)
        self.mask_colorbar.ax.tick_params(labelsize=6)

        self._mask_apply_overlay()
        self.mask_canvas.draw_idle()

    def _mask_apply_overlay(self):
        """Draw semi-transparent red overlay where mask is True."""
        if self.active_mask is None:
            return
        overlay = np.zeros((*self.active_mask.shape, 4), dtype=float)
        overlay[self.active_mask, 0] = 1.0
        overlay[self.active_mask, 3] = 0.4
        for img_artist in self.mask_ax.images[1:]:
            try:
                img_artist.remove()
            except Exception:
                pass
        if np.any(self.active_mask):
            self.mask_ax.imshow(overlay, interpolation='nearest', zorder=5)
        self._mask_update_count()

    def _mask_update_count(self):
        """Update the masked pixel count label."""
        if self.active_mask is not None:
            n = int(np.sum(self.active_mask))
            total = self.active_mask.size
            pct = 100.0 * n / total if total > 0 else 0.0
            self.mask_count_label.setText(f"Masked: {n} / {total} px ({pct:.1f}%)")
        else:
            self.mask_count_label.setText("Masked: 0 px")

    def _mask_on_threshold_apply(self):
        """Apply threshold masking on the current band."""
        if not self._mask_ensure_mask():
            return
        bands = self._get_bands_values(warn=False)
        if bands is None:
            return
        idx = self.mask_band_slider.value()
        img = self._get_band_image(idx)
        threshold = float(self.mask_thresh_spin.value())
        direction = self.mask_thresh_dir.currentText()

        if direction == "Below threshold":
            new_mask = img < threshold
        else:
            new_mask = img > threshold

        self.active_mask = self.active_mask | new_mask
        self._mask_update_map()
        self._mask_set_dirty()

    def _mask_tool_changed(self):
        """Ensure only one paint tool is active at a time."""
        sender = self.sender()
        if sender is self.mask_rect_btn and self.mask_rect_btn.isChecked():
            self.mask_brush_btn.setChecked(False)
            self.mask_poly_btn.setChecked(False)
            self._mask_poly_pts.clear()
        elif sender is self.mask_brush_btn and self.mask_brush_btn.isChecked():
            self.mask_rect_btn.setChecked(False)
            self.mask_poly_btn.setChecked(False)
            self._mask_poly_pts.clear()
        elif sender is self.mask_poly_btn and self.mask_poly_btn.isChecked():
            self.mask_rect_btn.setChecked(False)
            self.mask_brush_btn.setChecked(False)
            self._mask_poly_pts.clear()

    def _mask_on_press(self, event):
        """Handle mouse press on the mask canvas."""
        if event.inaxes != self.mask_ax or event.xdata is None:
            return
        if self.mask_toolbar.mode != '':
            return
        if not self._mask_ensure_mask():
            return

        if self.mask_brush_btn.isChecked():
            self._mask_painting = True
            self._mask_paint_at(event.xdata, event.ydata)
        elif self.mask_rect_btn.isChecked():
            self._mask_rect_start = (event.xdata, event.ydata)
        elif self.mask_poly_btn.isChecked():
            if event.button == 3 or event.dblclick:
                self._mask_polygon_close()
            else:
                self._mask_poly_pts.append((event.xdata, event.ydata))
                self._mask_draw_poly_preview()

    def _mask_on_motion(self, event):
        """Handle mouse motion for brush painting."""
        if not self._mask_painting:
            return
        if event.inaxes != self.mask_ax or event.xdata is None:
            return
        self._mask_paint_at(event.xdata, event.ydata)

    def _mask_on_release(self, event):
        """Handle mouse release."""
        if self.mask_rect_btn.isChecked() and hasattr(self, '_mask_rect_start'):
            if event.inaxes == self.mask_ax and event.xdata is not None:
                x0, y0 = self._mask_rect_start
                x1, y1 = event.xdata, event.ydata
                self._mask_apply_rect(x0, y0, x1, y1)
            if hasattr(self, '_mask_rect_start'):
                del self._mask_rect_start
        self._mask_painting = False

    def _mask_paint_at(self, x, y):
        """Paint mask at the given image coordinates with brush."""
        if self.active_mask is None:
            return
        ny, nx = self.active_mask.shape
        xi, yi = int(round(x)), int(round(y))
        r = int(self.mask_brush_size_spin.value())
        is_mask = self.mask_mode_combo.currentText() == "Mask"

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px, py = xi + dx, yi + dy
                    if 0 <= py < ny and 0 <= px < nx:
                        self.active_mask[py, px] = is_mask

        self._mask_update_map()
        self._mask_set_dirty()

    def _mask_apply_rect(self, x0, y0, x1, y1):
        """Apply mask to rectangle region."""
        if self.active_mask is None:
            return
        ny, nx = self.active_mask.shape
        c0, c1 = int(round(min(x0, x1))), int(round(max(x0, x1)))
        r0, r1 = int(round(min(y0, y1))), int(round(max(y0, y1)))
        c0 = max(0, c0); c1 = min(nx - 1, c1)
        r0 = max(0, r0); r1 = min(ny - 1, r1)

        is_mask = self.mask_mode_combo.currentText() == "Mask"
        self.active_mask[r0:r1 + 1, c0:c1 + 1] = is_mask

        self._mask_update_map()
        self._mask_set_dirty()

    def _mask_draw_poly_preview(self):
        """Draw current polygon vertices on the mask canvas."""
        for line in self.mask_ax.lines[:]:
            if getattr(line, '_mask_poly_preview', False):
                line.remove()

        if len(self._mask_poly_pts) >= 1:
            xs = [p[0] for p in self._mask_poly_pts]
            ys = [p[1] for p in self._mask_poly_pts]
            line, = self.mask_ax.plot(xs, ys, '-o', color='yellow', ms=4, lw=1.5, zorder=10)
            line._mask_poly_preview = True

        self.mask_canvas.draw_idle()

    def _mask_polygon_close(self):
        """Close and fill the current polygon."""
        if len(self._mask_poly_pts) < 3:
            self._mask_poly_pts.clear()
            return
        if self.active_mask is None:
            self._mask_poly_pts.clear()
            return

        from matplotlib.path import Path as MplPath
        ny, nx = self.active_mask.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        points = np.column_stack([xx.ravel(), yy.ravel()])

        path = MplPath(self._mask_poly_pts)
        inside = path.contains_points(points).reshape(ny, nx)

        is_mask = self.mask_mode_combo.currentText() == "Mask"
        if is_mask:
            self.active_mask[inside] = True
        else:
            self.active_mask[inside] = False

        self._mask_poly_pts.clear()
        self._mask_update_map()
        self._mask_set_dirty()

    def _mask_clear_all(self):
        """Reset mask to all unmasked."""
        if self.active_mask is not None:
            self.active_mask[:] = False
        self._mask_update_map()
        self._mask_set_dirty()

    def _mask_invert(self):
        """Invert the current mask."""
        if not self._mask_ensure_mask():
            return
        self.active_mask = ~self.active_mask
        self._mask_update_map()
        self._mask_set_dirty()

    def _mask_save(self):
        """Save mask as .npy file."""
        if self.active_mask is None:
            QMessageBox.information(self, "Mask", "No mask to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Mask", "", "NumPy array (*.npy);;All Files (*)")
        if not path:
            return
        try:
            np.save(path, self.active_mask)
            self.statusBar().showMessage(f"Mask saved: {os.path.basename(path)}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _mask_load(self):
        """Load mask from .npy file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Mask", "", "NumPy array (*.npy);;All Files (*)")
        if not path:
            return
        try:
            loaded = np.load(path)
            if loaded.dtype != bool:
                loaded = loaded.astype(bool)
            if self.ds is not None:
                first_var = next(iter(self.ds.data_vars))
                da = self.ds[first_var]
                spatial_dims = [d for d in da.dims if d != 'bands']
                expected_shape = tuple(da.sizes[d] for d in spatial_dims)
                if loaded.shape != expected_shape:
                    QMessageBox.warning(self, "Mask", f"Shape mismatch: mask is {loaded.shape}, expected {expected_shape}.")
                    return
            self.active_mask = loaded
            self._mask_update_map()
            self._mask_set_dirty()
            self.statusBar().showMessage(f"Mask loaded: {os.path.basename(path)}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _mask_set_dirty(self):
        """Flag that the mask preview has unapplied changes."""
        self._mask_is_dirty = True
        self.mask_dirty_label.setText(
            "\u26a0  Mask preview only — click 'Apply mask to cube' to use it.")
        self.mask_dirty_label.setVisible(True)

    def _mask_apply_to_cube(self):
        """Apply the current mask to the working dataset with progress feedback."""
        progress = QProgressDialog("Applying mask to cube...", None, 0, 3, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setLabelText(f"Applying mask to cube...{self._norm_mem_str()}")
        QApplication.processEvents()

        self._mask_propagate()

        progress.setValue(1)
        progress.setLabelText(f"Mask applied.{self._norm_mem_str()}")
        QApplication.processEvents()

        progress.setValue(2)
        progress.setLabelText(f"Updating map...{self._norm_mem_str()}")
        QApplication.processEvents()

        self._mask_is_dirty = False
        self.mask_dirty_label.setVisible(False)
        self.update_plot()
        progress.setValue(3)
        progress.close()
        self.statusBar().showMessage("Mask applied to cube", 3000)

    def _mask_propagate(self):
        """Apply current mask to the working dataset for downstream steps."""
        if self.original_ds is None:
            return
        if self.active_mask is not None and np.any(self.active_mask):
            if self.norm_combo.currentText() != "None":
                self.apply_normalization()
            else:
                self.ds = self.original_ds.where(~self.active_mask, other=0)
        else:
            if self.norm_combo.currentText() != "None":
                self.apply_normalization()
            else:
                self.ds = self.original_ds.copy()

    def _mask_on_tab_changed(self, index):
        """Refresh mask map when switching to the Mask tab."""
        if hasattr(self, 'tabMask') and self.tabs.widget(index) is self.tabMask:
            if self.ds is not None:
                bands = self._get_bands_values(warn=False)
                if bands is not None:
                    self.mask_band_slider.setMaximum(len(bands) - 1)
            self._mask_update_map()

    # ====== Shared interactions (draw line / pick pixel on LIBS image) ======
    def _ensure_mode_exclusive(self):
        s = self.sender()
        if s is self.line_mode_button and self.line_mode_button.isChecked():
            self.pixel_mode_button.setChecked(False)
        elif s is self.pixel_mode_button and self.pixel_mode_button.isChecked():
            self.line_mode_button.setChecked(False)
        self.is_drawing = False

    # --- clicks on LIBS image ---
    def on_click_libs(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None: return
        if self.line_mode_button.isChecked():
            self._start_line_from_libs(event.xdata, event.ydata)
        elif self.pixel_mode_button.isChecked():
            self._pick_pixel_from_libs(event.xdata, event.ydata)

    def on_click_img(self, event):
        if not hasattr(self, 'photo_ax_sci'):
            return
        if event.inaxes != self.photo_ax_sci or event.xdata is None or event.ydata is None: return
        if self.line_mode_button.isChecked():
            mapped = self._img_to_libs(event.xdata, event.ydata)
            if mapped is None: return
            xl, yl = mapped
            self._start_line_from_libs(xl, yl, coords_are_libs=True)
        elif self.pixel_mode_button.isChecked():
            mapped = self._img_to_libs(event.xdata, event.ydata)
            if mapped is None: return
            xl, yl = mapped
            self._pick_pixel_from_libs(xl, yl, coords_are_libs=True)

    def _start_line_from_libs(self, x, y, coords_are_libs=False):
        self.is_drawing = True
        if not coords_are_libs:
            x = int(round(x)); y = int(round(y))
        x = int(x); y = int(y)

        # clear existing lines
        self.erase_line(redraw=False)

        self.shared_line = [y, x, y, x]  # (y0,x0,y1,x1) in LIBS
        # draw on LIBS
        self.line_libs = Line2D([x, x], [y, y], color='red', linewidth=1.5, zorder=10)
        self.ax.add_line(self.line_libs); self.canvas.draw_idle()

        if self.photo_img is not None and hasattr(self, 'photo_ax_sci'):
            xy = self._libs_to_img(x, y)
            if xy is not None:
                xi, yi = xy
                self.line_img = Line2D([xi, xi], [yi, yi], color='red', linewidth=1.2, zorder=10)
                self.photo_ax_sci.add_line(self.line_img); self.photo_canvas_sci.draw_idle()

        self._update_line_label()

    def on_motion_shared(self, event):
        if not self.is_drawing or self.shared_line is None: return
        if event.inaxes is self.ax:
            if event.xdata is None or event.ydata is None: return
            xl, yl = int(round(event.xdata)), int(round(event.ydata))
        elif hasattr(self, 'photo_ax_sci') and event.inaxes is self.photo_ax_sci:
            if event.xdata is None or event.ydata is None: return
            mapped = self._img_to_libs(event.xdata, event.ydata)
            if mapped is None: return
            xl, yl = mapped
        else:
            return

        self.shared_line[2], self.shared_line[3] = yl, xl

        if self.line_libs is not None:
            y0, x0, y1, x1 = self.shared_line
            self.line_libs.set_xdata([x0, x1]); self.line_libs.set_ydata([y0, y1])
            self.canvas.draw_idle()

        if self.line_img is not None and self.photo_img is not None and hasattr(self, 'photo_ax_sci'):
            xy0 = self._libs_to_img(self.shared_line[1], self.shared_line[0])
            xy1 = self._libs_to_img(self.shared_line[3], self.shared_line[2])
            if xy0 is not None and xy1 is not None:
                self.line_img.set_xdata([xy0[0], xy1[0]]); self.line_img.set_ydata([xy0[1], xy1[1]])
                self.photo_canvas_sci.draw_idle()
        
        self._update_line_label()

    def on_release_shared(self, event):
        if not self.is_drawing: return
        self.is_drawing = False
        if self.line_mode_button.isChecked():
            self.update_line_plot()
        self._update_line_label()
        self._roi_sync_spinboxes_from_line()

    # --- pixel pick from LIBS ---
    def _pick_pixel_from_libs(self, x, y, coords_are_libs=False):
        if self.ds is None: return
        if not coords_are_libs:
            x = int(round(x)); y = int(round(y))
        x = int(x); y = int(y)

        self._set_pixel_marker(y, x)
        self.plot_spectrum(y, x)

    def _set_pixel_marker(self, yl, xl):
        if self.pixel_marker_libs is not None:
            try: self.pixel_marker_libs.remove()
            except Exception: pass
            self.pixel_marker_libs = None

        self.pixel_marker_libs = self.ax.plot([xl], [yl], marker='o', ms=4, mfc='none', mec='red', mew=1.2)[0]
        self.canvas.draw_idle()

        if self.pixel_marker_img is not None:
            try: self.pixel_marker_img.remove()
            except Exception: pass
            self.pixel_marker_img = None

        if self.photo_img is not None and hasattr(self, 'photo_ax_sci'):
            xy = self._libs_to_img(xl, yl)
            if xy is not None:
                xi, yi = xy
                self.pixel_marker_img = self.photo_ax_sci.plot([xi], [yi], marker='o', ms=4, mfc='none', mec='red', mew=1.2)[0]
                self.photo_canvas_sci.draw_idle()
                
        self._update_pixel_label()

    # ====== Line profile / Pixel spectrum ======
    def update_line_plot(self):
        if self.shared_line is None or self.current_data_array is None:
            return
    
        y0, x0, y1, x1 = self.shared_line
        num_points = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0))) + 1
        xs = np.linspace(x0, x1, num_points)
        ys = np.linspace(y0, y1, num_points)
    
        arr = self.current_data_array
        buffer = int(self.parallel_line_spinbox.value())
        ignore_zeros = bool(self.ignore_null_checkbox.isChecked())
    
        # Build stacked profiles for each parallel offset; use NaN to keep length
        stack = []
        for off in range(-buffer, buffer + 1):
            vals = np.full(num_points, np.nan, dtype=float)
            for i, (xf, yf) in enumerate(zip(xs, ys)):
                xi = int(round(xf + off))
                yi = int(round(yf + off))
                if 0 <= yi < arr.shape[0] and 0 <= xi < arr.shape[1]:
                    # If masked, keep NaN (gap)
                    if self.active_mask is not None and 0 <= yi < self.active_mask.shape[0] and 0 <= xi < self.active_mask.shape[1] and self.active_mask[yi, xi]:
                        continue  # leave NaN
                    v = float(arr[yi, xi])
                    if ignore_zeros and v == 0.0:
                        continue  # treat zero as no-data
                    vals[i] = v
            stack.append(vals)
    
        # stack: list of 1D arrays (length = num_points), each with NaNs for masked/ignored/out-of-bounds
        if stack:
            stack_arr = np.stack(stack, axis=0)              # (n_offsets, num_points)
            selected = np.nanmean(stack_arr, axis=0)         # mean of available offsets
            tol = int(self.nan_tolerance_spin.value())
            nan_counts = np.isnan(stack_arr).sum(axis=0)     # NaN count per position across offsets
            # If too many offsets are NaN, force the result to NaN at that position
            selected[nan_counts > tol] = np.nan
        else:
            selected = np.full(num_points, np.nan, dtype=float)
    
        # Update DataFrame for export (NaNs preserved, CSV will show empty cells)
        self.line_plot_data = pd.DataFrame({
            'Index': np.arange(num_points, dtype=int),
            'X_along': np.linspace(x0, x1, num_points),
            'Y_along': np.linspace(y0, y1, num_points),
            'Intensity': selected
        })
    
        # Plot — NaNs create visible gaps so the user sees masked sections
        self.line_ax.clear()
        self.line_ax.plot(selected, '-', lw=1.2)
        self.line_ax.set_title("Line Plot", fontsize=8)
        self.line_ax.set_xlabel("Distance along line (index)", fontsize=8)
        self.line_ax.set_ylabel("LIBS Intensity (a.u.)", fontsize=8)
        self.line_ax.tick_params(axis='both', which='major', labelsize=6)
    
        # Optional: annotate that gaps = masked/ignored
        if np.isnan(selected).any():
            self.line_ax.text(0.01, 0.98, "Gaps = masked / no‑data",
                              transform=self.line_ax.transAxes, fontsize=6, va='top', ha='left', color='gray')
    
        self.line_plot_canvas.draw_idle()
     
        
 
    
    def erase_line(self, redraw=True):
        self.shared_line = None
        for line in (self.line_libs, self.line_img):
            if line is not None:
                try: line.remove()
                except Exception: pass
        self.line_libs = None
        self.line_img = None
        if redraw:
            self.canvas.draw_idle()
            if hasattr(self, 'photo_canvas_sci'):
                self.photo_canvas_sci.draw_idle()
        self.line_ax.clear()
        self.line_ax.set_title("Line Plot", fontsize=8)
        self.line_ax.set_xlabel("Distance along line", fontsize=8)
        self.line_ax.set_ylabel("LIBS Intensity (a.u.)", fontsize=8)
        self.line_plot_canvas.draw_idle()
        if hasattr(self, 'line_plot_data'): del self.line_plot_data

        self._update_line_label()

    def save_line_plot_data(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot Data", "", "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)")
        if not file_path: return
        if self.pixel_mode_button.isChecked() and hasattr(self, 'spectrum_plot_data'):
            df = self.spectrum_plot_data
        elif hasattr(self, 'line_plot_data'):
            df = self.line_plot_data
        else:
            QMessageBox.information(self, "Nothing to save", "Draw a line or pick a pixel first."); return
        ext = file_path.lower().split('.')[-1]
        try:
            if ext in ('xlsx', 'xls'): df.to_excel(file_path, index=False)
            else: df.to_csv(file_path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def update_pixel_plot(self):
        if self.current_spectrum is not None and self.current_pixel is not None:
            self.plot_spectrum(self.current_pixel[0], self.current_pixel[1])

    def plot_spectrum(self, y, x):
        if self.ds is None: return
        try:
            first_var = list(self.ds.data_vars)[0]
            spatial_dims = [d for d in self.ds[first_var].dims if d != 'bands']
            sel_dict = {spatial_dims[0]: y, spatial_dims[1]: x}
            spectrum = self.ds.isel(**sel_dict).to_array().values
        except Exception as e:
            QMessageBox.critical(self, "Selection error", f"Could not select spectrum at ({x},{y}):\n{e}")
            return
        wavelengths = self._get_bands_values()
        if wavelengths is None:
            return
        if spectrum is None or spectrum.size == 0 or spectrum.shape[0] == 0:
            QMessageBox.warning(self, "Selection error", "No spectrum data found at the selected pixel.")
            return
        self.current_pixel = (y, x); self.current_spectrum = spectrum[0]
        # Preserve zoom/pan across pixel clicks: toolbar zoom/pan turns autoscale
        # off, so a disabled autoscale means the user set a custom view.
        keep_view = not (self.line_ax.get_autoscalex_on() and self.line_ax.get_autoscaley_on())
        if keep_view:
            prev_xlim = self.line_ax.get_xlim(); prev_ylim = self.line_ax.get_ylim()
        self.line_ax.clear(); self.line_ax.plot(wavelengths, self.current_spectrum)
        self.line_ax.set_title(f"Spectrum at pixel ({x}, {y})", fontsize=8)
        self.line_ax.set_xlabel("Wavelength (nm)", fontsize=8)
        self.line_ax.set_ylabel("Intensity", fontsize=8)
        self.line_ax.tick_params(axis='both', which='major', labelsize=6)
        self.spectrum_plot_data = pd.DataFrame({'Wavelength (nm)': wavelengths, 'Intensity': self.current_spectrum})
        if self.peak_detection_checkbox.isChecked():
            self.detect_peaks(self.current_spectrum, wavelengths)
        if keep_view:
            self.line_ax.set_xlim(prev_xlim); self.line_ax.set_ylim(prev_ylim)
        self.line_plot_canvas.draw_idle()

    def detect_peaks(self, data, wavelengths):
        prominence = float(self.prominence_spinbox.value()); distance = int(self.distance_spinbox.value())
        peaks, _ = find_peaks(data, prominence=prominence, distance=distance)
        self.line_ax.plot(wavelengths[peaks], data[peaks], "rx")
        for p in peaks:
            self.line_ax.annotate(f'({data[p]:.2f}, {wavelengths[p]:.2f} nm)', (wavelengths[p], data[p]),
                                  textcoords="offset points", xytext=(0, 10), ha='center', fontsize=6)

    # ====== Chemometrics (PCA / NMF / clustering) ======
    def _chem_icon(self, kind, color=None, size=16):
        """Small hand-drawn icon for the Chemometrics UI (no external files)."""
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        if kind == 'pca':
            # diverging score map: red / blue halves
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#c0392b'))
            p.drawRoundedRect(1, 2, 6, 12, 2, 2)
            p.setBrush(QColor('#2980b9'))
            p.drawRoundedRect(9, 2, 6, 12, 2, 2)
        elif kind == 'nmf':
            # additive non-negative components: stacked bars
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#1e8449')); p.drawRect(2, 8, 3, 6)
            p.setBrush(QColor('#2ecc71')); p.drawRect(6, 3, 3, 11)
            p.setBrush(QColor('#82e0aa')); p.drawRect(10, 10, 3, 4)
        elif kind == 'kmeans':
            # three cluster dots
            p.setPen(Qt.NoPen)
            for x, y, c in ((4.5, 4.5, '#1f77b4'), (11.5, 6.0, '#ff7f0e'), (6.5, 11.5, '#2ca02c')):
                p.setBrush(QColor(c))
                p.drawEllipse(QPointF(x, y), 3.0, 3.0)
        elif kind == 'spectrum':
            # loading spectrum with a peak
            p.setPen(QPen(QColor('#2c3e50'), 1.6))
            pts = [(1, 12), (4, 10), (6, 3), (8, 11), (11, 5), (13, 9), (15, 8)]
            for a, b in zip(pts[:-1], pts[1:]):
                p.drawLine(QPointF(*a), QPointF(*b))
        elif kind == 'scatter':
            # two point clouds
            p.setPen(Qt.NoPen)
            for x, y, c in ((3, 11, '#1f77b4'), (5, 8.5, '#1f77b4'), (7, 12, '#1f77b4'),
                            (10, 4, '#ff7f0e'), (12, 6.5, '#ff7f0e'), (13.5, 3, '#ff7f0e')):
                p.setBrush(QColor(c))
                p.drawEllipse(QPointF(x, y), 1.8, 1.8)
        elif kind == 'silhouette':
            # silhouette blades
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#1f77b4')); p.drawRect(1, 2, 12, 3)
            p.setBrush(QColor('#ff7f0e')); p.drawRect(1, 7, 9, 3)
            p.setBrush(QColor('#2ca02c')); p.drawRect(1, 12, 6, 3)
        elif kind == 'swatch':
            qc = color if isinstance(color, QColor) else QColor(color)
            p.setBrush(qc)
            p.setPen(QPen(qc.darker(140), 1))
            p.drawRoundedRect(2, 2, 12, 12, 3, 3)
        elif kind == 'sam':
            # two spectral vectors with the angle between them
            p.setPen(QPen(QColor('#8e44ad'), 1.6))
            p.drawLine(QPointF(2, 14), QPointF(14, 2))
            p.drawLine(QPointF(2, 14), QPointF(15, 11))
            p.setPen(QPen(QColor('#8e44ad'), 1.0))
            p.drawArc(-2, 10, 8, 8, 13 * 16, 32 * 16)
        elif kind == 'unmix':
            # one bar split into mixing fractions
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#1f77b4')); p.drawRect(1, 4, 7, 8)
            p.setBrush(QColor('#ff7f0e')); p.drawRect(8, 4, 4, 8)
            p.setBrush(QColor('#2ca02c')); p.drawRect(12, 4, 3, 8)
        p.end()
        return QIcon(pm)

    def _chem_reset(self):
        """Clear chemometrics results (called when a new cube is loaded)."""
        self.chem_result = None
        if not hasattr(self, 'chem_comp_combo'):
            return
        for combo in (self.chem_comp_combo, self.chem_scatter_x_combo, self.chem_scatter_y_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
        self.chem_status_label.setText("No analysis run yet.")
        self.chem_peaks_label.setText("Top peaks: —")
        self.chem_colorbar = None
        self.chem_canvas.figure.clf()
        self.chem_ax = self.chem_canvas.figure.add_subplot(111)
        self.chem_canvas.setMinimumHeight(0)
        self.chem_canvas.draw_idle()
        self.chem_plot_ax.clear(); self.chem_plot_canvas.draw_idle()
        self.chem_scatter_colorbar = None
        self.chem_scatter_canvas.figure.clf()
        self.chem_scatter_ax = self.chem_scatter_canvas.figure.add_subplot(111)
        self.chem_scatter_canvas.draw_idle()
        self.chem_sil_canvas.figure.clf()
        self.chem_sil_canvas.draw_idle()
        # Default the spectral range to the cube's band range
        bands = self._get_bands_values(warn=False)
        if bands is not None and bands.size:
            self.chem_wl_min_spin.setValue(float(np.min(bands)))
            self.chem_wl_max_spin.setValue(float(np.max(bands)))
        self._update_spectrometer_info()

    def _chem_prepare_matrix(self):
        """Flatten the current cube to (pixels x bands) restricted to the chosen
        wavelength range, and identify valid pixels (finite, non-empty, unmasked).

        Returns dict with keys: X (n_valid, n_bands float32), valid (H*W bool),
        H, W, wl (selected wavelengths), or None on failure."""
        if self.ds is None:
            QMessageBox.information(self, "Chemometrics", "Load a dataset first (Tab 1).")
            return None
        bands = self._get_bands_values()
        if bands is None:
            return None

        wl_min = float(self.chem_wl_min_spin.value())
        wl_max = float(self.chem_wl_max_spin.value())
        if wl_max <= wl_min:
            QMessageBox.warning(self, "Chemometrics", "λ max must be greater than λ min.")
            return None
        band_sel = np.where((bands >= wl_min) & (bands <= wl_max))[0]

        spectro_note = ""
        try:
            sp_idx = self._spectro_selected_band_indices(self.chem_spectro_edit.text())
        except ValueError as e:
            QMessageBox.warning(self, "Chemometrics", str(e))
            return None
        if sp_idx is not None:
            band_sel = np.intersect1d(band_sel, sp_idx)
            spectro_note = f"spectros {self.chem_spectro_edit.text().strip()}"

        if band_sel.size < 3:
            QMessageBox.warning(self, "Chemometrics",
                                "Fewer than 3 bands in the selected wavelength/spectrometer range.")
            return None

        first_var = next(iter(self.ds.data_vars))
        da = self.ds[first_var]
        spatial_dims = [d for d in da.dims if d != 'bands']
        if len(spatial_dims) != 2:
            QMessageBox.warning(self, "Chemometrics", "Expected a cube with 2 spatial dimensions.")
            return None
        da_t = da.isel(bands=band_sel).transpose(spatial_dims[0], spatial_dims[1], 'bands')
        cube = np.asarray(da_t.values, dtype=np.float32)
        Hh, Ww, Bb = cube.shape
        X_all = cube.reshape(Hh * Ww, Bb)

        valid = np.isfinite(X_all).all(axis=1)
        # all-zero spectra are masked-out or empty pixels
        valid &= np.nanmax(np.abs(X_all), axis=1) > 0
        if self.chem_exclude_mask_cb.isChecked() and self.active_mask is not None \
                and self.active_mask.shape == (Hh, Ww):
            valid &= ~self.active_mask.reshape(-1)

        n_valid = int(valid.sum())
        if n_valid < max(50, int(self.chem_ncomp_spin.value()) * 10):
            QMessageBox.warning(self, "Chemometrics",
                                f"Only {n_valid} valid pixels — not enough for a meaningful decomposition.")
            return None

        return {'X': X_all[valid], 'valid': valid, 'H': Hh, 'W': Ww,
                'wl': bands[band_sel], 'spectro_note': spectro_note}

    # ------ SAM (Spectral Angle Mapper) ------
    def _chem_on_method_changed(self, text):
        """Grey out settings the reference-based methods (SAM, NNLS unmixing)
        do not use; jump to the references tab for them."""
        is_ref_based = text.startswith("SAM") or text.startswith("Unmixing")
        for w in (self.chem_ncomp_spin, self.chem_max_fit_spin, self.chem_standardize_cb):
            w.setEnabled(not is_ref_based)
        # the max-angle cutoff is SAM-only
        self.chem_sam_thresh_spin.setEnabled(not text.startswith("Unmixing"))
        self.chem_sam_auto_btn.setEnabled(not text.startswith("Unmixing"))
        if is_ref_based:
            self.chem_settings_tabs.setCurrentIndex(1)   # "References" tab
        else:
            self.chem_settings_tabs.setCurrentIndex(0)   # "Analysis" tab

    def _sam_refresh_list(self):
        self.chem_sam_list.clear()
        colors = self._chem_cluster_colors(max(1, len(self.chem_sam_refs)))
        for i, ref in enumerate(self.chem_sam_refs):
            item = QListWidgetItem(self._chem_icon('swatch', QColor.fromRgbF(*colors[i][:3])),
                                   ref['name'])
            item.setToolTip(f"{ref['wl'].size} channels, "
                            f"{ref['wl'].min():.1f}–{ref['wl'].max():.1f} nm")
            self.chem_sam_list.addItem(item)

    def _sam_add_ref(self, name, wl, sp):
        wl = np.asarray(wl, dtype=float)
        sp = np.nan_to_num(np.asarray(sp, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if wl.size < 3 or not np.any(sp > 0):
            QMessageBox.warning(self, "SAM", f"Reference '{name}' is empty or all-zero.")
            return
        self.chem_sam_refs.append({'name': str(name), 'wl': wl, 'sp': sp})
        self._sam_refresh_list()

    def _sam_add_pixel(self):
        if self.current_spectrum is None or self.current_pixel is None:
            QMessageBox.information(self, "SAM",
                "No pixel selected. Pick a pixel in the Data Extraction tab first\n"
                "(Pixel mode, then click on the map).")
            return
        wl = self._get_bands_values()
        if wl is None or wl.size != np.asarray(self.current_spectrum).size:
            QMessageBox.warning(self, "SAM", "Pixel spectrum does not match the current band axis.")
            return
        y, x = self.current_pixel
        self._sam_add_ref(f"px ({x},{y})", wl, self.current_spectrum)

    def _sam_add_from_result(self):
        res = self.chem_result
        if res is None or res['method'] not in ("K-Means clustering", "NMF"):
            QMessageBox.information(self, "SAM",
                "Run K-Means or NMF first — their centroid / component spectra\n"
                "can then be added as SAM references.\n"
                "(PCA loadings have negative values and are not usable as endmembers.)")
            return
        tag = "K-Means C" if res['method'] == "K-Means clustering" else "NMF "
        for i in range(res['loadings'].shape[0]):
            self._sam_add_ref(f"{tag}{i+1}", res['wl'], res['loadings'][i])

    def _sam_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load SAM references (CSV)", "",
                                              "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                raise ValueError("CSV needs a wavelength column plus at least one spectrum column.")
            wl = df.iloc[:, 0].to_numpy(dtype=float)
            for col in df.columns[1:]:
                self._sam_add_ref(str(col), wl, df[col].to_numpy(dtype=float))
        except Exception as e:
            QMessageBox.critical(self, "SAM", f"Could not load references:\n{e}")

    def _sam_remove_selected(self):
        row = self.chem_sam_list.currentRow()
        if 0 <= row < len(self.chem_sam_refs):
            del self.chem_sam_refs[row]
            self._sam_refresh_list()

    def _sam_clear(self):
        self.chem_sam_refs = []
        self._sam_refresh_list()

    def _sam_rename_selected(self):
        row = self.chem_sam_list.currentRow()
        if not (0 <= row < len(self.chem_sam_refs)):
            QMessageBox.information(self, "SAM", "Select a reference to rename first.")
            return
        old = self.chem_sam_refs[row]['name']
        name, ok = QInputDialog.getText(self, "Rename SAM reference", "New name:",
                                        QLineEdit.Normal, old)
        name = (name or "").strip()
        if not ok or not name or name == old:
            return
        self.chem_sam_refs[row]['name'] = name
        self._sam_refresh_list()
        self.chem_sam_list.setCurrentRow(row)

        # propagate into the current SAM/unmixing result and its UI, if it uses
        # this reference
        res = self.chem_result
        if res is not None \
                and row < len(res.get('ref_names', [])) and res['ref_names'][row] == old:
            res['ref_names'][row] = name
            if res['method'].startswith("SAM"):
                frac = res.get('cluster_frac')
                pct = f"  ({100*frac[row]:.1f}% of px)" if frac is not None else ""
                self.chem_comp_combo.setItemText(row + 1, f"θ {name}{pct}")
                sym = "θ"
            elif res['method'].startswith("Unmixing"):
                mf = res.get('mean_frac')
                pct = f"  ({100*mf[row]:.1f}% mean)" if mf is not None else ""
                self.chem_comp_combo.setItemText(row, f"φ {name}{pct}")
                sym = "φ"
            else:
                return
            for combo in (self.chem_scatter_x_combo, self.chem_scatter_y_combo):
                if row < combo.count():
                    combo.setItemText(row, f"{sym} {name}")
            self._chem_update_map()
            self._chem_update_loading_plot()

    def _sam_apply_threshold(self, *_):
        """Re-apply the max-angle cutoff to the existing SAM result (no re-run):
        recomputes labels/fractions from the stored angle maps."""
        res = self.chem_result
        if res is None or not res['method'].startswith("SAM"):
            return
        thr = float(self.chem_sam_thresh_spin.value())
        angles = res['maps']
        k = angles.shape[1]
        labels = np.argmin(angles, axis=1)
        best = angles[np.arange(labels.size), labels]
        labels[best > thr] = -1
        res['labels'] = labels
        res['threshold'] = thr
        counts = np.bincount(labels[labels >= 0], minlength=k)
        res['cluster_frac'] = counts / max(1, labels.size)

        for i in range(k):
            self.chem_comp_combo.setItemText(
                i + 1, f"θ {res['ref_names'][i]}  ({100*res['cluster_frac'][i]:.1f}% of px)")
        pct = 100.0 * (labels >= 0).mean()
        self.chem_status_label.setText(
            res.get('status_base', "SAM") + f"  |  ≤{thr:g}°: {pct:.1f}% classified")
        self._chem_update_map()

    def _sam_auto_threshold(self):
        """Suggest the max-angle cutoff with Otsu's method on the best-angle
        histogram of the last SAM run, and plot the diagnostics."""
        res = self.chem_result
        if res is None or not res['method'].startswith("SAM"):
            QMessageBox.information(self, "SAM", "Run a SAM analysis first.")
            return
        best = np.sort(res['maps'].min(axis=1))
        n = best.size
        hi = float(min(90.0, best.max() + 1e-6))

        # Otsu's threshold: split the angle distribution into 'matched' and
        # 'unmatched' populations by maximizing the between-class variance.
        hist, edges = np.histogram(best, bins=256, range=(0.0, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        p = hist.astype(np.float64) / max(1, hist.sum())
        omega = np.cumsum(p)
        mu = np.cumsum(p * centers)
        mu_t = mu[-1]
        denom = omega * (1.0 - omega)
        denom[denom <= 0] = np.nan
        sigma_b = (mu_t * omega - mu) ** 2 / denom
        if not np.any(np.isfinite(sigma_b)):
            QMessageBox.warning(self, "SAM", "Could not compute a threshold "
                                             "(angle distribution is degenerate).")
            return
        # With well-separated populations the between-class variance is flat
        # across the whole valley; take the midpoint of that plateau instead of
        # its first bin (matches OpenCV's tie-breaking behavior).
        smax = np.nanmax(sigma_b)
        plateau = np.where(sigma_b >= smax * (1.0 - 1e-6))[0]
        thr = float(centers[int(round(0.5 * (plateau[0] + plateau[-1])))])
        pct_at_thr = 100.0 * np.searchsorted(best, thr, side='right') / n

        # ---- diagnostics in the Silhouette tab ----
        fig = self.chem_sil_canvas.figure
        fig.clf()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.hist(best, bins=100, range=(0.0, hi), color='#5dade2', edgecolor='none')
        ax1.axvline(thr, color='#c0392b', ls='--', lw=1.0)
        ax1.annotate(f"Otsu: {thr:.1f}°", (thr, ax1.get_ylim()[1] * 0.9),
                     textcoords="offset points", xytext=(6, 0), fontsize=7, color='#c0392b')
        ax1.set_xlabel("Best spectral angle per pixel (°)", fontsize=8)
        ax1.set_ylabel("Pixels", fontsize=8)
        ax1.set_title("Angle distribution — a valley between 'matched' and 'unmatched' "
                      "modes is the natural cutoff", fontsize=7)
        ax1.tick_params(axis='both', which='major', labelsize=6)

        ts = np.linspace(0.0, hi, 300)
        frac = np.searchsorted(best, ts, side='right') / n
        ax2.plot(ts, 100.0 * frac, lw=1.0)
        ax2.plot(thr, pct_at_thr, 'o', ms=6, mfc='none', mec='#c0392b', mew=1.4)
        ax2.axvline(thr, color='#c0392b', ls='--', lw=0.8)
        ax2.set_xlabel("Max angle threshold (°)", fontsize=8)
        ax2.set_ylabel("Classified (%)", fontsize=8)
        ax2.set_title("Classified fraction vs threshold — plateaus mean the threshold "
                      "is past a population", fontsize=7)
        ax2.tick_params(axis='both', which='major', labelsize=6)
        try:
            fig.tight_layout(pad=0.8)
        except Exception:
            pass
        self.chem_sil_canvas.draw_idle()
        self.chem_plot_tabs.setCurrentIndex(2)

        # setting the spin re-applies the classification instantly
        self.chem_sam_thresh_spin.setValue(round(thr, 1))
        self.statusBar().showMessage(
            f"Otsu-suggested max angle: {thr:.1f}° ({pct_at_thr:.1f}% classified)", 4000)

    @staticmethod
    def _sam_resample(ref_wl, ref_sp, target_wl):
        """Nearest-wavelength resampling of a reference onto the analysis band axis.

        Returns (values, coverage) where coverage is the fraction of target
        channels that found a reference channel within tolerance (others are 0)."""
        order = np.argsort(ref_wl)
        rw = np.asarray(ref_wl, dtype=float)[order]
        rs = np.asarray(ref_sp, dtype=float)[order]
        pos = np.clip(np.searchsorted(rw, target_wl), 1, rw.size - 1)
        left, right = rw[pos - 1], rw[pos]
        idx = np.where(np.abs(target_wl - left) <= np.abs(right - target_wl), pos - 1, pos)
        dist = np.abs(rw[idx] - target_wl)
        spacing = float(np.median(np.abs(np.diff(rw)))) if rw.size > 1 else 1.0
        ok = dist <= max(1.5 * spacing, 1e-9)
        return np.where(ok, rs[idx], 0.0), float(ok.mean())

    def _sam_ref_matrix(self, wl, notes, title):
        """Resample all stored references onto the analysis band axis.

        Returns (R, names) with R of shape (n_refs, n_bands), or None if a
        reference has no signal in the selected range (a warning is shown)."""
        refs = list(self.chem_sam_refs)
        R = np.empty((len(refs), wl.size), dtype=np.float64)
        for i, ref in enumerate(refs):
            R[i], coverage = self._sam_resample(ref['wl'], ref['sp'], wl)
            if coverage < 0.9:
                notes.append(f"'{ref['name']}' covers only {100*coverage:.0f}% of the range")
        r_norm = np.linalg.norm(R, axis=1)
        if np.any(r_norm <= 0):
            bad = [refs[i]['name'] for i in np.where(r_norm <= 0)[0]]
            QMessageBox.warning(self, title,
                "Reference(s) with no signal in the selected wavelength/spectrometer "
                f"range: {', '.join(bad)}")
            return None
        return R, [r['name'] for r in refs]

    def _sam_run_core(self, prep, _step):
        """Compute spectral angle maps + classification for the current references."""
        X = prep['X']; valid = prep['valid']
        Hh, Ww, wl = prep['H'], prep['W'], prep['wl']
        n_valid = X.shape[0]
        refs = list(self.chem_sam_refs)
        k = len(refs)
        thr = float(self.chem_sam_thresh_spin.value())

        notes = []
        if prep.get('spectro_note'):
            notes.append(prep['spectro_note'])

        if not _step(2, f"Resampling {k} references…"):
            self.chem_status_label.setText("Analysis cancelled.")
            return

        rm = self._sam_ref_matrix(wl, notes, "SAM")
        if rm is None:
            return
        R, _names = rm
        Rn = R / np.linalg.norm(R, axis=1)[:, None]

        if not _step(3, f"Computing spectral angles for {n_valid:,} pixels…"):
            self.chem_status_label.setText("Analysis cancelled.")
            return

        Xf = X.astype(np.float64, copy=False)
        x_norm = np.linalg.norm(Xf, axis=1)
        x_norm[x_norm <= 0] = 1.0
        cos = np.clip((Xf / x_norm[:, None]) @ Rn.T, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos))            # (n_valid, k)
        labels = np.argmin(angles, axis=1)
        best = angles[np.arange(n_valid), labels]
        labels[best > thr] = -1

        counts = np.bincount(labels[labels >= 0], minlength=k)
        frac = counts / max(1, n_valid)
        pct_classified = 100.0 * (labels >= 0).mean()

        result = {'method': "SAM (Spectral Angle Mapper)", 'wl': wl, 'H': Hh, 'W': Ww,
                  'valid': valid, 'maps': angles, 'labels': labels,
                  'loadings': R, 'ref_names': [r['name'] for r in refs],
                  'threshold': thr, 'cluster_frac': frac}

        if not _step(4, "Rendering maps and spectra…"):
            self.chem_status_label.setText("Analysis cancelled.")
            return
        self.chem_result = result

        colors = self._chem_cluster_colors(k)
        swatches = [self._chem_icon('swatch', QColor.fromRgbF(*c[:3])) for c in colors]
        self.chem_comp_combo.blockSignals(True)
        self.chem_comp_combo.clear()
        self.chem_comp_combo.addItem(self._chem_icon('kmeans'), "Classification map")
        for i, ref in enumerate(refs):
            self.chem_comp_combo.addItem(swatches[i], f"θ {ref['name']}  ({100*frac[i]:.1f}% of px)")
        self.chem_comp_combo.setCurrentIndex(0)
        self.chem_comp_combo.blockSignals(False)

        # scatter selectors: angle maps only (indices align with maps columns)
        for combo in (self.chem_scatter_x_combo, self.chem_scatter_y_combo):
            combo.blockSignals(True)
            combo.clear()
            for i, ref in enumerate(refs):
                combo.addItem(swatches[i], f"θ {ref['name']}")
            combo.blockSignals(False)
        self.chem_scatter_x_combo.setCurrentIndex(0)
        self.chem_scatter_y_combo.setCurrentIndex(min(1, k - 1))

        base = (f"SAM: {n_valid:,} px × {wl.size} bands ({wl.min():.1f}–{wl.max():.1f} nm)"
                f"  |  {k} reference{'s' if k != 1 else ''}")
        if notes:
            base += "  |  " + "; ".join(notes)
        result['status_base'] = base
        self.chem_status_label.setText(
            base + f"  |  ≤{thr:g}°: {pct_classified:.1f}% classified")

        self._chem_update_map()
        self._chem_update_loading_plot()
        self._chem_update_scatter()
        self.chem_settings_tabs.setCurrentIndex(2)   # jump to "Results"

    def _unmix_run_core(self, prep, _step, prog):
        """NNLS linear unmixing: model each pixel spectrum as a non-negative
        combination of the reference spectra; map abundance fractions + residual.

        The design matrix is the same for every pixel, so the problem is reduced
        once via the normal equations (Cholesky) and each pixel only needs a tiny
        k-dimensional NNLS solve."""
        from scipy.optimize import nnls as _nnls
        from scipy.linalg import solve_triangular

        X = prep['X']; valid = prep['valid']
        Hh, Ww, wl = prep['H'], prep['W'], prep['wl']
        n_valid = X.shape[0]
        k = len(self.chem_sam_refs)

        notes = []
        if prep.get('spectro_note'):
            notes.append(prep['spectro_note'])

        if not _step(2, f"Resampling {k} references…"):
            self.chem_status_label.setText("Analysis cancelled.")
            return
        rm = self._sam_ref_matrix(wl, notes, "Unmixing")
        if rm is None:
            return
        R, names = rm                                   # (k, B)

        if not _step(3, f"Unmixing {n_valid:,} pixels…"):
            self.chem_status_label.setText("Analysis cancelled.")
            return

        # Reduce ||R.T x - b||² to the k-dim problem ||L.T x - z||² (same for
        # every pixel): AtA = L L.T, z = L⁻¹ (R b).
        Xf = X.astype(np.float64, copy=False)
        AtA = R @ R.T                                   # (k, k)
        AtA[np.diag_indices_from(AtA)] += 1e-12 * max(1.0, np.trace(AtA))
        L = np.linalg.cholesky(AtA)
        AtB = R @ Xf.T                                  # (k, n_valid)
        Z = solve_triangular(L, AtB, lower=True)        # (k, n_valid)
        Ru = L.T
        btb = np.einsum('ij,ij->i', Xf, Xf)             # ||b||² per pixel
        ztz = np.einsum('ij,ij->j', Z, Z)

        weights = np.empty((n_valid, k))
        rn2 = np.empty(n_valid)
        chunk = 20000
        for start in range(0, n_valid, chunk):
            if prog.wasCanceled():
                self.chem_status_label.setText("Analysis cancelled.")
                return
            stop = min(start + chunk, n_valid)
            for i in range(start, stop):
                w, rn = _nnls(Ru, Z[:, i])
                weights[i] = w
                rn2[i] = rn * rn
            prog.setLabelText(f"Unmixing {stop:,} / {n_valid:,} pixels…")
            QApplication.processEvents()

        # abundance fractions (sum-to-one per pixel) + relative residual
        wsum = weights.sum(axis=1)
        frac = np.divide(weights, wsum[:, None],
                         out=np.zeros_like(weights), where=wsum[:, None] > 0)
        resid2 = np.maximum(0.0, rn2 + (btb - ztz))      # ||R.T x - b||² per pixel
        resid_rel = np.sqrt(np.divide(resid2, btb, out=np.zeros_like(resid2),
                                      where=btb > 0))
        maps = np.column_stack([frac, resid_rel])        # (n_valid, k + 1)
        mean_frac = frac.mean(axis=0)
        mean_resid = float(resid_rel.mean())

        result = {'method': "Unmixing (NNLS)", 'wl': wl, 'H': Hh, 'W': Ww,
                  'valid': valid, 'maps': maps,
                  'loadings': R, 'ref_names': list(names),
                  'mean_frac': mean_frac, 'evr': None}

        if not _step(4, "Rendering maps and spectra…"):
            self.chem_status_label.setText("Analysis cancelled.")
            return
        self.chem_result = result

        colors = self._chem_cluster_colors(k)
        swatches = [self._chem_icon('swatch', QColor.fromRgbF(*c[:3])) for c in colors]
        gray = self._chem_icon('swatch', QColor('#7f8c8d'))
        self.chem_comp_combo.blockSignals(True)
        self.chem_comp_combo.clear()
        for i, name in enumerate(names):
            self.chem_comp_combo.addItem(
                swatches[i], f"φ {name}  ({100*mean_frac[i]:.1f}% mean)")
        self.chem_comp_combo.addItem(gray, "Residual (unexplained fraction)")
        self.chem_comp_combo.setCurrentIndex(0)
        self.chem_comp_combo.blockSignals(False)

        for combo in (self.chem_scatter_x_combo, self.chem_scatter_y_combo):
            combo.blockSignals(True)
            combo.clear()
            for i, name in enumerate(names):
                combo.addItem(swatches[i], f"φ {name}")
            combo.addItem(gray, "Residual")
            combo.blockSignals(False)
        self.chem_scatter_x_combo.setCurrentIndex(0)
        self.chem_scatter_y_combo.setCurrentIndex(min(1, k))

        base = (f"Unmixing (NNLS): {n_valid:,} px × {wl.size} bands "
                f"({wl.min():.1f}–{wl.max():.1f} nm)"
                f"  |  {k} reference{'s' if k != 1 else ''}"
                f"  |  mean residual {100*mean_resid:.1f}%")
        if notes:
            base += "  |  " + "; ".join(notes)
        self.chem_status_label.setText(base)

        self._chem_update_map()
        self._chem_update_loading_plot()
        self._chem_update_scatter()
        self.chem_settings_tabs.setCurrentIndex(2)   # jump to "Results"

    _CHEM_BTN_STYLE_IDLE = "font-weight:bold; padding:4px;"
    _CHEM_BTN_STYLE_BUSY = ("font-weight:bold; padding:4px; "
                            "background-color:#f39c12; color:white;")

    def _chem_run(self):
        try:
            from sklearn.decomposition import PCA, NMF
            from sklearn.cluster import KMeans
        except ImportError:
            QMessageBox.critical(self, "Chemometrics",
                                 "scikit-learn is required for this tab.\n\nInstall it with:\n    pip install scikit-learn")
            return

        method = self.chem_method_combo.currentText()
        k = int(self.chem_ncomp_spin.value())
        max_fit = int(self.chem_max_fit_spin.value())
        standardize = bool(self.chem_standardize_cb.isChecked())

        if (method.startswith("SAM") or method.startswith("Unmixing")) \
                and not self.chem_sam_refs:
            QMessageBox.information(self, "Chemometrics",
                f"{method.split(' ')[0]} needs at least one reference spectrum.\n\n"
                "Add one in the 'References' tab: the current pixel spectrum,\n"
                "the centroids/components of a previous K-Means/NMF run, or a CSV file.")
            return

        # Busy indicators: colored Run button + modal progress dialog
        self.chem_run_btn.setEnabled(False)
        self.chem_run_btn.setText("Running…")
        self.chem_run_btn.setStyleSheet(self._CHEM_BTN_STYLE_BUSY)
        prog = QProgressDialog("Preparing data matrix…", "Cancel", 0, 5, self)
        prog.setWindowTitle("Chemometrics")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        QApplication.processEvents()

        def _step(value, text):
            """Advance the progress dialog; returns False if the user cancelled."""
            if prog.wasCanceled():
                return False
            prog.setValue(value)
            prog.setLabelText(text)
            QApplication.processEvents()
            return not prog.wasCanceled()

        try:
            prep = self._chem_prepare_matrix()
            if prep is None:
                return

            if method.startswith("SAM"):
                self._sam_run_core(prep, _step)
                prog.setValue(5)
                return

            if method.startswith("Unmixing"):
                self._unmix_run_core(prep, _step, prog)
                prog.setValue(5)
                return

            X = prep['X']; valid = prep['valid']
            Hh, Ww, wl = prep['H'], prep['W'], prep['wl']
            n_valid = X.shape[0]

            if k > min(X.shape) - 1:
                QMessageBox.warning(self, "Chemometrics",
                                    f"Too many components ({k}) for {n_valid} pixels × {X.shape[1]} bands.")
                return

            if not _step(1, "Preprocessing spectra…"):
                self.chem_status_label.setText("Analysis cancelled.")
                return

            notes = []
            if prep.get('spectro_note'):
                notes.append(prep['spectro_note'])
            scale = None
            Xw = X
            if standardize and method != "NMF":
                std = X.std(axis=0)
                std[std <= 0] = 1.0
                scale = std
                Xw = X / std[None, :]
                notes.append("bands standardized")

            rng = np.random.default_rng(0)
            if n_valid > max_fit:
                fit_idx = rng.choice(n_valid, size=max_fit, replace=False)
                notes.append(f"fitted on {max_fit:,} of {n_valid:,} px")
            else:
                fit_idx = None

            X_fit = Xw if fit_idx is None else Xw[fit_idx]

            fit_n = X_fit.shape[0]
            if not _step(2, f"Fitting {method} on {fit_n:,} pixels…"):
                self.chem_status_label.setText("Analysis cancelled.")
                return

            result = {'method': method, 'wl': wl, 'H': Hh, 'W': Ww, 'valid': valid}
            if method == "PCA":
                model = PCA(n_components=k, random_state=0)
                model.fit(X_fit)
                if not _step(3, f"Projecting all {n_valid:,} pixels…"):
                    self.chem_status_label.setText("Analysis cancelled.")
                    return
                scores = model.transform(Xw)                    # (n_valid, k)
                result['maps'] = scores
                result['loadings'] = np.asarray(model.components_, dtype=float)  # (k, B)
                result['evr'] = np.asarray(model.explained_variance_ratio_, dtype=float)
            elif method == "NMF":
                Xn = np.clip(Xw, 0.0, None)
                if np.any(Xw < 0):
                    notes.append("negative values clipped to 0")
                model = NMF(n_components=k, init='nndsvda', max_iter=400, random_state=0)
                X_fit_n = Xn if fit_idx is None else Xn[fit_idx]
                model.fit(X_fit_n)
                if not _step(3, f"Projecting all {n_valid:,} pixels…"):
                    self.chem_status_label.setText("Analysis cancelled.")
                    return
                scores = model.transform(Xn)
                result['maps'] = scores
                result['loadings'] = np.asarray(model.components_, dtype=float)
                result['evr'] = None
            else:  # K-Means
                model = KMeans(n_clusters=k, n_init=10, random_state=0)
                model.fit(X_fit)
                if not _step(3, f"Assigning all {n_valid:,} pixels to clusters…"):
                    self.chem_status_label.setText("Analysis cancelled.")
                    return
                labels = model.predict(Xw)
                result['labels'] = labels
                if scale is not None:
                    # centroids back in original intensity units for readability
                    result['loadings'] = np.asarray(model.cluster_centers_, dtype=float) * scale[None, :]
                else:
                    result['loadings'] = np.asarray(model.cluster_centers_, dtype=float)
                counts = np.bincount(labels, minlength=k)
                result['cluster_frac'] = counts / max(1, labels.size)

            if not _step(4, "Rendering maps and spectra…"):
                self.chem_status_label.setText("Analysis cancelled.")
                return

            self.chem_result = result

            # populate component selector
            if method == "K-Means clustering":
                # swatches matching the cluster colors on the map
                comp_icons = [self._chem_icon('swatch', QColor.fromRgbF(*c[:3]))
                              for c in self._chem_cluster_colors(k)]
            else:
                comp_icons = [self._chem_icon('pca' if method == "PCA" else 'nmf')] * k

            self.chem_comp_combo.blockSignals(True)
            self.chem_comp_combo.clear()
            if method == "PCA":
                for i in range(k):
                    self.chem_comp_combo.addItem(comp_icons[i], f"PC{i+1}  ({100*result['evr'][i]:.1f}% var)")
            elif method == "NMF":
                for i in range(k):
                    self.chem_comp_combo.addItem(comp_icons[i], f"Component {i+1}")
            else:
                for i in range(k):
                    self.chem_comp_combo.addItem(comp_icons[i],
                                                 f"Cluster {i+1}  ({100*result['cluster_frac'][i]:.1f}% of px)")
            self.chem_comp_combo.setCurrentIndex(0)
            self.chem_comp_combo.blockSignals(False)

            # scatter axis selectors mirror the component list (PCA/NMF only)
            items = [self.chem_comp_combo.itemText(i) for i in range(k)]
            for combo in (self.chem_scatter_x_combo, self.chem_scatter_y_combo):
                combo.blockSignals(True)
                combo.clear()
                if method != "K-Means clustering":
                    for i in range(k):
                        combo.addItem(comp_icons[i], items[i])
                combo.blockSignals(False)
            if method != "K-Means clustering":
                self.chem_scatter_x_combo.setCurrentIndex(0)
                self.chem_scatter_y_combo.setCurrentIndex(min(1, k - 1))

            txt = f"{method}: {n_valid:,} px × {wl.size} bands ({wl.min():.1f}–{wl.max():.1f} nm)"
            if method == "PCA":
                txt += f"  |  {100*result['evr'].sum():.1f}% variance in {k} PCs"
            if notes:
                txt += "  |  " + "; ".join(notes)
            self.chem_status_label.setText(txt)

            self._chem_update_map()
            self._chem_update_loading_plot()
            self._chem_update_scatter()
            self.chem_settings_tabs.setCurrentIndex(2)   # jump to "Results"
            prog.setValue(5)
        except Exception as e:
            QMessageBox.critical(self, "Chemometrics", f"Analysis failed:\n{e}")
            self.chem_status_label.setText("Analysis failed.")
        finally:
            prog.close()
            self.chem_run_btn.setEnabled(True)
            self.chem_run_btn.setText("Run analysis")
            self.chem_run_btn.setStyleSheet(self._CHEM_BTN_STYLE_IDLE)

    def _chem_on_component_changed(self, *_):
        if self.chem_result is None:
            return
        self._chem_update_map()
        self._chem_update_loading_plot()

    def _chem_cluster_colors(self, k):
        import matplotlib as mpl
        cmap = mpl.colormaps['tab10'] if k <= 10 else mpl.colormaps['tab20']
        return [cmap(i % cmap.N) for i in range(k)]

    def _chem_score_image(self, res, i):
        """Full-size (H, W) score/abundance map of component i (NaN outside valid px)."""
        full = np.full(res['H'] * res['W'], np.nan)
        full[res['valid']] = res['maps'][:, i]
        return full.reshape(res['H'], res['W'])

    def _chem_draw_score_map(self, ax, fig, res, i, title_fs=8):
        """Draw one component map on the given axes; returns the colorbar."""
        is_sam = res['method'].startswith("SAM")
        img = self._chem_score_image(res, i)
        vals = img[np.isfinite(img)]
        if vals.size:
            vmin, vmax = np.percentile(vals, [2, 98])
            if vmin == vmax:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0, 1
        if res['method'] == "PCA":
            # symmetric diverging scale so sign is meaningful
            lim = max(abs(vmin), abs(vmax))
            im = ax.imshow(img, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        elif is_sam:
            # inverted colormap: bright = small angle = good match
            im = ax.imshow(img, cmap='viridis_r', vmin=vmin, vmax=vmax, interpolation='nearest')
        else:
            im = ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.tick_params(labelsize=5)
        if is_sam:
            cbar.set_label("angle (°)", fontsize=6)
        elif res['method'].startswith("Unmixing"):
            cbar.set_label("fraction", fontsize=6)
        # SAM's combo has the classification map at index 0, angle maps after it
        ax.set_title(self.chem_comp_combo.itemText(i + 1 if is_sam else i), fontsize=title_fs)
        ax.tick_params(axis='both', which='major', labelsize=6)
        return cbar

    def _chem_update_map(self, *_):
        res = self.chem_result
        if res is None:
            return
        fig = self.chem_canvas.figure
        fig.clf()
        self.chem_colorbar = None
        self.chem_ax = None

        is_sam = res['method'].startswith("SAM")
        idx = max(0, self.chem_comp_combo.currentIndex())
        Hh, Ww, valid = res['H'], res['W'], res['valid']
        map_idx = idx - 1 if is_sam else idx   # SAM combo item 0 = classification map
        grid_mode = (self.chem_grid_cb.isChecked()
                     and res['method'] != "K-Means clustering"
                     and not (is_sam and idx == 0))

        if res['method'] == "K-Means clustering" or (is_sam and idx == 0):
            # ---- categorical label map ----
            self.chem_ax = fig.add_subplot(111)
            k = res['loadings'].shape[0]
            colors = list(self._chem_cluster_colors(k))
            if is_sam:
                names = [(n[:10] + '…') if len(n) > 11 else n for n in res['ref_names']]
                labels = np.where(res['labels'] < 0, k, res['labels']).astype(float)
                colors.append('#95a5a6')   # gray = unclassified (angle > threshold)
                names.append("n/a")
                title = f"SAM classification (best match, ≤ {res['threshold']:g}°)"
            else:
                names = [f"C{i+1}" for i in range(k)]
                labels = res['labels'].astype(float)
                title = f"Chemical domains (K-Means, k={k})"
            ncat = len(colors)
            full = np.full(Hh * Ww, np.nan)
            full[valid] = labels
            img = full.reshape(Hh, Ww)
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(colors)
            cmap.set_bad(color='black')
            im = self.chem_ax.imshow(np.ma.masked_invalid(img), cmap=cmap,
                                     vmin=-0.5, vmax=ncat - 0.5, interpolation='nearest')
            self.chem_colorbar = fig.colorbar(im, ax=self.chem_ax,
                                              ticks=range(ncat), shrink=0.85)
            self.chem_colorbar.ax.set_yticklabels(names, fontsize=6)
            self.chem_ax.set_title(title, fontsize=8)
            self.chem_ax.tick_params(axis='both', which='major', labelsize=6)
            self.chem_canvas.setMinimumHeight(0)
        elif grid_mode:
            k = res['maps'].shape[1]
            ncols = 2 if k > 1 else 1
            nrows = int(np.ceil(k / ncols))
            for i in range(k):
                ax = fig.add_subplot(nrows, ncols, i + 1)
                # combo index this subplot corresponds to (for click-to-select)
                ax._chem_comp_idx = i + 1 if is_sam else i
                self._chem_draw_score_map(ax, fig, res, i, title_fs=7)
                if i == map_idx:
                    ax.set_title(self.chem_comp_combo.itemText(ax._chem_comp_idx),
                                 fontsize=7, fontweight='bold', color='#c0392b')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#c0392b'); spine.set_linewidth(1.5)
            try:
                fig.tight_layout(pad=0.8)
            except Exception:
                pass
            # Give each row a fixed pixel height; the scroll area shows a
            # scrollbar when the grid grows taller than the visible panel.
            row_px = 260
            self.chem_canvas.setMinimumHeight(nrows * row_px)
        else:
            self.chem_ax = fig.add_subplot(111)
            if is_sam:
                title = self.chem_comp_combo.currentText() + " — spectral angle (lower = match)"
            elif res['method'].startswith("Unmixing"):
                title = self.chem_comp_combo.currentText()
                if idx < res['maps'].shape[1] - 1:
                    title += " — abundance fraction"
            else:
                title = self.chem_comp_combo.currentText() + (
                    " score map" if res['method'] == "PCA" else " abundance map")
            self._chem_draw_score_map(self.chem_ax, fig, res, map_idx)
            self.chem_ax.set_title(title, fontsize=8)
            self.chem_canvas.setMinimumHeight(0)

        self.chem_canvas.draw_idle()

    def _chem_on_map_click(self, event):
        """In grid mode, clicking a component map selects that component."""
        if self.chem_result is None or event.inaxes is None:
            return
        if getattr(self.chem_toolbar, 'mode', ''):  # zoom/pan active
            return
        i = getattr(event.inaxes, '_chem_comp_idx', None)
        if i is not None and i != self.chem_comp_combo.currentIndex():
            self.chem_comp_combo.setCurrentIndex(int(i))

    def _chem_update_scatter(self, *_):
        res = self.chem_result
        ax = self.chem_scatter_ax
        ax.clear()
        if getattr(self, 'chem_scatter_colorbar', None) is not None:
            try: self.chem_scatter_colorbar.remove()
            except Exception: pass
            self.chem_scatter_colorbar = None
        if res is None:
            self.chem_scatter_canvas.draw_idle()
            return
        if res['method'] == "K-Means clustering":
            ax.text(0.5, 0.5, "Score scatter is available for PCA / NMF results.\n"
                              "(K-Means produces discrete labels, not scores.)",
                    ha='center', va='center', fontsize=8, transform=ax.transAxes)
            self.chem_scatter_canvas.draw_idle()
            return

        ix = self.chem_scatter_x_combo.currentIndex()
        iy = self.chem_scatter_y_combo.currentIndex()
        k = res['maps'].shape[1]
        if ix < 0 or iy < 0 or ix >= k or iy >= k:
            self.chem_scatter_canvas.draw_idle()
            return

        xs = res['maps'][:, ix]
        ys = res['maps'][:, iy]
        hb = ax.hexbin(xs, ys, gridsize=70, bins='log', mincnt=1, cmap='inferno')
        self.chem_scatter_colorbar = self.chem_scatter_canvas.figure.colorbar(
            hb, ax=ax, shrink=0.85, label='px count (log)')
        self.chem_scatter_colorbar.ax.tick_params(labelsize=5)
        if res['method'] == "PCA":
            ax.axhline(0.0, color='gray', lw=0.6, ls='--')
            ax.axvline(0.0, color='gray', lw=0.6, ls='--')
        ax.set_xlabel(self.chem_scatter_x_combo.currentText(), fontsize=8)
        ax.set_ylabel(self.chem_scatter_y_combo.currentText(), fontsize=8)
        ax.set_title("Pixel scores — separate branches/blobs suggest distinct chemical domains",
                     fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=6)
        self.chem_scatter_canvas.draw_idle()

    def _chem_label_peaks(self, ax, wl, y, n_top, two_sided):
        """Mark the strongest spectral peaks of a loading/centroid and return their wavelengths."""
        if n_top <= 0:
            return []
        cands = []
        rng_y = float(np.nanmax(np.abs(y))) if y.size else 0.0
        if rng_y <= 0:
            return []
        prom = 0.02 * rng_y
        pk, _ = find_peaks(y, prominence=prom)
        cands.extend((int(p), float(y[p])) for p in pk)
        if two_sided:
            pk_neg, _ = find_peaks(-y, prominence=prom)
            cands.extend((int(p), float(y[p])) for p in pk_neg)
        if not cands:
            return []
        cands.sort(key=lambda t: abs(t[1]), reverse=True)
        picked = cands[:n_top]
        for p, v in picked:
            ax.plot(wl[p], v, 'rx', ms=4)
            ax.annotate(f"{wl[p]:.2f}", (wl[p], v), textcoords="offset points",
                        xytext=(0, 8 if v >= 0 else -12), ha='center', fontsize=6, color='red')
        return [float(wl[p]) for p, _ in sorted(picked, key=lambda t: abs(t[1]), reverse=True)]

    def _chem_update_loading_plot(self):
        res = self.chem_result
        if res is None:
            return
        idx = max(0, self.chem_comp_combo.currentIndex())
        wl = res['wl']
        n_top = int(self.chem_npeaks_spin.value())
        ax = self.chem_plot_ax
        ax.clear()

        if 'ref_names' in res:   # SAM / NNLS unmixing: reference spectra
            is_unmix = res['method'].startswith("Unmixing")
            k = res['loadings'].shape[0]
            colors = self._chem_cluster_colors(k)
            names = res['ref_names']
            # display-normalize each reference for comparability
            norm = res['loadings'] / np.maximum(res['loadings'].max(axis=1, keepdims=True), 1e-12)
            # SAM combo item 0 = classification map; unmixing's last item = residual
            sel = idx if is_unmix else idx - 1
            if sel < 0 or sel >= k:
                for i in range(k):
                    ax.plot(wl, norm[i], color=colors[i], lw=0.9, label=names[i])
                peaks = []
                ax.set_title(("Unmixing" if is_unmix else "SAM") +
                             " reference spectra (max-normalized)", fontsize=8)
            else:
                for i in range(k):
                    if i != sel:
                        ax.plot(wl, norm[i], color=colors[i], lw=0.7, alpha=0.3)
                ax.plot(wl, norm[sel], color=colors[sel], lw=1.4, label=names[sel])
                peaks = self._chem_label_peaks(ax, wl, norm[sel], n_top, two_sided=False)
                ax.set_title(f"Reference '{names[sel]}' (max-normalized)", fontsize=8)
            ax.set_ylabel("Intensity (norm.)", fontsize=8)
            ax.legend(fontsize=6, loc='best')
        elif res['method'] == "K-Means clustering":
            k = res['loadings'].shape[0]
            colors = self._chem_cluster_colors(k)
            for i in range(k):
                if i == idx:
                    continue
                ax.plot(wl, res['loadings'][i], color=colors[i], lw=0.7, alpha=0.35)
            ax.plot(wl, res['loadings'][idx], color=colors[idx], lw=1.4,
                    label=f"Cluster {idx+1} centroid")
            peaks = self._chem_label_peaks(ax, wl, res['loadings'][idx], n_top, two_sided=False)
            ax.set_title(f"Cluster centroid spectra (Cluster {idx+1} highlighted)", fontsize=8)
            ax.set_ylabel("Intensity (a.u.)", fontsize=8)
            ax.legend(fontsize=6, loc='best')
        else:
            loading = res['loadings'][idx]
            two_sided = res['method'] == "PCA"
            ax.plot(wl, loading, lw=0.9)
            if two_sided:
                ax.axhline(0.0, color='gray', lw=0.6, ls='--')
            peaks = self._chem_label_peaks(ax, wl, loading, n_top, two_sided=two_sided)
            name = self.chem_comp_combo.currentText().split("  ")[0]
            if res['method'] == "PCA":
                ax.set_title(f"{name} loadings — peaks pointing up co-vary with red map areas,\n"
                             f"peaks pointing down with blue areas", fontsize=7)
                ax.set_ylabel("Loading", fontsize=8)
            else:
                ax.set_title(f"{name} spectral signature", fontsize=8)
                ax.set_ylabel("Weight", fontsize=8)

        ax.set_xlabel("Wavelength (nm)", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        if peaks:
            self.chem_peaks_label.setText("Top peaks (nm): " + ", ".join(f"{p:.2f}" for p in peaks))
        else:
            self.chem_peaks_label.setText("Top peaks: —")
        self.chem_plot_canvas.draw_idle()

    def _chem_silhouette_scan(self):
        """Run K-Means for k = 2 … k_max and plot silhouette scores to suggest
        the number of chemical domains (plus the silhouette diagram of the best k)."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, silhouette_samples
        except ImportError:
            QMessageBox.critical(self, "Chemometrics",
                                 "scikit-learn is required for this tab.\n\nInstall it with:\n    pip install scikit-learn")
            return

        kmax = int(self.chem_kmax_spin.value())
        ks = list(range(2, kmax + 1))
        max_fit = int(self.chem_max_fit_spin.value())
        standardize = bool(self.chem_standardize_cb.isChecked())

        self.chem_sil_btn.setEnabled(False)
        self.chem_sil_btn.setText("Scanning…")
        self.chem_sil_btn.setStyleSheet(self._CHEM_BTN_STYLE_BUSY)
        prog = QProgressDialog("Preparing data matrix…", "Cancel", 0, len(ks) + 1, self)
        prog.setWindowTitle("Silhouette scan")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        QApplication.processEvents()

        try:
            prep = self._chem_prepare_matrix()
            if prep is None:
                return
            X = prep['X']
            n_valid = X.shape[0]
            if standardize:
                std = X.std(axis=0)
                std[std <= 0] = 1.0
                X = X / std[None, :]

            rng = np.random.default_rng(0)
            if n_valid > max_fit:
                X_fit = X[rng.choice(n_valid, size=max_fit, replace=False)]
            else:
                X_fit = X
            n_fit = X_fit.shape[0]
            # silhouette is O(n²): score a fixed-size subsample of the fitted pixels
            n_sil = min(4000, n_fit)
            sil_idx = (rng.choice(n_fit, size=n_sil, replace=False)
                       if n_fit > n_sil else np.arange(n_fit))
            X_sil = X_fit[sil_idx]

            scores = []
            labels_per_k = []
            for j, k in enumerate(ks):
                if prog.wasCanceled():
                    self.chem_status_label.setText("Silhouette scan cancelled.")
                    return
                prog.setValue(j + 1)
                prog.setLabelText(f"K-Means with k = {k}  ({j+1}/{len(ks)})…")
                QApplication.processEvents()

                km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X_fit)
                lab = km.labels_[sil_idx]
                if np.unique(lab).size < 2:
                    scores.append(np.nan)
                    labels_per_k.append(None)
                else:
                    scores.append(float(silhouette_score(X_sil, lab)))
                    labels_per_k.append(lab)

            scores = np.asarray(scores, dtype=float)
            if not np.any(np.isfinite(scores)):
                QMessageBox.warning(self, "Silhouette scan",
                                    "No valid silhouette scores could be computed.")
                return
            best_j = int(np.nanargmax(scores))
            best_k = ks[best_j]
            best_labels = labels_per_k[best_j]
            sil_vals = silhouette_samples(X_sil, best_labels)

            # ---- plot: score-vs-k curve (top) + silhouette diagram of best k (bottom) ----
            fig = self.chem_sil_canvas.figure
            fig.clf()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            ax1.plot(ks, scores, 'o-', lw=1.0, ms=4)
            ax1.plot(best_k, scores[best_j], 'o', ms=9, mfc='none', mec='#c0392b', mew=1.6)
            ax1.annotate(f"suggested k = {best_k}", (best_k, scores[best_j]),
                         textcoords="offset points", xytext=(8, 8), fontsize=7, color='#c0392b')
            ax1.set_xlabel("Number of clusters k", fontsize=8)
            ax1.set_ylabel("Mean silhouette", fontsize=8)
            ax1.set_xticks(ks)
            ax1.set_title(f"Silhouette scan ({n_sil:,} sampled px) — higher = better-separated domains",
                          fontsize=7)
            ax1.tick_params(axis='both', which='major', labelsize=6)

            colors = self._chem_cluster_colors(best_k)
            y0 = 0
            for ci in range(best_k):
                vals = np.sort(sil_vals[best_labels == ci])
                if vals.size == 0:
                    continue
                ax2.fill_betweenx(np.arange(y0, y0 + vals.size), 0, vals,
                                  facecolor=colors[ci], edgecolor=colors[ci], alpha=0.8)
                ax2.text(-0.02, y0 + vals.size / 2, f"C{ci+1}", fontsize=6,
                         ha='right', va='center', color=colors[ci])
                y0 += vals.size + max(10, n_sil // 100)
            ax2.axvline(float(np.mean(sil_vals)), color='#c0392b', ls='--', lw=0.8)
            ax2.set_xlabel("Silhouette coefficient", fontsize=8)
            ax2.set_yticks([])
            ax2.set_title(f"Per-pixel silhouettes at k = {best_k} "
                          f"(dashed line = mean; clusters cut by the line are weak)", fontsize=7)
            ax2.tick_params(axis='x', which='major', labelsize=6)

            try:
                fig.tight_layout(pad=0.8)
            except Exception:
                pass
            self.chem_sil_canvas.draw_idle()
            self.chem_plot_tabs.setCurrentIndex(2)

            self.chem_status_label.setText(
                f"Silhouette scan k=2–{kmax}: suggested k = {best_k} "
                f"(score {scores[best_j]:.3f}). Set 'Components / clusters' and run K-Means.")
        except Exception as e:
            QMessageBox.critical(self, "Silhouette scan", f"Scan failed:\n{e}")
            self.chem_status_label.setText("Silhouette scan failed.")
        finally:
            prog.close()
            self.chem_sil_btn.setEnabled(True)
            self.chem_sil_btn.setText("Silhouette scan (suggest k)")
            self.chem_sil_btn.setStyleSheet(self._CHEM_BTN_STYLE_IDLE)

    def _chem_export_spectra(self):
        res = self.chem_result
        if res is None:
            QMessageBox.information(self, "Chemometrics", "Run an analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save component spectra", "",
                                              "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        data = {'Wavelength (nm)': res['wl']}
        if 'ref_names' in res:   # SAM / NNLS unmixing: reference spectra
            for i, name in enumerate(res['ref_names']):
                data[name] = res['loadings'][i]
        elif res['method'] == "K-Means clustering":
            for i in range(res['loadings'].shape[0]):
                data[f"Cluster {i+1} centroid"] = res['loadings'][i]
        else:
            for i in range(res['loadings'].shape[0]):
                data[f"Component {i+1}"] = res['loadings'][i]
        try:
            pd.DataFrame(data).to_csv(path, index=False)
            self._refresh_status_bar(f"Component spectra saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Chemometrics", f"Could not save CSV:\n{e}")

    
    def _build_experiment_from_ui(self, name: str) -> ExperimentState:
        # figure out element label vs slider
        element_name = self._active_element()
        band_index = int(self.slider.value())
        divider_enabled = bool(self.divide_checkbox.isChecked())
        divider_wavelength = float(self.divider_spin.value()) if divider_enabled else None
        cmap_name = self.colormap_combo.currentText()
        axes_units = self.axes_units_combo.currentText()
        mm_per_px = float(self.mm_per_px_spin.value())
    
        autoscale = bool(self.autoscale_checkbox.isChecked())
        pmin = float(self.pmin_spin.value()); pmax = float(self.pmax_spin.value())
        vmin = float(self.vmin_spin.value()); vmax = float(self.vmax_spin.value())
    
        mask_enabled = bool(self.active_mask is not None and np.any(self.active_mask))
        mask_wl = None
        mask_thr = None
    
        line_coords = list(self.shared_line) if self.shared_line is not None else None
        parallel_buffer = int(self.parallel_line_spinbox.value())
        ignore_zeros = bool(self.ignore_null_checkbox.isChecked())
    
        pixel_coords = list(self.current_pixel) if self.current_pixel is not None else None
        peak_detection_enabled = bool(self.peak_detection_checkbox.isChecked())
        peak_prominence = float(self.prominence_spinbox.value())
        peak_distance = int(self.distance_spinbox.value())
    
        view_imaging_map = get_view(self.map_ax)
        view_science_libs = get_view(self.ax)
        view_right_plot = get_view(self.line_ax)
    
        mode_line = bool(self.line_mode_button.isChecked())
        mode_pixel = bool(self.pixel_mode_button.isChecked())
    
        nan_tolerance = int(self.nan_tolerance_spin.value())

        # Map Explorer baseline & display
        baseline_enabled = bool(self.baseline_checkbox.isChecked())
        baseline_method = self.baseline_method_combo.currentText()
        baseline_halfwidth = float(self.baseline_halfwidth_spin.value())
        baseline_gap = float(self.baseline_gap_spin.value())
        show_axes = bool(self.um_axes_checkbox.isChecked())
        locked_view = getattr(self, '_locked_view', None)

        # Normalization settings
        norm_method = self.norm_combo.currentText()
        norm_cont_start = float(self.norm_cont_start.value())
        norm_cont_end = float(self.norm_cont_end.value())
        norm_kernel = int(self.norm_kernel_spin.value())

        # Composite overlay settings
        composite_layers = []
        for i in range(5):
            composite_layers.append({
                'line': self.comp_layer_lines[i],
                'color': self.comp_layer_colors[i],
                'opacity': self.comp_layer_opacity_sliders[i].value(),
                'gain': float(self.comp_layer_gain_spins[i].value()),
                'auto': bool(self.comp_layer_auto_cbs[i].isChecked()),
                'vmin': float(self.comp_layer_min_spins[i].value()),
                'vmax': float(self.comp_layer_max_spins[i].value()),
                'baseline': bool(self.comp_layer_baseline_cbs[i].isChecked()),
                'colorbar': bool(self.comp_layer_cbar_cbs[i].isChecked()),
                'enabled': bool(self.comp_layer_enables[i].isChecked()),
            })
        composite_pmin = float(self.comp_pmin_spin.value())
        composite_pmax = float(self.comp_pmax_spin.value())
        composite_bl_method = self.comp_bl_method.currentText()
        composite_bl_hw = float(self.comp_bl_halfwidth.value())
        composite_bl_gap = float(self.comp_bl_gap.value())
        composite_bg = self.comp_bg_combo.currentText()

        # Photo / homography
        polygon_points = list(self.photo_polygon)
        calib_points = list(self.calib_pts_photo)
        drag_calib_enabled = bool(self.drag_calib_enabled)
        Hpl = ndarray_or_none_to_list(self.H_photo_to_libs)
        Hlp = ndarray_or_none_to_list(self.H_libs_to_photo)
        view_photo_tab = get_view(self.photo_ax_tab) if hasattr(self, 'photo_ax_tab') else {}
        view_science_photo = get_view(self.photo_ax_sci) if hasattr(self, 'photo_ax_sci') else {}

        return ExperimentState(
            name=name,
            cube_path=self.loaded_cube_path,
            photo_path=self.loaded_photo_path,
            polygon_points=polygon_points,
            calib_points=calib_points,
            drag_calib_enabled=drag_calib_enabled,
            H_photo_to_libs=Hpl,
            H_libs_to_photo=Hlp,
            view_photo_tab=view_photo_tab,
            view_science_photo=view_science_photo,
            element_name=element_name,
            band_index=band_index,
            divider_enabled=divider_enabled,
            divider_wavelength=divider_wavelength,
            cmap_name=cmap_name,
            axes_units=axes_units,
            mm_per_px=mm_per_px,
            autoscale=autoscale,
            pmin=pmin, pmax=pmax, vmin=vmin, vmax=vmax,
            band_label=self.last_band_label or "",
            mask_enabled=mask_enabled,
            mask_wavelength=mask_wl,
            mask_threshold=mask_thr,
            line_coords=line_coords,
            parallel_buffer=parallel_buffer,
            ignore_zeros=ignore_zeros,
            pixel_coords=pixel_coords,
            peak_detection_enabled=peak_detection_enabled,
            peak_prominence=peak_prominence,
            peak_distance=peak_distance,
            view_imaging_map=view_imaging_map,
            view_science_libs=view_science_libs,
            view_right_plot=view_right_plot,
            mode_line=mode_line,
            nan_tolerance=nan_tolerance,
            mode_pixel=mode_pixel,
            baseline_enabled=baseline_enabled,
            baseline_method=baseline_method,
            baseline_halfwidth=baseline_halfwidth,
            baseline_gap=baseline_gap,
            show_axes=show_axes,
            locked_view=locked_view,
            norm_method=norm_method,
            norm_cont_start=norm_cont_start,
            norm_cont_end=norm_cont_end,
            norm_kernel=norm_kernel,
            composite_layers=composite_layers,
            composite_pmin=composite_pmin,
            composite_pmax=composite_pmax,
            composite_bl_method=composite_bl_method,
            composite_bl_hw=composite_bl_hw,
            composite_bl_gap=composite_bl_gap,
            composite_bg=composite_bg,
            composite_overlay_pos=(
                self.comp_overlay_pos_combo.currentText()
                if hasattr(self, 'comp_overlay_pos_combo') else "Inside image"),
            divider_element=self._active_div_element(),
            divider_min=float(self.div_min_spin.value()),
            divider_scale=float(self.div_scale_spin.value()),
            cube_baseline_method=str(self._baseline_method_applied or "None"),
            cube_baseline_snip_iter=int(self.cube_bl_snip_iter_spin.value()),
            cube_baseline_window=int(self.cube_bl_window_spin.value()),
            cube_baseline_clip_negatives=bool(self.cube_bl_clip_neg_chk.isChecked()),
            cube_baseline_asls_log10_lam=float(self.cube_bl_asls_lam_spin.value()),
            cube_baseline_asls_p=float(self.cube_bl_asls_p_spin.value()),
            cube_baseline_asls_iter=int(self.cube_bl_asls_iter_spin.value()),
        )
    
    
    def _apply_experiment_to_ui(self, exp: ExperimentState):
        # Load cube if needed
        if exp.cube_path and (self.loaded_cube_path != exp.cube_path):
            try:
                if self.original_ds is not None:
                    try: self.original_ds.close()
                    except Exception: pass
                self.original_ds = xr.open_dataset(exp.cube_path)
                self.ds = self.original_ds.copy()
                self.loaded_cube_path = exp.cube_path
                base = os.path.basename(exp.cube_path)
                self.setWindowTitle(f"Hypercube Explorer — {base}")
                self._populate_metadata_tab(exp.cube_path)
                bands = self._get_bands_values()
                if bands is None:
                    return
                self.slider.setMaximum(len(bands) - 1)
                self._chem_reset()
            except Exception as e:
                QMessageBox.critical(self, "Project", f"Could not open cube:\n{e}")
                return

        # Load photo if needed
        if getattr(exp, 'photo_path', None):
            try:
                img = plt.imread(exp.photo_path)
                if img.dtype.kind == 'f' and img.max() <= 1.0: img = (img * 255).astype(np.uint8)
                self.photo_img = img
                self.loaded_photo_path = exp.photo_path
            except Exception as e:
                QMessageBox.warning(self, "Project", f"Could not load photo:\n{e}")
                self.photo_img = None
                self.loaded_photo_path = None
    
        # Controls — restore element into the correct combo
        element_name = exp.element_name
        if element_name in element_wavelengths_specialized and self.element_combo_spec.findText(element_name) >= 0:
            self.element_combo.setCurrentIndex(0)
            self.element_combo_spec.setCurrentText(element_name)
        elif element_name in element_wavelengths_common and self.element_combo.findText(element_name) >= 0:
            self.element_combo_spec.setCurrentIndex(0)
            self.element_combo.setCurrentText(element_name)
        else:
            self.element_combo.setCurrentText('None')
            self.element_combo_spec.setCurrentIndex(0)
        self.slider.setValue(int(exp.band_index))
        self.divide_checkbox.setChecked(bool(exp.divider_enabled))
        if exp.divider_wavelength is not None:
            self.divider_spin.setValue(float(exp.divider_wavelength))
        div_elem = getattr(exp, 'divider_element', 'None')
        self.div_element_combo.blockSignals(True)
        self.div_element_combo_spec.blockSignals(True)
        self.div_element_combo.setCurrentIndex(0)
        self.div_element_combo_spec.setCurrentIndex(0)
        if div_elem and div_elem != 'None':
            if div_elem in element_wavelengths_common:
                self.div_element_combo.setCurrentText(div_elem)
            elif div_elem in element_wavelengths_specialized:
                self.div_element_combo_spec.setCurrentText(div_elem)
        self.div_element_combo.blockSignals(False)
        self.div_element_combo_spec.blockSignals(False)
        self.div_min_spin.setValue(float(getattr(exp, 'divider_min', 0.0)))
        self.div_scale_spin.setValue(float(getattr(exp, 'divider_scale', 1000.0)))
        self.colormap_combo.setCurrentText(exp.cmap_name if exp.cmap_name in element_colormaps else 'Viridis')
        self.axes_units_combo.setCurrentText(exp.axes_units if exp.axes_units in ("µm","mm") else "mm")
        self.mm_per_px_spin.setValue(float(exp.mm_per_px))
        self.autoscale_checkbox.setChecked(bool(exp.autoscale))
        self.pmin_spin.setValue(float(exp.pmin)); self.pmax_spin.setValue(float(exp.pmax))
        self.vmin_spin.setValue(float(exp.vmin)); self.vmax_spin.setValue(float(exp.vmax))
        self.last_band_label = exp.band_label
    
        # Masking (handled by the Mask tab; legacy fields kept for backward compatibility)

        # Photo state
        self.photo_polygon = list(getattr(exp, 'polygon_points', None) or [])
        self.calib_pts_photo = list(getattr(exp, 'calib_points', None) or [])
        self.drag_calib_enabled = bool(getattr(exp, 'drag_calib_enabled', False))
        if hasattr(self, 'chk_drag_calib'):
            self.chk_drag_calib.setChecked(self.drag_calib_enabled)
        self.H_photo_to_libs = list_or_none_to_ndarray(getattr(exp, 'H_photo_to_libs', None))
        self.H_libs_to_photo = list_or_none_to_ndarray(getattr(exp, 'H_libs_to_photo', None))
    
        # Tools
        self.parallel_line_spinbox.setValue(int(exp.parallel_buffer))
        self.ignore_null_checkbox.setChecked(bool(exp.ignore_zeros))
    
        self.peak_detection_checkbox.setChecked(bool(exp.peak_detection_enabled))
        self.prominence_spinbox.setValue(float(exp.peak_prominence))
        self.distance_spinbox.setValue(int(exp.peak_distance))
    
        # Update images
        self.update_plot()
    
        # Line/pixel overlays (must happen after update_plot)
        self.erase_line(redraw=False)
        if exp.line_coords:
            self.shared_line = [float(v) for v in exp.line_coords]
            # draw line artists
            y0, x0, y1, x1 = self.shared_line
            self.line_libs = Line2D([x0, x1], [y0, y1], color='red', linewidth=1.5, zorder=10)
            self.ax.add_line(self.line_libs)
    
        if exp.pixel_coords:
            yl, xl = int(exp.pixel_coords[0]), int(exp.pixel_coords[1])
            self._set_pixel_marker(yl, xl)
            # also restore spectrum plot
            self.plot_spectrum(yl, xl)
    
        # Views (zoom/pan)
        set_view(self.map_ax, exp.view_imaging_map)
        set_view(self.ax, exp.view_science_libs)
        set_view(self.line_ax, exp.view_right_plot)
    
        self.canvas.draw_idle()
        self.map_canvas.draw_idle()
        self.line_plot_canvas.draw_idle()

        if hasattr(self, 'photo_ax_sci'):
            set_view(self.photo_ax_sci, getattr(exp, 'view_science_photo', None))
            self.photo_canvas_sci.draw_idle()
        if hasattr(self, 'photo_ax_tab'):
            set_view(self.photo_ax_tab, getattr(exp, 'view_photo_tab', None))
            self.photo_canvas_tab.draw_idle()
    
        # Tool mode
        self.line_mode_button.setChecked(bool(exp.mode_line))
        self.pixel_mode_button.setChecked(bool(exp.mode_pixel))
        
        self.nan_tolerance_spin.setValue(int(getattr(exp, "nan_tolerance", 5)))

        # Map Explorer baseline & display — restore
        self.baseline_checkbox.setChecked(bool(getattr(exp, "baseline_enabled", False)))
        bl_m = getattr(exp, "baseline_method", "Peak height")
        if bl_m in ("Peak height", "Peak area"):
            self.baseline_method_combo.setCurrentText(bl_m)
        self.baseline_halfwidth_spin.setValue(float(getattr(exp, "baseline_halfwidth", 0.10)))
        self.baseline_gap_spin.setValue(float(getattr(exp, "baseline_gap", 0.02)))
        self.um_axes_checkbox.setChecked(bool(getattr(exp, "show_axes", True)))
        self._locked_view = getattr(exp, "locked_view", None)
        if hasattr(self, 'record_view_btn'):
            self.clear_view_btn.setEnabled(self._locked_view is not None)
            self._update_view_info_label()

        # Cube-wide baseline correction — restore settings (UI only; apply on request)
        cb_method = getattr(exp, "cube_baseline_method", "None") or "None"
        if hasattr(self, 'cube_baseline_combo'):
            idx_b = self.cube_baseline_combo.findText(cb_method)
            try:
                self.cube_baseline_combo.blockSignals(True)
                self.cube_baseline_combo.setCurrentIndex(idx_b if idx_b >= 0 else 0)
            finally:
                self.cube_baseline_combo.blockSignals(False)
            self.cube_bl_snip_iter_spin.setValue(int(getattr(exp, "cube_baseline_snip_iter", 40)))
            self.cube_bl_window_spin.setValue(int(getattr(exp, "cube_baseline_window", 101)))
            self.cube_bl_clip_neg_chk.setChecked(bool(getattr(exp, "cube_baseline_clip_negatives", True)))
            self.cube_bl_asls_lam_spin.setValue(float(getattr(exp, "cube_baseline_asls_log10_lam", 6.0)))
            self.cube_bl_asls_p_spin.setValue(float(getattr(exp, "cube_baseline_asls_p", 0.01)))
            self.cube_bl_asls_iter_spin.setValue(int(getattr(exp, "cube_baseline_asls_iter", 10)))
            self._cube_baseline_on_method_changed()
            self._baseline_ds = None
            self._baseline_method_applied = "None"
            self.cube_bl_status_label.setText("")

        # Normalization — restore settings
        norm_method = getattr(exp, "norm_method", "None")
        idx = self.norm_combo.findText(norm_method)
        self.norm_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.norm_cont_start.setValue(float(getattr(exp, "norm_cont_start", 350.0)))
        self.norm_cont_end.setValue(float(getattr(exp, "norm_cont_end", 355.0)))
        self.norm_kernel_spin.setValue(int(getattr(exp, "norm_kernel", 21)))

        # Ask user if they want to re-run baseline + normalization pipeline
        if cb_method != "None" or norm_method != "None":
            steps = []
            if cb_method != "None":
                steps.append(f"Baseline: {cb_method}")
            if norm_method != "None":
                steps.append(f"Normalization: {norm_method}")
            reply = QMessageBox.question(
                self, "Re-apply preprocessing?",
                "This experiment was saved with preprocessing:\n\n  "
                + "\n  ".join(steps)
                + "\n\nRe-applying may take a while on large cubes.\nApply now?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                if cb_method != "None":
                    self._cube_baseline_apply()
                if norm_method != "None":
                    self.apply_normalization()
            else:
                if cb_method != "None":
                    self.cube_bl_status_label.setText("(not applied)")
                if norm_method != "None":
                    self.norm_status_label.setText("(not applied)")

        # Composite overlay — restore settings
        layers = getattr(exp, "composite_layers", None)
        if layers:
            for i, ld in enumerate(layers[:5]):
                self.comp_layer_lines[i] = ld.get('line')
                col = ld.get('color', list(COMPOSITE_DEFAULT_COLORS[i]))
                self.comp_layer_colors[i] = col
                r, g, b = col
                self.comp_layer_color_btns[i].setStyleSheet(
                    f"background:rgb({r},{g},{b});border:1px solid #888;border-radius:3px;"
                )
                if ld.get('line'):
                    self.comp_layer_labels[i].setText(
                        f"Layer {i+1}: <b>{ld['line']}</b>"
                    )
                else:
                    self.comp_layer_labels[i].setText(f"Layer {i+1}: <i>none</i>")
                self.comp_layer_opacity_sliders[i].blockSignals(True)
                self.comp_layer_opacity_sliders[i].setValue(int(ld.get('opacity', 80)))
                self.comp_layer_opacity_sliders[i].blockSignals(False)
                self.comp_layer_gain_spins[i].blockSignals(True)
                self.comp_layer_gain_spins[i].setValue(float(ld.get('gain', 1.0)))
                self.comp_layer_gain_spins[i].blockSignals(False)
                self.comp_layer_auto_cbs[i].blockSignals(True)
                self.comp_layer_auto_cbs[i].setChecked(bool(ld.get('auto', True)))
                self.comp_layer_auto_cbs[i].blockSignals(False)
                self.comp_layer_min_spins[i].blockSignals(True)
                self.comp_layer_min_spins[i].setValue(float(ld.get('vmin', 0)))
                self.comp_layer_min_spins[i].blockSignals(False)
                self.comp_layer_max_spins[i].blockSignals(True)
                self.comp_layer_max_spins[i].setValue(float(ld.get('vmax', 65535)))
                self.comp_layer_max_spins[i].blockSignals(False)
                self.comp_layer_baseline_cbs[i].blockSignals(True)
                self.comp_layer_baseline_cbs[i].setChecked(bool(ld.get('baseline', False)))
                self.comp_layer_baseline_cbs[i].blockSignals(False)
                self.comp_layer_cbar_cbs[i].blockSignals(True)
                self.comp_layer_cbar_cbs[i].setChecked(bool(ld.get('colorbar', False)))
                self.comp_layer_cbar_cbs[i].blockSignals(False)
                self.comp_layer_enables[i].blockSignals(True)
                self.comp_layer_enables[i].setChecked(bool(ld.get('enabled', True)))
                self.comp_layer_enables[i].blockSignals(False)
        self.comp_pmin_spin.blockSignals(True)
        self.comp_pmin_spin.setValue(float(getattr(exp, "composite_pmin", 0.5)))
        self.comp_pmin_spin.blockSignals(False)
        self.comp_pmax_spin.blockSignals(True)
        self.comp_pmax_spin.setValue(float(getattr(exp, "composite_pmax", 99.5)))
        self.comp_pmax_spin.blockSignals(False)
        comp_bl_m = getattr(exp, "composite_bl_method", "Peak height")
        if comp_bl_m in ("Peak height", "Peak area"):
            self.comp_bl_method.setCurrentText(comp_bl_m)
        self.comp_bl_halfwidth.blockSignals(True)
        self.comp_bl_halfwidth.setValue(float(getattr(exp, "composite_bl_hw", 0.10)))
        self.comp_bl_halfwidth.blockSignals(False)
        self.comp_bl_gap.blockSignals(True)
        self.comp_bl_gap.setValue(float(getattr(exp, "composite_bl_gap", 0.02)))
        self.comp_bl_gap.blockSignals(False)
        bg_idx = self.comp_bg_combo.findText(getattr(exp, "composite_bg", "Black"))
        if bg_idx >= 0:
            self.comp_bg_combo.setCurrentIndex(bg_idx)
        if hasattr(self, 'comp_overlay_pos_combo'):
            ov_idx = self.comp_overlay_pos_combo.findText(
                getattr(exp, "composite_overlay_pos", "Inside image"))
            if ov_idx >= 0:
                self.comp_overlay_pos_combo.blockSignals(True)
                self.comp_overlay_pos_combo.setCurrentIndex(ov_idx)
                self.comp_overlay_pos_combo.blockSignals(False)
        self._comp_refresh_ptable_highlights()
        self._update_rgb_composite()
        
        self._roi_refresh_lists()
        
        self._refresh_status_bar()
        
    def _proj_new(self):
        self.project = ProjectState(path=None, experiments=[], active_index=-1)

        self._refresh_status_bar("New project")

    
    def _proj_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Hypercube Explorer Project (*.hcxproj)")
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            exps = []
            _valid_fields = {f.name for f in ExperimentState.__dataclass_fields__.values()}
            for e in data.get('experiments', []):
                filtered = {k: v for k, v in e.items() if k in _valid_fields}
                exps.append(ExperimentState(**filtered))

            shared_lines = data.get('shared_lines', [])
            shared_pixels = data.get('shared_pixels', [])
            # ... keep existing ...
            self.project = ProjectState(path=path, experiments=exps,
                                        active_index=int(data.get('active_index', -1)),
                                        shared_lines=shared_lines, shared_pixels=shared_pixels)            # choose experiment to load
            names = [e.name for e in self.project.experiments]
            if not names:
                QMessageBox.information(self, "Project", "Project opened (no experiments yet).")
                self._roi_refresh_lists()
                self._refresh_status_bar("Project opened")
                return
            dlg = ExperimentPicker(self, "Load Experiment", names, initial_index=max(self.project.active_index,0))
            if dlg.exec_() == QDialog.Accepted:
                i = dlg.selected_row()
                if 0 <= i < len(self.project.experiments):
                    self.project.active_index = i
                    self._apply_experiment_to_ui(self.project.experiments[i])
        except Exception as e:
            QMessageBox.critical(self, "Project", f"Could not open project:\n{e}")
        
        self._refresh_status_bar("Project opened")    
            
    def _proj_save_payload(self):
        return {
            'active_index': int(self.project.active_index),
            'experiments': [asdict(e) for e in self.project.experiments],
            'shared_lines': self.project.shared_lines,
            'shared_pixels': self.project.shared_pixels,
        }
    
    def _proj_save(self):
        if self.project.path is None:
            return self._proj_save_as()
        # Update current experiment before saving the project
        self._proj_update_current_experiment(silent=True)
        try:
            with open(self.project.path, 'w', encoding='utf-8') as f:
                json.dump(self._proj_save_payload(), f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage(
                f"Saved project (experiment updated): {os.path.basename(self.project.path)}",
                3000
            )
        except Exception as e:
            QMessageBox.critical(self, "Project", f"Could not save project:\n{e}")
        
        self._refresh_status_bar(f"Saved project: {os.path.basename(self.project.path)}")
    
    def _proj_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", "Hypercube Explorer Project (*.hcxproj)")
        if not path: return
        if not path.lower().endswith('.hcxproj'):
            path += '.hcxproj'
        self.project.path = path
        self._proj_save()
        self._refresh_status_bar(f"Saved project: {os.path.basename(self.project.path)}")

    
    def _proj_add_experiment(self):
        if self.ds is None:
            QMessageBox.information(self, "Experiment", "Open a LIBS cube first.")
            return
        name, ok = QInputDialog.getText(self, "New Experiment", "Name:")
        if not ok or not name.strip(): return
        exp = self._build_experiment_from_ui(name.strip())
        self.project.experiments.append(exp)
        self.project.active_index = len(self.project.experiments) - 1
        self.statusBar().showMessage(f"Added experiment '{exp.name}'", 3000)

    def _proj_update_current_experiment(self, silent: bool = False):
        if not self.project.experiments:
            if not silent:
                QMessageBox.information(self, "Experiment", "No experiments in project.")
            return
        i = self.project.active_index
        if i < 0 or i >= len(self.project.experiments):
            if not silent:
                QMessageBox.information(self, "Experiment", "No active experiment selected.")
            return
    
        # Preserve the original experiment name
        old_name = self.project.experiments[i].name
        updated = self._build_experiment_from_ui(old_name)
        self.project.experiments[i] = updated
    
        if not silent:
            self._refresh_status_bar(f"Updated experiment '{old_name}'")

    
    def _proj_load_experiment(self):
        if not self.project.experiments:
            QMessageBox.information(self, "Experiment", "No experiments in project.")
            return
        names = [e.name for e in self.project.experiments]
        dlg = ExperimentPicker(self, "Load Experiment", names, initial_index=max(self.project.active_index,0))
        if dlg.exec_() == QDialog.Accepted:
            i = dlg.selected_row()
            if 0 <= i < len(self.project.experiments):
                self.project.active_index = i
                self._apply_experiment_to_ui(self.project.experiments[i])
                self._refresh_status_bar("Experiment loaded")   
                
    def _proj_rename_experiment(self):
        if not self.project.experiments:
            QMessageBox.information(self, "Experiment", "No experiments in project.")
            return
        names = [e.name for e in self.project.experiments]
        dlg = ExperimentPicker(self, "Rename Experiment (select one)", names, initial_index=max(self.project.active_index,0))
        if dlg.exec_() == QDialog.Accepted:
            i = dlg.selected_row()
            if 0 <= i < len(self.project.experiments):
                new_name, ok = QInputDialog.getText(self, "Rename Experiment", "New name:", text=self.project.experiments[i].name)
                if ok and new_name.strip():
                    self.project.experiments[i].name = new_name.strip()
                    self._refresh_status_bar("Experiment renamed")
    
    def _proj_delete_experiment(self):
        if not self.project.experiments:
            QMessageBox.information(self, "Experiment", "No experiments in project.")
            return
        names = [e.name for e in self.project.experiments]
        dlg = ExperimentPicker(self, "Delete Experiment (select one)", names, initial_index=max(self.project.active_index,0))
        if dlg.exec_() == QDialog.Accepted:
            i = dlg.selected_row()
            if 0 <= i < len(self.project.experiments):
                del self.project.experiments[i]
                if not self.project.experiments:
                    self.project.active_index = -1
                else:
                    self.project.active_index = min(i, len(self.project.experiments)-1)
                    self._refresh_status_bar("Experiment deleted")

    def _roi_store(self):
        """Return (lines_list, pixels_list) storage dicts, project-shared or per-experiment."""
        if self.chk_use_project_rois.isChecked() or not self.project.experiments or self.project.active_index < 0:
            return self.project.shared_lines, self.project.shared_pixels
        # per-experiment storage (requires the optional fields in ExperimentState)
        exp = self.project.experiments[self.project.active_index]
        if not hasattr(exp, 'roi_lines'):
            exp.roi_lines = []
        if not hasattr(exp, 'roi_pixels'):
            exp.roi_pixels = []
        return exp.roi_lines, exp.roi_pixels
    
    def _roi_sync_spinboxes_from_line(self):
        """Update the custom line spinboxes to reflect the current shared_line."""
        if self.shared_line is None:
            return
        y0, x0, y1, x1 = self.shared_line
        for spin in (self.roi_x0_spin, self.roi_y0_spin, self.roi_x1_spin, self.roi_y1_spin):
            spin.blockSignals(True)
        self.roi_x0_spin.setValue(int(round(x0)))
        self.roi_y0_spin.setValue(int(round(y0)))
        self.roi_x1_spin.setValue(int(round(x1)))
        self.roi_y1_spin.setValue(int(round(y1)))
        for spin in (self.roi_x0_spin, self.roi_y0_spin, self.roi_x1_spin, self.roi_y1_spin):
            spin.blockSignals(False)

    def _update_roi_spinbox_ranges(self):
        """Set custom line spinbox ranges based on the loaded dataset dimensions."""
        if self.ds is None:
            return
        try:
            var_name = list(self.ds.data_vars)[0]
            da = self.ds[var_name]
            dims = da.dims
            ny = da.shape[dims.index(dims[-2])] if len(dims) >= 2 else 1
            nx = da.shape[dims.index(dims[-1])] if len(dims) >= 1 else 1
            for spin in (self.roi_x0_spin, self.roi_x1_spin):
                spin.setRange(0, max(0, nx - 1))
            for spin in (self.roi_y0_spin, self.roi_y1_spin):
                spin.setRange(0, max(0, ny - 1))
        except Exception:
            pass

    def _roi_refresh_lists(self):
        lines, pixels = self._roi_store()
        # repopulate UI lists
        self.list_lines.clear()
        for item in lines:
            name = item.get('name', 'Line')
            coords = item.get('coords', None)
            if not coords or len(coords) < 4:
                self.list_lines.addItem(f"{name}  —  [no coords]")
                continue
            self.list_lines.addItem(f"{name}  —  [{coords[1]:.0f},{coords[0]:.0f}]→[{coords[3]:.0f},{coords[2]:.0f}]")
        self.list_pixels.clear()
        for item in pixels:
            name = item.get('name', 'Pixel')
            coords = item.get('coords', None)
            if not coords or len(coords) < 2:
                self.list_pixels.addItem(f"{name}  —  (no coords)")
                continue
            self.list_pixels.addItem(f"{name}  —  ({coords[1]:.0f},{coords[0]:.0f})")

    def _roi_add_line_from_current(self):
        if self.shared_line is None:
            QMessageBox.information(self, "ROIs", "Draw a line first.")
            return
        lines, _ = self._roi_store()
        name, ok = QInputDialog.getText(self, "Add Line", "Name:", text=f"Line {len(lines)+1}")
        if not ok or not name.strip(): return
        lines.append({'name': name.strip(), 'coords': [float(v) for v in self.shared_line]})
        self._roi_refresh_lists()
        self.statusBar().showMessage(f"Added line '{name.strip()}'", 2000)

    def _roi_apply_custom_coords(self):
        """Draw a line from the custom coordinate spinboxes onto the map."""
        x0 = self.roi_x0_spin.value()
        y0 = self.roi_y0_spin.value()
        x1 = self.roi_x1_spin.value()
        y1 = self.roi_y1_spin.value()
        self.erase_line(redraw=False)
        self.shared_line = [float(y0), float(x0), float(y1), float(x1)]
        self.line_libs = Line2D([x0, x1], [y0, y1], color='red', linewidth=1.5, zorder=10)
        self.ax.add_line(self.line_libs)
        if self.photo_img is not None and hasattr(self, 'photo_ax_sci'):
            p0 = self._libs_to_img(x0, y0)
            p1 = self._libs_to_img(x1, y1)
            if p0 is not None and p1 is not None:
                self.line_img = Line2D([p0[0], p1[0]], [p0[1], p1[1]], color='red', linewidth=1.2, zorder=10)
                self.photo_ax_sci.add_line(self.line_img)
        self._update_line_label()
        self.update_line_plot()
        self.canvas.draw_idle()
        if hasattr(self, 'photo_canvas_sci'):
            self.photo_canvas_sci.draw_idle()

    def _roi_draw_custom_line(self):
        """Draw a custom line on the map without saving it as an ROI."""
        self._roi_apply_custom_coords()
        self.statusBar().showMessage("Custom line drawn", 2000)

    def _roi_add_custom_line(self):
        """Draw a custom line and save it as a named ROI."""
        self._roi_apply_custom_coords()
        lines, _ = self._roi_store()
        name, ok = QInputDialog.getText(self, "Add Line", "Name:", text=f"Line {len(lines)+1}")
        if not ok or not name.strip():
            return
        lines.append({'name': name.strip(), 'coords': [float(v) for v in self.shared_line]})
        self._roi_refresh_lists()
        self.statusBar().showMessage(f"Added custom line '{name.strip()}'", 2000)

    def _roi_use_selected_line(self):
        idx = self.list_lines.currentRow()
        lines, _ = self._roi_store()
        if idx < 0 or idx >= len(lines):
            QMessageBox.information(self, "ROIs", "Select a line.")
            return
        coords = lines[idx]['coords']
        # apply as active shared line
        self.erase_line(redraw=False)
        self.shared_line = [float(v) for v in coords]
        y0, x0, y1, x1 = self.shared_line
        self.line_libs = Line2D([x0, x1], [y0, y1], color='red', linewidth=1.5, zorder=10)
        self.ax.add_line(self.line_libs)

        if self.photo_img is not None and hasattr(self, 'photo_ax_sci'):
            p0 = self._libs_to_img(x0, y0); p1 = self._libs_to_img(x1, y1)
            if p0 is not None and p1 is not None:
                self.line_img = Line2D([p0[0], p1[0]], [p0[1], p1[1]], color='red', linewidth=1.2, zorder=10)
                self.photo_ax_sci.add_line(self.line_img)

        self._update_line_label()
        self.update_line_plot()
        self._roi_sync_spinboxes_from_line()

        self.canvas.draw_idle()
        if hasattr(self, 'photo_canvas_sci'):
            self.photo_canvas_sci.draw_idle()
    
    def _roi_rename_selected_line(self):
        idx = self.list_lines.currentRow()
        lines, _ = self._roi_store()
        if idx < 0 or idx >= len(lines): return
        old = lines[idx]['name']
        name, ok = QInputDialog.getText(self, "Rename Line", "New name:", text=old)
        if not ok or not name.strip(): return
        lines[idx]['name'] = name.strip()
        self._roi_refresh_lists()
    
    def _roi_remove_selected_line(self):
        idx = self.list_lines.currentRow()
        lines, _ = self._roi_store()
        if idx < 0 or idx >= len(lines): return
        del lines[idx]
        self._roi_refresh_lists()
        
    def _roi_add_pixel_from_current(self):
        if self.current_pixel is None:
            QMessageBox.information(self, "ROIs", "Pick a pixel first (Pixel mode).")
            return
        _, pixels = self._roi_store()
        name, ok = QInputDialog.getText(self, "Add Pixel", "Name:", text=f"Pixel {len(pixels)+1}")
        if not ok or not name.strip(): return
        yl, xl = self.current_pixel
        pixels.append({'name': name.strip(), 'coords': [int(yl), int(xl)]})
        self._roi_refresh_lists()
        self.statusBar().showMessage(f"Added pixel '{name.strip()}'", 2000)
    
    def _roi_use_selected_pixel(self):
        idx = self.list_pixels.currentRow()
        _, pixels = self._roi_store()
        if idx < 0 or idx >= len(pixels):
            QMessageBox.information(self, "ROIs", "Select a pixel.")
            return
        yl, xl = pixels[idx]['coords']
        self._set_pixel_marker(int(yl), int(xl))
        self.plot_spectrum(int(yl), int(xl))
    
    def _roi_rename_selected_pixel(self):
        idx = self.list_pixels.currentRow()
        _, pixels = self._roi_store()
        if idx < 0 or idx >= len(pixels): return
        old = pixels[idx]['name']
        name, ok = QInputDialog.getText(self, "Rename Pixel", "New name:", text=old)
        if not ok or not name.strip(): return
        pixels[idx]['name'] = name.strip()
        self._roi_refresh_lists()
    
    def _roi_remove_selected_pixel(self):
        idx = self.list_pixels.currentRow()
        _, pixels = self._roi_store()
        if idx < 0 or idx >= len(pixels): return
        del pixels[idx]
        self._roi_refresh_lists()
        
    def _update_line_label(self):
        if self.shared_line is None:
            self.lbl_line_coords.setText("Line: —")
            return
        y0, x0, y1, x1 = self.shared_line
        self.lbl_line_coords.setText(f"Line: [{int(x0)},{int(y0)}] → [{int(x1)},{int(y1)}]")
    
    def _update_pixel_label(self):
        if self.current_pixel is None:
            self.lbl_pixel_coords.setText("Pixel: —")
            return
        y, x = self.current_pixel
        self.lbl_pixel_coords.setText(f"Pixel: ({int(x)},{int(y)})")

    # ====== Baseline Inspector ======

    def _bi_load_spectrum(self):
        if self.current_spectrum is None or self.current_pixel is None:
            QMessageBox.information(self, "Baseline Inspector",
                                    "Select a pixel on the LIBS map first.")
            return
        self._bi_spectrum = self.current_spectrum.copy()
        self._bi_wavelengths = self._get_bands_values().copy()
        self._bi_pixel = tuple(self.current_pixel)
        y, x = self._bi_pixel
        self.bi_pixel_label.setText(f"Pixel ({x}, {y})")

        bands = self._bi_wavelengths
        idx = self.slider.value()
        if bands is not None and 0 <= idx < len(bands):
            wl_sel = float(bands[idx])
        else:
            wl_sel = 0.0
        self.bi_center_spin.setValue(wl_sel)

        self._bi_update_plot()
        self.plot_tabs.setCurrentIndex(1)
        self.science_settings_tabs.setCurrentIndex(2)

    def _bi_update_plot(self):
        spec = getattr(self, '_bi_spectrum', None)
        wls = getattr(self, '_bi_wavelengths', None)
        if spec is None or wls is None:
            return

        center = float(self.bi_center_spin.value())
        hw = float(self.bi_hw_spin.value())
        gap = float(self.bi_gap_spin.value())
        view_range = float(self.bi_view_range_spin.value())

        bands = wls
        idx_center = int(np.abs(bands - center).argmin())
        center_snapped = float(bands[idx_center])

        left_mask = (bands >= center_snapped - hw) & (bands <= center_snapped - max(gap, 0.0))
        right_mask = (bands >= center_snapped + max(gap, 0.0)) & (bands <= center_snapped + hw)
        side_mask = left_mask | right_mask
        # Peak region: everything inside the sidebands (between inner edges)
        peak_mask = (bands >= center_snapped - max(gap, 0.0)) & (bands <= center_snapped + max(gap, 0.0))

        left_idx = np.where(left_mask)[0]
        right_idx = np.where(right_mask)[0]

        # Sloped baseline: linear interpolation between left and right sideband medians
        if left_idx.size > 0 and right_idx.size > 0:
            left_wl = float(np.mean(bands[left_idx]))
            left_val = float(np.nanmedian(spec[left_idx]))
            right_wl = float(np.mean(bands[right_idx]))
            right_val = float(np.nanmedian(spec[right_idx]))
            if abs(right_wl - left_wl) > 1e-12:
                slope = (right_val - left_val) / (right_wl - left_wl)
            else:
                slope = 0.0
            baseline = left_val + slope * (bands - left_wl)
        elif left_idx.size > 0:
            baseline = np.full_like(bands, float(np.nanmedian(spec[left_idx])))
            left_wl = float(np.mean(bands[left_idx]))
            left_val = float(np.nanmedian(spec[left_idx]))
            slope = 0.0
        elif right_idx.size > 0:
            baseline = np.full_like(bands, float(np.nanmedian(spec[right_idx])))
            left_wl = float(np.mean(bands[right_idx]))
            left_val = float(np.nanmedian(spec[right_idx]))
            slope = 0.0
        else:
            baseline = np.zeros_like(bands)
            slope = 0.0

        corrected = spec - baseline

        # Interpolate to fine grid within the full window for sub-channel accuracy
        full_win_mask = (bands >= center_snapped - hw) & (bands <= center_snapped + hw)
        full_win_idx = np.where(full_win_mask)[0]

        peak_height = 0.0
        peak_area = 0.0
        cross_left = center_snapped - hw
        cross_right = center_snapped + hw

        if full_win_idx.size > 1:
            n_fine = max(500, full_win_idx.size * 20)
            fine_wl = np.linspace(bands[full_win_idx[0]], bands[full_win_idx[-1]], n_fine)
            fine_spec = np.interp(fine_wl, bands, spec)
            fine_bl = np.interp(fine_wl, bands, baseline)
            fine_corr = fine_spec - fine_bl

            center_fi = int(np.argmin(np.abs(fine_wl - center_snapped)))

            # Find contiguous positive region around center on fine grid
            if fine_corr[center_fi] > 0:
                lo = center_fi
                while lo > 0 and fine_corr[lo - 1] > 0:
                    lo -= 1
                hi = center_fi
                while hi < n_fine - 1 and fine_corr[hi + 1] > 0:
                    hi += 1

                cross_left = float(fine_wl[lo])
                cross_right = float(fine_wl[hi])
                peak_area = float(np.trapz(fine_corr[lo:hi + 1], fine_wl[lo:hi + 1]))
                peak_height = float(np.max(fine_corr[lo:hi + 1]))
        else:
            fine_wl = fine_spec = fine_bl = fine_corr = None

        self.bi_height_label.setText(f"Peak height: {peak_height:.2f}")
        self.bi_area_label.setText(f"Peak area: {peak_area:.4f}")
        if peak_height > 0:
            self.bi_ratio_label.setText(f"Area / Height: {peak_area / peak_height:.4f}")
        else:
            self.bi_ratio_label.setText("Area / Height: —")

        ax = self.bi_ax
        ax.clear()

        view_mask = (bands >= center_snapped - view_range) & (bands <= center_snapped + view_range)
        view_idx = np.where(view_mask)[0]
        if view_idx.size == 0:
            view_idx = np.arange(len(bands))

        wl_v = bands[view_idx]
        sp_v = spec[view_idx]
        co_v = corrected[view_idx]
        bl_v = baseline[view_idx]

        # Raw channel data as dots + thin line
        ax.plot(wl_v, sp_v, 'o-', color='#888888', linewidth=0.6, markersize=2.5,
                label='Raw spectrum (channels)')

        # Interpolated curves (smooth)
        if fine_wl is not None:
            view_fine = (fine_wl >= wl_v[0]) & (fine_wl <= wl_v[-1])
            fwl = fine_wl[view_fine]
            fsp = fine_spec[view_fine]
            fbl = fine_bl[view_fine]
            fco = fine_corr[view_fine]

            ax.plot(fwl, fsp, color='#555555', linewidth=0.8, label='Interpolated spectrum')
            ax.plot(fwl, fco, color='#2196F3', linewidth=1.0, label='Baseline-corrected (interp.)')
            ax.plot(fwl, fbl, color='#FF9800', linewidth=0.8, linestyle='--',
                    label=f'Baseline (slope={slope:.1f}/nm)')
        else:
            ax.plot(wl_v, co_v, color='#2196F3', linewidth=1.0, label='Baseline-corrected')
            ax.plot(wl_v, bl_v, color='#FF9800', linewidth=0.8, linestyle='--',
                    label=f'Baseline (slope={slope:.1f}/nm)')

        ax.axhline(0, color='#FF9800', linewidth=0.4, linestyle=':')
        ax.axvline(center_snapped, color='#666', linewidth=0.5, linestyle=':')

        # Sideband fill on raw channels
        side_in_view = np.where(side_mask[view_idx[0]:view_idx[-1]+1])[0] + view_idx[0]
        if side_in_view.size > 0:
            ax.fill_between(bands[side_in_view],
                            np.nanmin(sp_v), spec[side_in_view],
                            color='#FF9800', alpha=0.15, label='Sideband windows')

        if left_idx.size > 0:
            ax.plot(float(np.mean(bands[left_idx])),
                    float(np.nanmedian(spec[left_idx])),
                    'o', color='#FF9800', markersize=5, zorder=5)
        if right_idx.size > 0:
            ax.plot(float(np.mean(bands[right_idx])),
                    float(np.nanmedian(spec[right_idx])),
                    'o', color='#FF9800', markersize=5, zorder=5)

        # Peak area fill using interpolated data between crossing points
        if fine_wl is not None and peak_area > 0:
            area_fine = (fine_wl >= cross_left) & (fine_wl <= cross_right)
            ax.fill_between(fine_wl[area_fine], fine_bl[area_fine], fine_spec[area_fine],
                            color='#4CAF50', alpha=0.25,
                            label=f'Peak area = {peak_area:.4f}')
            # Mark crossing points
            ax.plot(cross_left, np.interp(cross_left, bands, baseline), 'x',
                    color='#2E7D32', markersize=6, markeredgewidth=1.5, zorder=6)
            ax.plot(cross_right, np.interp(cross_right, bands, baseline), 'x',
                    color='#2E7D32', markersize=6, markeredgewidth=1.5, zorder=6)

        # Peak height using interpolated maximum
        if fine_wl is not None and peak_height > 0:
            peak_fine = (fine_wl >= cross_left) & (fine_wl <= cross_right)
            if np.any(peak_fine):
                max_fi = np.argmax(fine_corr * peak_fine.astype(float))
                ax.vlines(fine_wl[max_fi], fine_bl[max_fi], fine_spec[max_fi],
                          color='#F44336', linewidth=1.5,
                          label=f'Peak height = {peak_height:.1f}')
                ax.plot(fine_wl[max_fi], fine_spec[max_fi], 'v',
                        color='#F44336', markersize=6)

        ax.axvspan(center_snapped - hw, center_snapped - gap, alpha=0.08, color='orange')
        ax.axvspan(center_snapped + gap, center_snapped + hw, alpha=0.08, color='orange')
        ax.axvspan(center_snapped - gap, center_snapped + gap, alpha=0.08, color='#66BB6A')

        # Boundary lines at sideband / exclusion edges
        for bx in [center_snapped - hw, center_snapped - gap,
                    center_snapped + gap, center_snapped + hw]:
            ax.axvline(bx, color='#999', linewidth=0.4, linestyle=':')

        # Zone labels at top of plot
        ylo, yhi = ax.get_ylim()
        label_y = yhi - 0.04 * (yhi - ylo)
        ax.text(center_snapped - (hw + gap) / 2, label_y,
                "Left sideband\n(baseline estim.)",
                ha='center', va='top', fontsize=5.5, color='#E65100',
                fontstyle='italic')
        ax.text(center_snapped + (hw + gap) / 2, label_y,
                "Right sideband\n(baseline estim.)",
                ha='center', va='top', fontsize=5.5, color='#E65100',
                fontstyle='italic')
        ax.text(center_snapped, label_y,
                "Excl. zone\n(peak region)",
                ha='center', va='top', fontsize=5.5, color='#2E7D32',
                fontstyle='italic')

        y, x = getattr(self, '_bi_pixel', (0, 0))
        ax.set_title(f"Baseline Inspector — pixel ({x}, {y})  |  "
                     f"λ = {center_snapped:.3f} nm  |  hw = {hw:.3f}  gap = {gap:.3f}",
                     fontsize=8)
        ax.set_xlabel("Wavelength (nm)", fontsize=8)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.legend(fontsize=6, loc='upper right')

        self.bi_canvas.draw_idle()

    def _get_band_image(self, wl_or_idx):
        bands = self._get_bands_values()
        if bands is None:
            return np.zeros((1, 1), dtype=float)
    
        if isinstance(wl_or_idx, (int, np.integer)):
            idx = int(wl_or_idx)
        else:
            # snap wavelength to nearest index
            idx = int(np.abs(bands - float(wl_or_idx)).argmin())
    
        sel = self.ds.isel(bands=idx)  # <-- isel
        da = sel.to_array().squeeze() if isinstance(sel, xr.Dataset) else sel
        return np.asarray(da.values, dtype=float)
    
    def _local_baseline_map(self, center_wl, halfwidth_nm, gap_nm):
        bands = self._get_bands_values()
        if bands is None:
            return 0.0

        left_mask = (bands >= center_wl - halfwidth_nm) & (bands <= center_wl - max(gap_nm, 0.0))
        right_mask = (bands >= center_wl + max(gap_nm, 0.0)) & (bands <= center_wl + halfwidth_nm)

        left_idx = np.where(left_mask)[0]
        right_idx = np.where(right_mask)[0]

        def _median_map(indices):
            indices = np.unique(np.sort(indices))
            sel = self.ds.isel(bands=indices)
            da = sel.to_array().squeeze() if isinstance(sel, xr.Dataset) else sel
            cube = np.asarray(da.values, dtype=float)
            if cube.ndim == 2:
                return cube
            return np.nanmedian(cube, axis=0)

        if left_idx.size > 0 and right_idx.size > 0:
            left_wl = float(np.mean(bands[left_idx]))
            right_wl = float(np.mean(bands[right_idx]))
            left_val = _median_map(left_idx)
            right_val = _median_map(right_idx)
            denom = right_wl - left_wl
            if abs(denom) > 1e-12:
                t = (center_wl - left_wl) / denom
                return left_val + t * (right_val - left_val)
            else:
                return 0.5 * (left_val + right_val)
        elif left_idx.size > 0:
            return _median_map(left_idx)
        elif right_idx.size > 0:
            return _median_map(right_idx)
        else:
            return 0.0

    def _local_peak_area_map(self, center_wl, halfwidth_nm, gap_nm):
        """Compute baseline-corrected peak area map by integrating over center ± hw."""
        bands = self._get_bands_values()
        if bands is None:
            return np.zeros((1, 1), dtype=float)

        left_mask = (bands >= center_wl - halfwidth_nm) & (bands <= center_wl - max(gap_nm, 0.0))
        right_mask = (bands >= center_wl + max(gap_nm, 0.0)) & (bands <= center_wl + halfwidth_nm)
        # Integration over the full window (center ± hw)
        full_mask = (bands >= center_wl - halfwidth_nm) & (bands <= center_wl + halfwidth_nm)

        left_idx = np.where(left_mask)[0]
        right_idx = np.where(right_mask)[0]
        full_idx = np.where(full_mask)[0]

        if full_idx.size < 2:
            return np.zeros((1, 1), dtype=float)

        def _median_map(indices):
            indices = np.unique(np.sort(indices))
            sel = self.ds.isel(bands=indices)
            da = sel.to_array().squeeze() if isinstance(sel, xr.Dataset) else sel
            cube = np.asarray(da.values, dtype=float)
            if cube.ndim == 2:
                return cube
            return np.nanmedian(cube, axis=0)

        if left_idx.size > 0 and right_idx.size > 0:
            left_wl = float(np.mean(bands[left_idx]))
            right_wl = float(np.mean(bands[right_idx]))
            left_val = _median_map(left_idx)
            right_val = _median_map(right_idx)
            denom = right_wl - left_wl
        elif left_idx.size > 0:
            left_wl = float(np.mean(bands[left_idx]))
            left_val = _median_map(left_idx)
            right_wl = left_wl
            right_val = left_val
            denom = 0.0
        elif right_idx.size > 0:
            left_wl = float(np.mean(bands[right_idx]))
            left_val = _median_map(right_idx)
            right_wl = left_wl
            right_val = left_val
            denom = 0.0
        else:
            left_val = 0.0
            right_val = 0.0
            left_wl = center_wl
            denom = 0.0

        full_bands = bands[full_idx]
        full_sel = self.ds.isel(bands=full_idx.tolist())
        da = full_sel.to_array().squeeze() if isinstance(full_sel, xr.Dataset) else full_sel
        full_cube = np.asarray(da.values, dtype=float)
        if full_cube.ndim == 2:
            full_cube = full_cube[np.newaxis, :, :]

        n_bands = len(full_idx)
        shape_hw = full_cube.shape[1:]

        # Compute corrected cube and baseline per band
        corr_cube = np.empty_like(full_cube)
        for k in range(n_bands):
            wl_k = float(full_bands[k])
            if abs(denom) > 1e-12:
                t = (wl_k - left_wl) / denom
                bl_k = left_val + t * (right_val - left_val)
            else:
                bl_k = left_val
            corr_cube[k] = full_cube[k] - bl_k

        # Per-pixel mask: contiguous positive region around center band
        center_k = int(np.argmin(np.abs(full_bands - center_wl)))
        mask = np.zeros((n_bands,) + shape_hw, dtype=bool)
        mask[center_k] = corr_cube[center_k] > 0
        for k in range(center_k - 1, -1, -1):
            mask[k] = mask[k + 1] & (corr_cube[k] > 0)
        for k in range(center_k + 1, n_bands):
            mask[k] = mask[k - 1] & (corr_cube[k] > 0)

        # Trapezoidal integration with partial trapezoids at crossing points
        area_map = np.zeros(shape_hw, dtype=float)
        for k in range(1, n_bands):
            dwl = float(full_bands[k] - full_bands[k - 1])
            c_prev = corr_cube[k - 1]
            c_curr = corr_cube[k]

            both_in = mask[k - 1] & mask[k]
            exiting = mask[k - 1] & ~mask[k]
            entering = ~mask[k - 1] & mask[k]

            # Full trapezoid where both bands are inside the peak
            area_map += np.where(both_in, 0.5 * dwl * (c_prev + c_curr), 0.0)

            # Partial triangle at right exit (spectrum crosses below baseline)
            diff_exit = c_prev - c_curr
            safe_exit = np.where(np.abs(diff_exit) > 1e-12, diff_exit, 1.0)
            area_map += np.where(exiting & (c_prev > 0),
                                 0.5 * dwl * c_prev * c_prev / safe_exit, 0.0)

            # Partial triangle at left entry (spectrum crosses above baseline)
            diff_enter = c_curr - c_prev
            safe_enter = np.where(np.abs(diff_enter) > 1e-12, diff_enter, 1.0)
            area_map += np.where(entering & (c_curr > 0),
                                 0.5 * dwl * c_curr * c_curr / safe_enter, 0.0)

        return area_map

    def _line_or_doublet_map(self, target, apply_baseline, halfwidth_nm, gap_nm, method="Peak height"):
        bands = self._get_bands_values()
        def _snap(w):
            if bands is not None:
                return float(bands[int(np.abs(bands - float(w)).argmin())])
            return float(w)

        use_area = apply_baseline and method == "Peak area"

        if isinstance(target, (list, tuple)):
            parts = []
            for wl in target:
                snapped = _snap(wl)
                if use_area:
                    parts.append(self._local_peak_area_map(snapped, halfwidth_nm, gap_nm))
                else:
                    img = self._get_band_image(snapped)
                    if apply_baseline:
                        base = self._local_baseline_map(snapped, halfwidth_nm, gap_nm)
                        img = img - base
                    parts.append(img)
            return np.nansum(parts, axis=0)
        else:
            snapped = _snap(target)
            if use_area:
                return self._local_peak_area_map(snapped, halfwidth_nm, gap_nm)
            img = self._get_band_image(snapped)
            if apply_baseline:
                base = self._local_baseline_map(snapped, halfwidth_nm, gap_nm)
                img = img - base
            return img


    # ====== Close ======
    def closeEvent(self, event):
        self._save_config()
        try:
            if self.original_ds is not None: self.original_ds.close()
        except Exception: pass
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            font-size: 9pt;
            border: 1px solid #b0b0b0;
            border-radius: 4px;
            margin-top: 10px;
            padding: 8px 6px 6px 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 8px;
            padding: 0 4px;
            background-color: palette(window);
        }
        QPushButton {
            padding: 3px 10px;
            min-height: 20px;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            min-height: 20px;
        }
        QTabWidget::pane {
            border: 1px solid #c0c0c0;
        }
        QTabBar::tab {
            padding: 5px 12px;
        }
        QLabel#sectionLabel {
            font-weight: bold;
            color: #2c3e50;
            font-size: 9pt;
        }
        QSplitter::handle {
            background-color: #d0d0d0;
        }
        QSplitter::handle:horizontal {
            width: 4px;
        }
        QSplitter::handle:vertical {
            height: 4px;
        }
    """)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(BUILD_VERSION)

    # Splash screen (2 seconds) -- cave / LIBS / spectroscopy theme
    splash_w, splash_h = 640, 360
    pixmap = QPixmap(splash_w, splash_h)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # -- Background: dark cave gradient (deep navy -> near-black) --
    bg = QLinearGradient(0, 0, 0, splash_h)
    bg.setColorAt(0.0, QColor("#0a0e1a"))
    bg.setColorAt(0.4, QColor("#111827"))
    bg.setColorAt(1.0, QColor("#1a1206"))
    painter.fillRect(0, 0, splash_w, splash_h, bg)

    # -- Stalactites hanging from the top --
    import random as _rnd
    _rnd.seed(42)  # deterministic
    stalactite_color = QLinearGradient(0, 0, 0, 120)
    stalactite_color.setColorAt(0.0, QColor("#8a7560"))
    stalactite_color.setColorAt(0.6, QColor("#5c4a38"))
    stalactite_color.setColorAt(1.0, QColor("#2e2318"))
    painter.setPen(Qt.NoPen)
    painter.setBrush(stalactite_color)
    for cx in range(20, splash_w, 35):
        w = _rnd.randint(8, 22)
        h = _rnd.randint(25, 95)
        tri = QPolygonF([
            QPointF(cx - w / 2, 0),
            QPointF(cx + w / 2, 0),
            QPointF(cx + _rnd.randint(-3, 3), h),
        ])
        painter.drawPolygon(tri)

    # -- Laser beam (bright line cutting across the cave) --
    laser_y = 195
    # Glow
    for i, alpha in enumerate([15, 25, 40, 60]):
        pen = QPen(QColor(0, 200, 255, alpha))
        pen.setWidthF(12.0 - i * 2.5)
        painter.setPen(pen)
        painter.drawLine(0, laser_y, splash_w, laser_y)
    # Core
    pen = QPen(QColor(100, 240, 255, 220))
    pen.setWidthF(1.5)
    painter.setPen(pen)
    painter.drawLine(0, laser_y, splash_w, laser_y)

    # -- Plasma spark at impact point --
    spark_x = splash_w // 2
    for r, alpha in [(18, 30), (12, 50), (7, 90), (3, 160)]:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(180, 240, 255, alpha))
        painter.drawEllipse(QPointF(spark_x, laser_y), r, r)
    painter.setBrush(QColor(255, 255, 255, 200))
    painter.drawEllipse(QPointF(spark_x, laser_y), 2, 2)

    # -- Emission spectrum bar (rainbow strip below the laser) --
    spec_y, spec_h = laser_y + 20, 6
    spectrum_colors = [
        "#8b00ff", "#6600ff", "#0044ff", "#0088ff", "#00ccaa",
        "#00dd44", "#88dd00", "#cccc00", "#ff8800", "#ff3300", "#cc0000"
    ]
    seg_w = splash_w / len(spectrum_colors)
    for i, col in enumerate(spectrum_colors):
        painter.fillRect(int(i * seg_w), spec_y, int(seg_w) + 1, spec_h, QColor(col))
    # Fade edges
    fade_left = QLinearGradient(0, 0, 60, 0)
    fade_left.setColorAt(0, QColor(10, 14, 26, 255))
    fade_left.setColorAt(1, QColor(10, 14, 26, 0))
    painter.fillRect(0, spec_y, 60, spec_h, fade_left)
    fade_right = QLinearGradient(splash_w - 60, 0, splash_w, 0)
    fade_right.setColorAt(0, QColor(26, 18, 6, 0))
    fade_right.setColorAt(1, QColor(26, 18, 6, 255))
    painter.fillRect(splash_w - 60, spec_y, 60, spec_h, fade_right)

    # -- Title text --
    painter.setPen(QColor("#e0e7ef"))
    painter.setFont(QFont("Segoe UI", 26, QFont.Bold))
    painter.drawText(0, 108, splash_w, 50, Qt.AlignHCenter, APP_NAME)

    # -- Build version as subtitle --
    painter.setPen(QColor("#7eb8da"))
    painter.setFont(QFont("Segoe UI", 10))
    painter.drawText(0, 162, splash_w, 24, Qt.AlignHCenter, f"{BUILD_VERSION}")

    # -- Institution logos at the bottom --
    _assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    _logo_margin = 14
    _logo_gap = 10
    _logo_heights = {
        "N-be-logo_complet_png_transparent.png": 105,
        "newlogo_gsb.png": 80,
    }
    painter.setOpacity(0.92)
    _cursor_x = _logo_margin
    for _fname in ["N-be-logo_complet_png_transparent.png", "newlogo_gsb.png"]:
        _pm = QPixmap(os.path.join(_assets_dir, _fname))
        if not _pm.isNull():
            _h = _logo_heights[_fname]
            _pm = _pm.scaledToHeight(_h, Qt.SmoothTransformation)
            if _fname.startswith("N-be"):
                _img = _pm.toImage()
                _img.invertPixels(QImage.InvertRgb)
                _pm = QPixmap.fromImage(_img)
            _y = splash_h - _h - _logo_margin
            painter.drawPixmap(_cursor_x, _y, _pm)
            _cursor_x += _pm.width() + _logo_gap
    painter.setOpacity(1.0)

    painter.end()

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()

    ex = HypercubeExplorer()
    ex.setWindowTitle(APP_NAME)

    def show_main():
        ex.show()
        splash.finish(ex)

    QTimer.singleShot(3000, show_main)
    sys.exit(app.exec_())
