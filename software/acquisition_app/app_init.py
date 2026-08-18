# -*- coding: utf-8 -*-
"""
Application initialization: path setup, file verification, and logging.
"""
import sys
import os
import logging
from pathlib import Path
from datetime import datetime


# ===================================================================
# Path setup
# ===================================================================
def setup_application_paths():
    """Set up Python path and DLL directory. Returns the app directory."""
    app_dir = Path(__file__).parent.absolute()
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    try:
        os.add_dll_directory(str(app_dir))
    except AttributeError:
        pass  # os.add_dll_directory is Windows Python 3.8+ only
    os.chdir(app_dir)
    return app_dir


def setup_library_paths():
    """Add common library sub-directories to sys.path and DLL search."""
    app_dir = Path(__file__).parent.absolute()
    for sub in ("", "libs", "dlls", "bin"):
        lib_dir = app_dir / sub if sub else app_dir
        if lib_dir.exists() and str(lib_dir) not in sys.path:
            sys.path.insert(0, str(lib_dir))
            try:
                os.add_dll_directory(str(lib_dir))
            except AttributeError:
                pass
    return app_dir


def verify_required_files():
    """Return True if required application files are present."""
    app_dir = Path(__file__).parent.absolute()
    required_files = ["images/splash.png", "images/icon_aconvert.ico"]
    required_dlls = ["avaspecx64.dll"]
    missing = [f for f in required_files if not (app_dir / f).exists()]
    for dll in required_dlls:
        if not (app_dir / dll).exists():
            print(f"Warning: {dll} not found - some features may not work")
    if missing:
        print("Error: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return False
    return True


def setup_environment():
    """Complete environment setup (paths + file verification)."""
    print("Setting up LIBS Application environment...")
    app_dir = setup_application_paths()
    setup_library_paths()
    if not verify_required_files():
        print("Some required files are missing. The application may not work correctly.")
        return False
    print(f"Application directory: {app_dir}")
    print("Environment setup complete.")
    return True


# ===================================================================
# Logging
# ===================================================================
def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Configure file + console logging. Returns the root app logger."""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"libs_app_{timestamp}.log")

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("LIBS_App")
    logger.info("LIBS Application logging initialized")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Log level: {log_level}")
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a child logger under the LIBS_App namespace."""
    if name:
        return logging.getLogger(f"LIBS_App.{name}")
    return logging.getLogger("LIBS_App")
