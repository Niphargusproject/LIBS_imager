# -*- coding: utf-8 -*-
"""
Data processing utilities for LIBS application
"""
import logging
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class SpectrumProcessor:
    """Handles spectrum data processing and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger('LIBS_App.SpectrumProcessor')
    
    def dark_correction(self, spectrum_data: pd.DataFrame, 
                       dark_data: pd.DataFrame) -> pd.DataFrame:
        """Apply dark correction to spectrum data"""
        try:
            if dark_data.empty:
                self.logger.warning("No dark data provided, returning original spectrum")
                return spectrum_data
            
            # Ensure both DataFrames have the same wavelength range
            corrected_data = spectrum_data.copy()
            
            # Simple dark subtraction (can be improved with interpolation)
            corrected_data['intensity_dark_corrected'] = (
                spectrum_data['intensity_dark_corrected'] - 
                dark_data['dark']
            )
            
            self.logger.info("Dark correction applied")
            return corrected_data
            
        except Exception as e:
            self.logger.error(f"Failed to apply dark correction: {e}")
            return spectrum_data
    
    def save_spectrum(self, spectrum_data: pd.DataFrame, filename: str, 
                     folder_path: str = "") -> str:
        """Save spectrum data to CSV file"""
        try:
            full_path = folder_path + filename
            spectrum_data.to_csv(full_path, index=False)
            self.logger.info(f"Spectrum saved to: {full_path}")
            return full_path
            
        except Exception as e:
            self.logger.error(f"Failed to save spectrum: {e}")
            return ""


class PlotGenerator:
    """Handles plot generation and visualization"""
    
    def __init__(self, dark_mode: bool = True):
        self.dark_mode = dark_mode
        self.logger = logging.getLogger('LIBS_App.PlotGenerator')
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Setup matplotlib for dark/light mode"""
        if self.dark_mode:
            plt.rcParams.update({
                "lines.color": "white",
                "patch.edgecolor": "white",
                "text.color": "grey",
                "axes.facecolor": "white",
                "axes.edgecolor": "grey",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "grid.color": "grey",
                "figure.facecolor": "grey",
                "figure.edgecolor": "grey",
                "savefig.facecolor": "grey",
                "savefig.edgecolor": "grey"
            })
    
    def create_profile_plot(self, profile_data: pd.DataFrame, 
                          profile_steps: int) -> Figure:
        """Create profile plot for multiple shots"""
        try:
            fig = Figure()
            fig.set_tight_layout(True)
            ax = fig.add_subplot()
            
            # Plot each shot
            for n in range(profile_steps):
                if str(n+1) in profile_data.columns:
                    ax.plot(profile_data["wavelength_(nm)"], 
                           profile_data[str(n+1)], 
                           label=f'Shot {n+1}', 
                           linewidth=1)
            
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Intensity (counts)', fontsize=12)
            ax.set_title('LIBS Profile', fontsize=14)
            ax.legend()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create profile plot: {e}")
            return Figure()
    
class PeakIdentifier:
    """Identify peaks against a small reference line list for common LIBS elements."""
    
    def __init__(self):
        self.logger = logging.getLogger('LIBS_App.PeakIdentifier')
        self.ref_df = self._build_reference_lines()
    
    def _build_reference_lines(self) -> pd.DataFrame:
        """Build a practical reference line list (nm) for glass/rock-forming elements."""
        ref_lines = []
        
        def add_lines(element: str, ion: str, wavelengths_nm):
            for wl in wavelengths_nm:
                ref_lines.append({"element": element, "ion": ion, "wavelength_nm": float(wl)})
        
        # Network formers and common major elements
        add_lines("Si", "I", [251.43, 251.61, 252.41, 263.13, 288.16, 390.55, 410.29])
        add_lines("Al", "I", [257.51, 308.22, 309.27, 394.40, 396.15])
        add_lines("O", "I", [777.19])
        add_lines("C", "I", [193.09, 247.86])
        add_lines("H", "I", [486.13, 656.28])
        
        # Alkalis
        add_lines("Na", "I", [330.23, 330.30, 330.70, 568.26, 588.99, 589.59])
        add_lines("K", "I", [404.41, 404.72, 766.49, 769.90])
        add_lines("Li", "I", [610.36, 670.78])
        add_lines("Rb", "I", [780.03, 794.76])
        add_lines("Cs", "I", [852.11, 894.35])
        
        # Alkaline earths
        add_lines("Ca", "II", [315.89, 317.93, 373.69, 393.37, 396.85])
        add_lines("Ca", "I", [239.86, 422.67, 428.30, 430.25, 443.50, 445.48])
        add_lines("Mg", "I", [279.55, 280.27, 285.21, 383.83])
        add_lines("Mg", "II", [279.80, 280.35])
        
        add_lines("Ba", "II", [455.40, 493.41])
        add_lines("Sr", "II", [407.77, 421.55])
        
        # Transition metals
        add_lines("Fe", "I", [
            248.33, 252.28, 259.94, 271.44, 274.65,
            292.63, 302.06, 371.99, 373.71, 382.04,
            404.58, 438.35
        ])
        add_lines("Mn", "I", [257.61, 259.37, 260.57, 293.31, 294.92, 403.08])
        add_lines("Ti", "I", [334.94, 337.28, 368.52])
        add_lines("Cr", "I", [357.87, 359.35, 425.43])
        add_lines("Ni", "I", [341.48, 352.45, 361.94])
        add_lines("V", "I", [437.92, 438.91])
        add_lines("Cu", "I", [324.75, 327.40])
        add_lines("Co", "I", [340.51, 345.35, 350.23, 384.52])
        add_lines("Zn", "I", [202.55, 206.20, 213.86, 330.26, 334.50, 468.01])
        
        # Other heavy elements (technical glasses)
        add_lines("Pb", "I", [283.31, 368.35, 405.78])
        add_lines("Zr", "I", [339.20, 343.82])
        
        return pd.DataFrame(ref_lines)
    
    def detect_peaks(self, spectrum_data: pd.DataFrame,
                     min_height: float = 50.0,
                     min_prominence: float = 50.0) -> pd.DataFrame:
        """
        Detect peaks using scipy.signal.find_peaks.
        Returns subset of input DataFrame rows at peak indices with an added 'prominence' column.
        """
        try:
            intensities = spectrum_data['intensity_dark_corrected'].values
            indexes, props = signal.find_peaks(intensities,
                                               height=min_height,
                                               prominence=min_prominence)
            if len(indexes) == 0:
                return pd.DataFrame(columns=list(spectrum_data.columns) + ['prominence'])
            peaks_df = spectrum_data.iloc[indexes].copy()
            try:
                peaks_df['prominence'] = props.get('prominences', [])
            except Exception:
                pass
            return peaks_df.reset_index(drop=True)
        except Exception as e:
            self.logger.error(f"Failed to detect peaks: {e}")
            return pd.DataFrame(columns=list(spectrum_data.columns) + ['prominence'])
    
    def match_peaks_to_lines(self, peaks_df: pd.DataFrame, tol_nm: float = 0.05) -> pd.DataFrame:
        """
        Match detected peaks to reference lines within tolerance.
        Expects peaks_df to have 'wavelength_(nm)' and 'intensity_dark_corrected'.
        Returns a DataFrame with element, ion, ref_wavelength_nm, peak_wavelength_nm, intensity.
        """
        try:
            if peaks_df is None or peaks_df.empty:
                return pd.DataFrame(columns=[
                    "element", "ion", "ref_wavelength_nm", "peak_wavelength_nm", "intensity"
                ])
            matches = []
            for _, line in self.ref_df.iterrows():
                wl_ref = float(line['wavelength_nm'])
                mask = (
                    (peaks_df['wavelength_(nm)'] >= wl_ref - tol_nm) &
                    (peaks_df['wavelength_(nm)'] <= wl_ref + tol_nm)
                )
                if not mask.any():
                    continue
                for _, peak in peaks_df[mask].iterrows():
                    matches.append({
                        "element": line["element"],
                        "ion": line["ion"],
                        "ref_wavelength_nm": wl_ref,
                        "peak_wavelength_nm": float(peak['wavelength_(nm)']),
                        "intensity": float(peak['intensity_dark_corrected']),
                    })
            if not matches:
                return pd.DataFrame(columns=[
                    "element", "ion", "ref_wavelength_nm", "peak_wavelength_nm", "intensity"
                ])
            return pd.DataFrame(matches).sort_values("intensity", ascending=False).reset_index(drop=True)
        except Exception as e:
            self.logger.error(f"Failed to match peaks: {e}")
            return pd.DataFrame(columns=[
                "element", "ion", "ref_wavelength_nm", "peak_wavelength_nm", "intensity"
            ])
    
    def summarize_elements(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize matched peaks per element."""
        try:
            if matches_df is None or matches_df.empty:
                return pd.DataFrame(columns=["element", "n_lines", "n_matches", "total_intensity", "max_intensity"])
            return (
                matches_df.groupby("element")
                .agg(
                    n_lines=("ref_wavelength_nm", "nunique"),
                    n_matches=("ref_wavelength_nm", "size"),
                    total_intensity=("intensity", "sum"),
                    max_intensity=("intensity", "max"),
                )
                .sort_values("total_intensity", ascending=False)
                .reset_index()
            )
        except Exception as e:
            self.logger.error(f"Failed to summarize element matches: {e}")
            return pd.DataFrame(columns=["element", "n_lines", "n_matches", "total_intensity", "max_intensity"])
