"""
X-ray Reflectivity (XRR) processing library.

This library is used for the processing of Surface X-ray Scattering data
obtained on the ID10 beamline at ESRF.
"""

import copy
import logging
import os
import time
from math import sin, cos, pi
from typing import Optional, Tuple, Union, List, Dict, Any

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def rebin(q_vectors: np.ndarray,
          Refl: np.ndarray,
          Relf_E: np.ndarray,
          new_q: Optional[np.ndarray] = None,
          rebin_as: str = "linear",
          number_of_bins: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebin the data on a linear or logarithmic q-scale.

    This rebinning procedure is taken from islatu by Andrew R. McCluskey:
    https://github.com/DiamondLightSource/islatu

    Args:
        q_vectors (np.ndarray): The current q vectors.
        Refl (np.ndarray): The current reflected intensities.
        Relf_E (np.ndarray): The current reflected intensity errors.
        new_q (Optional[np.ndarray]): Array of potential q-values. Defaults to None.
            If this argument is not specified, then the new q, R values are binned
            according to rebin_as and number_of_bins.
        rebin_as (str): String specifying how the data should be rebinned.
            Options are "linear" and "log". This is only used if new_q is unspecified.
            Defaults to "linear".
        number_of_bins (int, optional): The max number of q-vectors to be using
            initially in the rebinning of the data. Defaults to 5000.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Containing:
            - cleaned_q: rebinned q-values.
            - cleaned_R: rebinned intensities.
            - cleaned_R_e: rebinned intensity errors.
    """
    # Unpack the arguments.
    q = q_vectors
    R = Refl
    R_e = Relf_E

    # Required so that logspace/linspace encapsulates the whole data.
    epsilon = 0.001

    if new_q is None:
        # Our new q vectors have not been specified, so we should generate some.
        if rebin_as == "log":
            new_q = np.logspace(
                np.log10(q[0]),
                np.log10(q[-1] + epsilon), number_of_bins)
        elif rebin_as == "linear":
            new_q = np.linspace(q.min(), q.max() + epsilon,
                                number_of_bins)

    binned_q = np.zeros_like(new_q)
    binned_R = np.zeros_like(new_q)
    binned_R_e = np.zeros_like(new_q)

    for i in range(len(new_q) - 1):
        indices = []
        inverse_var = []
        for j in range(len(q)):
            if new_q[i] <= q[j] < new_q[i + 1]:
                indices.append(j)
                inverse_var.append(1 / float(R_e[j] ** 2))

        # Don't bother doing maths if there were no recorded q-values between
        # the two bin points we were looking at.
        if len(indices) == 0:
            continue

        # We will be using inverse-variance weighting to minimize the variance
        # of the weighted mean.
        sum_of_inverse_var = np.sum(inverse_var)

        # If we measured multiple qs between these bin locations, then average
        # the data, weighting by inverse variance.
        for j in indices:
            binned_R[i] += R[j] / (R_e[j] ** 2)
            binned_q[i] += q[j] / (R_e[j] ** 2)

        # Divide by the sum of the weights.
        binned_R[i] /= sum_of_inverse_var
        binned_q[i] /= sum_of_inverse_var

        # The stddev of an inverse variance weighted mean is always:
        binned_R_e[i] = np.sqrt(1 / sum_of_inverse_var)

    # Get rid of any empty, unused elements of the array.
    cleaned_q = np.delete(binned_q, np.argwhere(binned_R == 0))
    cleaned_R = np.delete(binned_R, np.argwhere(binned_R == 0))
    cleaned_R_e = np.delete(binned_R_e, np.argwhere(binned_R == 0))

    return cleaned_q, cleaned_R, cleaned_R_e


class XRR:
    """
    Class for the processing of Surface X-ray Scattering data.

    This class handles loading h5 files, region-of-interest based integration
    of 2D detector data, corrections (footprint, transmission), and plotting.

    Attributes:
        file (str): Path to the HDF5 file containing the data.
        scans (np.ndarray): Array of scan numbers to process.
        alpha_i_name (str): Name of the incident angle motor/counter.
        detector_name (str): Name of the detector data entry.
        monitor_name (str): Name of the monitor counter.
        transmission_name (str): Name of the transmission counter.
        att_name (str): Name of the attenuator positioner.
        energy_name (str): Name of the energy positioner.
        cnttime_name (str): Name of the counting time counter.
        PX0 (int): Direct beam X pixel coordinate on detector.
        PY0 (int): Direct beam Y pixel coordinate on detector.
        dPX (int): Half-width of the ROI in X direction.
        dPY (int): Half-width of the ROI in Y direction.
        pixel_size_qxz (float): Pixel size in reciprocal space (x/z).
        pixel_size_qy (float): Pixel size in reciprocal space (y).
        I0 (float): Incident intensity normalization factor.
        footprint_correction_applied (bool): Flag if footprint correction was applied.
        corrected_doubles (bool): Flag if double points were corrected.
        replaced_transmission (bool): Flag if transmission was manually replaced.
        is_rebinned (bool): Flag if data has been rebinned.
        qz (Optional[np.ndarray]): Calculated qz vector.
        reflectivity (Optional[np.ndarray]): Calculated reflectivity.
        reflectivity_error (Optional[np.ndarray]): Calculated reflectivity error.
        Smap2D (List): List of 2D maps.
        Smap2D_e (List): List of 2D map errors.
        Qx_map (Optional[np.ndarray]): Qx map for 2D plotting.
        Qz_map (Optional[np.ndarray]): Qz map for 2D plotting.
        sample_name (str): Name of the sample.
    """

    def __init__(self,
                 file: str,
                 scans: Union[List[int], np.ndarray],
                 alpha_i_name: str = 'chi',
                 detector_name: str = 'mpx_cdte_22_eh1',
                 monitor_name: str = 'mon',
                 transmission_name: str = 'transm',
                 att_name: str = 'curratt',
                 cnttime_name: str = 'sec',
                 PX0: int = 404,
                 PY0: int = 165,
                 dPX: int = 5,
                 dPY: int = 5,
                 pixel_size_qxz: float = 0.055,
                 pixel_size_qy: float = 0.055,
                 energy_name: str = 'eccmono',
                 I0: float = 1e13):
        """
        Initialize the XRR processing class.

        Args:
            file (str): Path to the HDF5 file.
            scans (Union[List[int], np.ndarray]): List of scan numbers.
            alpha_i_name (str, optional): Motor name for incident angle. Defaults to 'chi'.
            detector_name (str, optional): Detector dataset name. Defaults to 'mpx_cdte_22_eh1'.
            monitor_name (str, optional): Monitor counter name. Defaults to 'mon'.
            transmission_name (str, optional): Transmission counter name. Defaults to 'transm'.
            att_name (str, optional): Attenuator name. Defaults to 'curratt'.
            cnttime_name (str, optional): Count time name. Defaults to 'sec'.
            PX0 (int, optional): Direct beam X pixel. Defaults to 404.
            PY0 (int, optional): Direct beam Y pixel. Defaults to 165.
            dPX (int, optional): ROI half-width X. Defaults to 5.
            dPY (int, optional): ROI half-width Y. Defaults to 5.
            pixel_size_qxz (float, optional): Pixel size factor. Defaults to 0.055.
            pixel_size_qy (float, optional): Pixel size factor. Defaults to 0.055.
            energy_name (str, optional): Energy motor name. Defaults to 'eccmono'.
            I0 (float, optional): Intensity normalization. Defaults to 1e13.
        """
        self.file = file
        self.scans = np.array(scans)
        self.alpha_i_name = alpha_i_name
        self.detector_name = detector_name
        self.monitor_name = monitor_name
        self.transmission_name = transmission_name
        self.att_name = att_name
        self.energy_name = energy_name
        self.cnttime_name = cnttime_name

        self.footprint_correction_applied = False
        self.corrected_doubles = False
        self.replaced_transmission = False
        self.is_rebinned = False

        self.PX0 = PX0
        self.PY0 = PY0
        self.dPX = dPX
        self.dPY = dPY

        self.I0 = I0

        self.pixel_size_qy = pixel_size_qy
        self.pixel_size_qxz = pixel_size_qxz

        self.qz = None
        self.reflectivity = None
        self.reflectivity_error = None
        self.Qx_map = None
        self.Qz_map = None
        self.Smap2D = []
        self.Smap2D_e = []
        self.sample_name = ""

        # Initialize data attributes to avoid attribute error if accessed before loading
        self.data = None
        self.alpha_i = None
        self.monitor = None
        self.transmission = None
        self.attenuator = None
        self.cnttime = None
        self.energy = None
        self.bckg = None
        self.raw_counts = None

        self.__load_data__()
        self.__process_2D_data__()

    def __load_single_scan__(self, scan_n: str) -> Dict[str, Any]:
        """
        Load data for a single scan.

        Args:
            scan_n (str): Scan number as a string.

        Returns:
            Dict[str, Any]: Dictionary containing loaded data components.
        """
        logger.info('Loading scan #%s', scan_n)
        with h5py.File(self.file, "r") as f:
            base_path = f"{scan_n}.1"
            meas_path = f"{base_path}/measurement/"

            data_dict = {
                'data': np.array(f.get(f"{meas_path}{self.detector_name}")),
                'alpha_i': np.array(f.get(f"{meas_path}{self.alpha_i_name}")),
                'monitor': np.array(f.get(f"{meas_path}{self.monitor_name}")),
                'transmission': np.array(f.get(f"{meas_path}{self.transmission_name}")),
                'attenuator': np.array(f.get(f"{meas_path}{self.att_name}")),
                'cnttime': np.array(f.get(f"{meas_path}{self.cnttime_name}")),
                'energy': np.array(f.get(f"{base_path}/instrument/positioners/{self.energy_name}")),
                'sample_name': str(f.get(f"{base_path}/sample/name/")[()])[2:-1:1]
            }

        logger.info('Loaded scan #%s', scan_n)
        return data_dict

    def __load_data__(self, skip_points: int = 1):
        """
        Load data from all scans and concatenate them.

        Args:
            skip_points (int, optional): Number of initial points to skip for subsequent scans.
                Defaults to 1.
        """
        t0 = time.time()
        logger.info("Start loading data.")

        first_scan_n = str(self.scans[0])
        first_scan_data = self.__load_single_scan__(first_scan_n)

        self.data = first_scan_data['data']
        self.alpha_i = first_scan_data['alpha_i']
        self.monitor = first_scan_data['monitor']
        self.transmission = first_scan_data['transmission']
        self.attenuator = first_scan_data['attenuator']
        self.cnttime = first_scan_data['cnttime']
        self.energy = first_scan_data['energy']
        self.sample_name = first_scan_data['sample_name']

        if len(self.scans) > 1:
            for scan_num in self.scans[1:]:
                scan_n = str(scan_num)
                scan_data = self.__load_single_scan__(scan_n)

                # Append with skipping points
                self.data = np.append(self.data, scan_data['data'][skip_points:], axis=0)
                self.alpha_i = np.append(self.alpha_i, scan_data['alpha_i'][skip_points:])
                self.monitor = np.append(self.monitor, scan_data['monitor'][skip_points:])
                self.transmission = np.append(self.transmission, scan_data['transmission'][skip_points:])
                self.attenuator = np.append(self.attenuator, scan_data['attenuator'][skip_points:])
                self.cnttime = np.append(self.cnttime, scan_data['cnttime'][skip_points:])
                self.energy = np.append(self.energy, scan_data['energy'])

        logger.info("Loading completed. Reading time %3.3f sec", time.time() - t0)

    def __process_2D_data__(self):
        """
        Process the 2D detector data to calculate reflectivity.

        Performs ROI integration for signal and background, calculates qz,
        and normalizes by monitor and transmission.
        """
        t0 = time.time()
        logger.info('Starting 2D data processing.')
        nic, nxc, nyc = np.shape(self.data)

        # Handle legacy data where energy might be an array
        try:
            if np.size(self.energy) > 1:
                self.energy = np.mean(self.energy)
        except Exception:
            pass

        Qzcut = np.ones(nxc)
        Qzcut_bckg1 = np.ones(nxc)
        Qzcut_bckg2 = np.ones(nxc)
        self.Smap2D = []
        self.Smap2D_e = []

        Is_cut = np.zeros(nic)  # array for signal
        Ib_cut = np.zeros(nic)  # array for background
        Is_cut_err = np.zeros(nic)
        Ib_cut_err = np.zeros(nic)
        I_err = np.zeros(nic)

        # Pre-calculate slice indices for performance and readability
        roi_y_slice = slice(self.PY0 - self.dPY, self.PY0 + self.dPY)
        roi_x_slice = slice(self.PX0 - self.dPX, self.PX0 + self.dPX)

        bkg_low_y_slice = slice(self.PY0 + 2 * self.dPY + 1 - self.dPY, self.PY0 + 2 * self.dPY + 1 + self.dPY)
        bkg_high_y_slice = slice(self.PY0 - 2 * self.dPY - 1 - self.dPY, self.PY0 - 2 * self.dPY - 1 + self.dPY)

        for i in range(nic):
            # Lower square background
            IqxyBL = np.sum(self.data[i, bkg_low_y_slice, roi_x_slice])
            # Higher square background
            IqxyBH = np.sum(self.data[i, bkg_high_y_slice, roi_x_slice])
            # Signal
            IqxyS = np.sum(self.data[i, roi_y_slice, roi_x_slice])

            if i < len(self.alpha_i):
                # Qz cuts (sum along X axis for the ROI Y range)
                Qzcut[:] = np.sum(self.data[i, roi_y_slice, :], axis=0)
                Qzcut_bckg1[:] = np.sum(self.data[i, bkg_low_y_slice, :], axis=0)
                Qzcut_bckg2[:] = np.sum(self.data[i, bkg_high_y_slice, :], axis=0)

                norm_factor = self.transmission[i] * self.monitor[i] / self.monitor[0]

                self.Smap2D.append(
                    (Qzcut[:] - (Qzcut_bckg1[:] + Qzcut_bckg2) / 2) / norm_factor
                )
                self.Smap2D_e.append(
                    ((np.sqrt(np.abs(Qzcut[:])) + np.sqrt((Qzcut_bckg1[:] + Qzcut_bckg2[:]) / 2)) / norm_factor)
                )

            Ib_cut[i] = (IqxyBL + IqxyBH) / 2  # subtract true background
            Is_cut[i] = IqxyS
            Is_cut_err[i] = np.sqrt(Is_cut[i])
            Ib_cut_err[i] = np.sqrt(Ib_cut[i])

            try:
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    I_err[i] = np.sqrt((Is_cut_err[i] / Is_cut[i]) ** 2 + (Ib_cut_err[i] / Ib_cut[i]) ** 2)
                    if not np.isfinite(I_err[i]):
                        I_err[i] = Is_cut_err[i] / Is_cut[i] if Is_cut[i] > 0 else 0
            except Exception:
                 I_err[i] = Is_cut_err[i] / Is_cut[i] if Is_cut[i] > 0 else 0

        logger.info('Number of points in the scan %6d', len(self.alpha_i))

        limit_len = len(self.alpha_i)
        I_Signal_cut = np.nan_to_num(Is_cut[:limit_len], nan=1e-11)
        I_Backgr_cut = np.nan_to_num(Ib_cut[:limit_len], nan=1e-11)
        I_error = np.nan_to_num(I_err[:limit_len], nan=1e-11)

        self.qz = 4 * pi * np.sin(np.deg2rad(self.alpha_i)) / (12.398 / self.energy)

        # Reflectivity calculation
        norm_factor = self.transmission * self.monitor / self.monitor[0]
        self.reflectivity = (I_Signal_cut - I_Backgr_cut) / norm_factor
        self.reflectivity_error = np.abs(I_error * self.reflectivity)

        self.bckg = I_Backgr_cut / norm_factor / self.I0

        self.raw_counts = I_Signal_cut / self.monitor * self.monitor[0]
        self.reflectivity = self.reflectivity / self.I0
        self.reflectivity_error = self.reflectivity_error / self.I0

        # Clip values
        self.reflectivity[self.reflectivity <= 1e-12] = 1e-12
        self.reflectivity_error[self.reflectivity_error <= 1e-13] = 1e-13

        logger.info("Processing completed. Processing time %3.3f sec", time.time() - t0)

    def footprint_correction(self, sample_size: float = 1.0, beamsize: float = 9.6, correct_dir_beam: bool = False):
        """
        Apply footprint correction to the reflectivity data.

        Args:
            sample_size (float, optional): Sample size in cm. Defaults to 1.
            beamsize (float, optional): Beam size in microns. Defaults to 9.6.
            correct_dir_beam (bool, optional): Whether to correct direct beam region.
                Defaults to False.
        """
        if not self.footprint_correction_applied:
            Si_critical_angle = 4 * np.pi * np.sin(np.deg2rad(8.103E-02)) / (12.398 / self.energy)
            samplesize_microns = sample_size * 10000  # converting cm to microns

            # Avoid division by zero by using a small epsilon or handling alpha_i == 0
            footprint = np.array([
                0.5 * beamsize / np.sin(np.deg2rad(alpha)) if alpha != 0 else 0.5 * beamsize / np.sin(np.deg2rad(1e-3))
                for alpha in self.alpha_i
            ])

            beam_fraction = np.array([
                (stats.norm.cdf(samplesize_microns / 2, 0, ftp) - stats.norm.cdf(-samplesize_microns / 2, 0, ftp))
                for ftp in footprint
            ])

            Icor = self.reflectivity / beam_fraction

            if correct_dir_beam:
                i = 0
                while i < len(self.qz) and self.qz[i] < Si_critical_angle:
                    if Icor[i] > 1.15:
                        Icor[i] = 1
                    i += 1

            self.reflectivity = Icor
            self.footprint_correction_applied = True
            logger.info('Footprint correction completed with beam size = %s microns and sample size = %s cm',
                        beamsize, sample_size)
        else:
            logger.warning('Footprint correction already applied! To apply again use reprocess() method.')

    def reprocess(self):
        """
        Reload and reprocess the data, resetting corrections.
        """
        self.__load_data__()
        self.__process_2D_data__()
        self.footprint_correction_applied = False
        self.corrected_doubles = False
        self.replaced_transmission = False
        self.is_rebinned = False
        logger.info("Reloaded and reprocessed data.")

    def produce_Qmap(self, SDD: float = 900):
        """
        Calculate Q-space map.

        Args:
            SDD (float, optional): Sample-to-Detector Distance in mm. Defaults to 900.
        """
        t0 = time.time()
        logger.info('Starting q-space mapping.')
        chi_r = np.deg2rad(self.alpha_i)
        pixels = np.array(range(516))

        k0 = 2 * pi / (12.398 / self.energy)

        # Using list comprehension inside array creation
        self.Qz_map = np.array(
            [[np.round(k0 * (sin(chi + ((self.PX0 - px) * self.pixel_size_qxz / SDD)) + sin(chi)), 10) for px in pixels]
             for chi in chi_r])
        self.Qx_map = np.array(
            [[np.round(k0 * (cos(chi + ((self.PX0 - px) * self.pixel_size_qxz / SDD)) - cos(chi)), 10) for px in pixels]
             for chi in chi_r])
        logger.info("2D map calculated. Processing time %3.3f sec", time.time() - t0)

    def plot_Qmap(self, save: bool = False, fig: Optional[plt.Figure] = None, axes: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the Q-space map.

        Args:
            save (bool, optional): Whether to save the plot. Defaults to False.
            fig (Optional[plt.Figure], optional): Matplotlib figure object. Defaults to None.
            axes (Optional[plt.Axes], optional): Matplotlib axes object. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
        """
        if axes is None:
            fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), layout='tight')
        else:
            ax0 = axes
            if fig is None:
                fig = ax0.get_figure()

        # Check if Qx_map and Qz_map are computed
        if self.Qx_map is None or self.Qz_map is None:
             self.produce_Qmap()

        ax0.pcolormesh(self.Qx_map, self.Qz_map, np.log10(self.Smap2D), cmap='jet', vmin=4, vmax=10,
                       shading='gouraud', snap=True)

        ax0.set_xlabel(r'$q_x, \AA^{-1}$')
        ax0.set_ylabel(r'$q_z, \AA^{-1}$')
        ax0.set_ylim(0, 0.5)
        ax0.set_xlim(-2e-4, 0.0005)
        ax0.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        if save:
            logger.info('Saving Q-space map.')
            plt.savefig('Qmap_{}_scan_{}.png'.format(self.sample_name, self.scans), dpi=300)

        return fig, ax0

    def get_reflectivity(self) -> np.ndarray:
        """
        Get the reflectivity data sorted by qz.

        Returns:
            np.ndarray: Array containing [qz, reflectivity, reflectivity_error].
        """
        # Ensure indices sort by qz
        sort_indices = self.qz.argsort()
        qz_tr = np.sort(self.qz) # or self.qz[sort_indices]
        R_tr = self.reflectivity[sort_indices]
        Rerr_tr = self.reflectivity_error[sort_indices]
        return np.array([qz_tr, R_tr, Rerr_tr])

    def plot_reflectivity(self, save: bool = False, ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the reflectivity curve.

        Args:
            save (bool, optional): Whether to save the plot. Defaults to False.
            ax (Optional[plt.Axes], optional): Matplotlib axes object. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
        """
        if ax is None:
            fig = plt.figure(figsize=(6, 6), layout='tight')
            ax = plt.gca()
        else:
            fig = ax.get_figure()

        ax.errorbar(*self.get_reflectivity(), capsize=1)
        ax.semilogy()
        ax.set_xlim(left=0)
        ax.set_ylim(top=2)
        ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
        ax.set_xlabel(r'$q_z, \AA^{-1}$')
        ax.set_ylabel(r'$\mathrm{Reflectivity}$')
        if save:
            logger.info('Saving reflectivity plot.')
            plt.savefig('XRR_{}_scan_{}.png'.format(self.sample_name, self.scans), dpi=300)

        return fig, ax

    def plot_reflectivity_qz4(self, save: bool = False, ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot reflectivity multiplied by qz^4 (Porod plot).

        Args:
            save (bool, optional): Whether to save the plot. Defaults to False.
            ax (Optional[plt.Axes], optional): Matplotlib axes object. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
        """
        if ax is None:
            fig = plt.figure(figsize=(6, 6), layout='tight')
            ax = plt.gca()
        else:
            fig = ax.get_figure()

        ax.errorbar(self.qz, self.reflectivity * self.qz ** 4, self.reflectivity_error * self.qz ** 4, capsize=1)
        ax.semilogy()
        ax.set_xlim(left=0)
        ax.set_xlabel(r'$q_z, \AA^{-1}$')
        ax.set_ylabel(r'$\mathrm{Reflectivity}\cdot q_z^4$')
        if save:
            logger.info('Saving reflectivity * qz**4 plot.')
            plt.savefig('XRR_{}_scan_qz4_{}.png'.format(self.sample_name, self.scans), dpi=300)

        return fig, ax

    def save_reflectivity(self, filename: Union[bool, str] = False, directory: str = None):
        """
        Save the reflectivity data to a text file.

        Args:
            filename (Union[bool, str], optional): Filename to save to. Defaults to False (auto-generated).
            directory (str, optional): Directory to save to. Defaults to None (current dir).
        """
        if not directory:
            directory = os.getcwd()
        if not filename:
            filename = self.sample_name + '_xrr_scan_{}.dat'.format(self.scans)

        filepath = os.path.join(directory, filename)
        _to_save = self.get_reflectivity().T
        np.savetxt(filepath, _to_save)
        logger.info('Reflectivity saved to: %s', filepath)

    def show_detector_image(self, frame_number: int = 50, ax: Optional[plt.Axes] = None, plot_cross: bool = True):
        """
        Show a single frame of the detector image with ROI and background regions.

        Args:
            frame_number (int, optional): Frame number to display. Defaults to 50.
            ax (Optional[plt.Axes], optional): Matplotlib axes object. Defaults to None.
            plot_cross (bool, optional): Whether to plot crosshairs at beam center. Defaults to True.
        """
        if ax is None:
            fig = plt.figure(figsize=(6, 6), layout='tight')
            ax = plt.gca()

        ax.imshow(np.log10(self.data[frame_number] + 1e-3))
        ax.set_ylim(self.PY0 - 10 * self.dPY, self.PY0 + 10 * self.dPY)
        ax.set_xlim(self.PX0 - 10 * self.dPY, self.PX0 + 10 * self.dPY)

        signal = patches.Rectangle((self.PX0 - self.dPX, self.PY0 - self.dPY), 2 * self.dPX, 2 * self.dPY, linewidth=1,
                                   edgecolor='r', facecolor='none', label='Signal')
        b1 = patches.Rectangle((self.PX0 - self.dPX, self.PY0 + 2 * self.dPY + 1 - self.dPY), 2 * self.dPX,
                               2 * self.dPY, linewidth=1, edgecolor='cyan', facecolor='none', label='Background')
        b2 = patches.Rectangle((self.PX0 - self.dPX, self.PY0 - 2 * self.dPY - 1 - self.dPY), 2 * self.dPX,
                               2 * self.dPY, linewidth=1, edgecolor='cyan', facecolor='none')
        ax.add_patch(signal)
        ax.add_patch(b1)
        ax.add_patch(b2)

        if plot_cross:
            ax.hlines(self.PY0, self.PX0 - 30, self.PX0 + 30)
            ax.vlines(self.PX0, self.PY0 - 30, self.PY0 + 30)

        ax.set_xlabel('Detector pixel, X')
        ax.set_ylabel('Detector pixel, Y')
        ax.set_title('Detector {}, frame #{}'.format(self.detector_name, frame_number))
        ax.legend()

    def replace_transmission(self, filter_transmission: Dict[Any, float]):
        """
        Replace transmission values using a dictionary of filter transmissions.

        Args:
            filter_transmission (Dict[Any, float]): Mapping of attenuator position to transmission.
        """
        _new_transmission = np.array([])
        for point in self.attenuator:
            _new_transmission = np.append(_new_transmission, [float(filter_transmission.get(point, 1.0))])

        # Calculate scaling factor
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
             scaling_factor = self.transmission / _new_transmission

        new_reflectivity = self.reflectivity * scaling_factor
        new_reflectivity_error = self.reflectivity_error * scaling_factor
        new_background = self.bckg * scaling_factor

        self.transmission = _new_transmission
        self.reflectivity = new_reflectivity
        self.reflectivity_error = new_reflectivity_error
        self.bckg = new_background
        self.replaced_transmission = True
        logger.info('Transmission corrected. Recalculation will reset it to defaults.')

    @staticmethod
    def _find_double_(x: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Find indices of repeated values in an array.

        Args:
            x (np.ndarray): Input array.

        Returns:
            Dict[float, np.ndarray]: Dictionary mapping values to their indices.
        """
        u, c = np.unique(x, return_counts=True)
        values_of_interest = u[c > 1]

        indexes_multiple_values = {value: np.where(x == value)[0] for value in values_of_interest}
        return indexes_multiple_values

    def calculate_corrected_transmission(self) -> Dict[Any, float]:
        """
        Calculate corrected transmission based on overlapping points (doubles).

        Returns:
            Dict[Any, float]: Dictionary of corrected transmissions.
        """
        _new_transmission_dict = dict(zip(self.attenuator, self.transmission))
        if self.replaced_transmission:
            logger.warning('Transmission was changed. Consider reprocessing data.')
        else:
            double_x = XRR._find_double_(self.qz)
            sorted_double_x = dict(sorted(double_x.items(), reverse=True))
            coeff = []

            # This logic assumes the array is sorted by angle/qz in general but contains overlaps
            # It calculates ratio between the last point of the first segment and first point of second segment
            # for a given Q value.
            # Note: i is the key (qz value), sorted_double_x[i] are indices
            for i in sorted_double_x:
                indices = sorted_double_x[i]
                # Assuming indices are ordered like [idx1, idx2], where idx1 is from higher intensity (lower attenuation) usually?
                # The original code used [-2] and [-1].
                if len(indices) >= 2:
                     coeff.append(self.reflectivity[indices[-2]] / self.reflectivity[indices[-1]])
                else:
                    coeff.append(1.0) # Fallback

            coeff_dict = {i: {'coeff': k, 'indexes': sorted_double_x[i]} for i, k in zip(sorted_double_x, coeff)}

            _new_transm = copy.copy(self.transmission)
            for i in coeff_dict:
                # Apply correction to all points before the overlap
                # This assumes scan direction and that overlaps are sequential adjustments
                idx_limit = coeff_dict[i]['indexes'][1]
                _new_transm[0:idx_limit] = _new_transm[0:idx_limit] * coeff_dict[i]['coeff']

            _new_transmission_dict = dict(zip(self.attenuator, _new_transm))

        return _new_transmission_dict

    def correct_doubles(self):
        """
        Correct transmission using overlapping points.
        """
        if not self.corrected_doubles:
            logger.info('Correcting transmission using double points.')
            new_transm = self.calculate_corrected_transmission()
            self.replace_transmission(new_transm)
            self.corrected_doubles = True
        else:
            logger.info('Double points already corrected.')

    def do_rebin(self, *args, **kwargs):
        """
        Rebin the reflectivity data.

        Args:
            *args: Arguments passed to rebin function.
            **kwargs: Keyword arguments passed to rebin function.
        """
        new_qz, new_R, new_R_err = rebin(*self.get_reflectivity(), **kwargs)
        self.qz = new_qz
        self.reflectivity = new_R
        self.reflectivity_error = new_R_err
        self.is_rebinned = True

    @staticmethod
    def find_i0_from_z_scan(z: np.ndarray, I: np.ndarray) -> Tuple[float, int]:
        """
        Find I0 (direct beam intensity) from a Z-scan.

        Args:
            z (np.ndarray): Z positions (or angles).
            I (np.ndarray): Intensity.

        Returns:
            Tuple[float, int]: I0 value and index of maximum intensity.
        """
        n_max = np.argmax(I)
        # Heuristic to find background cutoff
        I_cutoff = np.mean(I[:n_max]) - np.mean(I[:n_max]) ** 0.5
        zscan_new = I[np.where(I >= I_cutoff)]
        if len(zscan_new) > 0:
            I0 = np.median(zscan_new)
        else:
            I0 = I[n_max]
        return I0, n_max

    def find_i0(self, to_print: bool = True) -> Tuple[float, Any, float]:
        """
        Find I0, attenuator and transmission at the peak.

        Args:
            to_print (bool, optional): Whether to print details. Defaults to True.

        Returns:
            Tuple[float, Any, float]: I0, attenuator value, transmission value.
        """
        if to_print:
            logger.info('Processing scan of motor %s.', self.alpha_i_name)

        I0, n_max = XRR.find_i0_from_z_scan(self.alpha_i, self.reflectivity)

        if to_print:
            logger.info('I0 = %.5e, attenuator = %s, transmission = %.5e',
                        I0, self.attenuator[n_max], self.transmission[n_max])

        return I0, self.attenuator[n_max], self.transmission[n_max]

    def _assert_i0_from_calc(self, calc_I0: float, atten: Any, transmission: float):
        """
        Internal method to update I0 from calculated values.
        """
        if self.alpha_i_name.lower() == 'zgh':
            logger.warning('You are trying to replace I0 in zgH scan. Load reflectivity data.')
        else:
            if atten in self.attenuator:
                if self.corrected_doubles:
                    # Find indices where attenuator matches
                    indices = np.where(self.attenuator == atten)
                    # Use the transmission of the first matching point for adjustment
                    # This logic seems specific to the beamline workflow
                    current_transmission = self.transmission[indices][0]
                    I0 = calc_I0 * transmission / current_transmission
                    self.I0 = I0
                logger.info('I0 replaced.')
            else:
                logger.warning('Attenuator %s not found in the scan of motor %s.\nCheck inputs.',
                               atten, self.alpha_i)

    def assert_i0(self, zgH_scan: 'XRR'):
        """
        Assert I0 from a zgH scan (direct beam scan).

        Args:
            zgH_scan (XRR): Another XRR object representing the direct beam scan.
        """
        if self.alpha_i_name.lower() == 'zgh':
            logger.warning('You are trying to replace I0 in zgH scan. Load reflectivity data.')
        else:
            calc_I0, atten, transmission = zgH_scan.find_i0(to_print=False)
            calc_I0 = calc_I0 * zgH_scan.monitor[0] / self.monitor[0]

            if atten in self.attenuator:
                if self.corrected_doubles:
                    indices = np.where(self.attenuator == atten)
                    if len(indices[0]) > 0:
                        current_trans = self.transmission[indices[0][0]]
                        I0 = calc_I0 * transmission / current_trans
                        self.I0 = I0
                logger.info('I0 replaced.')
            else:
                logger.warning('Attenuator %s not found in the scan of motor %s.\nCheck inputs.',
                               atten, self.alpha_i)
