import h5py
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from math import sin, cos, pi
import scipy as sc
import copy


def rebin(q_vectors, Refl, Relf_E, new_q=None, rebin_as="linear",
          number_of_bins=5000):
    """
    This rebinning procedure is taken from islatu by Andrew R. McCluskey

    https://github.com/DiamondLightSource/islatu

    Rebin the data on a linear or logarithmic q-scale.

    Args:
        q_vectors:
            q - the current q vectors.
        reflected_intensity (:py:attr:`tuple`):
            (I, I_e) - The current reflected intensities, and their errors.
        new_q (:py:attr:`array_like`):
            Array of potential q-values. Defaults to :py:attr:`None`. If this
            argument is not specified, then the new q, R values are binned
            according to rebin_as and number_of_q_vectors.
        rebin_as (py:attr:`str`):
            String specifying how the data should be rebinned. Options are
            "linear" and "log". This is only used if the new_q are unspecified.
        number_of_bins (:py:attr:`int`, optional):
            The max number of q-vectors to be using initially in the rebinning
            of the data. Defaults to :py:attr:`400`.

    Returns:
        :py:attr:`tuple`: Containing:
            - q: rebinned q-values.
            - intensity: rebinned intensities.
            - intensity_e: rebinned intensity errors.
    """

    # Unpack the arguments.
    q = q_vectors
    R, R_e = Refl, Relf_E

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
    def __init__(self, file, scans, alpha_i_name='chi',
                 detector_name='mpx_cdte_22_eh1', monitor_name='mon',
                 transmission_name='transm', att_name='curratt', cnttime_name='sec',
                 PX0=404, PY0=165, dPX=5, dPY=5, pixel_size_qxz=0.055, pixel_size_qy=0.055,
                 energy_name='eccmono', I0=1e13):
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

        self.__load_data__()
        self.__process_2D_data__()

    def __load_single_scan__(self, ScanN):
        print('Loading scan #{}'.format(ScanN))
        f = h5py.File(self.file, "r")
        self.data = f.get(ScanN + '.1/measurement/' + self.detector_name)

        data_x = f.get(ScanN + '.1' + '/measurement/' + self.alpha_i_name)
        self.alpha_i = np.array(data_x)

        data_mon = f.get(ScanN + '.1' + '/measurement/' + self.monitor_name)
        self.monitor = np.array(data_mon)

        data_transm = f.get(ScanN + '.1' + '/measurement/' + self.transmission_name)
        self.transmission = np.array(data_transm)

        data_att = f.get(ScanN + '.1' + '/measurement/' + self.att_name)
        self.attenuator = np.array(data_att)

        cnttime = f.get(ScanN + '.1' + '/measurement/' + self.cnttime_name)
        self.cnttime = np.array(cnttime)

        energy = f.get(ScanN + '.1' + '/instrument/positioners/' + self.energy_name)
        self.energy = np.array(energy)

        self.sample_name = str(f.get(ScanN + '.1' + '/sample/name/')[()])[2:-1:1]

        print('Loaded scan #{}'.format(ScanN))

    def __load_data__(self, skip_points=1):
        t0 = time.time()
        print("Start loading data.")
        if len(self.scans) == 1:
            ScanN = str(self.scans[0])
            self.__load_single_scan__(ScanN)
        else:
            first_ScanN = str(self.scans[0])
            self.__load_single_scan__(first_ScanN)
            for each in self.scans[1:]:
                print('Loading scan ', each)
                f = h5py.File(self.file, "r")
                ScanN = str(each)

                data = f.get(ScanN + '.1/measurement/' + self.detector_name)[skip_points:]
                self.data = np.append(self.data, data, axis=0)

                data_x = f.get(ScanN + '.1' + '/measurement/' + self.alpha_i_name)[skip_points:]
                self.alpha_i = np.append(self.alpha_i, data_x)

                data_mon = f.get(ScanN + '.1' + '/measurement/' + self.monitor_name)[skip_points:]
                self.monitor = np.append(self.monitor, data_mon)

                data_transm = f.get(ScanN + '.1' + '/measurement/' + self.transmission_name)[skip_points:]
                self.transmission = np.append(self.transmission, data_transm)

                data_att = f.get(ScanN + '.1' + '/measurement/' + self.att_name)[skip_points:]
                self.attenuator = np.append(self.attenuator, data_att)

                cnttime = f.get(ScanN + '.1' + '/measurement/' + self.cnttime_name)[skip_points:]
                self.cnttime = np.append(self.cnttime, cnttime)

                energy = f.get(ScanN + '.1' + '/instrument/positioners/' + self.energy_name)
                self.energy = np.append(self.energy, energy)

                print('Loaded scan #{}'.format(ScanN))

        print("Loading completed. Reading time %3.3f sec" % (time.time() - t0))

    def __process_2D_data__(self):
        t0 = time.time()
        print('Starting 2D data processing.')
        nic, nxc, nyc = np.shape(self.data)
        try:
            if len(self.energy)>1:              # to deal with legacy data
                _energy = self.energy.mean()
                self.energy = _energy
        except:
            pass
        # print('Combined array of 2D images shape %6d, %6d, %6d \n' % (nic, nxc, nyc))

        # map2D = np.zeros((len(x), nxc))  # ,dtype=float32)
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
        for i in range(nic):
            IqxyBL = np.sum(
                np.sum(self.data[i, (self.PY0 + 2 * self.dPY + 1 - self.dPY):(self.PY0 + 2 * self.dPY + 1 + self.dPY),
                       (self.PX0 - self.dPX):(self.PX0 + self.dPX)]))   #lower square in the plotted detector image
            IqxyBH = np.sum(
                np.sum(self.data[i, (self.PY0 - 2 * self.dPY - 1 - self.dPY):(self.PY0 - 2 * self.dPY - 1 + self.dPY),
                       (self.PX0 - self.dPX):(self.PX0 + self.dPX)]))   #higher square in the plotted detector image
            IqxyS = np.sum(np.ravel(
                self.data[i, (self.PY0 - self.dPY):(self.PY0 + self.dPY), (self.PX0 - self.dPX):(self.PX0 + self.dPX)]))
            IqxyBF = np.sum(
                np.sum(self.data[i, (self.PY0 - self.dPY):(self.PY0 + self.dPY),
                       (self.PX0 + 2 * self.dPX + 1 - self.dPX):(self.PX0 + 2 * self.dPX + 1 + self.dPX)])) #offset into higher angles square in the plotted detector image
            IqxyBR = np.sum(
                np.sum(self.data[i, (self.PY0 - self.dPY):(self.PY0 + self.dPY),
                       (self.PX0 - 2 * self.dPX - 1 - self.dPX):(self.PX0 - 2 * self.dPX - 1 + self.dPX)])) #offset into lower angles square in the plotted detector image

            if i < len(self.alpha_i):
                Qzcut[:] = np.sum(self.data[i, (self.PY0 - self.dPY):(self.PY0 + self.dPY), :], axis=0)
                Qzcut_bckg1[:] = np.sum(
                    self.data[i, (self.PY0 + 2 * self.dPY + 1 - self.dPY):(self.PY0 + 2 * self.dPY + 1 + self.dPY), :],
                    axis=0)
                Qzcut_bckg2[:] = np.sum(
                    self.data[i, (self.PY0 - 2 * self.dPY - 1 + self.dPY):(self.PY0 - 2 * self.dPY - 1 - self.dPY), :],
                    axis=0)
                self.Smap2D.append(
                    (Qzcut[:] - (Qzcut_bckg1[:] + Qzcut_bckg2) / 2) / self.transmission[i]  /
                    self.monitor[i] * self.monitor[0])
                self.Smap2D_e.append(((np.sqrt(abs(Qzcut[:]))+ np.sqrt((Qzcut_bckg1[:]+Qzcut_bckg2[:])/2))/ self.transmission[i]  /
                    self.monitor[i] * self.monitor[0]))
                # Smap2D.append((Qzcut[:]) / m4[i] / m2[i] / m1[i] * m1[0])

            Ib_cut[i] = (IqxyBL + IqxyBH) / 2  # subtract true backgorund
            # Ib_cut[i] =(IqxyBF+IqxyBR)/ 2 #subtract diffuse scattering
            Is_cut[i] = IqxyS
            Is_cut_err[i] = np.sqrt(Is_cut[i])
            Ib_cut_err[i] = np.sqrt(Ib_cut[i])
            #I_err[i] = Is_cut_err[i]
            try:
                I_err[i] = np.sqrt((Is_cut_err[i]/Is_cut[i])**2 + (Ib_cut_err[i]/Ib_cut[i])**2)
            except:
                I_err[i] = Is_cut_err[i]/Is_cut[i]

        print('Number of points in the scan %6d \n' % (len(self.alpha_i)))
        I_Signal_cut = np.nan_to_num(Is_cut[:len(self.alpha_i)], nan=1e-11)
        I_Backgr_cut = np.nan_to_num(Ib_cut[:len(self.alpha_i)], nan=1e-11)
        I_error = np.nan_to_num(I_err[:len(self.alpha_i)], nan = 1e-11)

        self.qz = 4 * pi * np.sin(np.deg2rad(self.alpha_i)) / (12.398 / self.energy)
        self.reflectivity = (I_Signal_cut - I_Backgr_cut) / self.transmission  / self.monitor * \
                            self.monitor[0]
        self.reflectivity_error = abs(I_error * self.reflectivity)

        self.bckg = I_Backgr_cut / self.transmission / self.monitor * self.monitor[0] / self.I0

        self.raw_counts = I_Signal_cut /  self.monitor * self.monitor[0]
        self.reflectivity = self.reflectivity / self.I0
        self.reflectivity_error = self.reflectivity_error / self.I0

        for each in np.where(self.reflectivity<=1e-12):
            self.reflectivity[each] = 1e-12

        for each in np.where(self.reflectivity_error<=1e-13):
            self.reflectivity_error[each] = 1e-13

        print("Processing completed. Processing time %3.3f sec \n\n" % (time.time() - t0))

    def footprint_correction(self, sample_size=1, beamsize=9.6, correct_dir_beam=False):  # sample size in cm, beam size in microns
        if not self.footprint_correction_applied:
            Si_critical_angle = 4 * np.pi * np.sin(np.deg2rad(8.103E-02)) / (12.398 / self.energy)
            samplesize = sample_size * 10000  # converting cm to microns
            footprint = np.array([0.5 * beamsize / np.sin(np.deg2rad(alpha_i)) if (
                        alpha_i != 0) else 0.5 * beamsize / np.sin(np.deg2rad(1e-3)) for alpha_i in self.alpha_i])

            beam_fraction = np.array(
                [(sc.stats.norm.cdf(samplesize / 2, 0, ftp) - sc.stats.norm.cdf(-samplesize / 2, 0, ftp)) for ftp in
                 footprint])
            Icor = self.reflectivity / beam_fraction
            if correct_dir_beam:
                i = 0
                while self.qz[i] < Si_critical_angle:
                    if Icor[i] > 1.15:
                        Icor[i] = 1
                    i += 1
            self.reflectivity = Icor
            self.footprint_correction_applied = True
            print('Footprint correction completed with beam size = {} microns and sample size = {} cm\n'.format(beamsize,
                                                                                                              sample_size))
        else:
            print('Footprint correction already applied! To apply in again use reprocess() method. \n')

    def reprocess(self):
        self.__load_data__()
        self.__process_2D_data__()
        self.footprint_correction_applied = False
        self.corrected_doubles = False
        self.replaced_transmission = False
        self.is_rebinned = False
        print("Reloaded and reprocessed data. \n\n")

    def produce_Qmap(self, SDD=900):
        t0 = time.time()
        print('Starting q-space mapping.')
        chi_r = np.deg2rad(self.alpha_i)
        pixels = np.array(range(516))

        k0 = 2 * pi / (12.398 / self.energy)

        self.Qz_map = np.array(
            [[np.round(k0 * (sin(chi + ((self.PX0 - px) * self.pixel_size_qxz / SDD)) + sin(chi)), 10) for px in pixels]
             for chi in chi_r])
        self.Qx_map = np.array(
            [[np.round(k0 * (cos(chi + ((self.PX0 - px) * self.pixel_size_qxz / SDD)) - cos(chi)), 10) for px in pixels]
             for chi in chi_r])
        print("2D map calculated. Processing time %3.3f sec" % (time.time() - t0))

    def plot_Qmap(self, save=False, fig=None, axes=None):
        if axes is None:
            fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), layout='tight')
        else:
            ax0 = axes
        ax0.pcolormesh(self.Qx_map, self.Qz_map, np.log10(self.Smap2D), cmap='jet', vmin=4, vmax=10,
                            # np.log10(Zmin + 1E1)
                            shading='gouraud', snap=True)

        ax0.set_xlabel(r'$q_x, \AA^{-1}$')  # , size=18)
        ax0.set_ylabel(r'$q_z, \AA^{-1}$')  # , size=18)
        ax0.set_ylim(0, 0.5)
        ax0.set_xlim(-2e-4, 0.0005)
        ax0.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        if save:
            print('Saving Q-space map.')
            plt.savefig('Qmap_{}_scan_{}.png'.format(self.sample_name, self.scans), dpi=300)

        return fig, ax0

    def get_reflectivity(self):  # return np array of reflectivity and errors
        qz_tr =  np.sort(self.qz)
        R_tr = self.reflectivity[self.qz.argsort()]
        Rerr_tr = self.reflectivity_error[self.qz.argsort()]
        return np.array([qz_tr, R_tr, Rerr_tr])

    def plot_reflectivity(self, save=False):
        fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), layout='tight')
        ax0.errorbar(*self.get_reflectivity())
        ax0.semilogy()
        ax0.set_xlim(left=0)
        ax0.set_ylim(top=2)
        ax0.set_xlabel(r'$q_z, \AA^{-1}$')
        ax0.set_ylabel(r'$\mathrm{Reflectivity}$')
        if save:
            print('Saving reflectivity plot.')
            plt.savefig('XRR_{}_scan_{}.png'.format(self.sample_name, self.scans), dpi=300)

        return fig, ax0

    def save_reflectivity(self, filename=False, *directory):
        if not directory:
            directory = os.getcwd()
        if not filename:
            filename = self.sample_name + '_xrr_scan_{}.dat'.format(self.scans)
        _to_save = self.get_reflectivity().T
        np.savetxt(filename, _to_save)
        print('Reflectivity saved to dir: {} \n filename: {}'.format(directory, filename))

    def show_detector_image(self, frame_number=50, ax=None, plot_cross = True):
        fig = plt.figure()
        if ax is None:
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

    def replace_transmission(self, filter_transmission):
        _new_transmission = np.array([])
        for point in self.attenuator:
            _new_transmission = np.append(_new_transmission, [float(filter_transmission.get(point))])
        # print(_new_transmission)
        new_reflectivity = self.reflectivity * self.transmission / _new_transmission
        new_reflectivity_error = self.reflectivity_error * self.transmission / _new_transmission
        new_background = self.bckg * self.transmission / _new_transmission
        self.transmission = _new_transmission
        self.reflectivity = new_reflectivity
        self.reflectivity_error = new_reflectivity_error
        self.bckg = new_background
        self.replaced_transmission = True
        print('Transmission corrected. Recalculation will reset it to defaults.')

    @staticmethod
    def _find_double_(x):
        u, c = np.unique(x, return_counts=True)
        values_of_interest = u[c > 1]

        indexes_multiple_values = {value: np.where(x == value)[0] for value in values_of_interest}
        return indexes_multiple_values

    def calculate_corrected_transmission(self):
        _new_transmission_dict = dict(zip(self.attenuator, self.transmission))
        if self.replaced_transmission == True:
            print('Transmission was changed. Consider reprocessing data.')
        else:
            double_x = XRR._find_double_(self.qz)
            sorted_double_x = dict(sorted(double_x.items(), reverse=True))
            coeff = []
            for i in sorted_double_x.values():
                coeff.append(self.reflectivity[i[-2]] / self.reflectivity[i[-1]])

            coeff_dict = {i: {'coeff': k, 'indexes': sorted_double_x[i]} for i, k in zip(sorted_double_x, coeff)}

            _new_transm = copy.copy(self.transmission)
            for i in coeff_dict:
                _new_transm[0:coeff_dict[i]['indexes'][1]] = _new_transm[0:coeff_dict[i]['indexes'][1]] * coeff_dict[i][
                    'coeff']
            _new_transmission_dict = dict(zip(self.attenuator, _new_transm))
            # _new_transmission_dict.update({0:1})
        return _new_transmission_dict

    def correct_doubles(self):
        if not self.corrected_doubles:
            print('Correcting transmission using double points.')
            new_transm = self.calculate_corrected_transmission()
            self.replace_transmission(new_transm)
            self.corrected_doubles = True
        else:
            print('Double points already corrected.')

    def do_rebin(self, *args, **kwargs):
        new_qz, new_R, new_R_err = rebin(*self.get_reflectivity())
        self.qz = new_qz
        self.reflectivity = new_R
        self.reflectivity_error = new_R_err



    @staticmethod
    def find_i0_from_z_scan(z, I):
        n_max = np.argmax(I)
        I_cutoff = np.mean(I[:n_max]) - np.mean(I[:n_max]) ** 0.5
        zscan_new = I[np.where(I >= I_cutoff)]
        I0 = np.median(zscan_new)
        return I0, n_max

    def find_i0(self, to_print=True):
        if to_print:
            print('Processing scan of motor {}.'.format(self.alpha_i_name))
        else:
            pass
        I0, n_max = XRR.find_i0_from_z_scan(self.alpha_i, self.reflectivity)
        if to_print:
            print('I0 = {:.5e}, attenuator = {}, transmission = {:.5e}'.format(I0, self.attenuator[n_max], self.transmission[n_max]))
        else:
            pass
        return I0, self.attenuator[n_max], self.transmission[n_max]

    def assert_i0(self, calc_I0, atten, transmission):
        if self.alpha_i_name.lower()=='zgh':
            print('You are trying to replace I0 in zgH scan. Load reflectivity data.')
        else:
            if atten in self.attenuator:
                if self.corrected_doubles:
                    I0 = calc_I0*transmission/self.transmission[np.where(self.attenuator==atten)]
                    print(I0)
                    self.I0 = I0[0]
                print('I0 replaced.')
            else:
                print('Attenuator {} not found in the scan of motor {}.\nCheck inputs.'.format(atten, self.alpha_i))
