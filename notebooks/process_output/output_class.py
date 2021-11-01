import numpy as np
import os
import pickle

from astropy.cosmology import FlatLambdaCDM
from scipy.stats import gaussian_kde
from lenstronomy.Cosmo.kde_likelihood import KDELikelihood
from tqdm import tqdm_notebook, tnrange
import emcee
import gc
from astropy.io import fits

import lenstronomy.Util.util as util
from lenstronomy.Workflow.fitting_sequence import FittingSequence
import lenstronomy.Plots.output_plots as out_plot
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Analysis.lens_analysis import LensAnalysis
from lenstronomy.Analysis.lens_properties import LensProp
from lenstronomy.Plots.output_plots import ModelPlot
import lenstronomy.Util.mask as mask_util
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.class_creator as class_creator

cwd = os.getcwd()
base_path, _ = os.path.split(cwd)


class ModelOutput(object):
    """
    Class to load lens model posterior chain and other model setups.
    """

    TIME_DELAYS = np.array([-112.1, -155.5]) #, -42.4])
    SIGMA_DELAYS = np.array([2.1, 12.8]) #, 17.6])

    VEL_DIS = np.array([230., 236., 220., 227.])
    SIG_VEL_DIS = np.array([37., 42., 21., 9.])
    INV_COV_MAT = np.linalg.inv(np.diag(SIG_VEL_DIS**2+17**2) + (
            -1*np.eye(4, 4)+1)*17**2)
    PSF_FWHM = np.array([0.68, 0.76, 0.52, 0.60796])
    MOFFAT_BETA = np.array([2.97, 3.20, 3.06, 1.55])
    SLIT_WIDTH = np.array([1., 1., 0.75, 1.])
    APERTURE_LENGTH = np.array([1., 1., 1., 1.])



    def __init__(self, model_name, model_type, dir_prefix,
                 dir_suffix, is_test=False, microlensing=False):
        """
        Load the model output file and load the posterior chain and other model
        speification objects.
        """
        self.model_id = model_name
        self.model_type = model_type
        self.is_test = is_test
        self.file_path = dir_prefix + model_name + dir_suffix

        self.microlensing = microlensing
        if self.microlensing:
            self.ml_maps = []
            self.load_microlensing_delays()

        # input_temp = os.path.join(base_path, 'temp', job_name_out +'.txt')
        # output_temp = os.path.join(base_path, 'temp', job_name_out
        # +'_out.txt')

        f = open(self.file_path, 'rb')
        [input_, output_] = pickle.load(f)
        f.close()
        fitting_kwargs_list, kwargs_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, init_samples = input_
        # lens_result, source_result, lens_light_result, ps_result, cosmo_result, multi_band_list_out, fit_output, _ = output_
        kwargs_result, multi_band_list_out, fit_output, _ = output_

        self.kwargs_joint = kwargs_joint
        self.kwargs_likelihood = kwargs_likelihood

        self.kwargs_result = kwargs_result
        self.multi_band_list = multi_band_list_out

        self.samples_mcmc = fit_output[-1][1]
        # random.shuffle()
        self.num_param_mcmc = len(self.samples_mcmc)
        self.param_mcmc = fit_output[-1][2]
        self.logZ = fit_output[-1][4]

        self.kwargs_model = kwargs_model
        # if self.is_test:
        #    self.kwargs_model['cosmo'] = None
        self.kwargs_constraints = kwargs_constraints

        """if 'special' in kwargs_params:
            special = kwargs_params['special'][2]
        else:
            special = []

        if 'extinction_model' in kwargs_params:
            extinction_model = kwargs_params['extinction_model'][2]
        else:
            extinction_model = []"""

        self.param_class = Param(kwargs_model,
                                 kwargs_params['lens_model'][2],
                                 kwargs_params['source_model'][2],
                                 kwargs_params['lens_light_model'][2],
                                 kwargs_params['point_source_model'][2],
                                 kwargs_params['special'][2],
                                 kwargs_params['extinction_model'][2],
                                 kwargs_lens_init=kwargs_params[
                                     'lens_model'][0],
                                 **kwargs_constraints
                                )


        kwargs_result = self.param_class.args2kwargs(self.samples_mcmc[-1])

        self.lens_result = kwargs_result['kwargs_lens']
        self.lens_light_result = kwargs_result['kwargs_lens_light']
        self.ps_result = kwargs_result['kwargs_ps']
        self.source_result = kwargs_result['kwargs_source']

        self.lens_analysis = LensAnalysis(self.kwargs_model)

        self.r_eff = self.get_r_eff()
        self.r_ani = []

        if self.is_test:
            z_lens = kwargs_model['z_lens']
        else:
            z_lens = None

        self.cosmo = self.kwargs_model['cosmo']
        self.lens_model = LensModel(
            lens_model_list=self.kwargs_model['lens_model_list'],
            z_lens=z_lens, z_source=self.kwargs_model['z_source'],
            lens_redshift_list=self.kwargs_model['lens_redshift_list'],
            multi_plane=self.kwargs_model['multi_plane'],
            observed_convention_index=self.kwargs_model[
                'observed_convention_index'],
            z_source_convention=None,
            cosmo=self.cosmo
            )

        # declare following variables to populate later
        self.model_time_delays = None
        self.model_velocity_dispersion = None

        self.lens_prop = LensProp(self.kwargs_model['lens_redshift_list'][0],
                                  self.kwargs_model['z_source'],
                                  self.kwargs_model,
                                  cosmo=self.cosmo)


        # numerical options to perform the numerical integrals
        self.kwargs_galkin_numerics = {'sampling_number': 1000,
                                       'interpol_grid_num': 1000,
                                       'log_integration': True,
                                       'max_integrate': 100,
                                       'min_integrate': 0.001}

    def cleanup(self):
        del self.tiled_model_velocity_dispersion
        del self.tiled_model_time_delays

        gc.collect()

    def get_num_samples(self):
        """
        Get the number of samples.
        :return:
        :rtype:
        """
        return len(self.samples_mcmc)

    def get_r_eff(self, i=-1):
        """
        Compute effective radius of the light distribution in F160W band.
        :param i: index of sample in the mcmc chain
        """
        if i == -1:
            kwargs_result = self.kwargs_result
        else:
            kwargs_result = self.param_class.args2kwargs(self.samples_mcmc[i])

        self._imageModel = class_creator.create_im_sim(self.multi_band_list,
                                                       self.kwargs_joint[
                                                           'multi_band_type'],
                                                       self.kwargs_model,
                                                       bands_compute=
                                                       self.kwargs_likelihood[
                                                           'bands_compute'],
                                                       likelihood_mask_list=
                                                       self.kwargs_likelihood[
                                                           'image_likelihood_mask_list'],
                                                       band_index=0)

        model, error_map, cov_param, param = self._imageModel.image_linear_solve(
            inv_bool=True,
            **kwargs_result)

        lens_light_bool_list = [False] * len(
            self.kwargs_model['lens_light_model_list'])

        if self.is_test:
            lens_light_bool_list[0] = True
        else:
            if self.model_type == "powerlaw":
                lens_light_bool_list[2] = True  # F160W Sersic profile
                # indices: 6, 7
                lens_light_bool_list[3] = True
            elif self.model_type == "composite":
                lens_light_bool_list[2] = True  # Chameleon profile index: 6
            else:
                raise NotImplementedError

        r_eff = self.lens_analysis.half_light_radius_lens(
            kwargs_result['kwargs_lens_light'],
            center_x=self.lens_light_result[0]['center_x'],
            center_y=self.lens_light_result[0]['center_x'],
            model_bool_list=lens_light_bool_list,
            deltaPix=0.01, numPix=1000)
        return r_eff

    def compute_model_time_delays(self):
        """
        Compute time delays from the lens model and store it in class variable `model_time_delays`.
        """
        num_samples = self.get_num_samples()

        self.model_time_delays = []

        for i in tnrange(num_samples,
                         desc="{} model delays".format(self.model_id)):
            param_array = self.samples_mcmc[i]

            kwargs_result = self.param_class.args2kwargs(param_array)

            # print(param_array)

            model_arrival_times = self.lens_model.arrival_time(
                kwargs_result['kwargs_ps'][0]['ra_image'],
                kwargs_result['kwargs_ps'][0]['dec_image'],
                kwargs_result['kwargs_lens'])
            # print(model_arrival_times)
            dt_AB = model_arrival_times[0] - model_arrival_times[1]
            dt_AD = model_arrival_times[0] - model_arrival_times[3]
            dt_BD = model_arrival_times[1] - model_arrival_times[3]

            self.model_time_delays.append([dt_AB, dt_AD]) #, dt_BD])
            #print(dt_AB)

        self.model_time_delays = np.array(self.model_time_delays)

    def compute_model_velocity_dispersion(self,
                                          cGD_light=True, cGD_mass=True,
                                          aniso_param_min=0.5,
                                          aniso_param_max=5,
                                          start_index=0,
                                          num_compute=None,
                                          print_step=None,
                                          ):
        """
        Compute velocity dispersion from the lens model for different measurement setups.
        :param num_samples: default `None` to compute for all models in the
        chain, use lower number only for testing and keep it same between
        `compute_model_time_delays` and this method.
        :param start_index: compute velocity dispersion from this index
        :param num_compute: compute for this many samples
        :param print_step: print a notification after this many step
        """
        num_samples = self.get_num_samples()

        self.model_velocity_dispersion = []

        anisotropy_model = 'OsipkovMerritt'  # anisotropy model applied
        aperture_type = 'slit'  # type of aperture used

        if num_compute is None:
            num_compute = num_samples-start_index

        for i in range(start_index, start_index+num_compute):
            if print_step is not None:
                if (i-start_index)%print_step == 0:
                    print('Computing step: {}'.format(i-start_index))

            sample = self.samples_mcmc[i]

            vel_dis_array = []

            if self.is_test:
                aniso_param = 0.8
            else:
                r_eff = self.get_r_eff(i)
                aniso_param = np.random.uniform(aniso_param_min*r_eff,
                                                aniso_param_max*r_eff)

                self.r_ani.append(aniso_param)

            for n in range(len(self.VEL_DIS)):
                kwargs_result = self.param_class.args2kwargs(sample)

                lens_result = kwargs_result['kwargs_lens']
                lens_light_result = kwargs_result['kwargs_lens_light']
                #source_result = kwargs_result['kwargs_source']
                #ps_result = kwargs_result['kwargs_ps']

                kwargs_aperture = {'length': self.APERTURE_LENGTH[n],
                                   'width': self.SLIT_WIDTH[n],
                                   'center_ra': lens_light_result[0][
                                       'center_x'],
                                   'center_dec': lens_light_result[0][
                                       'center_y'], 'angle': 0}

                if self.is_test:
                    band_index = 0
                else:
                    band_index = 2
                image_model = class_creator.create_im_sim(
                    self.kwargs_joint['multi_band_list'],
                    self.kwargs_joint['multi_band_type'],
                    self.kwargs_model,
                    bands_compute=self.kwargs_likelihood['bands_compute'],
                    likelihood_mask_list=self.kwargs_likelihood[
                        'image_likelihood_mask_list'],
                    band_index=band_index)

                model, error_map, cov_param, param = image_model.image_linear_solve(
                    inv_bool=True, **kwargs_result)

                # set the anisotropy radius. r_eff is pre-computed half-light radius of the lens light
                kwargs_anisotropy = {'r_ani': aniso_param}
                # compute the velocity disperson in a pre-specified cosmology (see lenstronomy function)
                # We read out the lens light model kwargs that has previously been solved for the linear amplitude parameters.
                # This is necessary to perform an accurate kinematic estimate as we are using a superposition of different light profiles and their relative amplitudes matter.

                light_model_bool = [False] * len(
                    self.kwargs_model['lens_light_model_list'])

                # if self.is_test:
                #     light_model_bool[0] = True
                # else:
                #     for index in \
                #             self.kwargs_model['index_lens_light_model_list'][
                #                 2][:-1]:
                #         light_model_bool[index] = True

                if self.is_test:
                    light_model_bool[0] = True
                else:
                    if self.model_type == "powerlaw":
                        light_model_bool[2] = True  # F160W Sersic profile
                        # indices: 2, 3
                        light_model_bool[3] = True
                    elif self.model_type == "composite":
                        light_model_bool[2] = True # Chameleon profile indx: 2
                    else:
                        raise NotImplementedError

                lens_model_bool = [False] * len(
                    self.kwargs_model['lens_model_list'])

                if self.model_type == 'powerlaw':  # self.kwargs_model['lens_model_list'][0] == 'SPEMD':
                    lens_model_bool[0] = True

                    cGD_light = True
                    cGD_mass = False
                else:
                    lens_model_bool[0] = True
                    lens_model_bool[2] = True

                    cGD_light = True
                    cGD_mass = True

                # print(lens_model_bool, light_model_bool)
                vel_dispersion_temp = self.lens_prop.velocity_dispersion_numerical(
                    lens_result,
                    lens_light_result,
                    kwargs_anisotropy,
                    kwargs_aperture,
                    self.PSF_FWHM[n],
                    aperture_type,
                    anisotropy_model,
                    MGE_light=cGD_light,
                    MGE_mass=cGD_mass,
                    r_eff=self.r_eff,
                    psf_type='MOFFAT',
                    moffat_beta=self.MOFFAT_BETA[n],
                    kwargs_numerics=self.kwargs_galkin_numerics,
                    light_model_kinematics_bool=light_model_bool,
                    lens_model_kinematics_bool=lens_model_bool
                )
                vel_dis_array.append(vel_dispersion_temp[0])

            self.model_velocity_dispersion.append(vel_dis_array)

        self.model_velocity_dispersion = np.array(
            self.model_velocity_dispersion)

        return self.model_velocity_dispersion

    def populate_Ddt_Dsds(self, Ddt_low=70./150., Ddt_high=70./30.,
                          Dsds_low=0.1, Dsds_high=2., sample_multiplier=1000):
        """
        :param sample_multiplier: number of uniformly distributed Ddt or
        Dsds per lens model sample
        :return:
        :rtype:
        """
        self._uniform_Ddt_Dsds = np.array([
            np.random.uniform(low=Ddt_low, high=Ddt_high,
                              size=self.get_num_samples() * sample_multiplier),
            np.random.uniform(low=Dsds_low, high=Dsds_high,
                              size=self.get_num_samples() * sample_multiplier)
        ])

        #self.tiled_model_time_delays = np.array(list(
        #    self.model_time_delays) * sample_multiplier)

        #self.tiled_model_velocity_dispersion = np.array(list(
        #    self.model_velocity_dispersion) * sample_multiplier)

        self.tiled_model_time_delays = np.tile(self.model_time_delays, 
                                               [sample_multiplier, 1])
        self.tiled_model_velocity_dispersion = np.tile(
            self.model_velocity_dispersion,
            [sample_multiplier, 1]
        )
        
        #self.tiled_rani = np.tile(self.r_ani, sample_multiplier)

        self._time_delay_likelihoods = self.get_time_delay_likelihood(
            self.tiled_model_time_delays,
            self._uniform_Ddt_Dsds[0])
        self._time_delay_likelihoods -= np.max(self._time_delay_likelihoods)

        self._velocity_dispersion_likelihoods = self.get_velocity_dispersion_likelihood(
            self.tiled_model_velocity_dispersion,
            self._uniform_Ddt_Dsds[1])
        self._velocity_dispersion_likelihoods -= np.max(
            self._velocity_dispersion_likelihoods)

    def sample_Ddt_Dd(self, num_sample):
        """
        Jointly importance sample Ddt-Dd ratios. Multiply with fiducial
        values to get true values.
        :param num_sample: number of samples to return
        :type num_sample:
        :return:
        :rtype:
        """
        weights = np.exp(self._time_delay_likelihoods + \
                         self._velocity_dispersion_likelihoods)
        weights = weights / np.sum(weights)
        indices = np.random.choice(np.arange(len(self._uniform_Ddt_Dsds[0])),
                                   p=weights,
                                   size=num_sample)
        sample = self._uniform_Ddt_Dsds[:, indices]
        sample[1] = sample[0] / sample[1]
        
        return sample, indices #, self. #weights[indices], indices
        
    def simple_sample_Ddt_Dd(self, num_sample):
        """
        Return simple sampled Ddt-Dd ratios for importance sampling from the combined distribution.
        :param num_sample:
        :type num_sample:
        :return:
        :rtype:
        """
        vd_td_likelihoods = self._time_delay_likelihoods + \
                         self._velocity_dispersion_likelihoods
        #weights = weights / np.sum(weights)
        indices = np.random.choice(np.arange(len(self._uniform_Ddt_Dsds[0])),
                                   #p=weights,
                                   size=num_sample)
        sample = self._uniform_Ddt_Dsds[:, indices]
        sample[1] = sample[0] / sample[1]

        return sample, vd_td_likelihoods[indices]

    def sample_Ddt(self, num_sample):
        """
        Importance sample Ddt.
        :param num_sample: number of samples to return
        :type num_sample:
        :return:
        :rtype:
        """
        weights = np.exp(self._time_delay_likelihoods)
        weights = weights / np.sum(weights)
        sample = np.random.choice(self._uniform_Ddt_Dsds[0],
                                  p=weights,
                                  size=num_sample)

        return sample

    def sample_Dsds(self, num_sample):
        """
        Importance sample Ddt.
        :param num_sample: number of samples to return
        :type num_sample:
        :return:
        :rtype:
        """
        weights = np.exp(self._velocity_dispersion_likelihoods)
        weights = weights / np.sum(weights)
        sample = np.random.choice(self._uniform_Ddt_Dsds[1],
                                  p=weights,
                                  size=num_sample)

        return sample

    def load_microlensing_delays(self):
        names = ['A', 'B', 'D']
        for i, name in enumerate(names):
            ml_map = fits.open(
                '../data/J0408_microlensing/Img{}/TDmap_1.0R0_incl0.0_pa0.0.fits'.format(
                    name))[0].data

            self.ml_maps.append(ml_map.flatten())

    def get_time_delay_likelihood(self, model_delays, Ddt_ratio):
        """
        Get time delay likelihood from model delays and ratio = D_dt_true /
        D_dt_fiducial.
        :param model_delays:
        :type model_delays:
        :param Ddt_ratio: D_dt_true / D_dt_fiducial
        :type Ddt_ratio:
        :return:
        :rtype:
        """
        if self.microlensing:
            num = len(model_delays)

            ml_delay_A = np.random.choice(self.ml_maps[0], size=num)
            ml_delay_B = np.random.choice(self.ml_maps[1], size=num)
            ml_delay_D = np.random.choice(self.ml_maps[2], size=num)

            microlensing_delays = np.array([ml_delay_A - ml_delay_B,
                                            ml_delay_A - ml_delay_D
                                            ]).T
        else:
            microlensing_delays = 0.

        perturbed_delays = (model_delays + microlensing_delays) * Ddt_ratio[:,
                                                                  np.newaxis]

        likelihood_array = (perturbed_delays - self.TIME_DELAYS[np.newaxis,
                                               :]) ** 2 / self.SIGMA_DELAYS ** 2

        return -np.sum(likelihood_array, axis=1) / 2.

    def get_velocity_dispersion_likelihood(self, model_velocity_dispersion,
                                           Dsds_ratio):
        """
        Get velocity dispersion likelihood from model velocity dispersion and
        ratio = D_sds_true / D_sds_fiducial.
        :return:
        :rtype:
        """
        #likelihood_array = (model_velocity_dispersion * (np.sqrt(
        ## Dsds_ratio)[:,
        #                                                 np.newaxis])
        #                    - self.VEL_DIS[np.newaxis, :]) ** 2 \
        #                   / self.SIG_VEL_DIS ** 2

        # likelihood_array = np.matmul((model_velocity_dispersion * np.sqrt(
        #     Dsds_ratio)[:, np.newaxis] - self.VEL_DIS[np.newaxis, :]),
        #     np.matmul(
        #     self.INV_COV_MAT[:, np.newaxis], (model_velocity_dispersion * (np.sqrt(
        #     Dsds_ratio)[:,np.newaxis]) - self.VEL_DIS[np.newaxis, :])))

        diff_vector = model_velocity_dispersion * (np.sqrt(
             Dsds_ratio)[:,np.newaxis]) - self.VEL_DIS[np.newaxis, :]
        print(model_velocity_dispersion.shape, self.VEL_DIS[np.newaxis,
                                               :].shape)
        right_matmul = np.einsum('ij,kj->ki', self.INV_COV_MAT, diff_vector)
        print(right_matmul.shape)
        #left_matmul = np.einsum('ki,ji->k', diff_vector, right_matmul)
        left_matmul = np.multiply(diff_vector, right_matmul)
        print(left_matmul.shape)
        likelihood_array = -np.sum(left_matmul, axis=1) / 2.
        print(likelihood_array.shape)

        return likelihood_array #-np.sum(likelihood_array, axis=1) / 2.

    def load_velocity_dispersion(self, model_name, dir_prefix, dir_suffix,
                                 compute_chunk):
        """
        Load saved model velocity dispersions.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix: should include the slash at the end
        :param dir_suffix: example '_mod_out.txt'
        :type dir_suffix:
        :return:
        :rtype:
        """
        loaded_vel_dis = []

        for i in range(self.get_num_samples()/compute_chunk):
            start_index = i * compute_chunk
            file_path = dir_prefix+ '{}_'.format(start_index) + \
                        model_name + dir_suffix

            if loaded_vel_dis == []:
                loaded_vel_dis = np.loadtxt(file_path)
                #print(loaded_vel_dis.shape)
            else:
                #print(loaded_vel_dis.shape)
                loaded_vel_dis = np.append(loaded_vel_dis, np.loadtxt(file_path),
                                          axis=0)

        assert len(loaded_vel_dis) == self.get_num_samples()

        self.model_velocity_dispersion = loaded_vel_dis

    def load_r_ani(self, model_name, dir_prefix, dir_suffix,
                                 compute_chunk):
        """
        Load saved r_ani.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix: should include the slash at the end
        :param dir_suffix: example '_mod_out.txt'
        :type dir_suffix:
        :return:
        :rtype:
        """
        loaded_r_ani = []

        for i in range(self.get_num_samples()/compute_chunk):
            start_index = i * compute_chunk
            file_path = dir_prefix+ '{}_'.format(start_index) + \
                        model_name + dir_suffix

            if loaded_r_ani == []:
                loaded_r_ani = np.loadtxt(file_path)
                #print(loaded_vel_dis.shape)
            else:
                #print(loaded_r_ani.shape)
                loaded_r_ani = np.append(loaded_r_ani, np.loadtxt(file_path),
                                          axis=0)

        assert len(loaded_r_ani) == self.get_num_samples()

        self.r_ani = loaded_r_ani

    def load_time_delays(self, model_name, dir_prefix, dir_suffix):
        """
        Load saved time delays.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix:
        :param dir_suffix:
        :type dir_suffix:
        :return:
        :rtype:
        """
        file_path = dir_prefix + model_name + dir_suffix

        loaded_time_delays = np.loadtxt(file_path)

        assert len(loaded_time_delays) == self.get_num_samples()

        self.model_time_delays = loaded_time_delays


def get_relative_weights(logZs, sigma_numeric, neighbor_indices=None,
                         take_std=False, sigma_model=None):
    """

    :param logZs:
    :type logZs:
    :param sigma_numeric:
    :type sigma_numeric:
    :param neighbor_indices:
    :type neighbor_indices:
    :return:
    :rtype:
    """

    if sigma_model is not None:
        pass
    elif take_std:
        sigma_model = np.std(logZs)
    elif neighbor_indices is not None:
        dlogZ_neighbors = []
        for ni in neighbor_indices:
            dlogZ_neighbors.append(abs(logZs[ni[0] - 1] - logZs[ni[1] - 1]))

        sigma_model = np.std(dlogZ_neighbors)
    else:
        raise ValueError("Don't know what to do with sigma_model")

    print("sigma_mmodel:", sigma_model)
    max_logZ = np.max(logZs)
    # print(max_logZ)
    del_logZs = logZs - max_logZ

    sigma = np.sqrt(sigma_model ** 2 + sigma_numeric ** 2)

    effective_nums = []

    for dlogZ in del_logZs:
        distribution = np.random.normal(loc=dlogZ, scale=sigma, size=1000000)

        effective_num = float(len(distribution[distribution >= 0.]))

        effective_num += np.sum(np.exp(distribution[distribution < 0.]))

        effective_nums.append(effective_num / 1000000.)

    return effective_nums / np.max(effective_nums)


class CombinedModels(ModelOutput):
    """

    """
    def __init__(self, microlensing=False):
        self.model_id = "combined"
        #self.model_type = model_type
        #self.is_test = is_test
        #self.file_path = dir_prefix + model_name + dir_suffix

        # input_temp = os.path.join(base_path, 'temp', job_name_out +'.txt')
        # output_temp = os.path.join(base_path, 'temp', job_name_out
        # +'_out.txt')

        self.r_ani = []

        self.microlensing = microlensing
        if self.microlensing:
            self.ml_maps = []
            self.load_microlensing_delays()

        # declare following variables to populate later
        self.model_time_delays = None
        self.model_velocity_dispersion = None

        self.powerlaw_time_delays = None
        self.powerlaw_velocity_dispersion = None

        self.composite_time_delays = None
        self.composite_velocity_dispersion = None

    def get_num_samples(self):
        """
        Get the number of samples.
        :return:
        :rtype:
        """
        return len(self.model_time_delays)

    def combine_powerlaw_models(self, models, weights, max_sample):
        """

        :param models:
        :type models:
        :param weights:
        :type weights:
        :param num_sample: max sample from each model
        :type num_sample:
        :return:
        :rtype:
        """
        for model, w in zip(models, weights):
            indices = np.random.choice(np.arange(len(
                model.model_time_delays)), int(w*max_sample))

            if self.powerlaw_time_delays is None:
                self.powerlaw_time_delays = model.model_time_delays[indices, :]
                self.powerlaw_velocity_dispersion = model.model_velocity_dispersion[indices, :]
                if model.r_ani != []:
                    print(model.r_ani)
                    self.powerlaw_rani = model.r_ani[indices]
            else:
                self.powerlaw_time_delays = np.append(
                    self.powerlaw_time_delays,
                    model.model_time_delays[indices, :],
                    axis=0
                )
                self.powerlaw_velocity_dispersion = np.append(
                    self.powerlaw_velocity_dispersion,
                    model.model_velocity_dispersion[indices, :],
                    axis=0
                )
                if model.r_ani != []:
                    print(model.r_ani)
                    self.powerlaw_rani = np.append(
                        self.powerlaw_rani,
                        model.r_ani[indices]
                        )

    def combine_composite_models(self, models, weights, max_sample):
        """

        :param models:
        :type models:
        :param weights:
        :type weights:
        :param num_sample: max sample from each model
        :type num_sample:
        :return:
        :rtype:
        """
        for model, w in zip(models, weights):
            indices = np.random.choice(np.arange(len(
                model.model_time_delays)), int(w*max_sample))

            if self.composite_time_delays is None:
                self.composite_time_delays = model.model_time_delays[indices, :]
                self.composite_velocity_dispersion = model.model_velocity_dispersion[indices, :]
            else:
                self.composite_time_delays = np.append(
                    self.composite_time_delays,
                    model.model_time_delays[indices, :],
                    axis=0
                )
                self.composite_velocity_dispersion = np.append(
                    self.composite_velocity_dispersion,
                    model.model_velocity_dispersion[indices, :],
                    axis=0
                )

    def combine_powerlaw_composite(self, num_sample):
        """

        :param num_sample: number of sample from each model type
        :type num_sample:
        :return:
        :rtype:
        """
        assert num_sample <= len(self.powerlaw_time_delays) and num_sample \
               <= len(self.composite_time_delays)

        pl_indices = np.random.choice(np.arange(len(
            self.powerlaw_time_delays)), num_sample)
        comp_indices = np.random.choice(np.arange(len(
            self.composite_time_delays)), num_sample)

        self.model_time_delays = np.append(
            self.powerlaw_time_delays[pl_indices, :],
            self.composite_time_delays[comp_indices, :],
            axis=0
        )

        self.model_velocity_dispersion = np.append(
            self.powerlaw_velocity_dispersion[pl_indices, :],
            self.composite_velocity_dispersion[comp_indices, :],
            axis=0
        )


class Dist2Cosmology(object):
    """
    Class to extract cosmo posterior from distance posteriors.
    """
    def __init__(self, distance_posterior, z_deflector=0.597, z_source=2.375,
                 fid_h0=70., fid_omega_m=0.3, cosmo=None):
        """

        :param z_deflector: 
        :type z_deflector: 
        :param z_source: 
        :type z_source: 
        :param distance_posterior: fiducial divided 2D array/list, order D_d,
        D_dt
        :type distance_posterior: 
        """
        self.z_l = z_deflector  # 0.745 #0.597
        self.z_s = z_source  # 1.789 #2.375

        if cosmo is not None:
            self.cosmo_fid = cosmo
        else:
            self.cosmo_fid = FlatLambdaCDM(H0=fid_h0, Om0=fid_omega_m)

        self.D_d_fid = self.cosmo_fid.angular_diameter_distance(self.z_l).value
        self.D_s_fid = self.cosmo_fid.angular_diameter_distance(self.z_s).value
        self.D_ds_fid = self.cosmo_fid.angular_diameter_distance_z1z2(self.z_l, self.z_s).value

        self.D_dt_fid = (1 + self.z_l) * self.D_d_fid * self.D_s_fid / self.D_ds_fid

        # in units of Mpc
        self.physical_distance_posterior = [distance_posterior[1] *
                                           self.D_d_fid,
                          distance_posterior[0] * self.D_dt_fid]

        self.kde_likelihood = KDELikelihood(
            self.physical_distance_posterior[0],
            self.physical_distance_posterior[1],
            bandwidth=20)

    def log_likelihood(self, params):
        h0, om_m = params
        cosmo = FlatLambdaCDM(H0=h0, Om0=om_m)
        D_d = cosmo.angular_diameter_distance(self.z_l).value
        D_s = cosmo.angular_diameter_distance(self.z_s).value
        D_ds = cosmo.angular_diameter_distance_z1z2(self.z_l, self.z_s).value

        D_d_r = D_d  # / D_d_fid
        D_dt_r = (1 + self.z_l) * (D_d * D_s / D_ds)  # / D_dt_fid

        return self.kde_likelihood.logLikelihood(D_d_r, D_dt_r)

    def log_prior(self, params):
        h0, om_m = params

        if not 0. < h0 < 150.:
            return -np.inf

        if not 0.05 < om_m < 0.5:
            return -np.inf

        return 0.

    def log_probability(self, params):
        prior = self.log_prior(params)

        if not np.isinf(prior):
            return self.log_likelihood(params) + prior
        else:
            return prior

    def sample_cosmo(self, nwalkers=30, burnin=1000, sample_iteration=1000):
        # currently hard coded for fLCDM
        ndim = 2
        pos = np.random.normal(loc=[70., 0.3], size=[nwalkers, ndim],
                               scale=[1, 5e-2])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        state, _, _ = sampler.run_mcmc(pos, burnin);

        sampler.reset()
        sampler.run_mcmc(state, sample_iteration);

        return sampler