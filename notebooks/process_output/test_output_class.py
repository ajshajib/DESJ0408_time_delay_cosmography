import numpy as np
import numpy.testing as npt

from output_class import ModelOutput


# def test_ModelOutput():
#     """
#     Test the class `ModelOutput`.
#     """
#     print("Testing `class ModelOutput`")
#
#     model_output = ModelOutput('0408_run59_0_0_0_0_0_0_0_0', "powerlaw")
#
#     assert np.isclose(model_output.r_eff, 0.6915787162716911)
#
#     model_output.compute_model_time_delays(100)
#     assert model_output.model_time_delays.shape == (100, 3)
#
#     model_output.populate_Ddt(1000)
#     assert model_output.Ddt_sample.shape == (
#     3, len(model_output.model_time_delays) * 1000)
#
#     ##########
#
#     model_output = ModelOutput('0408_run59_0_0_0_0_0_0_0_0', "powerlaw")
#
#     model_output.get_Ddt_posterior(1000, 1000, num_lens_model=100)
#     assert model_output.model_time_delays.shape == (100, 3)
#     assert model_output.Ddt_sample.shape == (
#     3, len(model_output.model_time_delays) * 1000)
#
#     ##########
#
#     model_output.compute_model_velocity_dispersion(2)
#     assert model_output.model_velocity_dispersion.shape == (2, 3)
#
#     model_output.populate_Dsds(100)
#     assert model_output.Dsds_sample.shape == (
#     3, len(model_output.model_velocity_dispersion) * 100)
#
#     ##########
#     model_output = ModelOutput('0408_run59_0_0_0_0_0_0_0_0', "powerlaw")
#
#     model_output.get_Ddt_posterior(1000, 1000, num_lens_model=100)
#     model_output.get_Dd_Ddt_posterior(100, 1000, num_lens_model=2)
#
#     assert model_output.Ddt_Dd_posterior.shape == (2, 100)
#
#     print('All tests passed!')


# def test_ModelOutput_long():
#     """
#     """
#     print('Extensive testing...')
#     test_model = ModelOutput('cosmo_test_0_0_0_0_0_mod', "powerlaw",
#                              is_test=True)
#
#     td = np.array([-243.14971541, -147.96269473, -195.21014112, -206.07134407])
#     test_model.TIME_DELAYS = np.array(
#         [td[0] - td[1], td[0] - td[3], td[1] - td[3]])
#     test_model.SIGMA_DELAYS = np.array([1., 1, 1.])
#
#     test_model.VEL_DIS = np.array([322.31225448, 323.91335883, 323.54408908])
#     test_model.SIG_VEL_DIS = np.array([1, 1, 1]) * 0.5644663440116261
#
#     test_model.PSF_FWHM = np.array([1., 1., 1.])
#     test_model.SLIT_WIDTH = np.array([1., 1., 1.])
#     test_model.APERTURE_LENGTH = np.array([1., 1., 1.])
#
#     test_model.get_Ddt_posterior(num_sample=30000,
#                                  lens_model_posterior_mult=10000,
#                                  num_lens_model=3000)
#
#     Ddt_ratio = np.median(test_model.Ddt_posterior)
#     # assert np.abs(70./Ddt_ratio - 65.) < 0.15
#     print(70. / Ddt_ratio)
#     npt.assert_almost_equal(70. / Ddt_ratio, 65., decimal=1)
#
#     test_model.get_Dd_Ddt_posterior(num_sample=50,
#                                     lens_model_posterior_mult=10000,
#                                     num_lens_model=50)
#     Dsds_ratio = np.median(
#         test_model.Ddt_Dd_posterior[0] / test_model.Ddt_Dd_posterior[1])
#     print(Dsds_ratio)
#     npt.assert_almost_equal(Dsds_ratio, 1., decimal=2)
#
#     print('Extensive testing passed!')


# def test_ModelOutput_long():
#     """
#     """
#     print('Extensive testing...')
#     test_model = ModelOutput('test_output copy.txt', "powerlaw",
#                              is_test=True)
#
#     assert test_model.get_num_samples() == 5000
#
#     td = np.array([-144.18503279,  -75.13095228, -110.81730384, -118.84713161])
#     test_model.TIME_DELAYS = np.array(
#         [td[0] - td[1], td[0] - td[3], td[1] - td[3]])
#     test_model.SIGMA_DELAYS = np.array([1., 1., 1.])
#
#     test_model.VEL_DIS = np.array([288.28001364, 286.99466028, 285.03573312])
#     test_model.SIG_VEL_DIS = np.array([1, 1, 1]) * 10.
#
#     test_model.PSF_FWHM = np.array([.5, .75, 1.])
#     test_model.SLIT_WIDTH = np.array([1., 1., 1.])
#     test_model.APERTURE_LENGTH = np.array([1., 1., 1.])
#
#     test_model.get_Ddt_posterior(num_sample=50,
#                                  lens_model_posterior_mult=10000,
#                                  num_lens_model=50)
#
#     Ddt_ratio = np.median(test_model.Ddt_posterior)
#     # assert np.abs(70./Ddt_ratio - 65.) < 0.15
#     print(70. / Ddt_ratio, Ddt_ratio)
#     npt.assert_almost_equal(70 / Ddt_ratio, 65., decimal=1)
#
#     test_model.get_Dd_Ddt_posterior(num_sample=100,
#                                     lens_model_posterior_mult=10000,
#                                     num_lens_model=100)
#     Dsds_ratio = np.median(
#         test_model.Ddt_Dd_posterior[0] / test_model.Ddt_Dd_posterior[1])
#     print(Dsds_ratio)
#
#     npt.assert_almost_equal(Dsds_ratio, 1.0166440728064097, decimal=1)
#
#     print('Extensive testing passed!')

def test_new_functions():
    powerlaw_output_files = ['test_output_copy.txt']

    powerlaw_models = []

    for job_name_out in powerlaw_output_files:  # [-2:]:
        powerlaw_models.append(
            ModelOutput(job_name_out, "powerlaw", is_test=True))

    test_model = powerlaw_models[0]
    td = np.array([-144.18503279, -75.13095228, -110.81730384, -118.84713161])
    test_model.TIME_DELAYS = np.array(
        [td[0] - td[1], td[0] - td[3], td[1] - td[3]])
    test_model.SIGMA_DELAYS = np.array([1., 1., 1.])

    test_model.VEL_DIS = np.array([288.28001364, 286.99466028, 285.03573312])
    test_model.SIG_VEL_DIS = np.array([1, 1, 1]) * 10.

    test_model.PSF_FWHM = np.array([.5, .75, 1.])
    test_model.SLIT_WIDTH = np.array([1., 1., 1.])
    test_model.APERTURE_LENGTH = np.array([1., 1., 1.])

    test_model.compute_model_time_delays()

    test_model.model_velocity_dispersion = np.array(
        [list(test_model.VEL_DIS)] * test_model.get_num_samples()) * np.random.normal(loc=1.0
                                                                         ** 2,
                                                              scale=0.,
                                                              size=(test_model.get_num_samples(), 1))

    test_model.populate_Ddt_Dsds(sample_multiplier=1000, Ddt_high=1.25,
                                 Ddt_low=.75, Dsds_high=2.5, Dsds_low=0.)

    assert test_model._uniform_Dd_Dsds.shape == (2, test_model.get_num_samples()*1000)

    sample = test_model.sample_Ddt_Dd(num_sample=5000)
    print(np.median(sample[0]), np.median(sample[1]))
    npt.assert_almost_equal(70./np.median(sample[0]), 65., decimal=1)
    npt.assert_almost_equal(np.median(sample[0]/sample[1]), 1.,
                                   decimal=2)


if __name__ == '__main__':
    #test_ModelOutput()
    #test_ModelOutput_long()
    test_new_functions()