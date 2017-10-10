"""Tests for FreeSpace."""

import os
import unittest
import warnings
from glob import glob

import numpy as np
from scipy.io import loadmat

from free_space import FreeSpace
import utility as ut

warnings.filterwarnings('ignore')


class TestFreeSpace(unittest.TestCase):
    """
    Test class FreeSpace.
    """
    def test_init(self):
        """ Test init."""
        free_space = FreeSpace()
        self.assertTrue(free_space)

    def test_all_data(self):
        """ Check results for all data_*.mat in test_data.
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        for file_path in glob(os.path.join(path_to_mat_files, 'data_*.mat')):
            data = loadmat(file_path)['data']
            print(file_path)
            free_space = FreeSpace(**get_data_dict_for_free_space(data))
            result_signal = free_space.step(**get_data_dict(data))
            matlab_result_signal = data['y'][0, 0].astype('complex')
            # np.testing.assert_array_equal(y, matlab_result_signal)
            res = np.abs(
                [np.real(result_signal - matlab_result_signal),
                 np.imag(result_signal - matlab_result_signal)] /
                np.array([np.real(np.mean([result_signal, matlab_result_signal], axis=0)),
                          np.imag(np.mean([result_signal, matlab_result_signal], axis=0))]))

            res[np.isnan(res)] = 0
            self.assertLess(
                np.max(res),
                1e-9, msg=(file_path, 'mean_difference=', np.mean(res)))

    def test_all_example_data(self):
        """ Check results for all data_exmpl_*.mat in test_data.
            Data create in f-file example.mat.
        """
        free_space = None
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        for file_path in sorted(glob(os.path.join(path_to_mat_files, 'data_exmpl_*.mat'))):
            print(file_path)
            data = loadmat(file_path)['data']
            if free_space is None:
                free_space = FreeSpace(**get_data_dict_for_free_space(data))
            result_signal = free_space.step(**get_data_dict(data))
            matlab_result_signal = data['y'][0, 0].astype('complex')
            # np.testing.assert_array_equal(y, matlab_result_signal)
            res = np.abs(
                [np.real(result_signal - matlab_result_signal),
                 np.imag(result_signal - matlab_result_signal)] /
                np.array([np.real(np.mean([result_signal, matlab_result_signal], axis=0)),
                          np.imag(np.mean([result_signal, matlab_result_signal], axis=0))]))

            res[np.isnan(res)] = 0
            self.assertLess(
                np.max(res),
                1e-9, msg=(file_path, 'mean_difference=', np.mean(res)))

    def test_random_initialize(self):
        """ Check results for all data_*.mat in test_data.
        """
        free_space = FreeSpace()
        signal_shape = (int(1e7), 1)
        shape = (3, 1)
        result_signal = free_space.step(signal=np.random.random(signal_shape),
                                        dist_pos=np.random.randint(0, 1e4, shape),
                                        dist_vel=np.random.randint(0, 1e4, shape),
                                        origin_pos=np.random.randint(0, 1e4, shape),
                                        origin_vel=np.random.randint(0, 1e4, shape))
        self.assertTrue(result_signal is not None, result_signal.shape)

    def test_long_signal(self):
        """ Check results for all data_*.mat in test_data.
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        for file_path in glob(os.path.join(path_to_mat_files, 'data_signal_1e7.mat')):
            print(file_path)
            data = loadmat(file_path)['data']

            free_space = FreeSpace(**get_data_dict_for_free_space(data))
            result_signal = free_space.step(**get_data_dict(data))
            matlab_result_signal = data['y'][0, 0].astype('complex')
            # np.testing.assert_array_equal(y, matlab_result_signal)
            res = np.abs(
                np.array([np.real(result_signal - matlab_result_signal),
                          np.imag(result_signal - matlab_result_signal)]) /
                np.array([np.real(np.mean([result_signal, matlab_result_signal], axis=0)),
                          np.imag(np.mean([result_signal, matlab_result_signal], axis=0))]))

            res[np.isnan(res)] = 0
            self.assertLess(
                np.max(res),
                1e-9, msg=(np.mean(res)))

    def test_open_mat_file(self):
        """Test load mat-file.
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        for file_path in glob(os.path.join(path_to_mat_files, 'data_1.mat')):
            data = loadmat(file_path)['data']
            self.assertTrue(data['fs'][0, 0][0, 0] == 8000)
            self.assertTrue(data['fop'][0, 0][0, 0] == 3e8)
            np.testing.assert_array_equal(
                data['signal'][0, 0], np.ones((5, 1)))
            np.testing.assert_array_equal(
                data['origin_pos'][0, 0], np.array([[1000], [0], [0]]))
            np.testing.assert_array_equal(
                data['origin_vel'][0, 0], np.array([[0], [0], [0]]))
            np.testing.assert_array_equal(
                data['dist_vel'][0, 0], np.array([[0], [0], [0]]))
            np.testing.assert_array_equal(
                data['dist_pos'][0, 0], np.array([[3000], [200], [50]]))
            np.testing.assert_array_equal(
                np.around(data['y'][0, 0], decimals=8),
                np.around(1e-4 * np.array(
                    [[0.3734 + 0.0262j],
                     [0.3945 + 0.0277j],
                     [0.3945 + 0.0277j],
                     [0.3945 + 0.0277j],
                     [0.3945 + 0.0277j]]), decimals=8))


class TestUtility(unittest.TestCase):
    """ Test utilitu.
    """
    def test_linear_interpolation(self):
        """ Check result for linear_interpolation."""
        test = np.array([[0], [0], [1], [2]])
        delay = np.array([0])
        res = ut.linear_interpolation(test, delay)
        np.testing.assert_array_equal([[0], [0], [1], [2]], res)
        delay = np.array([1])
        res = ut.linear_interpolation(test, delay)
        np.testing.assert_array_equal([[0], [0], [0], [1]], res)
        delay = np.array([0.3])
        res = ut.linear_interpolation(test, delay)
        np.testing.assert_array_equal([[0], [0], [0.7], [1.7]], res)

    def test_fspl(self):
        """ Test utility.fspl. """
        distance = np.array([[1000000]])
        lambda_ = np.array([[3e8 / 1e3]])
        self.assertTrue(ut.fspl(distance, lambda_))

    def test_db2pow(self):
        """ Test utility.db2pow. """
        self.assertEqual(ut.db2pow(20), 100)

    def test_mag2db(self):
        """ Test utility.mag2db. """
        self.assertEqual(
            ut.mag2db(np.array([[1000]], dtype='float')), 60)

    def test_calc_radial_speed(self):
        """ Test utility.calc_radial_speed. """
        origin_pos = np.array([[1]])
        dist_pos = np.array([[1000]])
        origin_vel = np.array([[2]])
        dist_vel = np.array([[2000]])
        self.assertEqual(
            ut.calc_radial_speed(
                origin_pos, dist_pos, origin_vel, dist_vel),
            -1998)


def get_data_dict(data):
    """
    :param data:
    :return:
    """
    data_dict = dict(signal=data['signal'][0, 0],
                     dist_pos=data['dist_pos'][0, 0],
                     dist_vel=data['dist_vel'][0, 0],
                     origin_pos=data['origin_pos'][0, 0],
                     origin_vel=data['origin_vel'][0, 0])

    return data_dict


def get_data_dict_for_free_space(data):
    """
    :param data:
    :return:
    """
    data_dict = dict(sample_rate=data['fs'][0, 0],
                     operating_frequency=data['fop'][0, 0])

    return data_dict
