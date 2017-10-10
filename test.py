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

    def test_all_data(self):
        """
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        for file_path in glob(os.path.join(path_to_mat_files, 'data_*.mat')):
            data = loadmat(file_path)['data']

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
                1e-9, msg=file_path)

    def test_step_vel0(self):
        """
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_1.mat'))[0]
        data = loadmat(file_path)['data']

        free_space = FreeSpace(sample_rate=data['fs'])
        y = free_space.step(**get_data_dict(data))
        print((y - data['y'][0, 0].astype('complex')) / np.abs(y))
        # np.testing.assert_array_equal(y, data['y'][0, 0].astype('complex'))
        res = np.abs(
            [np.real(y - data['y'][0, 0].astype('complex')),
             np.imag(y - data['y'][0, 0].astype('complex'))] / np.array([np.real(y), np.imag(y)]))
        res[np.isnan(res)] = 0
        self.assertLess(
            np.max(res),
            1e-9)

    def test_step_delay_more1(self):
        """
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_delay_more1.mat'))[0]
        data = loadmat(file_path)['data']

        free_space = FreeSpace(sample_rate=data['fs'][0, 0].astype('float'))
        y = free_space.step(**get_data_dict(data))
        print((y - data['y'][0, 0].astype('complex')) / np.abs(y))
        # np.testing.assert_array_equal(y, data['y'][0, 0].astype('complex'))
        res = np.abs(
            [np.real(y - data['y'][0, 0].astype('complex')),
             np.imag(y - data['y'][0, 0].astype('complex'))] / np.array([np.real(y), np.imag(y)]))
        res[np.isnan(res)] = 0
        self.assertLess(
            np.max(res),
            1e-9)

    def test_step_delay_more1_big_speed(self):
        """
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_delay_more1_big_speed.mat'))[0]
        data = loadmat(file_path)['data']

        free_space = FreeSpace(sample_rate=data['fs'][0, 0].astype('float'))
        y = free_space.step(**get_data_dict(data))
        print((y - data['y'][0, 0].astype('complex')) / np.abs(y))
        # np.testing.assert_array_equal(y, data['y'][0, 0].astype('complex'))
        res = np.abs(
            [np.real(y - data['y'][0, 0].astype('complex')),
             np.imag(y - data['y'][0, 0].astype('complex'))] / np.array([np.real(y), np.imag(y)]))
        res[np.isnan(res)] = 0
        self.assertLess(
            np.max(res),
            1e-9)

    def test_step_big_speed(self):
        """
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_big_speed.mat'))[0]
        data = loadmat(file_path)['data']

        free_space = FreeSpace(sample_rate=data['fs'][0, 0].astype('float'))
        y = free_space.step(**get_data_dict(data))
        res = np.abs(
            [np.real(y - data['y'][0, 0].astype('complex')),
             np.imag(y - data['y'][0, 0].astype('complex'))]/np.array([np.real(y), np.imag(y)]))
        print(res)
        res[np.isnan(res)] = 0
        self.assertLess(
            np.max(res),
            1e-9)
        # p.testing.assert_array_equal(y - data['y'][0, 0].astype('complex'))

    def test_step_two_target(self):
        """
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_3.mat'))[0]
        data = loadmat(file_path)['data']

        free_space = FreeSpace(sample_rate=data['fs'])

        y = free_space.step(**get_data_dict(data))
        res = np.abs(
            [np.real(y - data['y'][0, 0].astype('complex')),
             np.imag(y - data['y'][0, 0].astype('complex'))]/np.array([np.real(y), np.imag(y)]))
        res[np.isnan(res)] = 0
        self.assertLess(
            np.max(res),
            1e-9)

    def test_step_radar_exapmle(self):
        """
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_example.mat'))[0]
        data = loadmat(file_path)['data']

        free_space = FreeSpace(sample_rate=data['fs'][0, 0].astype('float'),
                               operating_frequency=data['fop'][0, 0].astype('float'))

        y = free_space.step(**get_data_dict(data))
        res = np.abs(
            [np.real(y - data['y'][0, 0].astype('complex')),
             np.imag(y - data['y'][0, 0].astype('complex'))] / np.array([np.real(y), np.imag(y)]))
        res[np.isnan(res)] = 0
        print(np.sum(res))
        print(np.abs(
            [np.real(y - data['y'][0, 0].astype('complex')),
             np.imag(y - data['y'][0, 0].astype('complex'))] / np.array([np.real(y), np.imag(y)])))
        self.assertLess(
            np.max(res),
            1e-9)

    def test_open_mat_file(self):
        """
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
    """
    """
    def test_linear_interpolation(self):
        """ Test init."""
        test = np.array([[0], [0], [1], [2]])
        delay = np.array([0.3])
        res = ut.linear_interpolation(test, delay)
        np.testing.assert_array_equal([[0], [0], [0.7], [1.71324234]], res)


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
