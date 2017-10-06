
import os
import unittest
import numpy as np
from glob import glob
from scipy.io import loadmat

import free_space


class TestFreeSpace(unittest.TestCase):
    """
    Test class FreeSpace.
    """
    def test_init(self):
        """ Test init."""
        fs = free_space.FreeSpace()

    def test_step_vel0(self):
        """

        :return:
        """

        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_1.mat'))[0]
        data = loadmat(file_path)['data']

        fs = free_space.FreeSpace(sample_rate=data['fs'][0, 0].astype('float'))
        y = fs.step(**get_data_dict(data))
        print((y - data['y'][0, 0].astype('complex')) / np.abs(y))
        np.testing.assert_array_equal(y, data['y'][0, 0].astype('complex'))

    def test_step_delay_more1(self):
        """

        :return:
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_delay_more1.mat'))[0]
        data = loadmat(file_path)['data']

        fs = free_space.FreeSpace(sample_rate=data['fs'][0, 0].astype('float'))
        y = fs.step(**get_data_dict(data))
        print((y - data['y'][0, 0].astype('complex')) / np.abs(y))
        np.testing.assert_array_equal(y, data['y'][0, 0].astype('complex'))

    def test_step_delay_more1_big_speed(self):
        """

        :return:
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_delay_more1_big_speed.mat'))[0]
        data = loadmat(file_path)['data']

        fs = free_space.FreeSpace(sample_rate=data['fs'][0, 0].astype('float'))
        y = fs.step(**get_data_dict(data))
        print((y - data['y'][0, 0].astype('complex')) / np.abs(y))
        np.testing.assert_array_equal(y, data['y'][0, 0].astype('complex'))

    def test_step_big_speed(self):
        """

        :return:
        """
        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        file_path = glob(os.path.join(path_to_mat_files, 'data_big_speed.mat'))[0]
        data = loadmat(file_path)['data']

        fs = free_space.FreeSpace(sample_rate=data['fs'][0, 0].astype('float'))
        y = fs.step(**get_data_dict(data))
        print(([np.real(y - data['y'][0, 0].astype('complex')), np.imag(y - data['y'][0, 0].astype('complex'))]/np.array([np.real(y), np.imag(y)])))
        np.testing.assert_array_equal(y, data['y'][0, 0].astype('complex'))


    def test_open_mat_file(self):

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

def get_data_dict(data):
    """

    :param data:
    :return:
    """
    data_dict = dict(signal = data['signal'][0, 0].astype('float'),
         dist_pos=data['dist_pos'][0, 0].astype('float'),
         dist_vel=data['dist_vel'][0, 0].astype('float'),
         origin_pos=data['origin_pos'][0, 0].astype('float'),
         origin_vel=data['origin_vel'][0, 0].astype('float'))

    return data_dict
