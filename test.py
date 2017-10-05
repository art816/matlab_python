
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

    def test_step(self):
        """

        :return:
        """
        data = None
        y = None

        path_to_mat_files = os.path.join(os.path.dirname(__file__), 'test_data')
        for file_path in glob(os.path.join(path_to_mat_files, 'data_3.mat')):
            data = loadmat(file_path)['data']

        fs = free_space.FreeSpace(sample_rate=data['fs'][0, 0].astype('float'))
        y = fs.step(signal=data['signal'][0, 0].astype('float'),
                    dist_pos=data['dist_pos'][0, 0].astype('float'),
                    dist_vel=data['dist_vel'][0, 0].astype('float'),
                    origin_pos=data['origin_pos'][0, 0].astype('float'),
                    origin_vel=data['origin_vel'][0, 0].astype('float'))
        np.testing.assert_array_equal(y, data['y'][0, 0].astype('float'))

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

