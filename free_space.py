""" Matlab FreeSpace class"""

import time
import numpy as np
from scipy import constants

import utility as ut


DEBUG = True


def timer(func):
    """ Timer. """
    def wrapper(*args, **kw):
        """ Wrapper. """
        start_time = time.time()
        res = func(*args, **kw)
        if DEBUG is True:
            print('Время выполнения функции {}: {}'.format(func.__name__,
                                                           time.time() - start_time))
        return res
    return wrapper


class FreeSpace(object):
    """Matlab FreeSpace class"""

    def __init__(self,
                 propagation_speed=constants.speed_of_light,
                 operating_frequency=3e8,
                 two_way_propagation=False,
                 sample_rate=1e6,
                 maximum_distance_source='auto',
                 maximum_distance=1e4,
                 maximum_num_input_samples_source='auto',
                 maximum_num_input_samples=1e2):
        """
        :param propagation_speed:  Signal propagation speed
            Specify signal wave propagation speed in free space as a real positive scalar.
            Units are meters per second.
            Default: Speed of light
        :param operating_frequency: Signal carrier frequency
            A scalar containing the carrier frequency of the narrowband signal.
            Units are hertz.
            Default: 3e8
        :param two_way_propagation: Perform two-way propagation
            Set this property to True to perform round-trip propagation between
            the origin and destination that you specify in the step command.
            Set this property to false to perform one-way propagation from the
            origin to the destination.
            Default: False
        :param sample_rate: Sample rate
            A scalar containing the sample rate. Units of sample rate are hertz.
            The algorithm uses this value to determine the propagation delay in
            number of samples.
            Default: 1e6
        :param maximum_distance_source: Source of maximum number of samples
            Default: 'auto'
        :param maximum_distance: Maximum one-way propagation distance
            Default: 10000
        :param maximum_num_input_samples_source: Source of maximum number of samples
            Default: 'auto'
        :param maximum_num_input_samples: Maximum number of input signal samples
            Default: 100
        """
        self.propagation_speed = self._get_float(propagation_speed,
                                                 'propagation_speed')
        self.operating_frequency = self._get_float(operating_frequency,
                                                   'operating_frequency')
        self.two_way_propagation = two_way_propagation
        self.sample_rate = self._get_float(sample_rate, 'sample_rate')
        self.maximum_distance_source = maximum_distance_source
        self.maximum_distance = self._get_float(maximum_distance,
                                                'maximum_distance')
        self.maximum_num_input_samples_source = maximum_num_input_samples_source
        self.maximum_num_input_samples = self._get_float(maximum_num_input_samples,
                                                         'maximum_num_input_samples')
        self.lambda_ = self.propagation_speed / self.operating_frequency
        self._buffer = None

    @timer
    def step(self, signal, origin_pos, dist_pos, origin_vel, dist_vel):
        """
        :param signal: M-element complex-valued column vector
        :param origin_pos: Origin of the signal or signals, specified as a 3-by-1
            real-valued column vector.
            Position units are meters.
        :param dist_pos:
            Destination of the signal or signals, specified as a 3-by-N.
            Position units are meters
        :param origin_vel:
            Velocity of signal origin, specified as a 3-by-1 column vector.
            Velocity units are meters/second.
        :param dist_vel:
            Velocity of signal destinations, specified as a 3-by-N.
            Velocity units are meters/second.
        :return:
            Propagated signal, returned as a M-element complex-valued column vector
        """
        signal = self._get_complex_array(signal, 'signal')
        origin_pos = self._get_float_array(origin_pos, 'origin_pos')
        dist_pos = self._get_float_array(dist_pos, 'dist_pos')
        origin_vel = self._get_float_array(origin_vel, 'origin_vel')
        dist_vel = self._get_float_array(dist_vel, 'dist_vel')
        if self._check_shape(signal, origin_pos, dist_pos, origin_vel, dist_vel):
            return self._compute_multiple_propagated_signal(
                signal, origin_pos, dist_pos, origin_vel, dist_vel)

    def _push_buffer(self, signal):
        """

        :param signal:
        :return:
        """
        if self._buffer is None:
            self._buffer = signal
        else:
            self._buffer = np.concatenate((self._buffer, signal), axis=0)

    def _pull_buffer(self, number_of_rows, shape):
        """
        Return count last row from buffer.
        :param number_of_rows: N last row.
        :param shape: shape ndarray.
        :return: ndarray with shape == (number_of_rows, shape[1]) from self._buffer
        """
        if self._buffer is None:
            return np.zeros((number_of_rows, shape[1]))

        if number_of_rows == 0:
            return np.zeros((0, shape[1]))

        # Get number_of_rows last row, Add zeros if need
        buffered = self._buffer[-number_of_rows:, :]
        if buffered.shape[0] < number_of_rows:
            buffered = np.concatenate(
                (np.zeros((number_of_rows - buffered.shape[0], buffered.shape[1])),
                 buffered), axis=0)
        return buffered


    def _compute_multiple_propagated_signal(self, signal, origin_pos,
                                            dist_pos, origin_vel, dist_vel):
        """
        Compute multiple_propagated_signal.
        :param signal: M-by-N element complex-valued column vector
        :param origin_pos: Origin of the signal or signals, specified as a 3-by-1
            real-valued column vector.
            Position units are meters.
        :param dist_pos:
            Destination of the signal or signals, specified as a 3-by-N.
            Position units are meters
        :param origin_vel:
            Velocity of signal origin, specified as a 3-by-N column vector.
            Velocity units are meters/second.
        :param dist_vel:
            Velocity of signal destinations, specified as a 3-by-N.
            Velocity units are meters/second.
        :return:
        """
        two_way_factor = self._get_range_factor()
        #Get delay, distance, r-speed.

        prop_distance = self._compute_propagation_distance(origin_pos, dist_pos)
        prop_delay = self._compute_propagation_delay(prop_distance, two_way_factor)
        rspeed = self._compute_radial_velocity(origin_pos, dist_pos,
                                               origin_vel, dist_vel)

        #Loss of signal.
        sploss = two_way_factor * ut.fspl(prop_distance, self.lambda_)
        plossfactor = np.sqrt(ut.db2pow(sploss))

        #Get time step matrix for all targets.
        time_step = self._get_time_step(signal.shape)

        # Get result signal values without delay.
        result_signal = self._compute_signal(signal, plossfactor, rspeed,
                                             prop_distance, prop_delay, time_step)

        sample_delay = prop_delay * self.sample_rate
        # Get old signal from buffer.
        # Need add 1 for integer sample_delay value.
        buffered_signal = self._pull_buffer(np.max(np.floor(sample_delay)).astype('int') + 1,
                                            signal.shape)

        buffer_time_step = self._get_time_step(buffered_signal.shape, for_buffer=True)
        result_buffered_signal = self._compute_signal(buffered_signal, plossfactor, rspeed,
                                                      prop_distance, prop_delay, buffer_time_step)

        # Compute signal values with delay. Use linear interpolation.
        result_signal_after_interpolation = ut.linear_interpolation(
            result_signal, sample_delay, result_buffered_signal)

        # Push signal to buffer.
        self._push_buffer(signal)

        return result_signal_after_interpolation

    def _compute_signal(self, signal, plossfactor, rspeed, prop_distance,
                        prop_delay, time_step):
        """
        Compute result signal.
        :param signal: ndarray
        :param two_way_factor: {1 or 2}
        :param plossfactor: ndarray
        :param rspeed: ndarray
        :param prop_distance: ndarray
        :param prop_delay: ndarray
        :return:
        """
        two_way_factor = self._get_range_factor()
        result_signal = signal / plossfactor * np.exp(
            1j * 2 * np.pi * two_way_factor * rspeed / self.lambda_ * (
                prop_delay + time_step)
        ) * np.exp(
            -1j * 2 * np.pi * two_way_factor * prop_distance / self.lambda_)
        return result_signal

    def _get_time_step(self, shape, for_buffer=False):
        """ Get time step matrix for all targets.
            If 'for_buffer' is True, return time step matrix for old values.
        :param shape: signal shape.
        :return:
        """
        current_taсt = 0
        if self._buffer is not None:
            current_taсt = self._buffer.shape[0]

        if for_buffer:
            time_step = np.arange(current_taсt - shape[0],
                                  current_taсt) / self.sample_rate
        else:
            time_step = np.arange(current_taсt,
                                  current_taсt + shape[0]) / self.sample_rate
        time_step = time_step.reshape((shape[0], 1)) * np.ones(shape)
        return time_step

    def _get_range_factor(self):
        """
        Check self.two_way_propagation.
        :return: 2 or 1 way.
        """
        if self.two_way_propagation is True:
            return 2

        return 1

    @staticmethod
    def _compute_radial_velocity(origin_pos, dist_pos, origin_vel, dist_vel):
        """
        Compute radial speed.
        :param origin_pos: Origin of the signal or signals, specified as a 3-by-1
            real-valued column vector.
            Position units are meters.
        :param dist_pos:
            Destination of the signal or signals, specified as a 3-by-N.
            Position units are meters
        :param origin_vel:
            Velocity of signal origin, specified as a 3-by-1 column vector.
            Velocity units are meters/second.
        :param dist_vel:
            Velocity of signal destinations, specified as a 3-by-N.
            Velocity units are meters/second.
        :return: rspeed
        """
        rspeed = ut.calc_radial_speed(origin_pos, dist_pos, origin_vel, dist_vel)
        return rspeed

    def _compute_propagation_delay(self, prop_distance, two_way_factor):
        """
        Compute propagation delay.
        :param prop_distance: distance between targets and source.
        :param two_way_factor:
        :return: delay
        """
        prop_delay = two_way_factor * prop_distance / self.propagation_speed
        return prop_delay

    @staticmethod
    def _compute_propagation_distance(origin_pos, dist_pos):
        """
        Compute propdelay, prop_distance, rspeed.
        :param origin_pos: Origin of the signal or signals, specified as a 3-by-1
            real-valued column vector.
            Position units are meters.
        :param dist_pos:
            Destination of the signal or signals, specified as a 3-by-N.
            Position units are meters
        :return: prop_distance
        """
        # prop_distance = np.sqrt(np.sum(np.power(origin_pos - dist_pos, 2), 0))
        # Faster
        prop_distance = np.linalg.norm(origin_pos - dist_pos, axis=0)
        if not isinstance(prop_distance, np.ndarray):
            prop_distance = np.array([[prop_distance]])
        return prop_distance

    @staticmethod
    def _check_shape(signal, origin_pos, dist_pos, origin_vel, dist_vel):
        """"Check data shapes."""
        shapes = [origin_pos.shape,
                  dist_pos.shape,
                  origin_vel.shape,
                  dist_vel.shape]
        unique_shapes = list(set(shapes))
        if len(unique_shapes) == 1:
            try:
                if unique_shapes[0][1] == signal.shape[1]:
                    return True
                else:
                    raise Exception(("Signal, origin_pos, dist_pos, origin_vel,"
                                     "dist_vel should have the same second dimension."
                                     "\n{}").format([signal.shape] + shapes))
            except IndexError:
                raise Exception(("Signal, origin_pos, dist_pos, origin_vel,"
                                 "dist_vel should be a two-dimensional matrix."
                                 "\n{}").format([signal.shape] + shapes))

        elif shapes[0] == shapes[2] and shapes[1] == shapes[3]:
            try:
                if shapes[1][1] == signal.shape[1]:
                    return True
                else:
                    raise Exception(("Signal, origin_pos, dist_pos, origin_vel,"
                                     "dist_vel should have the same second dimension."
                                     "\n{}").format([signal.shape] + shapes))
            except IndexError:
                raise Exception(("Signal, origin_pos, dist_pos, origin_vel,"
                                 "dist_vel should be a two-dimensional matrix."
                                 "\n{}").format([signal.shape] + shapes))

        raise Exception(("Origin_pos, dist_pos, origin_vel, "
                         "dist_vel haves  different shapes {}").format((shapes)))

    @staticmethod
    def _get_float(value_, attribute_name):
        try:
            return float(value_)
        except TypeError as except_:
            raise type(except_)(
                'Error when try set {}.\nMessage: {}'.format(attribute_name,
                                                             str(except_)))

    @staticmethod
    def _get_float_array(array_, attribute_name):
        try:
            return np.array(array_, dtype='float')
        except TypeError as except_:
            raise type(except_)(
                'Error when try set {}.\nMessage: {}'.format(attribute_name,
                                                             str(except_)))
        except ValueError as except_:
            raise type(except_)(
                'Error when try set {}.\nMessage: {}'.format(attribute_name,
                                                             str(except_)))

    @staticmethod
    def _get_complex_array(array_, attribute_name):
        try:
            return np.array(array_, dtype='complex')
        except TypeError as except_:
            raise type(except_)(
                'Error when try set {}.\nMessage: {}.'.format(attribute_name,
                                                              str(except_)))
        except ValueError as except_:
            raise type(except_)(
                'Error when try set {}.\nMessage: {}.'.format(attribute_name,
                                                              str(except_)))
