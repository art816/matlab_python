import scipy
from scipy import constants
import numpy as np
from tqdm import tqdm

import utility as ut

class FreeSpace(object):
    """"""

    def __init__(self, propagation_speed=constants.speed_of_light,
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
            Set this property to false to perform one-way propagation from the origin to the destination.
            Default: False
        :param sample_rate: Sample rate
            A scalar containing the sample rate. Units of sample rate are hertz.
            The algorithm uses this value to determine the propagation delay in number of samples.
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
        self.propagation_speed = self._get_number(propagation_speed)
        self.operating_frequency = self._get_number(operating_frequency)
        self.two_way_propagation = two_way_propagation
        self.sample_rate = self._get_number(sample_rate)
        self.maximum_distance_source = maximum_distance_source
        self.maximum_distance = self._get_number(maximum_distance)
        self.maximum_num_input_samples_source = maximum_num_input_samples_source
        self.maximum_num_input_samples = self._get_number(maximum_num_input_samples)

    def step(self, signal, origin_pos, dist_pos, origin_vel, dist_vel):
        """
        :param signal: M-element complex-valued column vector
        :param origin_pos: Origin of the signal or signals, specified as a 3-by-1 real-valued column vector.
            Position units are meters.
        :param dist_pos:
            Destination of the signal or signals, specified as a 3-by-1.
            Position units are meters
        :param origin_vel:
            Velocity of signal origin, specified as a 3-by-1 column vector.
            Velocity units are meters/second.
        :param dist_vel:
            Velocity of signal destinations, specified as a 3-by-1.
            Velocity units are meters/second.
        :return:
            Propagated signal, returned as a M-element complex-valued column vector
        """
        signal = self._get_complex_array(signal)
        origin_pos = self._get_float_array(origin_pos)
        dist_pos = self._get_float_array(dist_pos)
        origin_vel = self._get_float_array(origin_vel)
        dist_vel = self._get_float_array(dist_vel)
        _lambda = self.propagation_speed / self.operating_frequency
        return self.compute_multiple_propagated_signal(signal, _lambda, origin_pos, dist_pos, origin_vel, dist_vel)


    def compute_multiple_propagated_signal(self, signal, _lambda, origin_pos, dist_pos, origin_vel, dist_vel):
        # y = np.zeros(signal.shape)
        k = self.get_range_factor()

        propdelay, propdistance, rspeed = self.compute_propagation_delay_velocity(origin_pos, dist_pos, origin_vel, dist_vel, k)

        sploss = k * ut.fspl(propdistance, _lambda)

        plossfactor = np.sqrt(ut.db2pow(sploss))
        #TODO
        z = np.array(list(range(signal.shape[0]))) / self.sample_rate
        z = z.reshape((signal.shape[0], 1)) * np.ones(signal.shape)

        y = np.exp(-1j * 2 * np.pi * k * propdistance / _lambda) / \
            plossfactor * np.exp(1j * 2 * np.pi * k * rspeed /_lambda * (propdelay + z)) * signal


        yy = ut.linear_interpolation(y, propdelay * self.sample_rate)
        # for i in range(-1, signal.shape[0] - 1):
        #     v = propdelay[0] * self.sample_rate[0, 0] + i
        #     vi = np.floor(v).astype('int')
        #     vf = v - vi
        #     if vi < signal.shape[0] - 1:
        #         if vi > -1:
        #             yy[i + 1] = ((1 - vf) * y[vi + 1])
        #             yy[i + 1] += (vf * y[vi])
        #             a = yy[i]
        #         elif vi == -1:
        #             yy[i + 1] = (1 - vf) * y[vi + 1]
        #     elif vi == signal.shape[0] - 1:
        #         yy[i + 1] += (vf * y[vi])


        return yy

    def get_range_factor(self):
        """

        :return: if self.two_way_propagation is True
        """
        if self.two_way_propagation is True:
            return 2
        else:
            return 1



    def compute_propagation_delay_velocity(self, origin_pos, dist_pos, origin_vel, dist_vel, k):
        """
        """
        propdistance = np.sqrt(np.sum(np.power(origin_pos - dist_pos, 2), 0))
        if not isinstance(propdistance, np.ndarray):
            propdistance = np.array([[propdistance]])
        rspeed = self.calc_radial_speed(origin_pos, dist_pos, origin_vel, dist_vel)
        propdelay = k * propdistance / self.propagation_speed
        return propdelay, propdistance, rspeed

    def calc_radial_speed(self, origin_pos, dist_pos, origin_vel, dist_vel):
        """CalcRadialSpeed    Compute radial speed
           RSPEED = calcRadialSpeed(POS,VEL,REFPOS,REFVEL) compute the relative
           speed RSPEED (in m/s) for a target at position POS (in meters) with a
           velocity VEL (in m/s) relative to the reference position REFPOS (in
           meters) and reference velocity REFVEL (in m/s).
           This is the same functionality as radialspeed function. However,
           because we already done the input validation here, we want to skip the
           validation to improve the performance. In addition, here all position
           and velocity are always column vectors and the target and reference can
           never be colocated, so it simplifies the computation too.
        """

        tgtdirec = dist_pos - origin_pos
        veldirec = dist_vel - origin_vel
        #Get distance between targets and source
        # rn = np.sqrt(np.sum(np.power(tgtdirec, 2), 0))
        rn = np.linalg.norm(tgtdirec, axis=0)

        # Negative sign to ensure that incoming relative speed is positive
        # Vr = (V, R)/|R|
        rspeed = -1 * np.sum((veldirec * tgtdirec) / rn, 0)
        return rspeed

    def _get_number(self, value_):
        try:
            return float(value_)
        except TypeError as e:
            raise type(e)(str(e) +
                      ' happens at %s' % value_)

    def _get_float_array(self, array_):
        try:
            return np.array(array_, dtype='float')
        except TypeError as e:
            raise type(e)(str(e) +
                      ' happens at %s' % array_)

    def _get_complex_array(self, array_):
        try:
            return np.array(array_, dtype='complex')
        except TypeError as e:
            raise type(e)(str(e) +
                          ' happens at %s' % array_)



    # [xbuf_in,nDelay] = ...
    #                         computeDelayedSignal(obj,complex(tempx),nDelay);
    #                 end
    #                 y_out = step(obj.cBuffer,xbuf_in,nDelay);