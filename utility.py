""" Utility."""

import numpy as np


def linear_interpolation(data, delay):
    """ Compute linear interpolation for array.
        Compute by columns. Finish concatenate columns.
    :param data: np.array with shape(n, m)
    :param delay: np.array with shape (m,)
    :return:
    """
    full_result = None
    for target_index in range(data.shape[1]):
        int_delay = np.floor(delay[target_index]).astype('int')
        remaind_float_delay = delay[target_index] - int_delay
        step = int_delay + 1

        try:
            result = np.concatenate(
                (np.zeros((step, 1)),
                 data[:, target_index].reshape(data.shape[0], 1)),
                axis=0)
        # Negative dimensions are not allowed. Delay < 0.
        except ValueError as except_:
            raise type(except_)(str(except_) + '\nInteger delay is {}'.format(int_delay))

        # If delay == 0
        if step == 0:
            previous_data = result[:data.shape[0]]
        else:
            previous_data = result[1:data.shape[0]+1]
        current_data = result[:data.shape[0]]
        result = (1 - remaind_float_delay)*previous_data + remaind_float_delay*current_data

        if full_result is None:
            full_result = result
        else:
            full_result = np.concatenate((full_result, result), axis=1)
    return full_result


def fspl(distance, lambda_):
    """ Free space path loss
        loss = fspl(R,LAMBDA) returns the free space path loss loss (in dB) suffered
        by a signal with wavelength LAMBDA (in meters) when it is propagated in
        free space for a distance of R (in meters). R can be a length-M vector
        and LAMBDA can be a length-N vector. loss has the same dimensionality as
        MxN. Each element in loss is the free space path loss for the
        corresponding propagation distance specified in R.

        Note that the best case is lossless so the loss is always greater than
        or equal to 0 dB.

        Example:
        Calculate the free space loss for a signal whose wavelength is 30
        cm. The signal is propagated for 1 km.
            loss = fspl(1000,0.3)
        See also phased, phased.FreeSpace.
        Reference
        [1] John Proakis, Digital Communications, 4th Ed., McGraw-Hill, 2001

    :param distance: numpy.array
    :param lambda_: numpy.array
    :return:
    """
    loss = 4 * np.pi * distance / lambda_
    loss = validate_loss(loss)
    loss = mag2db(loss)
    return loss


def validate_loss(loss):
    """ Change value < 1 to 1.
    :param loss: numpy.array
    :return:
    """
    loss[loss < 1] = 1
    return loss


def mag2db(magnitude):
    """ MAG2DB  Magnitude to dB conversion.
        YDB = MAG2DB(Y) converts magnitude data Y into dB values.
        Negative values of Y are mapped to NaN.
        See also DB2MAG.
        Copyright 1986-2011 The MathWorks, Inc.
    :param magnitude: numpy.array
    :return:
    """

    magnitude[magnitude < 0] = float('nan')
    db_value = 20*np.log10(magnitude)
    return db_value


def db2pow(db_value):
    """ DB2POW   dB to Power conversion
        Y = DB2POW(YDB) converts dB to its corresponding power value such that
        10*log10(Y)=YDB

        Example:
        Convert 12dB to Power.
          power = db2pow(12)
        Copyright 2006 The MathWorks, Inc.
    :param db_value: numpy.array
    :return:
    """
    power = np.power(10, db_value / 10)
    return power


def calc_radial_speed(origin_pos, dist_pos, origin_vel, dist_vel):
    """ Compute radial speed
       RSPEED = calcRadialSpeed(POS,VEL,REFPOS,REFVEL) compute the relative
       speed RSPEED (in m/s) for a target at position POS (in meters) with a
       velocity VEL (in m/s) relative to the reference position REFPOS (in
       meters) and reference velocity REFVEL (in m/s).
       This is the same functionality as radialspeed function. However,
       because we already done the input validation here, we want to skip the
       validation to improve the performance. In addition, here all position
       and velocity are always column vectors and the target and reference can
       never be colocated, so it simplifies the computation too.

    :param origin_pos: numpy.array
    :param dist_pos: numpy.array
    :param origin_vel: numpy.array
    :param dist_vel: numpy.array
    :return:
    """

    tgt_direc = dist_pos - origin_pos
    vel_direc = dist_vel - origin_vel
    # Get distance between targets and source
    # rn = np.sqrt(np.sum(np.power(tgtdirec, 2), 0))
    # Faster then np.sqrt(np.sum(np.power(tgtdirec, 2), 0))
    distance = np.linalg.norm(tgt_direc, axis=0)

    # Negative sign to ensure that incoming relative speed is positive
    # Vr = (V, R)/|R|
    rspeed = -1 * np.sum((vel_direc * tgt_direc) / distance, 0)
    return rspeed
