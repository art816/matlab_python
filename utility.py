import numpy as np

def linear_interpolation(data, delay):
    """

    :param data:
    :param delay:
    :return:
    """
    full_result = None
    for i in range(delay.shape[0]):
        i_delay = np.floor(delay[i]).astype('int')
        f_delay = delay[i] - i_delay
        if f_delay > 0:
            step = i_delay + 1
        else:
            step = i_delay
        zeros = np.zeros((step, data.shape[1]))
        a = data[:, i]
        result = np.concatenate((np.zeros((step, 1)), data[:, i].reshape(data.shape[0], 1)), axis=0)
        a = result[1:data.shape[0]+1]
        b = result[:data.shape[0]]
        result = (1 - f_delay)*a + f_delay*b
        # rrr = np.iscomplexobj(data[:, i])
        # result2 = np.interp(np.array(range(data.shape[0])) - delay[i], range(data.shape[0]), np.real(data[:, i]))
        if full_result is None:
            full_result = result
        else:
            full_result =  np.concatenate((full_result, result), axis=1)

    return full_result

def fspl(R, _lambda):
    """
        fspl     Free space path loss
        L = fspl(R,LAMBDA) returns the free space path loss L (in dB) suffered
        by a signal with wavelength LAMBDA (in meters) when it is propagated in
        free space for a distance of R (in meters). R can be a length-M vector
        and LAMBDA can be a length-N vector. L has the same dimensionality as
        MxN. Each element in L is the free space path loss for the
        corresponding propagation distance specified in R.

        Note that the best case is lossless so the loss is always greater than
        or equal to 0 dB.

        % Example:
        %   Calculate the free space loss for a signal whose wavelength is 30
        %   cm. The signal is propagated for 1 km.
        L = phased.internal.fspl(1000,0.3)
        See also phased, phased.FreeSpace.
        Reference
        [1] John Proakis, Digital Communications, 4th Ed., McGraw-Hill, 2001

    :param R:
    :param _lambda:
    :return:
    """
    L = 4 * np.pi * R / _lambda
    if not L.shape:
        L = np.array([[L]])
    L = validate_loss(L)
    L = mag2db(L);
    return L


def validate_loss(L):
    """

    :param L:
    :return:
    """
    L[L < 1] = 1
    return L

def mag2db(y):
    """
    MAG2DB  Magnitude to dB conversion.

    YDB = MAG2DB(Y) converts magnitude data Y into dB values.
    Negative values of Y are mapped to NaN.

    See also DB2MAG.
    Copyright 1986-2011 The MathWorks, Inc.

    :param y:
    :return:
    """

    y[y<0] = float('nan')
    ydb = 20*np.log10(y)
    return ydb


def db2pow(ydB):
    """
    DB2POW   dB to Power conversion
    Y = DB2POW(YDB) converts dB to its corresponding power value such that
    10*log10(Y)=YDB

    % Example:
    %   Convert 12dB to Power.
      y = db2pow(12)

    Copyright 2006 The MathWorks, Inc.

    :param ydB:
    :return:
    """

    y = np.power(10, ydB/10);
    return y