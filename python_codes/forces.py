"""Module containing all the forces that act on the satellite."""

import numpy as np
from scipy import special
from scipy.misc import factorial as fac


EGMCOEFFS = np.loadtext('EGM2008_coeff')
MU_MASS = 3986004.415E+8
SEMI_MAJ = 6378136.3


def gravpot(rad, lat, lon):
    """Get the gravitational potential at a point.

    Inputs:
    rad - radius of the point
    lat - latitude of the point
    lon - lonitude of the point

    Outputs:
    pot - gravitational potential
    """
    pot = 1
    for data in EGMCOEFFS:
        if data[1] == 0:
            kvar = 1
        else:
            kvar = 2
        pot += ((SEMI_MAJ/rad)**data[0]*(data[2]*np.cos(data[1]*lon) +
                                         data[3]*np.sin(data[1]*lon)) *
                special.lpmv(data[1], data[0], np.cos(lat)) *
                (kvar*(2*data[0]+1)*fac(data[0] - data[1]) /
                 fac(data[0] + data[1])))
    pot = pot*MU_MASS/rad
    return pot
