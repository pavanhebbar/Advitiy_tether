"""Module containing all the forces that act on the satellite."""

import numpy as np
from scipy import special
from scipy.misc import factorial as fac


EGMCOEFFS = np.loadtxt('EGM2008_coeff')
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
    for data in EGMCOEFFS[:10, :]:
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


def earthgrav(rad, lat, lon):
    """Get the acceleration due to gravity.

    Inputs:
    rad - dist of the point from centre of the earth
    lat - Latitude of the point
    lon - longitude of the point
    """
    colat = np.pi/2.0 - lat
    grav = np.zeros(3, dtype=float)
    grav[0] = 1.0
    max_order = EGMCOEFFS[-1, 1]
    max_deg = EGMCOEFFS[-1, 0]
    l_poly, l_diff = special.lpmn(max_order, max_deg, np.cos(colat))
    for data in EGMCOEFFS[:10, :]:
        dgrav = np.zeros(3)
        deg = (int)(data[0])
        order = (int)(data[1])
        if data[1] == 0:
            kvar = 1
        else:
            kvar = 2
        dgrav[0] += ((data[0] + 1) *
                     (data[2]*np.cos(data[1]*lon) +
                      data[3]*np.sin(data[1]*lon))*l_poly[order, deg])
        dgrav[1] += ((data[2]*np.cos(data[1]*lon) +
                      data[3]*np.sin(data[1]*lon))*l_diff[order, deg] *
                     np.sin(lat))
        dgrav[2] += ((-1*data[2]*np.sin(data[1]*lon) +
                      data[3]*np.cos(data[1]*lon))*data[1] *
                     l_poly[order, deg])
        dgrav = (dgrav*(SEMI_MAJ/rad)**data[0] *
                 (kvar*(2*data[0] + 1)*fac(data[0] - data[1]) /
                  fac(data[0] + data[1])))
        grav += dgrav
    grav = grav*MU_MASS/rad**2
    grav[0] = -1*grav[0]
    grav[1] = grav[1]
    if colat == 0:
        return grav[2] == 0.0
    else:
        grav[2] = grav[2]/(np.sin(colat))
    return grav


def gravity_wgs84(rad, lat):
    """Return gravity acc to wgs84 model."""
    j2val = 1.081874e-3
    aval = 6378137
    muval = 3.986004418e14
    g_rad = -1*muval/rad**2*(1.0 - 1.5*j2val*(aval/rad)**2*(3*np.sin(lat)**2 -
                                                            1))
    g_th = 3*muval/rad**4*aval**2*np.sin(lat)*np.cos(lat)*j2val
    return g_rad, g_th, 0


def gravity_eci(rad, lat, lon):
    """Get the gravity after removing centrifugal correction."""
    g_ecef = earthgrav(rad, lat, lon)
    g_eci = np.zeros_like(g_ecef)
    g_eci[0] = g_ecef[0] - (2*np.pi/86164.09164)**2*rad*np.cos(lat)**2
    g_eci[1] = g_ecef[1] - (2*np.pi/86164.09164)**2*rad*np.sin(lat)*np.cos(lat)
    g_eci[2] = g_ecef[2]
    return g_eci
