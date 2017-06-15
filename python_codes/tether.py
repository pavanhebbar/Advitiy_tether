"""Define a class to handle tether dynamics."""

import sys
import numpy as np
import igrf12


class tether:
    """Class to handle tether dynamics."""

    def __init__(self, tether_len=0, tether_res=0):
        """Initialize class."""
        self._len = tether_len
        self._res = tether_res

    def setlen(self, length):
        """Set length of tether."""
        self._len = length

    def setres(self, resistance):
        """Set resistance of tether."""
        self._res = resistance

    def getlen(self):
        """Return tether length."""
        return self._len

    def getres(self):
        """Return resistance of tether."""
        return self._res

    def getB(self, sat):
        """Return magnetic field at satellite position."""
        lat, lon = sat.getlatlong()
        r = sat.getpos_sph()[0]
        t = sat.gettime()
        return getB_sph(lat, lon, r, t)

    def getcurr(self, sat):
        """Return current."""
        B_r, B_t, B_p = self.getB(sat)
        v_r, v_t, v_p = sat.getvel_sph()
        r = sat.getpos_sph()[0]
        v_p = v_p - 2*np.pi/86164.09164*r  # v_p must be in ECEF coordinates
        return (v_t*B_p - v_p*B_t)*self._len/self._res

    def accln(self, sat):
        """Get acceleration due to tether."""
        a_r = 0
        B_r, B_t, B_p = self.getB(sat)
        mass = sat.getmass()
        a_t = -1*self.getcurr(sat)*self._len*B_p/mass
        a_p = self.getcurr(sat)*self._len*B_t/mass
        return a_r, a_t, a_p


def getB_NED(lat, longit, r, t):  # time in decimal year
    """Get B in NED frame."""
    longit = longit*180/np.pi
    colat = (np.pi/2.0 - lat)*180/np.pi
    t = 2017 + t/(365.3422*86400)
    B_x, B_y, B_z, B_t = igrf12.igrf12syn(0, t, 2, r*10**-3, colat, longit)
    return B_x*10**-9, B_y*10**-9, B_z*10**-9


def getB_ecef(lat, longit, r, t):
    """Return B in ECEF spherical coordinates."""
    B_north, B_east, B_down = getB_NED(lat, longit, r, t)
    B_r = -1*B_down
    B_t = -1*B_north
    B_p = B_east
    return B_r, B_t, B_p

def getB_sph(lat, longit, r, t):
    """Return B in spherical coordinates - ECI."""
    B_r_ec, B_t_ec, B_p_ec = getB_ecef(lat, longit, r, t)
    alp = 2*np.pi/86164.09164*t
    
