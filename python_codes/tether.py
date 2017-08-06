"""Define a class to handle tether dynamics."""


import numpy as np
import igrf12
import rk4_step as rk


class Tether(object):
    """Class to handle tether dynamics."""

    def __init__(self, tether_len=0, tether_con=0, em_current=0, t_cross=0):
        """Initialize class.

        Inputs:
        tether_len - length of tether_len
        tether_con - conductivity of tether
        em_current - current in emitter
        t_cross - array containing cross section dimensions of tether
            If array has 1 elem, it is assumed that tether is a cyindrical
            tether.
        _ang[0] - Angle of the tether from verticle
        _angle[1] = angle of tether from body x-z frame
        """
        self._len = tether_len
        self._sigma = tether_con
        self._ang = [0, 0]
        self._curr_c = em_current
        self._dim = t_cross

    def setlen(self, length):
        """Set length of tether."""
        self._len = length

    def setsigma(self, conductivity):
        """Set resistance of tether."""
        self._sigma = conductivity

    def setdim(self, t_cross):
        """Set the cross section dimensions of tether."""
        self._dim = t_cross

    def setemcurr(self, cath_curr):
        """Set the current in the emitter."""
        self._curr_c = cath_curr

    def getlen(self):
        """Return tether length."""
        return self._len

    def getsigma(self):
        """Return resistance of tether."""
        return self._sigma

    def getangle(self):
        """Return the angle of tether."""
        return self._ang

    def getemcurr(self):
        """Return current in cathode."""
        return self._curr_c

    def getdim(self):
        """Return dimensions of tether."""
        return self._dim

    def getmag(self, sat):
        """Return magnetic field at satellite position."""
        lat, lon = sat.getlatlong()
        rad = sat.getpos_sph()[0]
        time = sat.gettime()
        return getB_sph(lat, lon, rad, time)

    def getind_e(self, sat):
        """Return induced electric across tether.

        Positive in upward direction.
        """
        vel = sat.getvel_sph()
        v_abs = np.linalg.norm(vel)
        pos = sat.getpos_sph()
        vel[2] = vel[2] - 2*np.pi/86164.09164*pos[0]*np.sin(pos[1])
        l_cap_vel = [np.sin(self._ang[0])*np.cos(self._ang[1]),
                     np.sin(self._ang[0])*np.sin(self._ang[1]),
                     np.cos(self._ang[0])]
        vel_to_rtp = [[vel[0]/v_abs, vel[1]/v_abs, vel[2]/v_abs],
                      [0, -1*vel[2]/(vel[1]**2 + vel[2]**2)**0.5,
                       vel[1]/(vel[1]**2 + vel[2]**2)**0.5],
                      [(vel[1]**2 + vel[2]**2)**0.5/v_abs,
                       -1*(vel[0]*vel[1])/(v_abs*(vel[1]**2 + vel[2]**2)**0.5),
                       -1*(vel[0]*vel[2])/(v_abs*(vel[1]**2 + vel[2]**2)**0.5)]
                     ]
        l_cap_rtp = np.dot(np.transpose(vel_to_rtp), l_cap_vel)
        mag_vec = self.getmag(sat)
        return np.dot(np.cross(vel, mag_vec), l_cap_rtp)

    def getconst(self, sat):
        """Return constant parameters."""
        n_inf = 10**11
        e_ind = self.getind_e(sat)
        if len(self._dim) == 1:
            area = np.pi*self._dim[0]**2/4.0
        elif len(self._dim) == 2:
            area = self._dim[0]*self._dim[1]
        len0 = (9*np.pi*9.109E-31*(3.538E7)**2*e_ind*area/(128*(1.602E-19)**2 *
                                                           n_inf**2))
        curr0 = self._sigma*e_ind*area
        volt0 = e_ind*len0
        return len0, curr0, volt0


    def get_iv(self, sat):
        """Get the voltage and current in the tether.

        Assumed the ion currentis negligible.
        Assumed that lambda_c is 0
        """
        lambda_c = 0
        len0, curr0, volt0 = self.getconst()
        gamma_c = self._curr_c/curr0
        lambda_a = (lambda_c**1.5 - gamma_c**2 + 2*gamma_c)**(2/3)
        dlen = 0.1
        deps = dlen/len0
        nlen = (int)(self._len/dlen)
        lambda_arr = np.zeros(nlen)
        gamma_arr = np.zeros(nlen)

    def accln(self, sat):
        """Get acceleration due to tether."""
        a_r = 0
        B_r, B_t, B_p = self.getB(sat)
        mass = sat.getmass()
        a_t = -1*self.getcurr(sat)*self._len*B_p/mass
        a_p = self.getcurr(sat)*self._len*B_t/mass
        return a_r, a_t, a_p


def diff_iv(state):
    """Return differential of gamma ad lambda.

    state - [gamma, lambda]
    """
    return np.array([0.75*state[1]**0.5, state[0] - 1])


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
