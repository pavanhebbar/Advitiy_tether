"""Define a class to handle tether dynamics."""


import numpy as np
import igrf12
import rk4_step as rk
from scipy.optimize import fsolve


class Tether(object):
    """Class to handle tether dynamics."""

    def __init__(self, tether_len=0, tether_con=0, t_cross=0):
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
        self._dim = t_cross
        self._la = 0   # just an initialization. Donot use this value
        self._dlen = 0.1
        self._i_arr = np.zeros((int)(self._len/self._dlen))

    def setlen(self, length):
        """Set length of tether."""
        self._len = length

    def setsigma(self, conductivity):
        """Set resistance of tether."""
        self._sigma = conductivity

    def setdim(self, t_cross):
        """Set the cross section dimensions of tether."""
        self._dim = t_cross

    # def setemcurr(self, cath_curr):
    #    """Set the current in the emitter."""
    #    self._curr_c = cath_curr

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

    def getlcap_ecef(self, sat):
        """Return the direction of tether in ECEF frame."""
        vel = sat.getvel_sph()
        pos = sat.getpos_sph()
        vel[2] = vel[2] - 2*np.pi/86164.09164*pos[0]*np.sin(pos[1])
        v_abs = np.linalg.norm(vel)
        l_cap_vel = [np.sin(self._ang[0])*np.cos(self._ang[1]),
                     np.sin(self._ang[0])*np.sin(self._ang[1]),
                     np.cos(self._ang[0])]
        vel_to_rtp = [[vel[0]/v_abs, vel[1]/v_abs, vel[2]/v_abs],
                      [0, -1*vel[2]/(vel[1]**2 + vel[2]**2)**0.5,
                       vel[1]/(vel[1]**2 + vel[2]**2)**0.5],
                      [(vel[1]**2 + vel[2]**2)**0.5/v_abs,
                       -1*(vel[0]*vel[1])/(v_abs*(vel[1]**2 + vel[2]**2)**0.5),
                       -1*(vel[0]*vel[2]) /
                       (v_abs*(vel[1]**2 + vel[2]**2)**0.5)]]
        l_cap_rtp = np.dot(np.transpose(vel_to_rtp), l_cap_vel)
        return l_cap_rtp

    def getind_e(self, sat):
        """Return induced electric across tether.

        Positive in upward direction.
        """
        vel = sat.getvel_sph()
        pos = sat.getpos_sph()
        vel[2] = vel[2] - 2*np.pi/86164.09164*pos[0]*np.sin(pos[1])
        l_cap_rtp = self.getlcap_ecef(sat)
        mag_vec = getmag(sat)
        return np.dot(np.cross(vel, mag_vec), l_cap_rtp)

    def getconst(self, sat):
        """Return constant parameters."""
        n_inf = 10**11
        e_ind = self.getind_e(sat)
        if len(self._dim) == 1:
            area = np.pi*self._dim[0]**2/4.0
        elif len(self._dim) == 2:
            area = self._dim[0]*self._dim[1]
        len0 = ((9*np.pi*9.109E-31*self._sigma**2*abs(e_ind)*area) /
                (128*(1.602E-19)**3*n_inf**2))**(1.0/3)
        curr0 = self._sigma*e_ind*area
        volt0 = abs(e_ind)*len0
        return len0, curr0, volt0

    def setlamda_a(self, sat):
        """Solve lamda_a for given eps_b."""
        len0 = self.getconst(sat)[0]
        eps_b = self._len/len0
        lamda_a = fsolve(solvefunc_eps, 0.9999, eps_b)
        self._la = lamda_a

    def set_iv(self, sat):
        """Get the voltage and current in the tether.

        Assumed the ion currentis negligible.
        Assumed that lambda_c is 0
        """
        len0, curr0 = self.getconst(sat)[:-1]
        lambda_a = self._la
        deps = self._dlen/len0
        nlen = (int)(self._len/self._dlen)
        lambda_arr = np.zeros(nlen)
        gamma_arr = np.zeros(nlen)
        lambda_arr[0] = lambda_a
        for i, gamma in enumerate(gamma_arr[:-1]):
            state_new = rk.rk4_step(diff_iv, np.array([gamma, lambda_arr[i]]),
                                    deps)
            gamma_arr[i+1] = state_new[0]
            lambda_arr[i+1] = state_new[1]
        self._i_arr = gamma_arr*curr0

    def accln(self, sat):
        """Get acceleration due to tether."""
        mass = sat.get_satmass()
        mag_vec = getmag(sat)
        lcap_ecef = self.getlcap_ecef(sat)
        # field_ind = self.getind_e(sat)
        # curr = self._sigma*field_ind*np.pi*self._dim[0]**2/4.0
        # force = np.cross(lcap_ecef, mag_vec)*curr*self._len
        force = np.cross(lcap_ecef, mag_vec)*np.sum(self._i_arr)*self._dlen
        acc = force/mass
        return acc


def diff_iv(state):
    """Return differential of gamma ad lambda.

    state - [gamma, lambda]
    """
    state_n = state.copy()
    if state_n[1] <= 0:
        state_n[1] = 0
    return np.array([0.75*state_n[1]**0.5, state_n[0] - 1])


def diff_eps(state):
    """Return differenial to calculate the epsilon from lamda."""
    dstate = np.zeros(3)
    dstate[0] = (state[1]**1.5 - state[2]**1.5 + 1)**-0.5
    dstate[1] = 1.0
    return dstate


def get_epsb(lamda_a):
    """Return eps_b given lamda_a."""
    state = np.array([0, 0, lamda_a])
    dlamda = lamda_a/100.0
    if lamda_a == 1:
        state[0] = 4*dlamda**0.25
        state[1] = dlamda
    while state[1] < lamda_a:
        state = rk.rk4_step(diff_eps, state.copy(), dlamda)
    return state[0]


def solvefunc_eps(lamda_a, eps_b):
    """Return difference between eps for lamda_a and eps_b."""
    return get_epsb(lamda_a) - eps_b


def getmag_ned(lat, lon, rad, time):  # time in seconds
    """Get B in NED frame."""
    lon = lon*180/np.pi
    colat = (np.pi/2.0 - lat)*180/np.pi
    time = 2017 + time/(365.3422*86400)
    mag_x, mag_y, mag_z = igrf12.igrf12syn(0, time, 2, rad*10**-3, colat,
                                           lon)[:-1]
    return mag_x*10**-9, mag_y*10**-9, mag_z*10**-9


def getmag_ecef(lat, longit, rad, time):
    """Return B in ECEF spherical coordinates."""
    mag_north, mag_east, mag_down = getmag_ned(lat, longit, rad, time)
    mag_r = -1*mag_down
    mag_t = -1*mag_north
    mag_p = mag_east
    return mag_r, mag_t, mag_p


def getmag(sat):
    """Return magnetic field on satellite."""
    rad = sat.getpos_sph()
    time = sat.gettime()
    lat, lon = sat.getlatlon()
    return getmag_ecef(lat, lon, rad, time)
