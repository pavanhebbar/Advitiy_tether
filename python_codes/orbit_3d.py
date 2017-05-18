"""Defines a class to handle basic functions in 3D orbit."""

import sys
import numpy as np

G = 6.67408e-11


class orbit_3d:
    """Set of basic functions for the 3D orbit.

    All these functions assume reduced mass concept.
    """

    def __init__(self, r=0, v_r=0, theta=0, v_t=0, phi=0, v_p=0, t0=0):
        """Initialize the class.

        Only takes 0 or 7 arguments. If 0 arguments are passed object
        placed at the centre
        """
        if (len(locals()) == 0 or len(locals()) == 7):
            self._r = r
            self._th = theta
            self._phi = phi
            self._vr = v_r
            self._vt = v_t
            self._vp = v_p
            self.__t = t0
        else:
            print ('Give no arguments or all 7 arguments')
            sys.exit(2)

    def setstates(self, r, v_r, theta, v_t, phi, v_p):
        """Set the state of object in orbit."""
        self._r = r
        self._th = theta
        self._phi = phi
        self._vr = v_r
        self._vt = v_t
        self._vp = v_p

    def setpos(self, r, theta, phi):
        """Set position."""
        self._r = r
        self._th = theta
        self.phi = phi

    def setvel(self, v_r, v_t, v_p):
        """Set velocity."""
        self._vr = v_r
        self._vt = v_t
        self._vp = v_p

    def __settime(self, time):
        """Set time.

        Private
        """
        self.__t = time

    def getstate(self):
        """Return states."""
        states = [self._r, self._vr, self._th, self._vt,
                           self._phi, self._vp]
        return states

    def gettime(self):
        """Return time."""
        return self.__time

    def getpos(self):
        """Return position"""
        pos = [self._r, self._th, self._phi]
        return pos

    def getvel_sph(self):
        """Return velocity in spherical coordinates."""
        return [self._vr, self._vt, self._vp]

    def getvdot(self, a_r, a_t, a_p):
        """Return vrdot, vtdot, vpdot."""
        vrdot = a_r + (self._vt**2 + self._vp**2)/self._r
        vtdot = a_t - (self._vr*self._vt -
                       self._vp**2/np.tan(self._th))/self._r
        vpdot = a_p - (self._vr*self._vp +
                       self._vt*self._vp/np.tan(self._th))/self._r
        return [vrdot, vtdot, vpdot]

    def rk4_step(self, dt, acc_f):
        """Return the state after time dt."""
        state0 = self.getstate
