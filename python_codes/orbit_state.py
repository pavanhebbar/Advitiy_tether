"""Define a class to handle basic state of orbit.

state - [r, radial velocity, theta, vtheta, phi, vphi]
"""

import numpy as np


class OrbParam3D(object):
    """Set of basic functions for  3D orbit.

    All these functions assume reduced mass concept.
    """

    def __init__(self, pos=np.zeros(3), vel=np.zeros(3), time0=0, form='sph'):
        """Initialize the class.

        Inputs:
        pos - contains the initial position vetor
        vel - contains the initial velocity vector
        t0 - Initial time in decimal years
        format - frame in which vectors are stated - spherical (sph) or
                cartesian(xyz)
        """
        self._r = pos[0]
        self._th = pos[1]
        self._phi = pos[2]
        self._vr = vel[0]
        self._vt = vel[1]
        self._vp = vel[2]
        self.__t = time0
        if form == 'xyz':
            self.setpos_xyz(pos)
            self.setvel_xyz(vel)
        elif form != 'sph':
            print "form can conly take values 'xyz' or 'sph'"

    def setstates(self, states):
        """Set the state of object in orbit."""
        self._r = states[0]
        self._th = states[2]
        self._phi = states[4]
        self._vr = states[1]
        self._vt = states[3]
        self._vp = states[5]

    def setpos_sph(self, pos):
        """Set position.

        Inputs:
        pos - Position vector in spherical coordinates
        """
        self._r = pos[0]
        self._th = pos[1]
        self._phi = pos[2]

    def setpos_xyz(self, pos):
        """Set position given xyz coordinates.

        Inputs:
        pos - postion vector in cartesian coordinates
        """
        self._r = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
        self._th = np.arccos(pos[2]/self._r)
        self._phi = np.arctan2(pos[1], pos[0])
        if self._phi < 0:
            self._phi += 2*np.pi

    def setvel_sph(self, vel):
        """Set velocity.

        Inputs:
        vel - velocity in spherical coordinates
        """
        self._vr = vel[0]
        self._vt = vel[1]
        self._vp = vel[2]

    def setvel_xyz(self, vel):
        """Set velocity given velocity in xyz coordinates.

        Inputs:
        vel - velocity in cartesian coordinates
        """
        posxyz = self.getpos_xyz()
        self._vr = np.dot(posxyz, vel)/np.linalg.norm(posxyz)
        self._vt = (self._vr*posxyz[2] - vel[2]*self._r)/(self._r**2 -
                                                          posxyz[2]**2)**0.5
        self._vp = ((posxyz[0]*vel[1] - posxyz[1]*vel[0]) /
                    np.linalg.norm(posxyz[0:2]))

    def settime(self, time):
        """Set time.

        Private
        Inputs:
        time - time
        """
        self.__t = time

    def getstate(self):
        """Return states."""
        states = np.array([self._r, self._vr, self._th, self._vt,
                           self._phi, self._vp])
        return states

    def gettime(self):
        """Return time."""
        return self.__t

    def getpos_sph(self):
        """Return position."""
        pos = np.array([self._r, self._th, self._phi])
        return pos

    def getpos_xyz(self):
        """Return xyz coordinates."""
        xpos = self._r*np.sin(self._th)*np.cos(self._phi)
        ypos = self._r*np.sin(self._th)*np.sin(self._phi)
        zpos = self._r*np.cos(self._th)
        return np.array([xpos, ypos, zpos])

    def getvel_sph(self):
        """Return velocity in spherical coordinates."""
        return np.array([self._vr, self._vt, self._vp])

    def getvel_xyz(self):
        """Return velocity in cartesian coordinates."""
        velx = (self._vr*np.sin(self._th)*np.cos(self._phi) +
                self._vt*np.cos(self._th)*np.cos(self._phi) -
                self._vp*np.sin(self._phi))
        vely = (self._vr*np.sin(self._th)*np.sin(self._phi) +
                self._vt*np.cos(self._th)*np.sin(self._phi) +
                self._vp*np.cos(self._phi))
        velz = self._vr*np.cos(self._th) - self._vt*np.sin(self._th)
        return np.array([velx, vely, velz])

    def getvdot(self, a_r, a_t, a_p):
        """Return vrdot, vtdot, vpdot.

        Not yet done for theta=0
        Inputs:
        a_r - Acceleration in r direction
        a_t - Acceleration in theta direction
        a_p - Acceleration in phi direction

        Outputs:
        Array of [vrdot, vtdot, vpdot]
        """
        vrdot = a_r + (self._vt**2 + self._vp**2)/self._r
        vtdot = a_t - (self._vr*self._vt -
                       self._vp**2/np.tan(self._th))/self._r
        vpdot = a_p - (self._vr*self._vp +
                       self._vt*self._vp/np.tan(self._th))/self._r
        return np.array([vrdot, vtdot, vpdot])

    def rk4_step(self, tstep, acc_f):
        """Return the state after time dt.

        Inputs
        tstep - time step
        acc_f - function to calculate acceleration
        """
        state0 = self.getstate()
        time0 = self.gettime()
        k = np.zeros((6, 4))
        for i in range(4):
            state = self.getstate()
            k[0, i] = state[1]*tstep
            k[2, i] = state[3]/state[0]*tstep
            k[4, i] = state[5]/(state[0]*np.sin(state[2]))*tstep
            accs = acc_f(self)
            vdots = self.getvdot(accs[0], accs[1], accs[2])
            k[1, i] = vdots[0]*tstep
            k[3, i] = vdots[1]*tstep
            k[5, i] = vdots[2]*tstep
            if i < 2:
                state_n = state0 + k[:, i]/2.0
                self.setstates(state_n)
                self.settime(time0 + tstep/2.0)
            else:
                state_n = state0 + k[:, i]
                self.setstates(state_n)
                self.settime(time0 + tstep)
        state_n = state0 + (k[:, 0] + 2*k[:, 1] + 2*k[:, 2] + k[:, 3])/6.0
        state_n[4] = state[4] - np.floor(state[4]/(2*np.pi))*2*np.pi
        self.setstates(state_n)
        self.settime(time0 + tstep)

    def sp_ang_mom(self):
        """Return the specific angular momentum of satellite."""
        pos = self.getpos_xyz()
        vel = self.getvel_xyz()
        return np.cross(pos, vel)

    def get_inc(self):
        """Return inclination of the orbit."""
        h_mom = self.sp_ang_mom()
        return np.arccos(h_mom[2]/np.linalg.norm(h_mom))

    def get_ascnode(self):
        """Get longitude of ascending node."""
        h_mom = self.sp_ang_mom()
        if h_mom[0] == 0.0 and h_mom[1] == 0.0:
            cap_omega = 0.0
        else:
            cap_omega = np.arctan2(h_mom[0], -1*h_mom[1])
            if cap_omega < 0:
                cap_omega += 2*np.pi
        return cap_omega


def test_conv():
    """Test the conversions from cartesian to spherical frames."""
    xvar = OrbParam3D([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], 0, 'xyz')
    yvar = OrbParam3D([10.0, np.pi/2.0, np.pi/2.0], [5.0, 0.0, 0])
    assert (np.linalg.norm(xvar.getpos_sph() - np.array([1.0, np.pi/2.0, 0]))
            < 1e-5)
    assert (np.linalg.norm(xvar.getvel_sph() - np.array([0.0, 0.0, -1.0]))
            < 1e-5)
    assert (np.linalg.norm(yvar.getpos_xyz() - np.array([0.0, 10.0, 0.0]))
            < 1e-5)
    assert (np.linalg.norm(yvar.getvel_xyz() - np.array([0.0, 5.0, 0.0]))
            < 1e-5)


def zero_acc(body):
    """Return zeros."""
    if isinstance(body, OrbParam3D):
        return [0.0, 0.0, 0.0]
    else:
        print "Variable isn't of type OrbParam3D"


def test_rk4():
    """Test rk4 function."""
    var = OrbParam3D([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0, 'xyz')
    count = 0
    while count < 10000:
        var.rk4_step(0.001, zero_acc)
        count += 1
    assert (np.linalg.norm(var.getpos_xyz() - np.array([1.0, 10.0, 0.0]))
            < 1e-5)


if __name__ == '__main__':
    test_conv()
    test_rk4()
    print "All tests ok"
