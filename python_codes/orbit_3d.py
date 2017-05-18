"""Defines a class to handle basic functions in 3D orbit."""

import sys
import numpy as np
import matplotlib.pyplot as plt


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
        if (len(locals()) == 0 or len(locals()) == 8):
            self._r = r
            self._th = theta
            self._phi = phi
            self._vr = v_r
            self._vt = v_t
            self._vp = v_p
            self.__t = t0
        else:
            print ('Give no arguments or all 7 arguments')
            print (len(locals()))
            sys.exit(2)

    def setstates(self, states):
        """Set the state of object in orbit."""
        self._r = states[0]
        self._th = states[2]
        self._phi = states[4]
        self._vr = states[1]
        self._vt = states[3]
        self._vp = states[5]

    def setpos(self, pos):
        """Set position."""
        self._r = pos[0]
        self._th = pos[1]
        self.phi = pos[2]

    def setvel(self, vel):
        """Set velocity."""
        self._vr = vel[0]
        self._vt = vel[1]
        self._vp = vel[2]

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
        return self.__t

    def getpos(self):
        """Return position."""
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
        state0 = self.getstate()
        t0 = self.gettime()
        k = np.zeros((6, 4))
        for i in range(4):
            state = self.getstate()
            k[0, i] = state[1]*dt
            k[2, i] = state[3]/state[0]*dt
            k[4, i] = state[5]/(state[0]*np.sin(state[2]))*dt
            accs = acc_f(self)
            vdots = self.getvdot(accs[0], accs[1], accs[2])
            k[1, i] = vdots[0]*dt
            k[3, i] = vdots[1]*dt
            k[5, i] = vdots[2]*dt
            if (i < 2):
                state_n = state + k[:, i]/2.0
                self.setstates(state_n)
                self.__settime(t0 + dt/2.0)
            else:
                state_n = state + k[:, i]
                self.setstates(state_n)
                self.__settime(t0 + dt)
        state_n = state0 + (k[:, 0] + 2*k[:, 1] + 2*k[:, 2] + k[:, 3])/6.0
        state_n[4] = state[4] - np.floor(state[4]/(2*np.pi))*2*np.pi
        self.setstates(state_n)
        self.__settime(t0 + dt)


def two_body(orbit):
    """Give acceleartion in a two body problem."""
    M_Earth = 5.972e24
    pos = orbit.getpos()
    acc_r = -1.0*G*M_Earth/pos[0]**2
    return [acc_r, 0, 0]


def getorbit(orbit, tfinal, dt):
    """Calculate the orbit."""
    ntimes = (int)(tfinal/dt)
    n_t = (int)(tfinal/100)
    r_array = np.zeros(n_t)
    theta_array = np.zeros(n_t)
    phi_array = np.zeros(n_t)
    count = 0
    for i in range(ntimes):
        orbit.rk4_step(dt, two_body)
        if (i % (100/dt) == 0):
            r_array[count] = orbit.getpos()[0]
            theta_array[count] = orbit.getpos()[1]
            phi_array[count] = orbit.getpos()[2]
            print (r_array[count])
            print (count)
            count += 1
    return r_array, theta_array, phi_array


def plotfig(name, title, xlabel, ylabel, xdata, ydata, legend):
    """Plot figure."""
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(ydata)):
        if (xdata[i] == []):
            plt.plot(ydata[i], label=legend[i])
        else:
            plt.plot(xdata[i], ydata[i], label=legend[i])
    plt.legend()
    plt.savefig(name)
    plt.close()


def test_circular():
    """Test orbit function."""
    pratham = orbit_3d(7.048e6, 0, np.pi/2.0, -0.0010565973956662776, 0,
                       -0.0001484950799313115, 0)
    r_array, theta_array, phi_array = getorbit(pratham,
                                               60000, 0.001)
    time_array = np.linspace(0, 60000, len(r_array))
    plotfig("r_arr0.png", "r v/s t", "t", "r", [time_array], [r_array], ["r"])
    plotfig("t_arr0.png", "theta v/s t", "t", "theta", [time_array],
            [theta_array], ["theta"])
    plotfig("phi_arr0.png", "phi v/s t", "t", "phi", [time_array], [phi_array],
            ["phi"])

    assert abs((np.min(r_array) - r_array[0])/r_array[0]) < 1e-5
    assert abs((np.max(r_array) - r_array[0])/r_array[0]) < 1e-5
    assert np.min(theta_array) > 0
    assert np.max(theta_array) < np.pi
    assert np.min(phi_array) > 0
    assert np.max(phi_array) < 2*np.pi


if __name__ == "__main__":
    test_circular()
