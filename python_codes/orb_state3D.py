"""Defines a class to handle basic functions in 3D orbit.

state - [r, radial velocity, theta, vtheta, phi, vphi]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


class orb_param3D:
    """Set of basic functions for th    e 3D orbit.

    All these functions assume reduced mass concept.
    """

    def __init__(self, r=0, v_r=0, theta=0, v_t=0, phi=0, v_p=0, t0=0,
                 format='sph'):
        """Initialize the class.

        Only takes 0 or 7 arguments. If 0 arguments are passed object
        placed at the centre
        """
        if (len(locals()) == 0 or len(locals()) == 8 or (len(locals()) == 9)):
            self._r = r
            self._th = theta
            self._phi = phi
            self._vr = v_r
            self._vt = v_t
            self._vp = v_p
            self.__t = t0
            if (format == 'xyz'):
                self.setpos_xyz([r, theta, phi])
                self.setvel_xyz([v_r, v_t, v_p])
            elif (format != 'sph'):
                print ("format only takes arguments 'xyz' or 'sph'")
        else:
            print ('Give no arguments or atleast 7 arguments')
            print ('You have given ', len(locals()), ' arguments')
            print ('usage: orb_param3D() to set all elements to zero')
            print ('usage: orb_param3D(r, vr, theta, vtheta, phi, vphi, t)' +
                   ' for input in spherical coordinates')
            print ("usage: orb_param3D(r, vr, theta, vtheta, phi, vphi" +
                   ", t0, 'sph') for input in spherical coordinates")
            print ("usage: orb_param3D(x, vx, y, vy, z, vz" +
                   ", t0, 'xyz') for input in cartesian coordinates")
            sys.exit(2)

    def setstates(self, states):
        """Set the state of object in orbit."""
        self._r = states[0]
        self._th = states[2]
        self._phi = states[4]
        self._vr = states[1]
        self._vt = states[3]
        self._vp = states[5]

    def setpos_sph(self, pos):
        """Set position."""
        self._r = pos[0]
        self._th = pos[1]
        self._phi = pos[2]

    def setpos_xyz(self, pos):
        """Set position given xyz coordinates."""
        self._r = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
        self._th = np.arccos(pos[2]/self._r)
        self._phi = np.arctan2(pos[1], pos[0])
        if (self._phi < 0):
            self._phi += 2*np.pi

    def setvel_sph(self, vel):
        """Set velocity."""
        self._vr = vel[0]
        self._vt = vel[1]
        self._vp = vel[2]

    def setvel_xyz(self, vel):
        """Set velocity given velocity in xyz coordinates."""
        posxyz = self.getpos_xyz()
        self._vr = np.dot(posxyz, vel)/np.linalg.norm(posxyz)
        self._vt = (self._vr*posxyz[2] - vel[2]*self._r)/(self._r**2 -
                                                          posxyz[2]**2)**0.5
        self._vp = ((posxyz[0]*vel[1] - posxyz[1]*vel[2]) /
                    np.linalg.norm(posxyz[0:2]))

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

    def getpos_sph(self):
        """Return position."""
        pos = [self._r, self._th, self._phi]
        return pos

    def getpos_xyz(self):
        """Return xyz coordinates."""
        x = self._r*np.sin(self._th)*np.cos(self._phi)
        y = self._r*np.sin(self._th)*np.sin(self._phi)
        z = self._r*np.cos(self._th)
        return [x, y, z]

    def getvel_sph(self):
        """Return velocity in spherical coordinates."""
        return [self._vr, self._vt, self._vp]

    def getvel_xyz(self):
        """Return velocity in cartesian coordinates."""
        vx = (self._vr*np.sin(self._th)*np.cos(self._phi) +
              self._vt*np.cos(self._th)*np.cos(self._phi) -
              self._vp*np.sin(self._phi))
        vy = (self._vr*np.sin(self._th)*np.sin(self._phi) +
              self._vt*np.cos(self._th)*np.sin(self._phi) +
              self._vp*np.cos(self._phi))
        vz = self._vr*np.cos(self._th) - self._vt*np.sin(self._th)
        return [vx, vy, vz]

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
                state_n = state0 + k[:, i]/2.0
                self.setstates(state_n)
                self.__settime(t0 + dt/2.0)
            else:
                state_n = state0 + k[:, i]
                self.setstates(state_n)
                self.__settime(t0 + dt)
        state_n = state0 + (k[:, 0] + 2*k[:, 1] + 2*k[:, 2] + k[:, 3])/6.0
        state_n[4] = state[4] - np.floor(state[4]/(2*np.pi))*2*np.pi
        self.setstates(state_n)
        self.__settime(t0 + dt)


def two_body(orbit):
    """Give acceleartion in a two body problem."""
    G = 6.67408e-11
    M_Earth = 5.972e24
    pos = orbit.getpos_sph()
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
            r_array[count] = orbit.getpos_sph()[0]
            theta_array[count] = orbit.getpos_sph()[1]
            phi_array[count] = orbit.getpos_sph()[2]
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
    pratham = orb_param3D(7.048e6, 0, np.pi/2.0, -7446.8984446559243, 0,
                          -1046.5933233558835, 0)
    r_array, theta_array, phi_array = getorbit(pratham,
                                               200, 0.001)
    time_array = np.linspace(0, 200, len(r_array))
    plotfig("test_r.png", "r v/s t", "t", "r", [time_array], [r_array], ["r"])
    plotfig("test_th.png", "theta v/s t", "t", "theta", [time_array],
            [theta_array], ["theta"])
    plotfig("test_p.png", "phi v/s t", "t", "phi", [time_array], [phi_array],
            ["phi"])

    assert abs((np.min(r_array) - r_array[0])/r_array[0]) < 1e-5
    assert abs((np.max(r_array) - r_array[0])/r_array[0]) < 1e-5
    assert np.min(theta_array) > 0
    assert np.max(theta_array) < np.pi
    assert np.min(phi_array) > 0
    assert np.max(phi_array) < 2*np.pi


if __name__ == "__main__":
    test_circular()
