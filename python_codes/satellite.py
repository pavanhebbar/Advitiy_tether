"""Defines a class to handle basic parameters of satellite."""

import sys
import numpy as np
import matplotlib.pyplot as plt
import orb_state3D as orb
import tether as teth


G = 6.67408e-11
M_EARTH = 5.972e24


class satellite(orb.orb_param3D):
    """Set of basic functions to handle satellite.

    All these functions assume that the satellite is revolving around earth
    unless specified
    """

    def __init__(self, mass=0, I_matrix=np.zeros((3, 3)), bool_tether=0,
                 tether1=teth.tether(), r=0, v_r=0, theta=0, v_t=0, phi=0,
                 v_p=0, t0=0, form='sph', main_mass=M_EARTH):
        """Initialize the class.

        If form isn't given it will be assumed that the values are in
        spherical coordinates
        """
        super(satellite, self).__init__(r, v_r, theta, v_t, phi, v_p, t0, form)
        self._M = main_mass
        self._m = mass
        self._I = I_matrix
        self.bool_t = bool_tether
        self._tet = tether1
        """
        if (len(locals()) != 1 and len(locals()) != 4 and len(locals()) != 5
                and len(locals()) != 12 and len(locals()) != 13 and
                len(locals()) != 14):
            print ('You have given', len(locals()), 'arguments')
            print('usage: satellite()')
            print('usage: satellite(mass, I_mat, bool_tether)')
            print('usage: satellite(mass, I_mat, bool_tether, tether)')
            print('usage: satellite(mass, I_mat, bool_tether, tether, ' +
                  'r, vr, theta, vt, phi, vp, t0)')
            print('usage: satellite(mass, I_mat, bool_tether, tether, ' +
                  'r, vr, theta, vt, phi, vp, t0, "sph" or "xyz")')
            print('usage: satellite(mass, I_mat, bool_tether, tether, ' +
                  'r, vr, theta, vt, phi, vp, t0, "sph" or "xyz"' +
                  ', main_mass)')
            sys.exit(2)
        """

    def setmainmass(self, main_mass):
        """Set the mass of the main body around which satellite rotates."""
        self._M = main_mass

    def setmass(self, mass):
        """Set the mass of the satellite."""
        self._m = mass

    def settether(self, tether):
        """Set the tether.

        tether must be an instance of class tether
        """
        self._tet = tether

    def setI(self, I_mat):
        """Set the moment of Inertia matrix of satellite including tether."""
        self._I = I_mat

    def getmainmass(self):
        """Return main mass value."""
        return self._M

    def getmass(self):
        """Return satellite mass."""
        return self._m

    def gettether(self):
        """Return class instance of tether being used."""
        return self._tet

    def getI(self):
        """Return momnet of intertia matrix."""
        return self._I

    def getlatlong(self):
        """Return latitude and longitude of satellite."""
        lat = np.pi/2.0 - self._th
        t = self.gettime()
        lon = self._phi - 2*np.pi*t/86164.09164
        return lat, lon

    def geten(self):
        """Return total energy of the system."""
        return (0.5*self._m*(self._vr**2 + self._vt**2 + self._vp**2) -
                G*self._M*self._m/self._r)

    def get_a(self):
        """Return semi major axis of satellite."""
        return G*self._M*self._m/self.geten()

    def get_ecc(self):
        """Return eccentricity of satellite."""
        mu = G*(self._M + self._m)
        h = self.sp_ang_mom()
        v = self.getvel_xyz()
        r = self.getpos_xyz()
        e_vec = 1.0/mu*(np.cross(v, h) - mu*r/np.linalg.norm(r))
        return e_vec

    def arg_per(self):
        """Get the argument of perigee."""
        h = self.sp_ang_mom()
        n_vec = np.cross([0, 0, 1], h)
        e_vec = self.get_ecc()
        small_omega = np.arccos(np.dot(n_vec, e_vec)/(np.linalg.norm(n_vec) *
                                                      np.linalg.norm(e_vec)))
        if (e_vec[2] < 0):
            small_omega = 2*np.pi - small_omega
        return small_omega

    def true_an(self):
        """Get the true anomaly of satellite."""
        v = self.getpos_sph()
        e_vec = self.get_ecc()
        r = self.getpos_xyz()
        true_th = np.arccos(np.dot(e_vec, r)/(np.linalg.norm(e_vec) *
                                              np.linalg.norm(r)))
        if (v[0] < 0):
            true_th = 2*np.pi - true_th
        return true_th

    def orb_elem(self):
        """Return 6 orbital elements."""
        h_vec = self.sp_ang_mom()
        h = np.linalg.norm(h_vec)
        i = self.get_inc()
        cap_omega = self.get_ascnode()
        e_vec = self.get_ecc()
        e = np.linalg.norm(e_vec)
        small_omega = self.arg_per()
        true_th = self.true_an()
        return h, i, cap_omega, e, small_omega, true_th


def tot_acc(sat):
    """Get the total acceleration to the satellite."""
    main_mass = sat.getmainmass()
    r = sat.getpos_sph()[0]
    tether = sat._tet
    if sat.bool_t == 1:
        at_r, at_t, at_p = tether.accln(sat)
    else:
        at_r, at_t, at_p = [0, 0, 0]
    g_r = -G*main_mass/r**2
    g_t = 0
    g_p = 0
    return at_r + g_r, at_t + g_t, at_p + g_p


def getorbit(sat, tfinal, dt):
    """Calculate the orbit."""
    ntimes = (int)(tfinal/dt)
    n_t = (int)(tfinal/100)
    state_arr = np.zeros((6, n_t))
    orbelem_arr = np.zeros((6, n_t))
    acc_arr = np.zeros((3, n_t))
    vdot_arr1 = np.zeros((3, n_t))
    vdot_arr2 = np.zeros((3, n_t))
    pow_arr = np.zeros(n_t)
    en_arr = np.zeros(n_t)
    en0 = sat.geten()
    v0 = sat.getvel_sph()
    count = 0
    for i in range(ntimes):
        sat.rk4_step(dt, tot_acc)
        en1 = sat.geten()
        pow1 = (en1 - en0)/dt
        en0 = en1
        v1 = sat.getvel_sph()
        vdot1 = (v1 - v0)/dt
        v0 = v1
        if (i % (100/dt) == 0):
            state_arr[:, count] = sat.getstate()
            orbelem_arr[:, count] = sat.orb_elem()
            acc_arr[:, count] = tot_acc(sat)
            vdot_arr1[:, count] = sat.getvdot(acc_arr[0, count],
                                              acc_arr[1, count],
                                              acc_arr[2, count])
            vdot_arr2[:, count] = vdot1
            pow_arr[count] = pow1
            en_arr[count] = sat.geten()
            print (state_arr[0, count])
            print (count)
            count += 1
    return (state_arr, orbelem_arr, pow_arr, en_arr, acc_arr, vdot_arr1,
            vdot_arr2)


def plotfig(name, title, xlabel, ylabel, xdata, ydata, legend):
    """Plot figure."""
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(ydata)):
        plt.plot(xdata[i], ydata[i], label=legend[i])
    plt.legend()
    plt.savefig(name)
    plt.close()


def test_circular():
    """Test orbit function."""
    pratham = satellite(10.0, np.zeros((3, 3)), 0.0, teth.tether(100, 10),
                        7.048e6, 0.0, np.pi/2.0, -7446.8984446559243, 0.0,
                        -1046.5933233558835, 0.0)
    state_arr = getorbit(pratham, 400, 0.001)[0]
    time_array = np.linspace(0, 400, len(state_arr[0, :]))
    plotfig("test_r.png", "r v/s t", "t", "r", [time_array], [state_arr[0, :]],
            ["r"])
    plotfig("test_th.png", "theta v/s t", "t", "theta", [time_array],
            [state_arr[2, :]], ["theta"])
    plotfig("test_p.png", "phi v/s t", "t", "phi", [time_array],
            [state_arr[4, :]], ["phi"])
    r_array = state_arr[0, :]
    theta_array = state_arr[2, :]
    phi_array = state_arr[4, :]

    assert abs((np.min(r_array) - r_array[0])/r_array[0]) < 1e-5
    assert abs((np.max(r_array) - r_array[0])/r_array[0]) < 1e-5
    assert np.min(theta_array) > 0
    assert np.max(theta_array) < np.pi
    assert np.min(phi_array) > 0
    assert np.max(phi_array) < 2*np.pi


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
                self.__settime(time0 + tstep/2.0)
            else:
                state_n = state0 + k[:, i]
                self.setstates(state_n)
                self.__settime(time0 + tstep)
        state_n = state0 + (k[:, 0] + 2*k[:, 1] + 2*k[:, 2] + k[:, 3])/6.0
        state_n[4] = state[4] - np.floor(state[4]/(2*np.pi))*2*np.pi
        self.setstates(state_n)
        self.__settime(time0 + tstep)


if __name__ == "__main__":
    test_circular()
