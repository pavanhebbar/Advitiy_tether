"""Defines a class to handle basic parameters of satellite."""

import numpy as np
import orbit_state as orb
import tether as teth
import forces
import matplotlib.pyplot as plt

G = 6.67408e-11
M_EARTH = 5.972e24


class Satellite(orb.OrbParam3D):
    """Set of basic functions to handle satellite.

    All these functions assume that the satellite is revolving around earth
    unless specified
    """

    def __init__(self, satmass, state0, form='sph', time0=0,
                 bool_tether=0, tether=teth.Tether()):
        """Initialize class.

        Inputs:
        satmass - satellite mass
        main_mass - main body mass
        state0 - [r, vr, theta, vth, phi, vp]
        form - format of state specified
        time0 - initial time
        bool_tether - 0 if tether is not deployed, 1 if deployed
        tether - tether attached to satellite
        """
        super(Satellite, self).__init__([state0[0], state0[2], state0[4]],
                                        [state0[1], state0[3], state0[5]],
                                        time0, form)
        self._mm = M_EARTH
        self._sm = satmass
        self._boolt = bool_tether
        self._teth = tether

    def set_tether(self, tether):
        """Set tether."""
        self._teth = tether
        self._boolt = 1

    def set_satmass(self, satmass):
        """Set satellite mass."""
        self._sm = satmass

    def get_tether(self):
        """Return tether."""
        return self._teth

    def get_satmass(self):
        """Return satellite mass."""
        return self._sm

    def getlatlon(self):
        """Get the laitude and longitude of the satellite."""
        lat = np.pi/2.0 - self._th
        time = self.gettime()
        lon = self._phi - 2*np.pi*time/86164.09164
        return lat, lon

    def geten(self):
        """Return the energy of the satellite."""
        return (0.5*self._sm*(self._vr**2 + self._vt**2 + self._vp**2) -
                G*self._mm*self._sm/self._r)

    def get_a(self):
        """Return the semi major axis of the satellite."""
        return -G*self._mm*self._sm/self._r

    def get_ecc(self):
        """Return eccentricity of orbit."""
        mu_mass = G*(self._mm + self._sm)
        h_mom = self.sp_ang_mom()
        vel = self.getvel_xyz()
        pos = self.getpos_xyz()
        e_vec = 1.0/mu_mass*(np.cross(vel, h_mom) -
                             mu_mass*pos/np.linalg.norm(pos))
        return e_vec

    def arg_per(self):
        """Return argument of perigee."""
        h_mom = self.sp_ang_mom()
        n_vec = np.cross([0, 0, 1], h_mom)
        e_vec = self.get_ecc()
        if np.linalg.norm(e_vec) == 0:
            small_omega = 0.0
        else:
            small_omega = np.arccos(np.dot(n_vec, e_vec) /
                                    (np.linalg.norm(n_vec) *
                                     np.linalg.norm(e_vec)))
            if e_vec[2] < 0:
                small_omega = 2*np.pi - small_omega
        return small_omega

    def true_an(self):
        """Get the true anomaly of satellite."""
        vel = self.getvel_sph()
        e_vec = self.get_ecc()
        pos = self.getpos_xyz()
        if np.linalg.norm(e_vec) == 0:
            h_mom = self.sp_ang_mom()
            e_vec = np.cross([0, 0, 1], h_mom)
        true_th = np.arccos(np.dot(e_vec, pos)/(np.linalg.norm(e_vec) *
                                                np.linalg.norm(pos)))
        if vel[0] < 0:
            true_th = 2*np.pi - true_th
        return true_th

    def orb_elem(self):
        """Return 6 orbital elements."""
        h_vec = self.sp_ang_mom()
        h_mom = np.linalg.norm(h_vec)
        inc = self.get_inc()
        cap_omega = self.get_ascnode()
        e_vec = self.get_ecc()
        ecc = np.linalg.norm(e_vec)
        small_omega = self.arg_per()
        true_th = self.true_an()
        return h_mom, inc, cap_omega, ecc, small_omega, true_th


def twobody_acc(sat):
    """Get the two body acceleration on the satellite."""
    pos = sat.getpos_sph()
    g_acc = [-G*M_EARTH/pos[0]**2, 0, 0]
    return g_acc


def tot_acc(sat):
    """Get the total acceleration on the satellite."""
    pos = sat.getpos_sph()
    lat = sat.getlatlon()[0]
    g_acc = forces.gravity_wgs84(pos[0], lat)
    return g_acc


def getorbit(sat, tfinal, tstep, trec):
    """Calculate the orbit.

    sat - Satellite object.
    tfinal - Duration for which the simulation is to be continued.
    tstep - Time steps for rk4 method.
    trec - Time durations after which the data are to be recorded.
    """
    ntimes = (int)(tfinal/tstep)
    n_tvals = (int)(tfinal/trec)
    state_arr = np.zeros((6, n_tvals))
    orbelem_arr = np.zeros((6, n_tvals))
    s_major_arr = np.zeros(n_tvals)
    count = 0
    for i in range(ntimes):
        sat.rk4_step(tstep, tot_acc)
        if i % (trec/tstep) == 0:
            state_arr[:, count] = sat.getstate()
            orbelem_arr[:, count] = sat.orb_elem()
            s_major_arr[count] = sat.get_a()
            print state_arr[0, count]
            print count
            count += 1
    return (state_arr, orbelem_arr, s_major_arr)


def plotfig(name, title, xlabel, ylabel, xdata, ydata, legend):
    """Plot figure."""
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, elem in enumerate(ydata):
        plt.plot(xdata[i], elem, label=legend[i])
    plt.legend()
    plt.savefig(name)
    plt.close()


def test_circular():
    """Test orbit function."""
    state = np.array([7.048e6, 0.0, np.pi/2.0, -7446.8984446559243, 0.0,
                      -1046.5933233558835])
    pratham = Satellite(10.0, state)
    state_arr, orbelem_arr = getorbit(pratham, 86400, 0.1, 100)[:2]
    time_array = np.linspace(0, 400, len(state_arr[0, :]))
    plotfig("test_r.png", "r v/s t", "t", "r", [time_array], [state_arr[0, :]],
            ["r"])
    plotfig("test_th.png", "theta v/s t", "t", "theta", [time_array],
            [state_arr[2, :]], ["theta"])
    plotfig("test_p.png", "phi v/s t", "t", "phi", [time_array],
            [state_arr[4, :]], ["phi"])
    plotfig("test_comega.png", "omega v/s t", "t", "omega", [time_array],
            [orbelem_arr[2, :]], ["omega"])
    # r_array = state_arr[0, :]
    # theta_array = state_arr[2, :]
    # phi_array = state_arr[4, :]

    # assert abs((np.min(r_array) - r_array[0])/r_array[0]) < 1e-5
    # assert abs((np.max(r_array) - r_array[0])/r_array[0]) < 1e-5
    # assert np.min(theta_array) > 0
    # assert np.max(theta_array) < np.pi
    # assert np.min(phi_array) > 0
    # assert np.max(phi_array) < 2*np.pi
    return orbelem_arr[2, :]*180/np.pi


test_circular()
