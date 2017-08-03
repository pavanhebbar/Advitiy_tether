"""Defines a class to handle basic parameters of satellite."""

import numpy as np
import orbit_state as orb
import tether as teth


G = 6.67408e-11
M_EARTH = 5.972e24


class Satellite(orb.orb_param3D):
    """Set of basic functions to handle satellite.

    All these functions assume that the satellite is revolving around earth
    unless specified
    """

    def __init__(self, satmass, state0, form='sph', time0=0,
                 bool_tether=0, tether=teth.tether()):
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


def tot_acc(sat):
    """Get the total acceleration on the satellite."""
    return 0, 0, 0


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
            s_major_arr[:, count] = sat.get_a()
            print state_arr[0, count]
            print count
            count += 1
    return (state_arr, orbelem_arr, s_major_arr)
