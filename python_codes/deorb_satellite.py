"""Program to deorbit the satellite."""

import numpy as np
import sat_orbit as sat
import tether


def deorbit(pos0, v_r0, v_tan0, inc, tether_param, tfinal, tstep):
    """Return the state and the orbital parameters during deorbiting."""
    v_t0 = -1*v_tan0*np.sin(inc)
    v_p0 = v_tan0*np.cos(inc)
    state = np.array([pos0[0], v_r0, pos0[1], v_t0, pos0[2], v_p0])
    satellite = sat.Satellite(10.0, state)
    sat_tether = tether.Tether(tether_param[0], tether_param[1],
                               tether_param[2])
    sat_tether.setlamda_a(satellite)
    sat_tether.set_iv(satellite)
    satellite.set_tether(sat_tether)
    state_arr, orbelem_arr, a_array = sat.getorbit(satellite, tfinal, tstep,
                                                   100)
    return state_arr, orbelem_arr, a_array


def plotparam(n_prefix, n_suffix, tfinal, states, orb_elems, a_arr):
    """Plot the orbital state vectors and elements."""
    time_arr = np.linspace(0, tfinal, len(states[0, :]))
    sat.plotfig(n_prefix + '_r_' + n_suffix, "Plot of r with time", 't', 'r',
                [time_arr], [states[0, :]], ['r'])
    sat.plotfig(n_prefix + '_th_' + n_suffix, "Plot of theta with time", 't',
                'theta', [time_arr], [states[2, :]], ['theta'])
    sat.plotfig(n_prefix + '_p_' + n_suffix, "Plot of phi with time", 't',
                'phi', [time_arr], [states[4, :]], ['phi'])
    sat.plotfig(n_prefix + '_vr_' + n_suffix, "Plot of vr with time", 't',
                'vr', [time_arr], [states[1, :]], ['vr'])
    sat.plotfig(n_prefix + '_vt_' + n_suffix, "Plot of vt with time", 't',
                'vt', [time_arr], [states[3, :]], ['vt'])
    sat.plotfig(n_prefix + '_vp_' + n_suffix, "Plot of vp with time", 't',
                'vp', [time_arr], [states[5, :]], ['vp'])
    sat.plotfig(n_prefix + '_h_' + n_suffix, "Plot of h with time", 't',
                'h', [time_arr], [orb_elems[0, :]], ['h'])
    sat.plotfig(n_prefix + '_i_' + n_suffix, "Plot of inc with time", 't',
                'i', [time_arr], [orb_elems[1, :]], ['i'])
    cap_omega = orb_elems[2, :]
    for i, elem in enumerate(cap_omega):
        if elem > np.pi:
            cap_omega[i] = elem - 2*np.pi
    sat.plotfig(n_prefix + '_O_' + n_suffix, "Plot of lon_node with time", 't',
                'lon_node', [time_arr], [cap_omega*180/np.pi], ['node'])
    sat.plotfig(n_prefix + '_e_' + n_suffix, "Plot of ecc with time", 't',
                'e', [time_arr], [orb_elems[3, :]], ['e'])
    sat.plotfig(n_prefix + '_ap_' + n_suffix, "Plot of arg_per with time", 't',
                'ap', [time_arr], [orb_elems[4, :]], ['arg of perigee'])
    sat.plotfig(n_prefix + '_an_' + n_suffix, "Plot of anomaly with time", 't',
                'anomaly', [time_arr], [orb_elems[5, :]], ['true anomaly'])
    sat.plotfig(n_prefix + '_a_' + n_suffix,
                "Plot of semimajor axis with time",
                't', 'a', [time_arr], [a_arr], ['energy'])


def main():
    """Deorbit satellite."""
    state_arr, orb_arr, a_arr = deorbit([7.048e6, np.pi/2.0, 0.0], 0.0,
                                        7520.083379159564, 98.0*np.pi/180.0,
                                        [500.0, 3.4014e+7, [0.001]], 518400.0,
                                        0.1)
    plotparam('98_wgs84', '6days', 518400.0, state_arr, orb_arr, a_arr)
    return state_arr, orb_arr, a_arr


if __name__ == '__main__':
    main()
