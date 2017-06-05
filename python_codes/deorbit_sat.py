"""Program to analyse the deorbitting of satellite."""

import numpy as np
import satellite as sat
import tether as teth


def deorbit(r0, vr0, v_tan0, inc, theta0, phi0, tether_len, tether_res,
            sat_m, sat_I, tfinal, tstep):
    """Return the state and orbital parameters during deorbitting."""
    v_r = vr0
    v_t = -1*v_tan0*np.sin(inc*np.pi/180)
    v_p = v_tan0*np.cos(inc*np.pi/180)
    tether_sat = teth.tether(tether_len, tether_res)
    satel = sat.satellite(sat_m, sat_I, 1, tether_sat, r0, v_r, theta0, v_t,
                          phi0, v_p, 0)
    state_arr, orb_arr, pow_arr, en_arr, acc_arr, emf_arr = sat.getorbit(
        satel, tfinal, tstep)
    pow_arr2 = np.zeros(len(pow_arr))
    for i in range(len(pow_arr)):
        pedot = sat.G*sat.M_EARTH*sat_m/state_arr[0, i]**2*state_arr[1, i]
        vel = [state_arr[1, i], state_arr[3, i], state_arr[5, i]]
        pow_arr2[i] = np.dot(acc_arr[:, i], vel)*sat_m + pedot
    return state_arr, orb_arr, pow_arr, pow_arr2, en_arr, acc_arr, emf_arr


def plotparam(n_prefix, n_suffix, tfinal, states, orb_elems, pow_diff, en_arr,
              emf_arr):
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
    sat.plotfig(n_prefix + '_O_' + n_suffix, "Plot of lon_node with time", 't',
                'lon_node', [time_arr], [orb_elems[2, :]], ['node'])
    sat.plotfig(n_prefix + '_e_' + n_suffix, "Plot of ecc with time", 't',
                'e', [time_arr], [orb_elems[3, :]], ['e'])
    sat.plotfig(n_prefix + '_ap_' + n_suffix, "Plot of arg_per with time", 't',
                'ap', [time_arr], [orb_elems[4, :]], ['arg of perigee'])
    sat.plotfig(n_prefix + '_an_' + n_suffix, "Plot of anomaly with time", 't',
                'anomaly', [time_arr], [orb_elems[5, :]], ['true anomaly'])
    sat.plotfig(n_prefix + '_en_' + n_suffix, "Plot of energy with time", 't',
                'en', [time_arr], [en_arr], ['energy'])
    sat.plotfig(n_prefix + '_emf_' + n_suffix, "Plot of emf with time", 't',
                'emf', [time_arr], [-1*emf_arr], ['emf'])
    sat.plotfig(n_prefix + '_pow_' + n_suffix, "Comparin power arrays", 't',
                'diff', [time_arr], [pow_diff], ['pow_arr - pow_arr2'])


def main():
    """Deorbit the satellite."""
    state_arr, orb_arr, pow_arr, pow_arr2, en_arr, acc_arr, emf_arr = deorbit(
        7.378e6, 0.0, 7349.982044408744, 91.0, np.pi/2.0, 0.0, 500.0, 10.0,
        10.0, np.zeros((3, 3)), 86400.0, 0.1)
    powdiff = pow_arr - pow_arr2
    plotparam('90', 'test', 86400.0, state_arr, orb_arr, powdiff, en_arr,
              emf_arr)
    return state_arr, orb_arr, pow_arr, pow_arr2, en_arr, acc_arr, emf_arr
