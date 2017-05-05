import numpy as np 
import tether_orbit as torb 

def deorbit_sat(r0, v_r0, v_tan0, inc, theta0, phi0, leng, resis, mass, tfinal, tstep):
    rdot0 = v_r0
    t_dot0 = -1*v_tan0*np.sin(inc*np.pi/180)/r0
    p_dot0 = v_tan0*np.cos(inc*np.pi/180)/r0
    sat = torb.sat_param(r0, rdot0, theta0, t_dot0, phi0, p_dot0, leng, mass, resis, 1)
    r_arr, t_arr, p_arr, lat_arr, long_arr = torb.orbit(sat, tfinal, tstep)
    return r_arr, t_arr, p_arr, lat_arr, long_arr

def plot_param(n_suffix, n_prefix, tfinal, r_arr, t_arr, p_arr, lat_arr, long_arr):
    time_arr = np.linspace(0, tfinal, len(r_arr))
    torb.plotfig(n_prefix + "r_" + n_suffix, "Plot of r v/s t", "t (s)", "r (m)", [time_arr], [r_arr], ["r"])
    torb.plotfig(n_prefix + "thet_" + n_suffix, "Plot of theta v/s t", "t (s)", "theta (deg)", [time_arr], [t_arr*180/np.pi], ["theta"])
    torb.plotfig(n_prefix + "p_" + n_suffix, "Plot of r v/s t", "t (s)", "phi (deg)", [time_arr], [p_arr*180/np.pi], ["phi"])
    torb.plotfig(n_prefix + "latlong_" + n_suffix, "Plot of r v/s t", "t (s)", "lat (m)", [long_arr*180/np.pi], [lat_arr*180/np.pi], ["sat path"])
    torb.plotfig(n_prefix + "t_p_" + n_suffix, "Comparing theta and phi", "t (s)", "theta, phi", [time_arr, time_arr], [t_arr*180/np.pi, p_arr*180/np.pi], ["theta", "phi"])

def main():
    r_arr, t_arr, p_arr, lat_arr, long_arr  = deorbit_sat(7.048e6, 0, 7520.083379159564, 98, np.pi/2, 0, 100, 10, 10, 172800, 0.001)
    plot_param("", "2day", 172800, r_arr, t_arr, p_arr, lat_arr, long_arr)

if __name__=="__main__":
    main()