import numpy as np 
import matplotlib.pyplot as plt 
import igrf12

G = 6.67408e-11
M = 5.972e24  

def getB_NED(lat, longit, r):
    B_x, B_y, B_z, B_t = igrf12.igrf12syn(0, 2017, 2, r, (np.pi/2 - lat)*180/np.pi, longit*180/np.pi)
    return B_x, B_y, -1*B_z

def getB_sph(lat, longit, r):
    B_north, B_east, B_down = getB_NED(lat, longit, r)
    B_r = -1*B_down
    B_t = -1*B_north
    B_p = B_east
    return B_r, B_t, B_p

class orbit_state:
    def __init__(self, r, rdot, theta, t_dot, phi, p_dot):
        self.r = r
        self.rdot = rdot
        self.theta = theta
        self.t_dot = t_dot
        self.phi = phi
        self.p_dot = p_dot

    def set_state(self, r, rdot, theta, t_dot, phi, p_dot):
        self.r = r
        self.rdot = rdot
        self.theta = theta
        self.t_dot = t_dot
        self.phi = phi
        self.p_dot = p_dot

    def get_state(self):
        return self.r, self.rdot, self.theta, self.t_dot, self.phi, self.p_dot

    def get_pos(self):
        return self.r, self.theta, self.phi

    def getvel_sph(self):
        return self.rdot, self.r*self.t_dot, self.r*np.sin(self.theta)*self.p_dot

    def getdotdot(self, a_r, a_t, a_p):
        rdotdot = a_r + self.r*self.t_dot**2 + self.r*self.p_dot**2*np.sin(self.theta)**2
        t_dotdot = (a_t - 2*self.rdot*self.t_dot + self.r*self.p_dot**2*np.sin(self.theta)*np.cos(self.theta))/(self.r)
        p_dotdot = (a_p - 2*self.rdot*self.p_dot*np.sin(self.theta) - 2*self.r*self.t_dot*self.p_dot*np.cos(self.theta))/(self.r*np.sin(self.theta))
        return rdotdot, t_dotdot, p_dotdot

class sat_param(orbit_state):
    def __init__(self, r, rdot, theta, t_dot, phi, p_dot, length, mass, resis, bool_B):
        self.r = r
        self.rdot = rdot
        self.theta = theta
        self.t_dot = t_dot
        self.phi = phi
        self.p_dot = p_dot
        self.leng = length
        self.mass = mass
        self.time = 0
        self.res = resis
        self.b_B = bool_B

    def getm(self):
        return self.mass

    def getlen(self):
        return self.leng

    def settime(self, t):
        self.time = t

    def get_res(self):
        return self.res

    def gettime(self):
        return self.time

    def getlatlong(self):
        lat = np.pi/2.0 - self.theta
        longit = self.phi - 2*np.pi*self.time/86164.09164
        longit = longit - np.floor(longit/(2*np.pi))*2*np.pi
        return lat, longit

    def getB(self):
        lat, longit = self.getlatlong()
        B_r, B_t, B_p = getB_sph(lat, longit, self.r)
        return B_r, B_t, B_p

    def getcurr(self):
        B_r, B_t, B_p = self.getB()
        v_r, v_t, v_p = self.getvel_sph()
        return (v_t*B_p - v_p*B_t)*self.leng/self.res     ## only r component

    def accln(self):
        a_r = -G*M/self.r**2
        if (self.b_B == 1):
            B_r, B_t, B_p = self.getB()
            a_t = -1*self.getcurr()*self.leng*B_p
            a_p = self.getcurr()*self.leng*B_t
        elif (self.b_B == 0):
            a_t = 0
            a_p = 0
        return a_r, a_t, a_p

def rk4_step(sat, tstep):
    r0, rdot0, theta0, t_dot0, phi0, p_dot0 = sat.get_state()
    t0 = sat.gettime()
    k = np.zeros((6, 4))
    for i in range(3):
        r, rdot, theta, t_dot, phi, p_dot = sat.get_state()
        k[0, i] = rdot*tstep
        k[2, i] = t_dot*tstep
        k[4, i] = p_dot*tstep
        acc_r, acc_t, acc_p = sat.accln()
        rdotdot, t_dotdot, p_dotdot = sat.getdotdot(acc_r, acc_t, acc_p)
        k[1, i] = rdotdot*tstep
        k[3, i] = t_dotdot*tstep
        k[5, i] = p_dotdot*tstep
        if (i < 2):
            sat.settime(t0 + tstep/2)
            sat.set_state(r0 + k[0, i]/2, rdot0 + k[1, i]/2, theta0 + k[2, i]/2, t_dot0 + k[3, i]/2, phi0 + k[4, i]/2, p_dot0 + k[5, i]/2)
        elif (i == 2):
            sat.settime(t0 + tstep)
            sat.set_state(r0 + k[0, i], rdot0 + k[1, i], theta0 + k[2, i], t_dot0 + k[3, i], phi0 + k[4, i], p_dot0 + k[5, i])
    r_new = r0 + (k[0, 0] + 2*k[0, 1] + 2*k[0, 2] + k[0, 3])/6.0
    rdot_new = rdot0 + (k[1, 0] + 2*k[1, 1] + 2*k[1, 2] + k[1, 3])/6.0
    theta_new = theta0 + (k[2, 0] + 2*k[2, 1] + 2*k[2, 2] + k[2, 3])/6.0
    t_dot_new = t_dot0 + (k[3, 0] + 2*k[3, 1] + 2*k[3, 2] + k[3, 3])/6.0
    phi_new = phi0 + (k[4, 0] + 2*k[4, 1] + 2*k[4, 2] + k[4, 3])/6.0
    phi_new = phi_new - np.floor(phi_new/(2*np.pi))*2*np.pi
    p_dot_new = p_dot0 + (k[5, 0] + 2*k[5, 1] + 2*k[5, 2] + k[5, 3])/6.0
    sat.settime(t0 + tstep)
    sat.set_state(r_new, rdot_new, theta_new, t_dot_new, phi_new, p_dot_new)

def orbit(sat, tfinal, tstep):
    ntimes = (int)(tfinal/tstep)
    n_t = (int)(tfinal/100)
    r_array = np.zeros(n_t)
    theta_array = np.zeros(n_t)
    phi_array = np.zeros(n_t)
    long_arr = np.zeros(n_t)
    lat_arr = np.zeros(n_t)
    count = 0
    for i in range(ntimes):
        rk4_step(sat, tstep)
        #print i
        if (i % (int)(100/tstep) == 0):
            r_array[count] = sat.get_state()[0]
            print (r_array[count])
            theta_array[count] = sat.get_state()[2]
            phi_array[count] = sat.get_state()[4]
            lat_arr[count], long_arr[count] = sat.getlatlong()
            count += 1
            print (count)

    return r_array, theta_array, phi_array, lat_arr, long_arr

def plotfig(name, title, xlabel, ylabel, xdata, ydata, legend):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(ydata)):
        if (xdata[i] == []):
            plt.plot(ydata[i], label = legend[i])
        else:
            plt.plot(xdata[i], ydata[i], label = legend[i])
    plt.legend()
    plt.savefig(name)
    plt.close()

def test_circular():
    pratham = sat_param(7.048e6, 0, np.pi/2.0, -0.0010565973956662776, 0, -0.0001484950799313115, 100, 10, 10, 0)
    r_array, theta_array, phi_array, lat_arr, long_arr = orbit(pratham, 10800, 0.001)
    time_array = np.linspace(0, 10800, len(r_array))
    plotfig("r_arr0.png", "r v/s t", "t", "r", [time_array], [r_array], ["r"])
    plotfig("t_arr0.png", "theta v/s t", "t", "theta", [time_array], [theta_array], ["theta"])
    plotfig("phi_arr0.png", "phi v/s t", "t", "phi", [time_array], [phi_array], ["phi"])

    assert abs((np.min(r_array) - r_array[0])/r_array[0]) < 1e-5
    assert abs((np.max(r_array) - r_array[0])/r_array[0]) < 1e-5
    assert np.min(theta_array) > 0
    assert np.max(theta_array) < np.pi 
    assert np.min(phi_array) > 0
    assert np.max(phi_array) < 2*np.pi

if __name__ == "__main__":
    test_circular()
