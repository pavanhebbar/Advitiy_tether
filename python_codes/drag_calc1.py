"""Program to calculate drag on tether."""

import numpy as np


def genparticles(num_den, dim, mass, temp):
    """Generate the particles and assign position and velocity.

    Inputs:
    num_den - Number density of particles
    dim - Dimensions of simulation box
    mass - Mass of the particle
    temp - Temperature

    Outputs:
    pos - Sorted array of coordinates as per x coordinates
    vel_x = x velocity
    """
    pos = np.random.rand((int)(num_den*dim[0]*dim[1]*dim[2]), 3)*dim
    sigma_v = (1.38064852*10**-23*temp/mass)**0.5
    vel_x = np.random.normal(0, sigma_v, (int)(num_den*dim[0]*dim[1]*dim[2]))
    vel = vel[pos[:, 0]
    pos = pos[pos[:, 0].argsort()]
    return pos, vel_x.sort()
