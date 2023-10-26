import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time


def refl_time(offset, angle, velocity, depth=405):
    refl_timing = 2*np.sqrt(
                            (depth ** 2) +
                            ((offset/2)**2 * np.cos(np.deg2rad(angle))**2)
    ) / velocity
    return refl_timing


def depthvel_gridsearch(primary, secondary):
    gridtime = time.time()
    depth_array = np.arange(450, 500, 1)
    vel_array = np.arange(3600, 3800, 1)
    sumsq_array = np.zeros((len(depth_array), len(vel_array)))
    for i in range(len(depth_array)):
        for j in range(len(vel_array)):
            for k in range(len(primary)):
                sumsq_array[i, j] += np.sum((refl_time(primary[k].dist, 0, vel_array[j], depth_array[i]) - secondary[k].tmax)**2)
    gridtime = time.time() - gridtime
    print('Gridsearch time: ' + str(gridtime) + ' seconds')
    return depth_array, vel_array, sumsq_array


def depthvel_gridsearch_plot(primary, secondary, prior=[3630, 405]):
    depth_array, vel_array, sumsq_array = depthvel_gridsearch(primary, secondary)
    # Plot with a logarithmic scale contour plot
    plt.contourf(vel_array, depth_array, np.log(sumsq_array))
    plt.plot(prior[0], prior[1], 'r*')
    # Plot with a linear scale contour plot
    # plt.contourf(vel_array, depth_array, sumsq_array)
    plt.colorbar()
    plt.title('Sum of squares of differences between theoretical and actual traveltime curves')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Depth (m)')
    plt.grid()
    plt.grid(which='minor', linestyle='--')
    plt.minorticks_on()
    plt.show()