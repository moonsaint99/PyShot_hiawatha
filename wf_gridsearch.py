import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
import obspy as op
import multiprocessing
import wf_model as wfm

def refl_time(offset, angle, velocity, depth=405):
    refl_timing = 2*np.sqrt(
                            (depth ** 2) +
                            ((offset/2)**2 * np.cos(np.deg2rad(angle))**2)
    ) / velocity
    return refl_timing

# def depthvel_gridsearch_wf(streamdict):
#     gridtime = time.time()
#     depth_array = np.arange(380, 500, 1)
#     vel_array = np.arange(3300, 4200, 1)
#     sumsq_array = np.zeros((len(depth_array), len(vel_array)))
#     tracecount = 0
#     for key, stream in streamdict.items():
#         tracecount += len(stream)
#     operation_length = len(depth_array)* len(vel_array) * tracecount
#     print('Operation length: ' + str(operation_length))
#     operations_done = 0
#     for i in range(len(depth_array)):
#         for j in range(len(vel_array)):
#             for filename, stream in streamdict.items():
#                 for trace in stream:
#                     operations_done += 1
#                     # print('\rOperation ' + str(operations_done/operation_length*100) + '% complete')
#                     print('Estimated time remaining: ' + str((time.time() - gridtime) * (operation_length - operations_done) / operations_done) + ' seconds', end='\r', flush=True)
#                     theor_timing = refl_time(trace.stats.distance, 0, vel_array[j], depth_array[i])
#                     sumsq_array[i, j] += np.sum((trace.data[int(theor_timing/trace.stats.delta)])**2)
#     return depth_array, vel_array, sumsq_array
#
# def depthvel_gridsearch_wf_plot(streamdict, prior=[3630, 405]):
#     depth_array, vel_array, sumsq_array = depthvel_gridsearch_wf(streamdict)
#     # Plot with a logarithmic scale contour plot
#     plt.contourf(vel_array, depth_array, np.log(sumsq_array))
#     plt.plot(prior[0], prior[1], 'r*')
#     # Plot with a linear scale contour plot
#     # plt.contourf(vel_array, depth_array, sumsq_array)
#     plt.colorbar()
#     plt.title('Sum of squares of differences between theoretical and actual traveltime curves')
#     plt.xlabel('Velocity (m/s)')
#     plt.ylabel('Depth (m)')
#     plt.show()


def corrected_amplitude(trace, time):
    if int(time / trace.stats.delta) < len(trace.data):
        return trace.data[int(time / trace.stats.delta)] * trace.stats.distance
    else:
        return 0


def amplitude_summing_task(params):
    depth_index, velocity_index, depth, velocity, streamdict = params
    sumsq = 0
    for filename, stream in streamdict.items():
        for trace in stream:
            angle = np.arctan(trace.stats.distance / depth)
            theor_timing = refl_time(trace.stats.distance, angle, velocity, depth)
            if int(theor_timing / trace.stats.delta) < len(trace.data):
                sumsq += np.sum((corrected_amplitude(trace, theor_timing) ** 2))
            else:
                pass
    return depth_index, velocity_index, sumsq


def depthvel_gridsearch_wf(streamdict):
    gridtime = time.time()
    depth_array = np.arange(380, 500, 1)
    vel_array = np.arange(3300, 4200, 1)
    sumsq_array = np.zeros((len(depth_array), len(vel_array)))

    params_list = [(i, j, depth_array[i], vel_array[j], streamdict)
                   for i in range(len(depth_array)) for j in range(len(vel_array))]

    with multiprocessing.Pool() as pool:
        results = pool.map(amplitude_summing_task, params_list)
    pool.close()

    for result in results:
        depth_index, velocity_index, sumsq = result
        sumsq_array[depth_index, velocity_index] = sumsq

    print("Elapsed time:", time.time() - gridtime, "seconds")

    return depth_array, vel_array, sumsq_array


# define a function that takes a trace and returns the L2 misfit to a 2-layer model
def l2_misfit_task(trace, depth, mat1, mat2):
    synthetic = wfm.retrieve_seismogram_2layer(np.arange(0, trace.stats.npts * trace.stats.delta, trace.stats.delta),
                                               depth, mat1, mat2, trace.stats.distance, trace.stats.delta)
    return np.linalg.norm(synthetic - trace.data)


def synthetic_summing_task(params):
    depth_index, mat1_index, mat2_index, depth, mat1, mat2, streamdict = params
    sumsq = 0
    for filename, stream in streamdict.items():
        for trace in stream:
            L2_misfit = l2_misfit_task(trace, depth, mat1, mat2)
            sumsq += L2_misfit
    return depth_index, mat1_index, mat2_index, sumsq


def depthmat_gridsearch_wf(streamdict):
    gridtime = time.time()
    depth_array = np.arange(390, 420, 1)
    mat1_rho_array = np.arange(916, 922, 1)
    mat1_vp_array = np.arange(3500, 3800, 1)
    mat1_vs_array = mat1_vp_array/2
    mat2_rho_array = np.arange(2700, 3300, 50)
    mat2_vp_array = np.arange(500, 6000, 50)
    mat2_vs_array = mat2_vp_array/1.82



def depthvel_gridsearch_wf_plot(streamdict, prior=[3630, 405]):
    depth_array, vel_array, sumsq_array = depthvel_gridsearch_wf(streamdict)

    # plt.figure(figsize=(10, 10))
    plt.contourf(vel_array, depth_array, sumsq_array)
    plt.plot(prior[0], prior[1], 'r*')
    plt.colorbar()
    plt.title('Sum of squares of trace amplitudes')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Depth (m)')
    plt.show()
