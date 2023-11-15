import numpy as np
import matplotlib as mpl
# mpl.use('macosx')
import matplotlib.pyplot as plt
import scipy as sp

import op_pickfile
import pick_gridsearch as gs
import os

import su_pickfile as supf
import op_pickfile as pf
import pickanalysis as pa
import firn_analysis as fa

# This script will use picked amplitudes of the primary and secondary seismic
# arrivals to calculate the ratio of the two amplitudes.


lithic_smallstack_names, primary_lithic_smallstacks, secondary_lithic_smallstacks, firn_lithic_smallstacks = op_pickfile.assimilate_pickdata('pickdata_lithic_smallstack')

primary_lithicstack = pf.opPickfile('primary_shotpicks_output/stack_onspread_good.csv')
secondary_lithicstack = pf.opPickfile('secondary_shotpicks_output/stack_onspread_good.csv')
primary_lithic_bigstack = [primary_lithicstack]
secondary_lithic_bigstack = [secondary_lithicstack]


# Perform a firn velocity inversion
inversion_results = fa.double_linear_exponential_fit(primary_lithic_smallstacks)

# ref_lithic_stack_pair = pa.reflectivity(primary_lithic_fullstack, secondary_lithic_fullstack, 'pair', refl_polarity='min')
# ref_lithic_stack_dir_lin = pa.reflectivity(primary_lithic_fullstack, secondary_lithic_fullstack, 'dir_lin', refl_polarity='min')
# ref_high_error = reflectivity(primary_lithic_fullstack, secondary_lithic_fullstack, attn_low, polarity='min')
# ref_low_error = reflectivity(primary_lithic_fullstack, secondary_lithic_fullstack, attn_high, polarity='min')
# ref_shots_pair = []
# ref_shots_dir_lin = []
# for i in range(len(primary_shots)):
#     pair = pa.reflectivity(primary_shots[i], secondary_shots[i], 'pair', refl_polarity='min')
#     if pair != 'No pairs found':
#         ref_shots_pair.append(pair)
#         plt.plot(np.rad2deg(primary_shots[i].angle), pair, zorder=1, marker='o', markersize=5, linestyle='none')
#     ref_shots_dir_lin.append(pa.reflectivity(primary_shots[i], secondary_shots[i], 'dir_lin', attn_coeff=0.27e-3, refl_polarity='min'))
#     plt.plot(np.rad2deg(primary_shots[i].angle), ref_shots_dir_lin[i], zorder=1, marker='o', markersize=5, linestyle='none')
ref_shots_pair_smallstack = []
ref_shots_dir_lin_smallstack = []
ref_shots_simple_smallstack = []
ref_shots_simple_smallstack_upper_error = []
ref_shots_simple_smallstack_lower_error = []
for i in range(len(primary_lithic_smallstacks)):
#     # pair = pa.reflectivity(primary_lithic_smallstacks[i], secondary_lithic_smallstacks[i], 'pair', attn_coeff=0.27e-3, polarity='min')
#     # if pair != 'No pairs found':
#     #     ref_shots_pair_smallstack.append(pair)
#     #     plt.plot(np.rad2deg(primary_lithic_smallstacks[i].angle), pair, zorder=1, marker='o', markersize=5, linestyle='none')
#     # ref_shots_dir_lin_smallstack.append(pa.reflectivity(primary_lithic_smallstacks[i], secondary_lithic_smallstacks[i], 'dir_lin', attn_coeff=0.27e-3, polarity='min'))
#     # plt.plot(np.rad2deg(primary_lithic_smallstacks[i].angle), ref_shots_dir_lin_smallstack[i], zorder=1, marker='o', markersize=5, linestyle='none')
    simple_reflectivity = pa.reflectivity(primary_lithic_smallstacks[i], secondary_lithic_smallstacks[i], 'simple', inversion_results, attn_coeff=2.7e-4, polarity='min')
    ref_shots_simple_smallstack.append(simple_reflectivity[0])
    ref_shots_simple_smallstack_upper_error.append(simple_reflectivity[1] - simple_reflectivity[0])
    ref_shots_simple_smallstack_lower_error.append(simple_reflectivity[0] - simple_reflectivity[2])
#
#     angle = np.rad2deg(primary_lithic_smallstacks[i].angle)
#     refl = ref_shots_simple_smallstack[i]
#     plt.plot(angle,refl, zorder=1, marker='o', markersize=5, linestyle='none')
#     plt.errorbar(np.rad2deg(primary_lithic_smallstacks[i].angle), ref_shots_simple_smallstack[i], yerr=[ref_shots_simple_smallstack_lower_error[i], ref_shots_simple_smallstack_upper_error[i]], zorder=0, fmt='none', ecolor='k', capsize=3)

ref_shots_simple_bigstack = []
ref_shots_simple_bigstack_upper_error = []
ref_shots_simple_bigstack_lower_error = []
for i in range(len(primary_lithic_bigstack)):
    simple_reflectivity = pa.reflectivity(primary_lithic_bigstack[i], secondary_lithic_bigstack[i], 'simple', inversion_results, attn_coeff=2.7e-4, polarity='min')
    ref_shots_simple_bigstack.append(simple_reflectivity[0])
    ref_shots_simple_bigstack_upper_error.append(simple_reflectivity[1] - simple_reflectivity[0])
    ref_shots_simple_bigstack_lower_error.append(simple_reflectivity[0] - simple_reflectivity[2])

    angle = np.rad2deg(primary_lithic_bigstack[i].angle)
    refl = ref_shots_simple_bigstack[i]
    plt.plot(angle,refl, zorder=1, marker='o', markersize=5, linestyle='none')
    plt.errorbar(np.rad2deg(primary_lithic_bigstack[i].angle), ref_shots_simple_bigstack[i], yerr=[ref_shots_simple_bigstack_lower_error[i], ref_shots_simple_bigstack_upper_error[i]], zorder=0, fmt='none', ecolor='k', capsize=3)



# ref_stack_simple = pa.reflectivity(primary_lithic_fullstack, secondary_lithic_fullstack, 'simple', attn_coeff=2.7e-4, polarity='min')

# # Plot reflectivity as a function of angle
plt.grid()
# plt.scatter(np.rad2deg(primary_lithic_fullstack.angle), ref_stack_simple, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_lithic_fullstack.angle), ref_lithic_stack_pair, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_lithic_fullstack.angle), ref_lithic_stack_dir_lin, zorder=1, s=20)
# plt.errorbar(np.rad2deg(primary_lithic_fullstack.angle), ref_lithic_stack, yerr=[ref_low_error, ref_high_error], zorder=0, fmt='none', ecolor='k', capsize=3)
plt.ylim([-1, 4])
plt.title('Reflectivity as fxn of angle')
plt.ylabel('Reflectivity')
plt.xlabel('Angle (deg)')
plt.show()


def refl_time(offset, angle, velocity, depth=405):
    refl_timing = 2*np.sqrt(
                            (depth ** 2) +
                            ((offset/2)**2 * np.cos(np.deg2rad(angle))**2)
    ) / velocity
    return refl_timing

def zerodip_depth(reflectiontiming, offset, velocity):
    return np.sqrt((reflectiontiming*velocity/2)**2 - (offset/2)**2)
# # Plot theoretical correct time difference between primary and secondary
# # We'll do this using the primary arrival travel time curve generated by the inversion
# refl_tdiff = np.array([])
# refl_tdiff_inv = np.array([])
# refl_timingarray_0deg = np.array([])
# refl_timingarray_10deg = np.array([])
# refl_timingarray_20deg = np.array([])
# refl_deep = np.array([])
# for i in range(1, 200, 1):
#     # refl_timing = 2*np.sqrt(i**2 + 405**2)/3830
#     refl_timingarray_0deg = np.append(refl_timingarray_0deg, refl_time(i, 0, 3600, 405))
#     refl_timingarray_10deg = np.append(refl_timingarray_10deg, refl_time(i, -10, 3600, 405))
#     refl_timingarray_20deg = np.append(refl_timingarray_20deg, refl_time(i, -20, 3600, 405))
#     # refl_tdiff = np.append(refl_tdiff, refl_time(i, 0, 3600, 405) - i/3600)
#     refl_deep = np.append(refl_deep, refl_time(i, 0, 3710, 477))
#     # refl_tdiff_inv = np.append(refl_tdiff_inv, refl_timing - inv1(i, inversion_results[0][0], inversion_results[0][1],
#     # inversion_results[0][2], inversion_results[0][3], inversion_results[0][4]))

# # Plot data vs theoretical reflection traveltime
# # plt.plot(primary_stack.dist, secondary_stack.tmin)
# #plt.plot(primary_stack.dist, secondary_stack.tmax)
# plt.plot(primary_72.dist, secondary_72.tmax)
# plt.plot(primary_73.dist, secondary_73.tmax)
# plt.plot(primary_74.dist, secondary_74.tmax)
# plt.plot(primary_76.dist, secondary_76.tmax)
# plt.plot(primary_77.dist, secondary_77.tmax)
# plt.plot(primary_78.dist, secondary_78.tmax)
# plt.plot(primary_79.dist, secondary_79.tmax)
# plt.plot(primary_80.dist, secondary_80.tmax)
# plt.plot(primary_81.dist, secondary_81.tmax)
# plt.plot(primary_82.dist, secondary_82.tmax)
# plt.plot(np.arange(1,200,1), refl_timingarray_0deg, zorder=0)
# plt.plot(np.arange(1,200,1), refl_timingarray_10deg, zorder=0)
# plt.plot(np.arange(1,200,1), refl_timingarray_20deg, zorder=0)
# # plt.legend(['Data', 'Theoretical'])
# plt.title('Data vs theoretical reflection traveltime')
# # plt.legend(['Shot 1', 'Shot 2', 'Shot 3', '0 deg bed', '10 deg bed', '20 deg bed'])
# plt.xlabel('Offset (m)')
# plt.ylabel('Traveltime (s)')
# plt.show()

# # Plot zero dip reflector depth
# plt.title('Reflector depth assuming zero reflector dip')
# #plt.plot(primary_stack.dist/2, zerodip_depth(secondary_stack.tmin, secondary_stack.dist, 3800))
# plt.plot(primary_73.dist/2, zerodip_depth(secondary_73.tmax, secondary_73.dist, 3650))
# plt.plot(primary_74.dist/2, zerodip_depth(secondary_74.tmax, secondary_74.dist, 3650))
# plt.ylim(400, 450)
# plt.show()


# # Plot direct arrival time vs offset
# primary_known_slope, primary_known_intercept = sp.stats.linregress(primary_lithic_fullstack.tmin, primary_lithic_fullstack.dist)[0:2]
#
# plt.scatter(primary_lithic_fullstack.dist, primary_lithic_fullstack.tmin, s=50, marker="o", zorder=0, facecolors='none', edgecolors='k')
# # plt.plot(np.arange(0,0.06,0.001)*primary_known_slope+primary_known_intercept, np.arange(0,0.06,0.001), zorder=0, linewidth=0.5)
# plt.plot(np.arange(0, 125, 200), pa.inv1(np.arange(0, 125, 200), *inversion_results[0]), zorder=0, linewidth=0.5)
# plt.title('Direct arrival time vs offset')
# plt.xlabel('Offset (m)')
# plt.ylabel('Traveltime (s)')
# plt.show()

# Plot traveltime of lithic reflector vs offset
# plt.plot(primary_lithic_fullstack.dist_no_outliers, secondary_lithic_fullstack.tmin_no_outliers)
# plt.plot(primary_29.dist_no_outliers, secondary_29.tmin_no_outliers)
# plt.plot(primary_30.dist_no_outliers, secondary_30.tmin_no_outliers)
# plt.plot(primary_31.dist_no_outliers, secondary_31.tmin_no_outliers)
# plt.plot(primary_33.dist_no_outliers, secondary_33.tmin_no_outliers)
# plt.plot(primary_34.dist_no_outliers, secondary_34.tmin_no_outliers)
# plt.xlabel('Offset (m)')
# plt.ylabel('Traveltime (s)')
# plt.plot(np.arange(1,200,1), refl_deep, zorder=0)
# plt.show()

# gs.depthvel_gridsearch_plot([primary_lithic_fullstack], [secondary_lithic_fullstack], prior=[3710, 477])
