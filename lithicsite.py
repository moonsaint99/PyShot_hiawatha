import numpy as np
import matplotlib as mpl
# mpl.use('macosx')
import matplotlib.pyplot as plt
import scipy as sp
import pick_gridsearch as gs
import os

# This script will use picked amplitudes of the primary and secondary seismic
# arrivals to calculate the ratio of the two amplitudes.


import pickfile as pf
import pickanalysis as pa


# primary_29 = pf.Pickfile('pickdata_lithic/29_primary.info', 'incr', 115, outliers=3)
# primary_30 = pf.Pickfile('pickdata_lithic/30_primary.info', 'incr', 115, outliers=3)
# primary_31 = pf.Pickfile('pickdata_lithic/31_primary.info', 'incr', 115, outliers=3)
# primary_33 = pf.Pickfile('pickdata_lithic/33_primary.info', 'incr', 100, outliers=5)
# primary_34 = pf.Pickfile('pickdata_lithic/34_primary.info', 'incr', 100, outliers=5)
#
# secondary_29 = pf.Pickfile('pickdata_lithic/29_secondary.info', 'incr', 'secondary', outliers=3)
# secondary_30 = pf.Pickfile('pickdata_lithic/30_secondary.info', 'incr', 'secondary', outliers=3)
# secondary_31 = pf.Pickfile('pickdata_lithic/31_secondary.info', 'incr', 'secondary', outliers=3)
# secondary_33 = pf.Pickfile('pickdata_lithic/33_secondary.info', 'incr', 'secondary', outliers=5)
# secondary_34 = pf.Pickfile('pickdata_lithic/34_secondary.info', 'incr', 'secondary', outliers=5)

p29 = pf.Pickfile('lithic_shotpicks/29_primary.info', 'incr', 115, outliers=3, maxrows=22)
p30 = pf.Pickfile('lithic_shotpicks/30_primary.info', 'incr', 115, outliers=3, maxrows=22)
p31 = pf.Pickfile('lithic_shotpicks/31_primary.info', 'incr', 115, outliers=3, maxrows=22)
p32 = pf.Pickfile('lithic_shotpicks/32_primary.info', 'incr', 100, outliers=3, maxrows=22)
p33 = pf.Pickfile('lithic_shotpicks/33_primary.info', 'incr', 100, outliers=5, maxrows=22)
p34 = pf.Pickfile('lithic_shotpicks/34_primary.info', 'incr', 100, outliers=5, maxrows=22)
p35 = pf.Pickfile('lithic_shotpicks/35_primary.info', 'incr', 85, outliers=5, maxrows=22)
p36 = pf.Pickfile('lithic_shotpicks/36_primary.info', 'incr', 85, outliers=5, maxrows=22)
p37 = pf.Pickfile('lithic_shotpicks/37_primary.info', 'incr', 70, outliers=5, maxrows=22)
p39 = pf.Pickfile('lithic_shotpicks/39_primary.info', 'incr', 70, outliers=5, maxrows=22)
p40 = pf.Pickfile('lithic_shotpicks/40_primary.info', 'incr', 70, outliers=5, maxrows=22)
p41 = pf.Pickfile('lithic_shotpicks/41_primary.info', 'incr', 55, outliers=5, maxrows=22)
p42 = pf.Pickfile('lithic_shotpicks/42_primary.info', 'incr', 55, outliers=5, maxrows=22)
p43 = pf.Pickfile('lithic_shotpicks/43_primary.info', 'incr', 40, outliers=5, maxrows=22)
p44 = pf.Pickfile('lithic_shotpicks/44_primary.info', 'incr', 40, outliers=5, maxrows=22)
p45 = pf.Pickfile('lithic_shotpicks/45_primary.info', 'incr', 25, outliers=5, maxrows=22)
p46 = pf.Pickfile('lithic_shotpicks/46_primary.info', 'incr', 25, outliers=5, maxrows=22)
p47 = pf.Pickfile('lithic_shotpicks/47_primary.info', 'incr', 25, outliers=5, maxrows=22)
p50 = pf.Pickfile('lithic_shotpicks/50_primary.info', 'incr', 0, outliers=5, maxrows=22)
p51 = pf.Pickfile('lithic_shotpicks/51_primary.info', 'incr', 0, outliers=5, maxrows=22)
p53 = pf.Pickfile('lithic_shotpicks/53_primary.info', 'incr', 0, outliers=5, maxrows=22)
primary_shots = [p29, p30, p31, p32, p33, p34, p35, p36, p37, p39, p40, p41, p42, p43, p44, p45, p46, p47, p50, p51, p53]

s29 = pf.Pickfile('lithic_shotpicks/29_secondary.info', 'incr', 'secondary', outliers=3, maxrows=22)
s30 = pf.Pickfile('lithic_shotpicks/30_secondary.info', 'incr', 'secondary', outliers=3, maxrows=22)
s31 = pf.Pickfile('lithic_shotpicks/31_secondary.info', 'incr', 'secondary', outliers=3, maxrows=22)
s32 = pf.Pickfile('lithic_shotpicks/32_secondary.info', 'incr', 'secondary', outliers=3, maxrows=22)
s33 = pf.Pickfile('lithic_shotpicks/33_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s34 = pf.Pickfile('lithic_shotpicks/34_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s35 = pf.Pickfile('lithic_shotpicks/35_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s36 = pf.Pickfile('lithic_shotpicks/36_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s37 = pf.Pickfile('lithic_shotpicks/37_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s39 = pf.Pickfile('lithic_shotpicks/39_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s40 = pf.Pickfile('lithic_shotpicks/40_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s41 = pf.Pickfile('lithic_shotpicks/41_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s42 = pf.Pickfile('lithic_shotpicks/42_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s43 = pf.Pickfile('lithic_shotpicks/43_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s44 = pf.Pickfile('lithic_shotpicks/44_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s45 = pf.Pickfile('lithic_shotpicks/45_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s46 = pf.Pickfile('lithic_shotpicks/46_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s47 = pf.Pickfile('lithic_shotpicks/47_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s50 = pf.Pickfile('lithic_shotpicks/50_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s51 = pf.Pickfile('lithic_shotpicks/51_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
s53 = pf.Pickfile('lithic_shotpicks/53_secondary.info', 'incr', 'secondary', outliers=5, maxrows=22)
secondary_shots = [s29, s30, s31, s32, s33, s34, s35, s36, s37, s39, s40, s41, s42, s43, s44, s45, s46, s47, s50, s51, s53]

primary_lithic_fullstack = pf.Pickfile('pickdata_lithic/fullstack_stacked_primary.info', 'incr', 0, 3, maxrows=22)
secondary_lithic_fullstack = pf.Pickfile('pickdata_lithic/fullstack_stacked_secondary.info', 'incr', 0, 3, maxrows=22)

p29_30_31 = pf.Pickfile('lithic_smallstacks/29_30_31_primary.info', 'decr', 115, outliers=3, maxrows=22)
p32_33_34 = pf.Pickfile('lithic_smallstacks/32_33_34_primary.info', 'decr', 100, outliers=3, maxrows=22)
p35_36 = pf.Pickfile('lithic_smallstacks/35_36_primary.info', 'decr', 85, outliers=3, maxrows=22)
p37_39_40 = pf.Pickfile('lithic_smallstacks/37_39_40_primary.info', 'decr', 70, outliers=3, maxrows=22)
p41_42 = pf.Pickfile('lithic_smallstacks/41_42_primary.info', 'decr', 55, outliers=3, maxrows=22)
p43_44 = pf.Pickfile('lithic_smallstacks/43_44_primary.info', 'decr', 40, outliers=3, maxrows=22)
p45_46_47 = pf.Pickfile('lithic_smallstacks/45_46_47_primary.info', 'decr', 25, outliers=3, maxrows=22)
p50_51_53 = pf.Pickfile('lithic_smallstacks/50_51_53_primary.info', 'decr', 0, outliers=3, maxrows=22)
primary_lithic_smallstacks = [p29_30_31, p32_33_34, p35_36, p37_39_40, p41_42, p43_44, p45_46_47, p50_51_53]

s29_30_31 = pf.Pickfile('lithic_smallstacks/29_30_31_secondary.info', 'decr', 115, outliers=3, maxrows=22)
s32_33_34 = pf.Pickfile('lithic_smallstacks/32_33_34_secondary.info', 'decr', 100, outliers=3, maxrows=22)
s35_36 = pf.Pickfile('lithic_smallstacks/35_36_secondary.info', 'decr', 85, outliers=3, maxrows=22)
s37_39_40 = pf.Pickfile('lithic_smallstacks/37_39_40_secondary.info', 'decr', 70, outliers=3, maxrows=22)
s41_42 = pf.Pickfile('lithic_smallstacks/41_42_secondary.info', 'decr', 55, outliers=3, maxrows=22)
s43_44 = pf.Pickfile('lithic_smallstacks/43_44_secondary.info', 'decr', 40, outliers=3, maxrows=22)
s45_46_47 = pf.Pickfile('lithic_smallstacks/45_46_47_secondary.info', 'decr', 25, outliers=3, maxrows=22)
s50_51_53 = pf.Pickfile('lithic_smallstacks/50_51_53_secondary.info', 'decr', 0, outliers=3, maxrows=22)
secondary_lithic_smallstacks = [s29_30_31, s32_33_34, s35_36, s37_39_40, s41_42, s43_44, s45_46_47, s50_51_53]

attn_opt, attn_cov = pa.attn_fit([primary_lithic_fullstack])
# Next we'll fit attenuation using picks from the shot_loc_data_primary array
# attn_opt_new, attn_cov = attn_fit(shot_loc_data_primary)


inversion_results = pa.inv1_fit([primary_lithic_fullstack])
incidence_angle = pa.inc_angle(primary_lithic_fullstack, inversion_results[0])

# depth, vel = pa.inv1_depth(np.arange(0, 200, 0.1), inversion_results[0])
# np.save('tmp/depthvel_lithic.npy', np.array([depth, vel]))
# plt.plot(vel, depth)
# plt.ylim([0, 40])
# plt.gca().invert_yaxis()
# plt.show()


def source_amp(primary, attn_coeff, inc_angle):
    geom_corr = np.cos(inc_angle)/(primary.dist)
    attn_corr = np.exp(-attn_coeff*primary.dist)
    return primary.min/attn_corr/geom_corr


# pair_amplitude = pa.pair_source_amplitudes(primary_lithic_fullstack)
# pair_amplitude = pair_amplitude[1:]
# plt.scatter(pair_amplitude,np.zeros(len(pair_amplitude)))
# pair_mean = np.mean(pair_amplitude)
# pair_std = np.std(pair_amplitude)
#
# dir_lin_amp_results = pa.dir_lin_source_amplitudes(primary_lithic_fullstack)
# dir_lin_amp = np.exp(dir_lin_amp_results[1])
# dir_lin_attn = -dir_lin_amp_results[0]
# plt.scatter(dir_lin_amp,0)
# plt.show()


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
for i in range(len(primary_lithic_smallstacks)):
    # pair = pa.reflectivity(primary_lithic_smallstacks[i], secondary_lithic_smallstacks[i], 'pair', attn_coeff=0.27e-3, polarity='min')
    # if pair != 'No pairs found':
    #     ref_shots_pair_smallstack.append(pair)
    #     plt.plot(np.rad2deg(primary_lithic_smallstacks[i].angle), pair, zorder=1, marker='o', markersize=5, linestyle='none')
    # ref_shots_dir_lin_smallstack.append(pa.reflectivity(primary_lithic_smallstacks[i], secondary_lithic_smallstacks[i], 'dir_lin', attn_coeff=0.27e-3, polarity='min'))
    # plt.plot(np.rad2deg(primary_lithic_smallstacks[i].angle), ref_shots_dir_lin_smallstack[i], zorder=1, marker='o', markersize=5, linestyle='none')
    ref_shots_simple_smallstack.append(pa.reflectivity(primary_lithic_smallstacks[i], secondary_lithic_smallstacks[i], 'simple', attn_coeff=2.7e-4, polarity='min'))
    plt.plot(np.rad2deg(primary_lithic_smallstacks[i].angle), ref_shots_simple_smallstack[i], zorder=1, marker='o', markersize=5, linestyle='none')

ref_stack_simple = pa.reflectivity(primary_lithic_fullstack, secondary_lithic_fullstack, 'simple', attn_coeff=2.7e-4, polarity='min')

# # Plot reflectivity as a function of angle
plt.grid()
# plt.scatter(np.rad2deg(primary_lithic_fullstack.angle), ref_stack_simple, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_lithic_fullstack.angle), ref_lithic_stack_pair, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_lithic_fullstack.angle), ref_lithic_stack_dir_lin, zorder=1, s=20)
# plt.errorbar(np.rad2deg(primary_lithic_fullstack.angle), ref_lithic_stack, yerr=[ref_low_error, ref_high_error], zorder=0, fmt='none', ecolor='k', capsize=3)
plt.ylim([-0.5, 100])
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
