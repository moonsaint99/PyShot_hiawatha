import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pick_gridsearch as gs
import time

# This script will use picked amplitudes of the primary and secondary seismic
# arrivals to calculate the ratio of the two amplitudes.


pvel = 3800

import pickfile as pf
import pickanalysis as pa


primary_72 = pf.Pickfile('pickdata_aquifer/72_wind_primary.info', 'decr', 195)
primary_73 = pf.Pickfile('pickdata_aquifer/73_wind_primary.info', 'decr', 195)
primary_74 = pf.Pickfile('pickdata_aquifer/74_wind_primary.info', 'decr', 195)
primary_80 = pf.Pickfile('pickdata_aquifer/80_wind_primary.info', 'decr', 60, outliers=3)
primary_81 = pf.Pickfile('pickdata_aquifer/81_wind_primary.info', 'decr', 60, outliers=3)
primary_82 = pf.Pickfile('pickdata_aquifer/82_wind_primary.info', 'decr', 60, outliers=3)


# For some of these shots, we don't have an accurate GPS or measuring tape estimate of where the shots were fired.
# For these shots, we'll run a linear regression on the primary arrival timing curve for the shots of known location
# and use that to estimate the location of the shots of unknown location.

# First we can concatenate the arrays for tmin and dist for the shots of known location
primary_known_tmin = np.concatenate((primary_72.tmin, primary_73.tmin, primary_74.tmin, primary_80.tmin, primary_81.tmin, primary_82.tmin))
primary_known_dist = np.concatenate((primary_72.dist, primary_73.dist, primary_74.dist, primary_80.dist, primary_81.dist, primary_82.dist))

# Next we can run a linear regression on the known data
primary_known_slope, primary_known_intercept = sp.stats.linregress(primary_known_tmin, primary_known_dist)[0:2]
# print('Slope: ' + str(primary_known_slope))
# print('Intercept: ' + str(primary_known_intercept))

# Now we regenerate the pickfiles with time delay corrections for the direct arrivals
primary_72 = pf.Pickfile('pickdata_aquifer/72_wind_primary.info', 'decr', 195)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_73 = pf.Pickfile('pickdata_aquifer/73_wind_primary.info', 'decr', 195)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_74 = pf.Pickfile('pickdata_aquifer/74_wind_primary.info', 'decr', 195)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_80 = pf.Pickfile('pickdata_aquifer/80_wind_primary.info', 'decr', 60, 3)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_81 = pf.Pickfile('pickdata_aquifer/81_wind_primary.info', 'decr', 60, 3)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_82 = pf.Pickfile('pickdata_aquifer/82_wind_primary.info', 'decr', 60, 3)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_stack_onspread = pf.Pickfile('pickdata_aquifer/stack_onspread_primary.info', 'incr', 0, 3)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_stack_offspread = pf.Pickfile('pickdata_aquifer/stack_offspread_primary.info', 'incr', 80, 0, maxrows=24)#, timecorrection=primary_known_intercept/primary_known_slope)

# Now we can use the linear regression to estimate the location of the shots of unknown location
# Right now we've put in 125 based on the results of our regression
primary_76 = pf.Pickfile('pickdata_aquifer/76_wind_primary.info', 'decr', 'unknown', pvel=primary_known_slope, pintercept=primary_known_intercept)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_77 = pf.Pickfile('pickdata_aquifer/77_wind_primary.info', 'decr', 'unknown', pvel=primary_known_slope, pintercept=primary_known_intercept)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_78 = pf.Pickfile('pickdata_aquifer/78_wind_primary.info', 'decr', 'unknown', pvel=primary_known_slope, pintercept=primary_known_intercept)#, timecorrection=primary_known_intercept/primary_known_slope)
primary_79 = pf.Pickfile('pickdata_aquifer/79_wind_primary.info', 'decr', 'unknown', pvel=primary_known_slope, pintercept=primary_known_intercept)#, timecorrection=primary_known_intercept/primary_known_slope)

aquifer_primary_list = [primary_72, primary_73, primary_74, primary_76, primary_77, primary_78, primary_79, primary_80, primary_81, primary_82]

secondary_74 = pf.Pickfile('pickdata_aquifer/74_wind_secondary.info', 'decr', 'secondary')#, timecorrection=primary_known_intercept/primary_known_slope)
secondary_73 = pf.Pickfile('pickdata_aquifer/73_wind_secondary.info', 'decr', 'secondary')#, timecorrection=primary_known_intercept/primary_known_slope)
secondary_72 = pf.Pickfile('pickdata_aquifer/72_wind_secondary.info', 'decr', 'secondary')#, timecorrection=primary_known_intercept/primary_known_slope)
secondary_76 = pf.Pickfile('pickdata_aquifer/76_wind_secondary.info', 'decr', 'secondary')#, timecorrection=primary_known_intercept/primary_known_slope) # This value of 125 was computed using a linear interpolation
secondary_77 = pf.Pickfile('pickdata_aquifer/77_wind_secondary.info', 'decr', 'secondary')#, timecorrection=primary_known_intercept/primary_known_slope)
secondary_78 = pf.Pickfile('pickdata_aquifer/78_wind_secondary.info', 'decr', 'secondary')#, timecorrection=primary_known_intercept/primary_known_slope)
secondary_79 = pf.Pickfile('pickdata_aquifer/79_wind_secondary.info', 'decr', 'secondary')#, timecorrection=primary_known_intercept/primary_known_slope)
secondary_80 = pf.Pickfile('pickdata_aquifer/80_wind_secondary.info', 'decr', 'secondary')#, 3, timecorrection=primary_known_intercept/primary_known_slope)
secondary_81 = pf.Pickfile('pickdata_aquifer/81_wind_secondary.info', 'decr', 'secondary')#, 3, timecorrection=primary_known_intercept/primary_known_slope)
secondary_82 = pf.Pickfile('pickdata_aquifer/82_wind_secondary.info', 'decr', 'secondary')#, 3, timecorrection=primary_known_intercept/primary_known_slope)
secondary_stack_onspread = pf.Pickfile('pickdata_aquifer/stack_onspread_secondary.info', 'incr', 'secondary', 3)#, timecorrection=primary_known_intercept/primary_known_slope)
secondary_stack_offspread = pf.Pickfile('pickdata_aquifer/stack_offspread_secondary.info', 'incr', 'secondary', 0, maxrows=24)#, timecorrection=primary_known_intercept/primary_known_slope)

aquifer_secondary_list = [secondary_72, secondary_73, secondary_74, secondary_76, secondary_77, secondary_78, secondary_79, secondary_80, secondary_81, secondary_82]

print(primary_stack_onspread)
print(primary_stack_offspread)
print([primary_stack_onspread]+[primary_stack_offspread])
attn_opt, attn_cov = pa.attn_fit([primary_stack_onspread])


# Next we'll fit attenuation using picks from the shot_loc_data_primary array
# attn_opt_new, attn_cov = attn_fit(shot_loc_data_primary)


# def primary_amp(primary, attn_coeff, inc_angle):
#     geom_corr = np.cos(inc_angle)/primary.dist_no_outliers
#     attn_corr = np.exp(-attn_coeff*primary.dist_no_outliers)
#     return primary.min_no_outliers/attn_corr/geom_corr*100
def primary_amp(primary, attn_coeff, inc_angle):
    geom_corr = np.cos(inc_angle)/primary.dist
    attn_corr = np.exp(-attn_coeff*primary.dist)
    return primary.min/attn_corr/geom_corr*100

inversion_results = pa.inv1_fit(aquifer_primary_list)
# incidence_angle = inc_angle(primary_fullstack, inversion_results[0])
incidence_angle = pa.inc_angle(primary_stack_onspread, inversion_results[0])
incidence_angle_long = pa.inc_angle(primary_stack_offspread, inversion_results[0])

depth, vel = pa.inv1_depth(np.arange(0, 200, 0.1), inversion_results[0])
# Pickle depth and vel for later use
np.save('tmp/depthvel_aquifer.npy', np.array([depth, vel]))
plt.plot(vel, depth)
plt.ylim([0, 40])
plt.gca().invert_yaxis()
plt.xlabel('Velocity (m/s)')
plt.ylabel('Depth (m)')
plt.title('Firn Velocity Profile')
plt.show()

# amp_72 = primary_amp(primary_72, attn_opt[0], incidence_angle)
# amp_73 = primary_amp(primary_73, attn_opt[0], incidence_angle)
# amp_74 = primary_amp(primary_74, attn_opt[0], incidence_angle)
# amp_76 = primary_amp(primary_76, attn_opt[0], incidence_angle)
# amp_77 = primary_amp(primary_77, attn_opt[0], incidence_angle)
# amp_78 = primary_amp(primary_78, attn_opt[0], incidence_angle)
# amp_79 = primary_amp(primary_79, attn_opt[0], incidence_angle)
# amp_80 = primary_amp(primary_80, attn_opt[0], incidence_angle)
# amp_81 = primary_amp(primary_81, attn_opt[0], incidence_angle)
# amp_82 = primary_amp(primary_82, attn_opt[0], incidence_angle)
amp_stack_onspread = primary_amp(primary_stack_onspread, attn_opt[0], incidence_angle)
amp_stack_offspread = primary_amp(primary_stack_offspread, attn_opt[0], incidence_angle_long)


def reflectivity(primary, secondary, attn_coeff, polarity='max'):
    path_length = 2*np.sqrt(primary.dist**2 + 406**2)
    geom_corr = np.cos(primary.angle)/path_length
    attn_corr = np.exp(-attn_coeff*primary.dist)
    if polarity == 'max':
        return secondary.max/primary_amp(primary, attn_coeff, pa.inc_angle(primary, inversion_results[0]))/attn_corr/geom_corr
    elif polarity == 'min':
        return secondary.min/primary_amp(primary, attn_coeff, pa.inc_angle(primary, inversion_results[0]))/attn_corr/geom_corr


# ref_72 = reflectivity(primary_72, secondary_72, attn_opt[0])
# ref_73 = reflectivity(primary_73, secondary_73, attn_opt[0])
# ref_74 = reflectivity(primary_74, secondary_74, attn_opt[0])
# ref_76 = reflectivity(primary_76, secondary_76, attn_opt[0])
# ref_77 = reflectivity(primary_77, secondary_77, attn_opt[0])
# ref_78 = reflectivity(primary_78, secondary_78, attn_opt[0])
# ref_79 = reflectivity(primary_79, secondary_79, attn_opt[0])
# ref_80 = reflectivity(primary_80, secondary_80, attn_opt[0])
# ref_81 = reflectivity(primary_81, secondary_81, attn_opt[0])
# ref_82 = reflectivity(primary_82, secondary_82, attn_opt[0])
ref_stack_onspread = reflectivity(primary_stack_onspread, secondary_stack_onspread, attn_opt[0])
ref_stack_offspread = reflectivity(primary_stack_offspread, secondary_stack_offspread, attn_opt[0])


# plt.scatter(np.rad2deg(primary_72.angle), ref_72, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_73.angle), ref_73, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_74.angle), ref_74, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_76.angle), ref_76, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_77.angle), ref_77, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_78.angle), ref_78, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_79.angle), ref_79, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_80.angle), ref_80, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_81.angle), ref_81, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_82.angle), ref_82, zorder=1, s=20)
plt.scatter(np.rad2deg(primary_stack_onspread.angle), ref_stack_onspread, zorder=0, s=20)
plt.scatter(np.rad2deg(primary_stack_offspread.angle), ref_stack_offspread, zorder=0, s=20)
plt.ylim([-1, 1])
plt.title('Reflectivity as fxn of angle')
plt.ylabel('Reflectivity')
plt.xlabel('Angle (deg)')
plt.grid()
plt.show()


depth = pa.inv1_depth(np.arange(1, 200, 1), inversion_results[0])

#depthvel=depthvel_fit(aquifer_primary_list, aquifer_secondary_list)[0]


def zerodip_depth(reflectiontiming, offset, velocity):
    return np.sqrt((reflectiontiming*velocity/2)**2 - (offset/2)**2)


# Plot theoretical correct time difference between primary and secondary
# We'll do this using the primary arrival travel time curve generated by the inversion
refl_tdiff = np.array([])
refl_tdiff_inv = np.array([])
refl_timingarray_0deg = np.array([])
refl_timingarray_10deg = np.array([])
refl_timingarray_20deg = np.array([])
refl_deep = np.array([])
for i in range(1, 200, 1):
    # refl_timing = 2*np.sqrt(i**2 + 405**2)/3830
    refl_timingarray_0deg = np.append(refl_timingarray_0deg, pa.refl_time(i, 0, primary_known_slope, 405))
    refl_timingarray_10deg = np.append(refl_timingarray_10deg, pa.refl_time(i, -10, primary_known_slope, 405))
    refl_timingarray_20deg = np.append(refl_timingarray_20deg, pa.refl_time(i, -20, primary_known_slope, 405))
    # inversion_results[0][2], inversion_results[0][3], inversion_results[0][4]))

# Plot data vs theoretical reflection traveltime
# plt.plot(primary_stack.dist, secondary_stack.tmin)
#plt.plot(primary_stack.dist, secondary_stack.tmax)
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
plt.plot(primary_stack_onspread.dist, secondary_stack_onspread.tmax)
plt.plot(primary_stack_offspread.dist, secondary_stack_offspread.tmax)
plt.plot(np.arange(1,200,1), refl_timingarray_0deg, zorder=0)
plt.plot(np.arange(1,200,1), refl_timingarray_10deg, zorder=0)
plt.plot(np.arange(1,200,1), refl_timingarray_20deg, zorder=0)
# plt.legend(['Data', 'Theoretical'])
plt.title('Data vs theoretical reflection traveltime')
# plt.legend(['Shot 1', 'Shot 2', 'Shot 3', '0 deg bed', '10 deg bed', '20 deg bed'])
plt.xlabel('Offset (m)')
plt.ylabel('Traveltime (s)')
plt.grid()
plt.show()

#gs.depthvel_gridsearch_plot(aquifer_primary_list, aquifer_secondary_list)

# # Plot zero dip reflector depth
# plt.title('Reflector depth assuming zero reflector dip')
# #plt.plot(primary_stack.dist/2, zerodip_depth(secondary_stack.tmin, secondary_stack.dist, 3800))
# plt.plot(primary_73.dist/2, zerodip_depth(secondary_73.tmax, secondary_73.dist, primary_known_slope))
# plt.plot(primary_74.dist/2, zerodip_depth(secondary_74.tmax, secondary_74.dist, primary_known_slope))
# plt.ylim(400, 450)
# plt.show()


# Plot direct arrival time vs offset
plt.scatter(primary_72.dist, primary_72.tmin, s=16, marker=".")
plt.scatter(primary_74.dist, primary_74.tmin, s=16, marker=".")
plt.scatter(primary_73.dist, primary_73.tmin, s=16, marker=".")
plt.scatter(primary_76.dist, primary_76.tmin, s=16, marker=".")
plt.scatter(primary_77.dist, primary_77.tmin, s=16, marker=".")
plt.scatter(primary_78.dist, primary_78.tmin, s=16, marker=".")
plt.scatter(primary_79.dist, primary_79.tmin, s=16, marker=".")
plt.scatter(primary_80.dist, primary_80.tmin, s=16, marker=".")
plt.scatter(primary_81.dist, primary_81.tmin, s=16, marker=".")
plt.scatter(primary_82.dist, primary_82.tmin, s=16, marker=".")
plt.scatter(primary_stack_onspread.dist, primary_stack_onspread.tmin, s=50, marker="o", zorder=0, facecolors='none', edgecolors='k')
plt.scatter(primary_stack_offspread.dist, primary_stack_offspread.tmin, s=50, marker="o", zorder=0, facecolors='none', edgecolors='k')
plt.plot(np.arange(0,0.06,0.001)*primary_known_slope+primary_known_intercept, np.arange(0,0.06,0.001), zorder=0, linewidth=0.5)
plt.title('Direct arrival time vs offset')
plt.xlabel('Offset (m)')
plt.ylabel('Traveltime (s)')
plt.grid()
plt.show()