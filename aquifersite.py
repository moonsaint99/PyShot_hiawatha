import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import firn_analysis as fa
import pick_gridsearch as gs
import time
import pickanalysis as pa
import op_pickfile as opf

# This script will use picked amplitudes of the primary and secondary seismic
# arrivals to calculate the ratio of the two amplitudes.


pvel = 3800

import su_pickfile as pf
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
primary_stack_offspread = pf.Pickfile('pickdata_aquifer/stack_offspread_primary.info', 'incr', 80, 0, maxrows=23)#, timecorrection=primary_known_intercept/primary_known_slope)

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
secondary_stack_offspread = pf.Pickfile('pickdata_aquifer/stack_offspread_secondary.info', 'incr', 'secondary', 0, maxrows=23)#, timecorrection=primary_known_intercept/primary_known_slope)

aquifer_secondary_list = [secondary_72, secondary_73, secondary_74, secondary_76, secondary_77, secondary_78, secondary_79, secondary_80, secondary_81, secondary_82]

aquifer_good_primary = opf.opPickfile('primary_shotpicks_output/aquifer_stack.csv')
aquifer_good_secondary = opf.opPickfile('secondary_shotpicks_output/aquifer_stack.csv')
aquifer_bigstack_primary = [aquifer_good_primary]
aquifer_bigstack_secondary = [aquifer_good_secondary]

# print(primary_stack_onspread)
# print(primary_stack_offspread)
# print([primary_stack_onspread]+[primary_stack_offspread])
attn_opt, attn_cov = pa.attn_fit([primary_stack_onspread])

inversion_results = fa.double_linear_exponential_fit(aquifer_primary_list)
incidence_angle = fa.inc_angle(primary_stack_onspread, inversion_results[0])
incidence_angle_long = fa.inc_angle(primary_stack_offspread, inversion_results[0])

depth, vel = fa.firn_depth_vs_velocity(np.arange(0, 200, 0.1), inversion_results[0])
# Pickle depth and vel for later use
np.save('tmp/depthvel_aquifer.npy', np.array([depth, vel]))
plt.plot(vel, depth)
plt.ylim([0, 40])
plt.gca().invert_yaxis()
plt.xlabel('Velocity (m/s)')
plt.ylabel('Depth (m)')
plt.title('Firn Velocity Profile')
plt.show()

ref_bigstack = pa.reflectivity(aquifer_good_primary, aquifer_good_secondary, 'simple', inversion_results, attn_coeff=2.7e-4, polarity='max')

plt.scatter(np.rad2deg(aquifer_good_primary.angle), ref_bigstack[0], zorder=0, s=20)
# plt.ylim([-1, 1])
plt.title('Reflectivity as fxn of angle')
plt.ylabel('Reflectivity')
plt.xlabel('Angle (deg)')
plt.grid()
plt.show()